"""CPU regressions: real prefix method + tiny model, no checkpoints/downloads required."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.finetune_indextts25 import _common as io
from tools.finetune_indextts25 import train_lora as train
from tools.finetune_indextts25 import prepare_manifest as manifest
from tools.finetune_indextts25 import precompute_features as precompute
from tools.finetune_indextts25 import eval_suite as evaluation
from tools.finetune_indextts25 import score_wer as scoring
from tools.finetune_indextts25.merge_lora import merge_weights

torch.set_num_threads(1)
BASE = {"config_sha256": "config", "gpt_sha256": "base", "feature_format": io.FEATURE_FORMAT}


def actual_prefix_method():
    # Execute only this real method, avoiding optional dependencies/model construction.
    path = ROOT / "indextts/gpt/model_v2.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnifiedVoice")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_gpt_inputs")
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


class Pos(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(32, 8)

    def forward(self, tokens):
        return self.emb(torch.arange(tokens.shape[1], device=tokens.device))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.c_attn = nn.Linear(8, 8)
        block.attn.c_proj = nn.Linear(8, 8)
        self.h = nn.ModuleList([block])
        self.drop = nn.Dropout(.15)
        self.config = SimpleNamespace(n_positions=64)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        assert gradient_checkpointing_kwargs == {"use_reentrant": False}

    def forward(self, inputs_embeds, attention_mask, use_cache, return_dict):
        assert not use_cache and return_dict
        self.last_attention = attention_mask.detach().clone()
        self.last_input = inputs_embeds.detach().clone()
        # Causal fixture: no future token can influence an earlier output.
        x = (inputs_embeds * attention_mask.unsqueeze(-1)).cumsum(1)
        block = self.h[0].attn
        return SimpleNamespace(last_hidden_state=self.drop(block.c_proj(torch.tanh(block.c_attn(x)))))


class TinyVoice(nn.Module):
    prepare_gpt_inputs = actual_prefix_method()
    start_text_token, stop_text_token, start_mel_token, stop_mel_token = 0, 1, 6, 7
    number_mel_codes, spk_cond_mode = 8, "campplus"

    def __init__(self):
        super().__init__()
        self.gpt = TinyGPT()
        self.text_embedding = nn.Embedding(32, 8)
        self.mel_embedding = nn.Embedding(8, 8)
        self.lang_embedding = nn.Embedding(5, 8)
        self.text_pos_embedding, self.mel_pos_embedding = Pos(), Pos()
        self.spk_emb_proj = nn.Linear(192, 8)
        self.emovec_layer = nn.Linear(1024, 8)
        self.emo_layer = nn.Linear(8, 8)
        self.final_norm = nn.LayerNorm(8)
        self.mel_head = nn.Linear(8, 8)

    def get_emo_conditioning(self, emo, lengths):
        assert emo.shape[-1] == int(lengths[0])
        return emo.mean(-1)


def feature(**updates):
    value = {"format": io.FEATURE_FORMAT, "base_identity": BASE, "split": "train",
             "text_tokens": torch.tensor([5, 2, 3]), "mel_codes": torch.tensor([1, 2, 3]),
             "lang_id": 3, "language_token_id": 5, "campplus": torch.randn(192),
             "emo_condition": torch.randn(4, 1024), "audio_sha256": "target",
             "ref_audio_sha256": "reference", "pipeline_source": io.source_identity(),
             "extraction_id": "extractor"}
    value.update(updates)
    return value


def configured():
    torch.manual_seed(11)
    model = TinyVoice().eval()
    targets = train.inject(model, train.DEFAULT_TARGET, 2, 4., .1)
    for name in targets:
        nn.init.normal_(model.get_submodule(name).parametrizations.weight[0].B, std=.05)
    return model, targets


def save_adapter(path, model, targets):
    data = train.adapter_state(model, targets, {"base_identity": BASE, "rank": 2,
                               "alpha": 4., "dropout": .1, "step": 1})
    io.save_tensor(path, data)
    return data


def test_inference_tokenizer_prefix():
    calls = []
    tokenizer = SimpleNamespace(encode=lambda text, **kw: calls.append(text) or [5, 2])
    assert precompute.tokenize(SimpleNamespace(tokenizer=tokenizer), "HOLA", "es") == [5, 2]
    assert calls == ["<|es|> HOLA"]


def test_teacher_prefix_eos_and_causality():
    model = TinyVoice().eval()
    obj = feature()
    logits, targets = train.teacher_logits(model, obj, "cpu", False)
    assert targets.tolist() == [[1, 2, 3, 7]]
    assert logits.shape == (1, 4, 8)
    assert model.gpt.last_attention.shape[1] == model.gpt.last_input.shape[1]
    assert model.gpt.last_attention[0, 0] == 0  # Real inference's masked left pad.
    expected = (model.text_embedding(torch.tensor([0, 5, 2, 3, 1])) +
                model.text_pos_embedding.emb(torch.arange(5)) + model.lang_embedding(torch.tensor(3)))
    torch.testing.assert_close(model.gpt.last_input[:, 4:9], expected.unsqueeze(0))
    manual = F.cross_entropy(logits.reshape(-1, 8), targets.reshape(-1))
    torch.testing.assert_close(train.one_loss(model, obj, "cpu", False), manual)
    changed = {**obj, "mel_codes": torch.tensor([1, 2, 4])}
    other, _ = train.teacher_logits(model, changed, "cpu", False)
    torch.testing.assert_close(logits[:, :3], other[:, :3])
    logits.retain_grad()
    manual.backward()
    assert logits.grad[0, -1, 7] < 0  # EOS participates in loss.


@pytest.mark.parametrize("change", [
    {"format": "legacy"}, {"text_tokens": torch.tensor([2, 3])},
    {"mel_codes": torch.tensor([7])}, {"mel_codes": torch.tensor([9])},
    {"text_tokens": torch.tensor([5., 2.])}, {"lang_id": 99},
    {"campplus": torch.full((192,), float("nan"))},
    {"text_tokens": torch.tensor([5] * 32)}, {"emo_condition": torch.zeros(2, 12)},
])
def test_bad_features_fail_before_training(change):
    with pytest.raises(ValueError):
        train.validate_feature(feature(**change), TinyVoice(), BASE)


def test_combined_capacity_and_base_mismatch():
    model = TinyVoice()
    model.gpt.config.n_positions = 10
    with pytest.raises(ValueError, match="combined"):
        train.validate_feature(feature(), model)
    with pytest.raises(ValueError, match="base mismatch"):
        train.validate_feature(feature(), TinyVoice(), {"wrong": "base"})


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_adapter_device_and_fp32_master_parameters(device):
    model = nn.Sequential(nn.Linear(8, 4, device=device, dtype=torch.bfloat16))
    train.inject(model, "0", 2, 4., 0.)
    delta = model[0].parametrizations.weight[0]
    assert delta.A.device.type == device and delta.B.device.type == device
    assert delta.A.dtype == torch.float32 and delta.B.dtype == torch.float32
    assert not model[0].parametrizations.weight.original.requires_grad


def test_adapter_reload_and_merge_logits(tmp_path):
    model, targets = configured()
    obj = feature()
    model.eval()
    before, _ = train.teacher_logits(model, obj, "cpu", False)
    data = save_adapter(tmp_path / "a.pt", model, targets)
    torch.manual_seed(11)
    fresh = TinyVoice().eval()
    # Reproduce aliases created by post_init_gpt2_config during real inference.
    fresh.inference_model = nn.Module()
    fresh.inference_model.transformer = fresh.gpt
    metadata = train.load_adapter(fresh, tmp_path / "a.pt", BASE)
    json.dumps(metadata)  # No tensors or optimizer state leak into eval receipts.
    fresh.eval()
    loaded, _ = train.teacher_logits(fresh, obj, "cpu", False)
    torch.testing.assert_close(before, loaded)
    assert set(train.adapter_weights(fresh)) == set(data["state_dict"])
    merge_weights(fresh, targets)
    merged, _ = train.teacher_logits(fresh, obj, "cpu", False)
    torch.testing.assert_close(before, merged)
    assert not any("parametrizations" in name for name, _ in fresh.named_parameters())


@pytest.mark.parametrize("corrupt", ["extra", "missing", "shape", "nonfinite", "base", "format", "target", "rank", "step", "dropout"])
def test_corrupt_adapters_rejected(tmp_path, corrupt):
    model, targets = configured()
    path = tmp_path / "a.pt"
    data = save_adapter(path, model, targets)
    key = next(iter(data["state_dict"]))
    if corrupt == "extra":
        data["state_dict"]["mel_head.bias"] = torch.zeros(8)
    elif corrupt == "missing":
        del data["state_dict"][key]
    elif corrupt == "shape":
        data["state_dict"][key] = torch.zeros(1)
    elif corrupt == "nonfinite":
        data["state_dict"][key].fill_(float("nan"))
    elif corrupt == "base":
        data["base_identity"] = {}
    elif corrupt == "format":
        data["format"] = "indextts25-native-lora-v1"
    elif corrupt in {"rank", "step", "dropout"}:
        data[corrupt] = -1
    else:
        data["targets"] = ["missing"]
    io.save_tensor(path, data)
    fresh = TinyVoice()
    bias = fresh.mel_head.bias.detach().clone()
    with pytest.raises(ValueError):
        train.load_adapter(fresh, path, BASE)
    torch.testing.assert_close(bias, fresh.mel_head.bias)


def test_gradient_failure_does_not_step():
    parameter = nn.Parameter(torch.tensor([1.]))
    opt = torch.optim.AdamW([parameter])
    for value in (float("nan"), float("inf"), 0.):
        parameter.grad = torch.tensor([value])
        with pytest.raises(RuntimeError):
            train.step_optimizer(opt, [parameter], .5)
        assert parameter.item() == 1.


def test_evaluate_restores_mixed_modes_on_error(monkeypatch, tmp_path):
    model, _ = configured()
    model.train()
    model.emo_layer.eval()
    modes = [m.training for m in model.modules()]
    path = tmp_path / "f.pt"
    io.save_tensor(path, feature())
    def fail(*args):
        raise ValueError("test failure")
    monkeypatch.setattr(train, "one_loss", fail)
    with pytest.raises(ValueError):
        train.evaluate(model, [path], "cpu", False)
    assert modes == [m.training for m in model.modules()]


def run_args(tmp_path, output, steps, resume=None):
    return argparse.Namespace(features=str(tmp_path / "features"), dev_features=None,
        config="unused", model_dir="unused", output=str(output), device="cpu", bf16=False,
        rank=2, alpha=4., dropout=.1, target_regex=train.DEFAULT_TARGET, learning_rate=.01,
        max_steps=steps, grad_accum=2, warmup_steps=3, weight_decay=0., clip_grad=.5,
        save_every=2, eval_every=2, seed=20260912, max_items=0, resume=resume)


def setup_tiny_run(monkeypatch, tmp_path):
    directory = tmp_path / "features"
    directory.mkdir()
    for i in range(3):
        io.save_tensor(directory / f"{i}.pt", feature())
    io.write_json(directory / "feature_receipt.json", {"records": [
        {"row_id": str(i), "sha256": io.sha256(directory / f"{i}.pt")} for i in range(3)]})
    def load(*args):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            return TinyVoice().eval(), None
    monkeypatch.setattr(train, "load_gpt", load)
    monkeypatch.setattr(io, "base_identity", lambda *args: BASE)


def test_true_resume_matches_uninterrupted_training(monkeypatch, tmp_path):
    setup_tiny_run(monkeypatch, tmp_path)
    for name, steps, resume in (("full", 6, None), ("part", 2, None),
                ("resumed", 6, str(tmp_path / "part/adapter-step-000002.pt"))):
        out = tmp_path / name
        with io.output_lock(out, empty=True):
            train.run(run_args(tmp_path, out, steps, resume), out)
    full = io.load_tensor(tmp_path / "full/adapter-step-000006.pt")
    resumed = io.load_tensor(tmp_path / "resumed/adapter-step-000006.pt")
    for key in full["state_dict"]:
        torch.testing.assert_close(full["state_dict"][key], resumed["state_dict"][key], rtol=0, atol=0)
    assert full["training"]["sampler"] == resumed["training"]["sampler"]


def test_nan_training_saves_no_adapter(monkeypatch, tmp_path):
    setup_tiny_run(monkeypatch, tmp_path)
    monkeypatch.setattr(train, "one_loss", lambda *args: torch.tensor(float("nan"), requires_grad=True))
    out = tmp_path / "failed"
    with io.output_lock(out, empty=True), pytest.raises(RuntimeError, match="BEFORE"):
        train.run(run_args(tmp_path, out, 2), out)
    assert not list(out.glob("adapter*.pt"))
    assert json.loads((out / "run_receipt.json").read_text())["status"] == "failed_or_interrupted"


def test_resume_changed_dataset_rejected(monkeypatch, tmp_path):
    setup_tiny_run(monkeypatch, tmp_path)
    out = tmp_path / "part"
    with io.output_lock(out, empty=True):
        train.run(run_args(tmp_path, out, 2), out)
    io.save_tensor(tmp_path / "features/0.pt", feature())
    dest = tmp_path / "next"
    with io.output_lock(dest, empty=True), pytest.raises(ValueError, match="checksum"):
        train.run(run_args(tmp_path, dest, 4, str(out / "adapter-step-000002.pt")), dest)


def source_rows(tmp_path):
    rows = []
    for i in range(12):
        path = tmp_path / f"a{i}.wav"
        sf.write(path, np.sin(np.arange(8000) * (.02 + i * .001)) * .2, 16000)
        rows.append({"audio": path.name, "text": f"Frase número {i}", "speaker": "ana",
                     "session": f"s{i // 2}", "row_id": f"row{i}"})
    source = tmp_path / "raw.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows))
    return source, rows


def test_split_relative_paths_references_and_determinism(tmp_path):
    source, _ = source_rows(tmp_path)
    kw = dict(seed=3, train_frac=.6, dev_frac=.2)
    result = manifest.prepare(source, **kw)
    assert result == manifest.prepare(source, **kw)
    seen = set()
    for name, rows in result.items():
        assert rows
        hashes = {r["audio_sha256"] for r in rows}
        assert not hashes & seen
        seen.update(hashes)
        for row in rows:
            assert row["ref_audio_sha256"] in hashes
            assert not row["ref_is_target"]
            assert row["split"] == name
            assert Path(row["audio"]).is_absolute()


@pytest.mark.parametrize("field,value", [("row_id", "../escape"), ("text", None), ("speaker", ""), ("lang", "unknown")])
def test_invalid_manifest_metadata(tmp_path, field, value):
    source, rows = source_rows(tmp_path)
    rows[0][field] = value
    source.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError):
        manifest.prepare(source, seed=1, train_frac=.8, dev_frac=.1)


def test_duplicate_text_and_audio_stay_together(tmp_path):
    source, rows = source_rows(tmp_path)
    rows[4]["text"] = rows[0]["text"]
    rows[8]["audio"] = rows[2]["audio"]
    source.write_text("\n".join(json.dumps(r) for r in rows))
    result = manifest.prepare(source, seed=1, train_frac=.6, dev_frac=.2)
    where = {r["row_id"]: split for split, rs in result.items() for r in rs}
    assert where["row4"] == where["row0"]
    assert where["row8"] == where["row2"]


def test_too_few_groups_and_external_reference_fail(tmp_path):
    source, rows = source_rows(tmp_path)
    for row in rows:
        row["session"] = "same"
    source.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="too few"):
        manifest.prepare(source, seed=1, train_frac=.8, dev_frac=.1)
    rows[0]["ref_audio"] = "nonexistent.wav"
    source.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(FileNotFoundError):
        manifest.prepare(source, seed=1, train_frac=.8, dev_frac=.1)


def test_audio_never_silently_truncated(monkeypatch):
    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(load=lambda *a, **kw: (np.zeros(240001), 16000)))
    with pytest.raises(ValueError, match="not truncated"):
        precompute.audio16("unused.wav")


def test_evaluation_persists_failures_and_truncations(tmp_path):
    class FakeTTS:
        gpt = nn.Linear(1, 1)
        def infer(self, reference, text, output, lang, **kwargs):
            if text == "fail":
                raise RuntimeError("synthesis failure")
            if text == "truncate":
                warnings.warn("max_mel_tokens reached", RuntimeWarning)
            sf.write(output, np.sin(np.arange(4000) * .03) * .2, 16000)
    rows = [{"id": f"id{i}", "text": t, "speaker_ref": "ref", "reference_sha256": "hash"}
            for i, t in enumerate(("good", "fail", "truncate"))]
    records = evaluation.synthesize(FakeTTS(), rows, [42], tmp_path, {}, {}, "cpu")
    assert [r["status"] for r in records] == ["ok", "failed", "truncated"]
    assert len(json.loads((tmp_path / "receipt.json").read_text())["records"]) == 3
    assert records[0]["rtf"] > 0
    with pytest.raises(FileExistsError):
        evaluation.synthesize(FakeTTS(), rows, [42], tmp_path, {}, {}, "cpu")


@pytest.mark.parametrize("records,expected", [
    ([{"id": "a", "seed": 1, "text": "dos palabras", "ok": False}], 1.),
    ([{"id": "a", "seed": 1, "text": "dos palabras", "ok": True},
      {"id": "b", "seed": 1, "text": "tres palabras más", "ok": False}], .6),
    ([{"id": "a", "seed": 1, "text": "pendiente", "status": "pending"}], 1.),
])
def test_failed_generations_count_in_corpus_denominator(records, expected):
    result = scoring.score(records, lambda r: "dos palabras")
    assert result["corpus_wer"] == expected
    assert result["reference_words"] == sum(len(scoring.norm(r["text"]).split()) for r in records)


def test_asr_error_empty_and_duplicate_scores():
    row = {"id": "a", "seed": 1, "text": "hola mundo", "ok": True}
    def fail(row):
        raise RuntimeError("ASR offline")
    assert scoring.score([row], fail)["corpus_wer"] == 1.
    assert scoring.score([row], fail)["failed_records"] == 1
    with pytest.raises(ValueError):
        scoring.score([], fail)
    with pytest.raises(ValueError):
        scoring.score([row, row], fail)
    assert scoring.norm("¡Niño, sí!") == "niño sí"


def test_atomic_json_and_nonempty_output(tmp_path):
    path = tmp_path / "r.json"
    io.write_json(path, {"ok": 1})
    with pytest.raises(ValueError):
        io.write_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"ok": 1}
    with pytest.raises(FileExistsError):
        with io.output_lock(tmp_path, empty=True):
            pass


@pytest.mark.parametrize("script", ["train_lora", "precompute_features", "prepare_manifest", "eval_suite", "score_wer", "merge_lora"])
def test_cli_help_direct_and_module(script):
    direct = subprocess.run([sys.executable, str(ROOT / f"tools/finetune_indextts25/{script}.py"), "--help"],
                            cwd="/tmp", capture_output=True, text=True, timeout=20)
    assert direct.returncode == 0, direct.stderr
    module = subprocess.run([sys.executable, "-m", f"tools.finetune_indextts25.{script}", "--help"],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert module.returncode == 0, module.stderr


def test_fraction_roundoff_and_invalid_fractions(tmp_path):
    source, _ = source_rows(tmp_path)
    result = manifest.prepare(source, seed=2, train_frac=.9, dev_frac=.1)
    assert not result["test"]
    for a, b in ((0, .1), (1.1, 0), (.8, .3), (.8, -.1), (float("nan"), .1)):
        with pytest.raises(ValueError):
            manifest.prepare(source, seed=2, train_frac=a, dev_frac=b)


def test_duplicate_row_ids_fail(tmp_path):
    source, rows = source_rows(tmp_path)
    rows[0]["row_id"] = rows[1]["row_id"]
    source.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="duplicate row_id"):
        manifest.prepare(source, seed=2, train_frac=.8, dev_frac=.1)


def test_resume_changed_hyperparameters_rejected(monkeypatch, tmp_path):
    setup_tiny_run(monkeypatch, tmp_path)
    out = tmp_path / "part"
    with io.output_lock(out, empty=True):
        train.run(run_args(tmp_path, out, 2), out)
    dest = tmp_path / "next"
    args = run_args(tmp_path, dest, 4, str(out / "adapter-step-000002.pt"))
    args.learning_rate = .02
    with io.output_lock(dest, empty=True), pytest.raises(ValueError, match="resume requires"):
        train.run(args, dest)


def test_resume_without_optimizer_state_rejected(monkeypatch, tmp_path):
    setup_tiny_run(monkeypatch, tmp_path)
    model, targets = configured()
    path = tmp_path / "weights_only.pt"
    save_adapter(path, model, targets)
    dest = tmp_path / "next"
    with io.output_lock(dest, empty=True), pytest.raises(ValueError, match="resume requires"):
        train.run(run_args(tmp_path, dest, 4, str(path)), dest)
