#!/usr/bin/env python3
"""Single-GPU, EOS-supervised IndexTTS 2.5 adaptation. No production writes."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
import random
import re
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

try:
    from . import _common as io
except ImportError:
    import _common as io

DEFAULT_TARGET = r"^gpt\.h\.\d+\.(attn\.(c_attn|c_proj)|mlp\.(c_fc|c_proj))$"


class LoRAWeight(nn.Module):
    """Native low-rank weight update; dropout is on A, NOT PEFT input dropout."""
    def __init__(self, weight, rank, alpha, dropout):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid rank, alpha or dropout")
        self.scale, self.dropout = alpha / rank, dropout
        self.A = nn.Parameter(torch.empty(weight.shape[0], rank, device=weight.device,
                                          dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(rank, weight.shape[1], device=weight.device,
                                          dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, weight):
        a = F.dropout(self.A, self.dropout, self.training)
        return weight + (a @ self.B).to(weight.dtype) * self.scale


def inject(model, pattern, rank, alpha, dropout, targets=None):
    modules = dict(model.named_modules())  # Do not mutate a live module iterator.
    chosen = list(targets) if targets is not None else [n for n in modules if re.fullmatch(pattern, n)]
    if not chosen or len(chosen) != len(set(chosen)):
        raise ValueError("no targets or duplicate adapter targets")
    for name in chosen:
        module = modules.get(name)
        w = getattr(module, "weight", None)
        if not isinstance(w, nn.Parameter) or w.ndim != 2 or parametrize.is_parametrized(module):
            raise ValueError(f"missing, incompatible or already parametrized target: {name}")
    for p in model.parameters():
        p.requires_grad_(False)
    for name in chosen:
        module = modules[name]
        delta = LoRAWeight(module.weight, rank, alpha, dropout)
        delta.train(module.training)
        parametrize.register_parametrization(module, "weight", delta)
    return chosen


def adapter_weights(model):
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters()
            if ".parametrizations.weight.0." in k}


def adapter_state(model, targets, meta):
    return {**meta, "format": io.ADAPTER_FORMAT, "targets": list(targets),
            "state_dict": adapter_weights(model)}


def load_adapter(model, path, expected_base, dropout_override=None):
    data = io.load_tensor(path)
    if data.get("format") != io.ADAPTER_FORMAT or data.get("base_identity") != expected_base:
        raise ValueError("legacy adapter or base/config mismatch; restart with v2 features")
    targets = data["targets"]
    if (not isinstance(targets, list) or not targets or any(not isinstance(n, str) for n in targets)
            or len(targets) != len(set(targets)) or type(data.get("rank")) is not int or data["rank"] < 1
            or type(data.get("step")) is not int or data["step"] < 0
            or not math.isfinite(data["alpha"]) or data["alpha"] <= 0 or not 0 <= data["dropout"] < 1):
        raise ValueError("invalid adapter metadata")
    expected = {f"{n}.parametrizations.weight.0.{ab}" for n in targets for ab in ("A", "B")}
    if set(data["state_dict"]) != expected:
        raise ValueError("adapter keys must contain exactly the declared A/B tensors")
    # Reject non-adapter keys BEFORE load_state_dict can touch frozen base weights.
    modules = dict(model.named_modules())
    rank = int(data["rank"])
    for name in targets:
        weight = getattr(modules.get(name), "weight", None)
        if weight is None or weight.ndim != 2:
            raise ValueError(f"invalid adapter target: {name}")
        for ab, shape in (("A", (weight.shape[0], rank)), ("B", (rank, weight.shape[1]))):
            tensor = data["state_dict"][f"{name}.parametrizations.weight.0.{ab}"]
            if not isinstance(tensor, torch.Tensor) or tensor.shape != shape or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(f"invalid adapter tensor: {name}.{ab}")
    dropout = data["dropout"] if dropout_override is None else dropout_override
    inject(model, "", rank, data["alpha"], dropout, targets)
    # Inference registers aliases to gpt inside inference_model. Load each unique
    # parametrization directly, so alias keys are neither missing nor double-loaded.
    for name in targets:
        delta = model.get_submodule(name).parametrizations.weight[0]
        delta.load_state_dict({ab: data["state_dict"][f"{name}.parametrizations.weight.0.{ab}"]
                               for ab in ("A", "B")}, strict=True)
    return {k: data[k] for k in ("format", "targets", "step", "rank", "alpha", "dropout", "base_identity")}


def load_gpt(config, model_dir, device, bf16=False):
    from omegaconf import OmegaConf
    from indextts.gpt.model_v2 import UnifiedVoice
    cfg = OmegaConf.load(config)
    kwargs = dict(cfg.gpt)
    kwargs.update(use_accel=False, spk_cond_mode="campplus")
    model = UnifiedVoice(**kwargs)
    checkpoint = io.load_tensor(Path(model_dir) / cfg.gpt_checkpoint)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    # FP32 master weights/parameters; BF16 is a CUDA autocast choice, never a CPU cast.
    return model.to(device).eval(), cfg


def validate_feature(obj, model, expected_base=None):
    if obj.get("format") != io.FEATURE_FORMAT:
        raise ValueError("stale feature format: regenerate with the v2 precompute script")
    if expected_base is not None and obj.get("base_identity") != expected_base:
        raise ValueError("feature/base mismatch")
    for name in ("text_tokens", "mel_codes"):
        t = obj[name]
        if not isinstance(t, torch.Tensor) or t.ndim != 1 or not t.numel() or t.dtype != torch.int64:
            raise ValueError(f"invalid {name}")
    text, codes = obj["text_tokens"], obj["mel_codes"]
    if int(text[0]) != obj["language_token_id"]:
        raise ValueError("missing language-prefix token")
    if text.min() < 0 or text.max() >= model.text_embedding.num_embeddings:
        raise ValueError("text token outside vocabulary")
    if codes.min() < 0 or codes.max() >= model.number_mel_codes or (codes == model.start_mel_token).any() or (codes == model.stop_mel_token).any():
        raise ValueError("semantic target contains invalid/reserved codes")
    if not 0 <= obj["lang_id"] < model.lang_embedding.num_embeddings:
        raise ValueError("invalid language ID")
    if text.numel() + 2 > model.text_pos_embedding.emb.num_embeddings or codes.numel() + 1 > model.mel_pos_embedding.emb.num_embeddings:
        raise ValueError("example exceeds position capacity")
    # Inference adds one stop token before prepare_gpt_inputs: one masked left pad.
    if 3 + text.numel() + 3 + codes.numel() + 1 > model.gpt.config.n_positions:
        raise ValueError("combined conditioning/text/audio exceeds GPT context")
    for name, shape in (("campplus", (192,)), ("emo_condition", (None, 1024))):
        t = obj[name]
        if t.ndim != len(shape) or any(s is not None and t.shape[i] != s for i, s in enumerate(shape)) or not t.numel() or not torch.isfinite(t).all():
            raise ValueError(f"invalid {name} shape or values")


def teacher_logits(model, obj, device, bf16):
    validate_feature(obj, model)
    text = obj["text_tokens"].to(device).unsqueeze(0)
    codes = obj["mel_codes"].to(device).unsqueeze(0)
    lang = torch.tensor([obj["lang_id"]], device=device)
    enabled = bf16 and torch.device(device).type == "cuda"
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=enabled):
        with torch.no_grad():
            spk = model.spk_emb_proj(obj["campplus"].to(device).float().unsqueeze(0)).unsqueeze(1)
            emo = obj["emo_condition"].to(device).float().unsqueeze(0)
            lengths = torch.tensor([emo.shape[1]], device=device)
            ev = model.get_emo_conditioning(emo.transpose(1, 2), lengths)
            ev = model.emo_layer(model.emovec_layer(ev))
            conds = torch.cat((spk + ev.unsqueeze(1), spk.new_zeros(1, 2, spk.shape[-1])), 1)
            # Both the tokenizer prefix AND lang_embedding are used by real inference.
            padded = F.pad(text, (0, 1), value=model.stop_text_token)
            _, prefix, attention = model.prepare_gpt_inputs(conds, padded, lang)
        mel_in = F.pad(codes, (1, 0), value=model.start_mel_token)
        mel_emb = model.mel_embedding(mel_in) + model.mel_pos_embedding(mel_in)
        embeds = torch.cat((prefix, mel_emb), 1)
        attention = F.pad(attention, (0, codes.shape[1]), value=1)
        hidden = model.gpt(inputs_embeds=embeds, attention_mask=attention,
                           use_cache=False, return_dict=True).last_hidden_state
        logits = model.mel_head(model.final_norm(hidden[:, -mel_in.shape[1]:])).float()
    targets = F.pad(codes, (0, 1), value=model.stop_mel_token)
    return logits, targets


def one_loss(model, obj, device, bf16):
    logits, targets = teacher_logits(model, obj, device, bf16)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@contextlib.contextmanager
def eval_mode(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


@torch.no_grad()
def evaluate(model, paths, device, bf16):
    total = count = 0
    with eval_mode(model):
        for path in paths:
            obj = io.load_tensor(path)
            loss = float(one_loss(model, obj, device, bf16))
            if not math.isfinite(loss):
                raise RuntimeError("nonfinite validation loss")
            n = obj["mel_codes"].numel() + 1
            total += loss * n
            count += n
    if not count:
        raise ValueError("empty validation set")
    return total / count


class SampleOrder:
    def __init__(self, size, seed):
        self.rng = random.Random(seed)
        self.order, self.cursor = list(range(size)), 0
        self.rng.shuffle(self.order)

    def next(self):
        if self.cursor == len(self.order):
            self.rng.shuffle(self.order)
            self.cursor = 0
        result = self.order[self.cursor]
        self.cursor += 1
        return result

    def state(self):
        return {"order": self.order[:], "cursor": self.cursor, "rng": self.rng.getstate()}

    def restore(self, state):
        if sorted(state["order"]) != list(range(len(self.order))) or not 0 <= state["cursor"] <= len(self.order):
            raise ValueError("invalid sampler resume state")
        self.order, self.cursor = state["order"][:], state["cursor"]
        self.rng.setstate(state["rng"])


def step_optimizer(opt, parameters, clip):
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    if not torch.isfinite(norm) or norm <= 0:
        raise RuntimeError("nonfinite or zero adapter gradient")
    opt.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise RuntimeError("nonfinite adapter after optimizer step; no checkpoint saved")
    opt.zero_grad(set_to_none=True)
    return float(norm)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ("features", "output"):
        ap.add_argument("--" + flag, required=True)
    ap.add_argument("--dev-features")
    ap.add_argument("--config", default="checkpoints/config.yaml")
    ap.add_argument("--model-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--target-regex", default=DEFAULT_TARGET)
    for flag, default in (("rank", 8), ("max-steps", 500), ("grad-accum", 8),
                          ("warmup-steps", 25), ("save-every", 50), ("eval-every", 50),
                          ("seed", 20260912), ("max-items", 0)):
        ap.add_argument("--" + flag, type=int, default=default)
    for flag, default in (("alpha", 16.), ("dropout", .05), ("learning-rate", 1e-5),
                          ("weight-decay", 0.), ("clip-grad", .5)):
        ap.add_argument("--" + flag, type=float, default=default)
    ap.add_argument("--resume", help="v2 full-state checkpoint; use a new output directory")
    a = ap.parse_args()
    if any(getattr(a, k) < 1 for k in ("rank", "max_steps", "grad_accum", "save_every", "eval_every")) or min(a.warmup_steps, a.max_items, a.seed) < 0:
        ap.error("counts must be positive; warmup/max-items/seed must be nonnegative")
    if any(not math.isfinite(getattr(a, k)) for k in ("alpha", "dropout", "learning_rate", "weight_decay", "clip_grad")) or min(a.alpha, a.learning_rate, a.clip_grad) <= 0 or a.weight_decay < 0 or not 0 <= a.dropout < 1:
        ap.error("invalid optimizer/adapter settings")
    if a.bf16 and torch.device(a.device).type == "cuda" and not torch.cuda.is_bf16_supported():
        ap.error("selected CUDA runtime does not support BF16; use --no-bf16")
    with io.output_lock(a.output, empty=True) as out:
        run(a, out)


def run(a, out):
    torch.manual_seed(a.seed)
    base = io.base_identity(a.config, a.model_dir)
    model, _ = load_gpt(a.config, a.model_dir, a.device)
    groups, dataset = {}, {}
    audio_sets = {}
    extraction_ids = set()
    source = io.source_identity()
    for split, root in (("train", a.features), ("dev", a.dev_features)):
        paths = sorted(Path(root).glob("*.pt")) if root else []
        if split == "train" and a.max_items:
            paths = paths[:a.max_items]
        if root and not paths:
            raise ValueError(f"no feature files for {split}")
        hashes, audios = {}, set()
        report = json.loads((Path(root) / "feature_receipt.json").read_text()) if root else {}
        expected_hashes = {r["row_id"] + ".pt": r["sha256"] for r in report.get("records", [])}
        for path in paths:
            if expected_hashes.get(path.name) != io.sha256(path):
                raise ValueError(f"feature checksum missing/mismatched: {path}")
            obj = io.load_tensor(path)
            validate_feature(obj, model, base)
            if obj.get("pipeline_source") != source:
                raise ValueError("feature cache was generated with different code; regenerate")
            extraction_ids.add(obj["extraction_id"])
            if obj.get("split") != split:
                raise ValueError(f"wrong split in {path}; test data cannot train/select checkpoints")
            audios.update((obj["audio_sha256"], obj["ref_audio_sha256"]))
            hashes[path.name] = io.sha256(path)
        groups[split], dataset[split], audio_sets[split] = paths, hashes, audios
    if len(extraction_ids) != 1:
        raise ValueError("mixed feature extractor provenance")
    if audio_sets["train"] & audio_sets["dev"]:
        raise ValueError("train/dev audio or reference leakage")
    contract = {k: getattr(a, k) for k in ("rank", "alpha", "dropout", "target_regex", "learning_rate",
                "grad_accum", "warmup_steps", "weight_decay", "clip_grad", "seed", "bf16")}
    contract.update(dataset=dataset, source=source, environment=io.environment(),
                    device_type=torch.device(a.device).type)
    sampler = SampleOrder(len(groups["train"]), a.seed)
    start = 0
    saved = None
    if a.resume:
        saved = io.load_tensor(a.resume)
        if saved.get("training", {}).get("contract") != contract:
            raise ValueError("resume requires identical data, code, environment and training settings")
        meta = load_adapter(model, a.resume, base)
        targets, start = meta["targets"], meta["step"]
        sampler.restore(saved["training"]["sampler"])
    else:
        targets = inject(model, a.target_regex, a.rank, a.alpha, a.dropout)
    if start >= a.max_steps:
        raise ValueError("max-steps must exceed the resumed optimizer step")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.learning_rate, weight_decay=a.weight_decay)
    # Frozen conditioning encoders stay in eval; transformer/adapter dropout trains.
    model.eval()
    model.gpt.train()
    model.gpt.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if saved:
        opt.load_state_dict(saved["training"]["optimizer"])
        torch.set_rng_state(saved["training"]["rng"])
        if saved["training"]["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["training"]["cuda_rng"])
    receipt = {"base_identity": base, "contract": contract, "targets": targets,
               "resumed_from": a.resume, "status": "running", "max_steps": a.max_steps}
    io.write_json(out / "run_receipt.json", receipt)
    wall = time.monotonic()
    try:
        for step in range(start + 1, a.max_steps + 1):
            opt.zero_grad(set_to_none=True)
            acc = 0.
            for _ in range(a.grad_accum):
                obj = io.load_tensor(groups["train"][sampler.next()])
                loss = one_loss(model, obj, a.device, a.bf16)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss BEFORE optimizer step {step}")
                (loss / a.grad_accum).backward()
                acc += float(loss.detach())
            lr = a.learning_rate * min(1., step / max(1, a.warmup_steps))
            for pg in opt.param_groups:
                pg["lr"] = lr
            rec = {"step": step, "train_loss": acc / a.grad_accum, "lr": lr,
                   "grad_norm": step_optimizer(opt, params, a.clip_grad),
                   "elapsed_s": time.monotonic() - wall}
            if groups["dev"] and (step == start + 1 or step % a.eval_every == 0):
                rec["dev_loss"] = evaluate(model, groups["dev"], a.device, a.bf16)
            with (out / "train_log.jsonl").open("a") as log:
                log.write(json.dumps(rec, allow_nan=False) + "\n")
            print(json.dumps(rec), flush=True)
            if step % a.save_every == 0 or step == a.max_steps:
                data = adapter_state(model, targets, {"step": step, "rank": a.rank,
                    "alpha": a.alpha, "dropout": a.dropout, "base_identity": base})
                data["training"] = {"contract": contract, "optimizer": opt.state_dict(),
                    "sampler": sampler.state(), "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.device(a.device).type == "cuda" else []}
                path = out / f"adapter-step-{step:06d}.pt"
                io.save_tensor(path, data)
                # CPU reload avoids a second full GPU model; preserve training RNG.
                with torch.random.fork_rng(devices=[]):
                    fresh, _ = load_gpt(a.config, a.model_dir, "cpu")
                    load_adapter(fresh, path, base)
                    actual = adapter_weights(fresh)
                    if any(not torch.equal(v, actual[k]) for k, v in data["state_dict"].items()):
                        raise RuntimeError("adapter tensor reload mismatch")
                    del fresh
                io.write_json(path.with_suffix(".json"), {"step": step, "sha256": io.sha256(path),
                    "strict_cpu_tensor_reload": True, "speech_quality_validated": False})
        receipt["status"] = "completed"
    except BaseException:
        receipt["status"] = "failed_or_interrupted"
        raise
    finally:
        io.write_json(out / "run_receipt.json", receipt)


if __name__ == "__main__":
    main()
