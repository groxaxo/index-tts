#!/usr/bin/env python3
"""Cache v2 features without truncating target audio or retaining autograd graphs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

try:
    from . import _common as io
except ImportError:
    import _common as io


def clean_text(tts, text, lang):
    from indextts.infer_v2_5 import apply_pronunciation_annotations
    from indextts.utils.nemo_tn import normalize_text
    if lang != "es":
        raise ValueError("this pilot supports Spanish only")
    text = tts.text_process.clean_pattern.sub(lambda x: tts.text_process.char_rep_map[x.group()], text)
    text = normalize_text(text, lang).upper()
    text = apply_pronunciation_annotations(text)
    return re.sub(r"<\|([^|]+)\|>", lambda m: f"<|{m.group(1).upper()}|>", text)


def tokenize(tts, text, lang):
    # infer_v2_5.py deliberately uses BOTH a tokenizer prefix and language embeddings.
    return tts.tokenizer.encode(f"<|{lang}|> " + text, allowed_special="all")


def audio16(path):
    import librosa
    import numpy as np
    import torch
    wav, sr = librosa.load(path, sr=16000, mono=True)
    if not 4000 <= len(wav) <= 15 * sr or not np.isfinite(wav).all():
        raise ValueError(f"invalid/overlong audio (not truncated): {path}")
    return torch.from_numpy(wav).unsqueeze(0)


def w2v(tts, waveform):
    z = tts.extract_features(waveform, sampling_rate=16000, return_tensors="pt")
    return tts.get_emb(z["input_features"].to(tts.device), z["attention_mask"].to(tts.device))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-dir", default="checkpoints")
    ap.add_argument("--config", default="checkpoints/config.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-items", type=int, default=0)
    a = ap.parse_args()
    if a.max_items < 0:
        ap.error("max-items must be nonnegative")
    rows = io.read_jsonl(a.manifest)
    if a.max_items:
        rows = rows[:a.max_items]
    ids = [io.safe_id(row["row_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate row IDs")
    model_dir = Path(a.model_dir).resolve()
    if Path(a.output_dir).resolve().is_relative_to(model_dir):
        raise ValueError("feature output must be outside the model directory")
    for row in rows:
        if row.get("split") not in {"train", "dev", "test"} or row.get("lang") != "es":
            raise ValueError("use the split manifests from prepare_manifest.py")
        for key in ("audio", "ref_audio"):
            if io.sha256(row[key]) != row[key + "_sha256"]:
                raise ValueError(f"audio changed after manifest validation: {row[key]}")
            if not .25 <= io.audio_info(row[key])["duration_s"] <= 15:
                raise ValueError("audio exceeds cache duration bounds")
    import torch
    import torchaudio
    from indextts.infer_v2_5 import IndexTTS2
    from indextts.utils.tokenizer import lang_to_token
    try:
        from .train_lora import validate_feature
    except ImportError:
        from train_lora import validate_feature
    with io.output_lock(a.output_dir) as out, torch.no_grad():
        # FP32 extraction is a fixed cache contract. BF16 is only used by training/eval.
        tts = IndexTTS2(cfg_path=a.config, model_dir=a.model_dir, device=a.device,
                       use_bf16=False, use_cuda_kernel=False, use_qwen_emo=False)
        for value in vars(tts).values():
            if isinstance(value, torch.nn.Module):
                value.eval().requires_grad_(False)
        codec_checkpoint = io.load_tensor(model_dir / "codec.pth")
        tts.semantic_codec.load_state_dict(codec_checkpoint.get("model", codec_checkpoint), strict=True)
        base = io.base_identity(a.config, a.model_dir)
        suffixes = {".pth", ".pt", ".bin", ".safetensors", ".yaml", ".yml", ".json", ".model", ".tiktoken"}
        assets = {str(p.relative_to(model_dir)): io.sha256(p) for p in sorted(model_dir.rglob("*"))
                  if p.is_file() and p.suffix in suffixes and not any(x.startswith(".") for x in p.relative_to(model_dir).parts)}
        provenance = {"base_identity": base, "assets": assets, "source": io.source_identity(),
                      "environment": io.environment(), "precision": "fp32"}
        extraction_id = io.fingerprint(provenance)
        expected_paths = {f"{rid}.pt" for rid in ids}
        if {p.name for p in out.glob("*.pt")} - expected_paths:
            raise ValueError("cache contains rows outside this manifest; use a new directory")
        prior_path = out / "feature_receipt.json"
        prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
        if prior and prior.get("extraction_id") != extraction_id:
            raise ValueError("extractor provenance changed; use a new cache directory")
        receipts = {r["row_id"]: r for r in prior.get("records", [])}
        for row in rows:
            rid = row["row_id"]
            path = out / f"{rid}.pt"
            key = io.fingerprint({"row": row, "extraction": extraction_id})
            if path.exists():
                if receipts.get(rid, {}).get("sha256") != io.sha256(path):
                    raise ValueError(f"cache checksum missing/mismatched: {path}")
                old = io.load_tensor(path)
                if old.get("cache_key") != key:
                    raise ValueError(f"stale cache {path}; use a new output directory")
                validate_feature(old, tts.gpt, base)
            else:
                text = clean_text(tts, row["text"], row["lang"])
                tokens = torch.tensor(tokenize(tts, text, row["lang"]), dtype=torch.long)
                target = w2v(tts, audio16(row["audio"]))
                if not torch.isfinite(target).all():
                    raise ValueError(f"nonfinite target semantic features: {rid}")
                codes = tts.get_scode(target)
                if codes.ndim != 2 or codes.shape[0] != 1:
                    raise ValueError(f"unexpected semantic-code shape: {tuple(codes.shape)}")
                ref = audio16(row["ref_audio"])
                emo = w2v(tts, ref)
                fbank = torchaudio.compliance.kaldi.fbank(ref.to(tts.device), num_mel_bins=80,
                                                        dither=0, sample_frequency=16000)
                fbank -= fbank.mean(dim=0, keepdim=True)
                spk = tts.campplus_model(fbank.unsqueeze(0))
                obj = {**row, "format": io.FEATURE_FORMAT, "base_identity": base,
                       "cache_key": key, "extraction_id": extraction_id, "pipeline_source": provenance["source"],
                       "lang_id": int(lang_to_token(row["lang"])), "normalized_text": text,
                       "language_token_id": int(tts.tokenizer.encode("<|es|>", allowed_special="all")[0]),
                       "text_tokens": tokens, "mel_codes": codes[0].long().cpu(),
                       "campplus": spk[0].float().cpu(), "emo_condition": emo[0].float().cpu()}
                validate_feature(obj, tts.gpt, base)
                io.save_tensor(path, obj)
            receipts[rid] = {"row_id": rid, "sha256": io.sha256(path)}
            io.write_json(out / "feature_receipt.json", {"provenance": provenance,
                          "extraction_id": extraction_id, "records": list(receipts.values())})
            print(f"cached {rid}", flush=True)


if __name__ == "__main__":
    main()
