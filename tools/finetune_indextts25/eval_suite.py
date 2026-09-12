#!/usr/bin/env python3
"""Multi-seed evaluation with incremental receipts, including every failed item."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import time
import warnings

try:
    from . import _common as io
except ImportError:
    import _common as io


def validate_suite(rows, root, seeds):
    if not seeds or len(seeds) != len(set(seeds)) or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("seeds must be distinct integers in [0, 2**32)")
    ids, clean = set(), []
    for row in rows:
        row = dict(row)
        rid = io.safe_id(row.get("id"))
        if rid in ids:
            raise ValueError(f"duplicate evaluation ID: {rid}")
        ids.add(rid)
        row["text"] = io.require_text(row.get("text"), "evaluation text")
        ref = io.resolve_audio(row.get("speaker_ref"), root)
        if not .25 <= io.audio_info(ref)["duration_s"] <= 15:
            raise ValueError("reference must be 0.25-15 seconds")
        row.update(speaker_ref=str(ref), reference_sha256=io.sha256(ref))
        if row.get("lang", "es") != "es":
            raise ValueError("this evaluation pilot supports Spanish only")
        clean.append(row)
    return clean


def synthesize(tts, rows, seeds, out, receipt, settings, device):
    import numpy as np
    import soundfile as sf
    import torch
    out = Path(out)
    records = [{"id": r["id"], "seed": s, "text": r["text"], "lang": "es",
                "speaker_ref": r["speaker_ref"], "reference_sha256": r["reference_sha256"],
                "wav": str((out / f"{r['id']}__seed{s}.wav").resolve()),
                "ok": False, "status": "pending"} for r in rows for s in seeds]
    receipt["records"] = records
    io.write_json(out / "receipt.json", receipt)
    is_cuda = torch.device(device).type == "cuda"
    tts.gpt.eval()
    for rec in records:
        random.seed(rec["seed"])
        np.random.seed(rec["seed"])
        torch.manual_seed(rec["seed"])
        path = Path(rec["wav"])
        if path.exists():
            raise FileExistsError(f"refusing stale output: {path}")
        start = time.perf_counter()
        try:
            if is_cuda:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
            with torch.inference_mode(), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                tts.infer(rec["speaker_ref"], rec["text"], str(path), "es", verbose=False, **settings)
            if is_cuda:
                torch.cuda.synchronize(device)
                rec["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
            if not samples.size or not np.isfinite(samples).all() or np.max(np.abs(samples)) == 0:
                raise ValueError("empty, silent or nonfinite generated audio")
            rec.update(warnings=[str(w.message) for w in caught], audio_s=len(samples) / sr,
                       wav_sha256=io.sha256(path), clipped_fraction=float(np.mean(np.abs(samples) >= .999)))
            truncated = any("max_mel_tokens" in text for text in rec["warnings"])
            rec.update(ok=not truncated, status="truncated" if truncated else "ok")
        except Exception as exc:
            rec.update(ok=False, status="failed", error=f"{type(exc).__name__}: {exc}")
            # A CUDA failure may poison the context. Record all remaining entries as pending.
            if is_cuda and (isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA" in str(exc)):
                raise
        finally:
            rec["elapsed_s"] = time.perf_counter() - start
            if rec.get("audio_s"):
                rec["rtf"] = rec["elapsed_s"] / rec["audio_s"]
            io.write_json(out / "receipt.json", receipt)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="checkpoints/config.yaml")
    ap.add_argument("--model-dir", default="checkpoints")
    ap.add_argument("--adapter")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--temperature", type=float, default=.8)
    ap.add_argument("--top-p", type=float, default=.8)
    ap.add_argument("--top-k", type=int, default=30)
    a = ap.parse_args()
    if not math.isfinite(a.temperature) or a.temperature <= 0 or not 0 < a.top_p <= 1 or a.top_k < 0:
        ap.error("invalid sampling settings")
    seeds = [int(s) for s in a.seeds.split(",")]
    rows = validate_suite(io.read_jsonl(a.suite), Path(a.suite).resolve().parent, seeds)
    from indextts.infer_v2_5 import IndexTTS2
    try:
        from .train_lora import load_adapter
    except ImportError:
        from train_lora import load_adapter
    with io.output_lock(a.output, empty=True) as out:
        base = io.base_identity(a.config, a.model_dir)
        tts = IndexTTS2(cfg_path=a.config, model_dir=a.model_dir, device=a.device,
                       use_bf16=a.bf16, use_cuda_kernel=False, use_qwen_emo=False)
        metadata = load_adapter(tts.gpt, a.adapter, base, dropout_override=0.) if a.adapter else None
        settings = {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k,
                    "num_beams": 3, "repetition_penalty": 10., "max_mel_tokens": 1500,
                    "max_text_tokens_per_segment": 120}
        receipt = {"suite_sha256": io.sha256(a.suite), "base_identity": base,
                   "adapter_metadata": metadata, "adapter_sha256": io.sha256(a.adapter) if a.adapter else None,
                   "settings": settings, "seeds": seeds, "bf16_requested": a.bf16,
                   "environment": io.environment(), "source": io.source_identity(),
                   "timing": "synchronized full-utterance wall time; includes first-call warmup; not TTFA"}
        records = synthesize(tts, rows, seeds, out, receipt, settings, a.device)
        if any(not r["ok"] for r in records):
            raise SystemExit("evaluation contains failures; inspect receipt.json (no entries excluded)")


if __name__ == "__main__":
    main()
