#!/usr/bin/env python3
"""Failure-inclusive corpus WER/CER. Accent and naturalness still require listening."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import unicodedata

try:
    from . import _common as io
except ImportError:
    import _common as io


def norm(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\wáéíóúüñ]+", " ", text)).strip()


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score(records, transcribe):
    if not records:
        raise ValueError("cannot score an empty evaluation")
    rows, keys = [], set()
    we = ww = ce = cw = failures = 0
    for original in records:
        row = dict(original)
        key = (row["id"], row["seed"])
        if key in keys:
            raise ValueError("duplicate evaluation record")
        keys.add(key)
        ref = norm(io.require_text(row["text"], "reference text"))
        if not ref:
            raise ValueError("reference normalizes to empty text")
        hypothesis = ""
        if row.get("ok"):
            try:
                hypothesis = transcribe(row)
                if not isinstance(hypothesis, str):
                    raise TypeError("ASR must return a string")
            except Exception as exc:
                row.update(ok=False, asr_error=f"{type(exc).__name__}: {exc}")
        if not row.get("ok"):
            failures += 1
            hypothesis = ""  # Every reference word/character counts as a deletion.
        hn = norm(hypothesis)
        words = ref.split()
        e, c = edit(words, hn.split()), edit(list(ref), list(hn))
        we += e
        ww += len(words)
        ce += c
        cw += len(ref)
        row.update(transcript=hypothesis.strip(), wer=e / len(words), cer=c / len(ref))
        rows.append(row)
    return {"corpus_wer": we / ww, "corpus_cer": ce / cw, "word_edits": we,
            "reference_words": ww, "char_edits": ce, "reference_chars": cw,
            "failed_records": failures, "total_records": len(rows),
            "failure_rate": failures / len(rows), "cer_includes_spaces": True, "records": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--language", default="es")
    ap.add_argument("--device")
    a = ap.parse_args()
    path = Path(a.receipt)
    data = json.loads(path.read_text(encoding="utf-8"))
    model = None

    def transcribe(row):
        nonlocal model
        wav = io.resolve_audio(row["wav"], path.resolve().parent)
        if row.get("wav_sha256") and io.sha256(wav) != row["wav_sha256"]:
            raise ValueError("generated WAV changed after evaluation")
        if model is None:
            import whisper
            model = whisper.load_model(a.model, device=a.device)
        return model.transcribe(str(wav), language=a.language, temperature=0,
                                fp16=model.device.type == "cuda")["text"]

    result = score(data["records"], transcribe)
    result.update(whisper_model=a.model, language=a.language, receipt_sha256=io.sha256(path),
                  environment=io.environment(), normalization="NFKC/lowercase/punctuation-to-spaces; accents retained")
    io.write_json(path.with_name("wer.json"), result)
    print(json.dumps({k: result[k] for k in ("corpus_wer", "corpus_cer", "failure_rate")}))
    if result["failed_records"]:
        raise SystemExit("failure-inclusive scores written; failed/pending/ASR-error records require review")


if __name__ == "__main__":
    main()
