#!/usr/bin/env python3
"""Build leakage-checked audio manifests before any model is loaded."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import unicodedata

try:
    from . import _common as io
except ImportError:
    import _common as io


def normalize_group(text):
    # Deliberately conservative; accents remain significant.
    return " ".join("".join(c if c.isalnum() else " " for c in
                            unicodedata.normalize("NFKC", text).casefold()).split())


def split_rows(rows, seed, train_frac, dev_frac, allow_self=False):
    fractions = {"train": train_frac, "dev": dev_frac, "test": max(0., 1 - train_frac - dev_frac)}
    if not 0 < train_frac <= 1 or not 0 <= dev_frac < 1 or train_frac + dev_frac > 1 + 1e-12:
        raise ValueError("invalid split fractions")
    # Union connected sessions, duplicate audio/text, template groups and explicit refs.
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a, b):
        parent[root(b)] = root(a)

    seen, audio_owner = {}, {}
    missing_session_speakers = {r["speaker"] for r in rows if not r.get("session")}
    for i, row in enumerate(rows):
        session = "__all__" if row["speaker"] in missing_session_speakers else row["session"]
        keys = [("session", row["speaker"], session), ("audio", row["audio_sha256"]),
                ("text", normalize_group(row["text"]))]
        if row.get("text_group"):
            keys.append(("template", row["text_group"]))
        for key in keys:
            if key in seen:
                join(i, seen[key])
            else:
                seen[key] = i
        audio_owner.setdefault(row["audio_sha256"], []).append(i)
    for i, row in enumerate(rows):
        if row.get("ref_audio"):
            owners = audio_owner.get(row["ref_audio_sha256"], [])
            if not owners or any(rows[j]["speaker"] != row["speaker"] for j in owners):
                raise ValueError("explicit reference must be a manifest row from the same speaker")
            for j in owners:
                join(i, j)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(dict(row))
    groups = list(groups.values())
    active = [k for k, v in fractions.items() if v > 1e-12]
    if len(groups) < len(active):
        raise ValueError("too few independent groups for nonempty splits; add sessions or explicitly disable holdouts")
    random.Random(seed).shuffle(groups)
    splits = {k: [] for k in fractions}
    hours = dict.fromkeys(fractions, 0.)
    total = sum(r["duration_s"] for r in rows)
    for i, group in enumerate(groups):
        empty = [k for k in active if not splits[k]]
        candidates = empty if len(groups) - i == len(empty) else active
        name = max(candidates, key=lambda k: fractions[k] * total - hours[k])
        splits[name].extend(group)
        hours[name] += sum(r["duration_s"] for r in group)
    for name, items in splits.items():
        for row in items:
            row["split"] = name
            if not row.get("ref_audio"):
                choices = sorted((r for r in items if r["speaker"] == row["speaker"]
                                  and r["audio_sha256"] != row["audio_sha256"]), key=lambda r: r["row_id"])
                if not choices and not (allow_self and name == "train"):
                    raise ValueError(f"{row['row_id']}: no distinct same-speaker reference within {name}")
                ref = choices[0] if choices else row
                row["ref_audio"], row["ref_audio_sha256"] = ref["audio"], ref["audio_sha256"]
            row["ref_is_target"] = row["audio_sha256"] == row["ref_audio_sha256"]
            if row["ref_is_target"] and not (allow_self and name == "train"):
                raise ValueError("target-as-reference is forbidden outside explicit training diagnostics")
    return splits


def prepare(source, *, seed, train_frac, dev_frac, allow_self=False):
    source = Path(source).resolve()
    if source.suffix.lower() == ".jsonl":
        rows = io.read_jsonl(source)
    else:
        with source.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty input manifest")
    clean, ids = [], set()
    for i, r in enumerate(rows):
        row = dict(r)
        for key in ("text", "speaker"):
            row[key] = io.require_text(row.get(key), key)
        rid = io.safe_id(row.get("row_id") or f"{i:07d}")
        if rid in ids:
            raise ValueError(f"duplicate row_id: {rid}")
        ids.add(rid)
        path = io.resolve_audio(row.get("audio"), source.parent)
        row.update(row_id=rid, audio=str(path), audio_sha256=io.sha256(path), **io.audio_info(path))
        if not 0.25 <= row["duration_s"] <= 15:
            raise ValueError(f"{rid}: segment audio/transcript together into 0.25-15 second clips; never truncate")
        row["lang"] = str(row.get("lang") or "es").lower()
        if row["lang"] != "es":
            raise ValueError("this validated pilot supports lang=es only")
        if row.get("ref_audio"):
            ref = io.resolve_audio(row["ref_audio"], source.parent)
            row.update(ref_audio=str(ref), ref_audio_sha256=io.sha256(ref))
        clean.append(row)
    return split_rows(clean, seed, train_frac, dev_frac, allow_self)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--train-frac", type=float, default=.8)
    ap.add_argument("--dev-frac", type=float, default=.1)
    ap.add_argument("--allow-self-reference", action="store_true", help="training-only diagnostic; not release qualification")
    a = ap.parse_args()
    splits = prepare(a.input, seed=a.seed, train_frac=a.train_frac, dev_frac=a.dev_frac,
                     allow_self=a.allow_self_reference)
    with io.output_lock(a.output_dir, empty=True) as out:
        receipts = {}
        for name, rows in splits.items():
            path = out / f"{name}.jsonl"
            data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
            io.atomic_write(path, lambda stream, data=data: stream.write(data))
            by_gender, by_dialect = defaultdict(float), defaultdict(float)
            for row in rows:
                by_gender[row.get("gender") or "unknown"] += row["duration_s"]
                by_dialect[row.get("dialect") or "unknown"] += row["duration_s"]
            receipts[name] = {"rows": len(rows), "sha256": io.sha256(path),
                              "seconds_by_gender": dict(by_gender), "seconds_by_dialect": dict(by_dialect)}
        io.write_json(out / "dataset_receipt.json", {"source_sha256": io.sha256(a.input),
            "seed": a.seed, "train_frac": a.train_frac, "dev_frac": a.dev_frac, "splits": receipts})


if __name__ == "__main__":
    main()
