"""Shared, local-only artifact validation. No model imports at CLI help time."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FEATURE_FORMAT = "indextts25-features-v2-prefix-eos"
ADAPTER_FORMAT = "indextts25-native-lora-v2"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", value):
        raise ValueError(f"unsafe or invalid ID: {value!r}")
    return value


def require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def resolve_audio(value, root):
    path = Path(require_text(value, "audio path")).expanduser()
    path = (Path(root) / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSONL objects: {path}")
    return rows


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    atomic_write(path, lambda stream: stream.write(data))


def save_tensor(path, value):
    import torch
    atomic_write(path, lambda stream: torch.save(value, stream))


def load_tensor(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


@contextlib.contextmanager
def output_lock(directory, *, empty=False):
    """Advisory local-process lock; never kills another GPU user or overwrites a run."""
    import fcntl  # This toolkit targets the Ubuntu workstation.
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if empty and any(p.name != ".lock" for p in directory.iterdir()):
            raise FileExistsError(f"use a new, empty output directory: {directory}")
        yield directory


def base_identity(config, model_dir):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config)
    return {"config_sha256": sha256(config),
            "gpt_sha256": sha256(Path(model_dir) / cfg.gpt_checkpoint),
            "feature_format": FEATURE_FORMAT}


def source_identity():
    files = sorted((ROOT / "indextts").rglob("*.py"))
    files += sorted(Path(__file__).parent.glob("*.py"))
    return fingerprint({str(p.relative_to(ROOT)): sha256(p) for p in files})


def environment():
    versions = {}
    for name in ("torch", "torchaudio", "transformers", "librosa", "soundfile",
                 "omegaconf", "openai-whisper", "tiktoken"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version.split()[0], "packages": versions}


def audio_info(path):
    import soundfile as sf
    info = sf.info(str(path))
    if info.samplerate <= 0 or info.frames <= 0:
        raise ValueError(f"empty or invalid audio: {path}")
    return {"duration_s": info.frames / info.samplerate,
            "sample_rate": info.samplerate, "channels": info.channels}
