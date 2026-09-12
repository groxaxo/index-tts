#!/usr/bin/env python3
"""Export selected adapters without overwriting the base checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import _common as io
except ImportError:
    import _common as io


def merge_weights(model, targets):
    import torch
    from torch.nn.utils import parametrize
    model.eval()
    with torch.no_grad():
        for name in targets:
            module = model.get_submodule(name)
            if not parametrize.is_parametrized(module, "weight"):
                raise ValueError(f"not an adapter target: {name}")
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("nonfinite merged weights")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="checkpoints/config.yaml")
    ap.add_argument("--model-dir", default="checkpoints")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", choices=["cpu"], default="cpu", help="merge/reload use host RAM, not a second GPU copy")
    a = ap.parse_args()
    out = Path(a.output).resolve()
    if out.exists() or out.with_suffix(out.suffix + ".json").exists() or out.is_relative_to(Path(a.model_dir).resolve()):
        raise FileExistsError("merge output must be new and outside the model directory")
    import torch
    try:
        from .train_lora import load_adapter, load_gpt
    except ImportError:
        from train_lora import load_adapter, load_gpt
    base = io.base_identity(a.config, a.model_dir)
    model, _ = load_gpt(a.config, a.model_dir, "cpu")
    meta = load_adapter(model, a.adapter, base, dropout_override=0.)
    merge_weights(model, meta["targets"])
    receipt = {"format": "indextts25-merged-gpt-v2", "adapter_sha256": io.sha256(a.adapter),
               "adapter_metadata": meta, "source": io.source_identity(), "speech_quality_validated": False}
    # Verify a fresh model BEFORE publishing the release checkpoint.
    fresh, _ = load_gpt(a.config, a.model_dir, "cpu")
    state = model.state_dict()
    fresh.load_state_dict(state, strict=True)
    fresh_state = fresh.state_dict()
    if any(not torch.equal(v, fresh_state[k]) for k, v in state.items()):
        raise ValueError("merged tensor reload mismatch")
    with io.output_lock(out.parent / ("." + out.name + ".lockdir")):
        if out.exists():
            raise FileExistsError(out)
        io.save_tensor(out, {"model": state, "finetune_receipt": receipt})
        reloaded = io.load_tensor(out)["model"]
        fresh.load_state_dict(reloaded, strict=True)
        if any(not torch.equal(v, reloaded[k]) for k, v in state.items()):
            raise ValueError("serialized merged tensor mismatch")
        receipt.update(sha256=io.sha256(out), strict_cpu_tensor_reload=True)
        io.write_json(out.with_suffix(out.suffix + ".json"), receipt)
    print(f"Merged checkpoint: {out}; independent synthesis qualification still required")


if __name__ == "__main__":
    main()
