# IndexTTS 2.5 LATAM pilot — integrity-reviewed v2

Local, single-GPU adaptation of the autoregressive GPT, with frozen semantic codec,
CAMPPlus, emotion-conditioning encoders, S2Mel and vocoder. No production endpoint,
service, model directory, or GitHub Actions workflow is changed by these tools.

**Migration:** recreate manifests and features in a new work directory. Adapters and
unversioned feature caches from PR #1 are rejected, not silently resumed or converted.
Keep them archived for comparison; do not delete your known-good release.

## Corrections to the first pilot

The previous review incorrectly removed the tokenizer's `<|es|>` prefix. The actual
`indextts/infer_v2_5.py` path uses **both** this prefix and `lang_embedding`.
Training now calls the real `UnifiedVoice.prepare_gpt_inputs()` method, including
its masked left padding, rather than maintaining a divergent copy of that logic.

The autoregressive objective is now:

```text
input:   conditioning + language-conditioned text + [BOS, code_0, ..., code_N]
target:                                               [code_0, ..., code_N, EOS]
loss:    cross entropy over audio codes AND EOS; prefix positions excluded
```

This corrects the missing stop-token supervision; it does not prove speech quality.
Other fixes include device-local FP32 adapter parameters, strict checkpoint loading,
full optimizer/RNG/sampler resume, split-local references, duplicate grouping,
cache checksums, failure-inclusive WER, and JSON-safe evaluation receipts.

## Environment and local tests

Use the repository's existing dependency environment (Python 3.10/3.11 and its
pinned PyTorch/Transformers versions). The toolkit adds no dependencies or lockfile
changes; SoundFile is already in the audio dependency stack.

```bash
# From the repository root, using its prepared environment:
python -m pytest -q tools/finetune_indextts25/tests
python -m compileall -q tools/finetune_indextts25
```

Every script supports direct invocation and `python -m tools.finetune_indextts25.NAME`.
`--help` does not initialize IndexTTS, download models, or require Transformers.
The CPU suite uses a tiny model and the actual inference-prefix method extracted
from the repository source. It is not a full-model or CUDA qualification; see
[VALIDATION.md](VALIDATION.md).

## 1. Validate and split data

Input CSV or JSONL requires `audio,text,speaker`. Recommended fields:
`session,gender,dialect,lang,ref_audio,text_group,row_id`.
Paths are relative to the input manifest, not the shell's current directory.
Use `lang=es`; other languages are deliberately rejected by this pilot.

```csv
audio,text,speaker,lang,gender,dialect,session
ana/session01_01.wav,"Che, ¿querés venir mañana?",ana,es,f,es-AR,session01
ana/session01_02.wav,"Dale, nos vemos a las ocho.",ana,es,f,es-AR,session01
```

The two-row example illustrates syntax, not enough independent data for three splits.
Provide at least three independent groups and enough distinct utterances to assign
same-speaker references inside each split. Missing session metadata groups the entire
speaker together. Shared audio hashes, normalized transcripts and optional `text_group`
labels connect groups transitively. Explicit references must appear as same-speaker
rows in the source manifest and are assigned to the same split as their targets.

```bash
WORK=/data/indextts-v2
CKPT=/data/models/IndexTTS-2.5
python tools/finetune_indextts25/prepare_manifest.py \
  --input /data/indextts/raw.csv --output-dir "$WORK/manifests"
```

Review `dataset_receipt.json`, including speech duration by gender and dialect.
The splitter targets duration fractions but does not promise demographic balance
or detect perceptual audio duplicates. Confirm source-recording/session boundaries
and transcript accuracy manually. Training-only `--allow-self-reference` is for
explicit diagnostics, never held-out evaluation. Clips must be 0.25–15 seconds;
segment audio and transcript together instead of truncating only audio.

## 2. Precompute features

Select an **idle** GPU. These scripts never stop another workload or reserve the
whole workstation. Output locks protect artifacts, not GPU scheduling.

```bash
# Start with a small cache to qualify extraction.
CUDA_VISIBLE_DEVICES=0 python tools/finetune_indextts25/precompute_features.py \
  --manifest "$WORK/manifests/train.jsonl" --output-dir "$WORK/features/smoke" \
  --model-dir "$CKPT" --config "$CKPT/config.yaml" --max-items 24
```

Extraction is FP32 under `no_grad`, with all loaded modules frozen/evaluating.
There is no extraction `--bf16` switch in v2. It loads the existing inference stack,
so its initialization memory requirement must still be measured on the workstation.
Targets are never silently truncated. Features record tokenizer/model assets,
config/base hashes, source fingerprint, environment, reference hashes and format.
Existing files are reused only when their key and recorded checksum both match.
Use a new directory after changing code, assets or manifests. An unreceipted file
left by interruption between file-save and receipt-save is rejected for inspection.

For the actual pilot, extract full `train` and `dev` caches with the same code,
environment and model directory:

```bash
for SPLIT in train dev; do
  CUDA_VISIBLE_DEVICES=0 python tools/finetune_indextts25/precompute_features.py \
    --manifest "$WORK/manifests/$SPLIT.jsonl" --output-dir "$WORK/features/$SPLIT" \
    --model-dir "$CKPT" --config "$CKPT/config.yaml"
done
```

Do not use `test` features for training or checkpoint selection.

## 3. Qualify the trainer

```bash
CUDA_VISIBLE_DEVICES=0 python tools/finetune_indextts25/train_lora.py \
  --features "$WORK/features/smoke" --model-dir "$CKPT" --config "$CKPT/config.yaml" \
  --output "$WORK/runs/smoke" --max-items 24 --max-steps 30 \
  --grad-accum 1 --learning-rate 3e-5 --save-every 10
```

Gates enforced in code: compatible feature/checkpoint identity, valid code/position
bounds, finite loss before backward, finite nonzero gradients before optimizer update,
only low-rank trainable parameters, strict adapter tensors, and fresh CPU tensor reload.
The fresh copy is on CPU to avoid a second full model consuming training VRAM.
Checkpoints also contain optimizer moments, CPU/CUDA RNG, sampler order/cursor and
the run contract. Logs are flushed each optimizer step and receipts written atomically.

Gates **you must still judge**: the tiny-subset loss trend, audible reconstruction,
voice identity, naturalness, EOS behavior and real VRAM/throughput. A successful
reload is not a speech-quality certificate. There is no automatic checkpoint promotion.

Defaults are rank 8, alpha 16, dropout 0.05, FP32 master weights, CUDA BF16 autocast,
non-reentrant gradient checkpointing and gradient clip 0.5. This native
`torch.nn.utils.parametrize` implementation applies dropout to low-rank factor A,
**not** PEFT-style input dropout. It is not a PEFT adapter format. No NF4 is used.

## 4. Two short learning-rate arms

Run only after the smoke qualification passes. These are hypotheses, not benchmarked
optimal settings. Use the same feature caches, seed and sampling in both arms.

```bash
# Arm A on an idle GPU 0. For arm B use another idle GPU, a new output directory,
# and --learning-rate 3e-5; leave other training settings identical.
CUDA_VISIBLE_DEVICES=0 python tools/finetune_indextts25/train_lora.py \
  --features "$WORK/features/train" --dev-features "$WORK/features/dev" \
  --model-dir "$CKPT" --config "$CKPT/config.yaml" \
  --output "$WORK/runs/lr1e-5" --learning-rate 1e-5 \
  --max-steps 500 --grad-accum 8 --warmup-steps 25 --save-every 50 --eval-every 50
```

Use `train_log.jsonl` for full-dev token-weighted loss and training-step receipts.
Generation/listening evaluation is a separate operation; it is not secretly invoked
by `--eval-every`. Pause when held-out speech degrades. More steps are not necessarily
better. Single-GPU microbatch 1 plus accumulation 8 gives effective batch 8.

**True resume** requires identical data checksums, source, environment and critical
training settings. Specify the original settings and a **new** output directory;
`--max-steps` is the total step target, not extra updates. Example for the smoke run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/finetune_indextts25/train_lora.py \
  --features "$WORK/features/smoke" --model-dir "$CKPT" --config "$CKPT/config.yaml" \
  --output "$WORK/runs/smoke-resumed" --max-items 24 --max-steps 60 \
  --grad-accum 1 --learning-rate 3e-5 --save-every 10 \
  --resume "$WORK/runs/smoke/adapter-step-000030.pt"
```

Changing optimizer settings is a new experiment, not a silently accepted resume.
CPU fixture tests establish resume equality in that fixture, not CUDA determinism.

## 5. Fixed-suite synthesis and scoring

Make an independently held-out JSONL suite. IDs must be unique safe filenames.
Use distinct reference recordings, never the target utterance as reference.
Use multiple voices and seeds; keep final-test prompts out of checkpoint selection.

```json
{"id":"ar_001_ana","text":"Che, ¿querés venir mañana?","lang":"es","speaker_ref":"refs/ana.wav"}
```

```bash
CUDA_VISIBLE_DEVICES=0 python tools/finetune_indextts25/eval_suite.py \
  --suite /data/indextts/eval/dev.jsonl --output "$WORK/eval/stock" \
  --model-dir "$CKPT" --config "$CKPT/config.yaml" --seeds 42,43,44

CUDA_VISIBLE_DEVICES=0 python tools/finetune_indextts25/eval_suite.py \
  --suite /data/indextts/eval/dev.jsonl --output "$WORK/eval/step200" \
  --model-dir "$CKPT" --config "$CKPT/config.yaml" --seeds 42,43,44 \
  --adapter "$WORK/runs/lr1e-5/adapter-step-000200.pt"

python tools/finetune_indextts25/score_wer.py \
  --receipt "$WORK/eval/step200/receipt.json" --model medium --language es
```

Output directories must be new. Evaluation persists the complete planned matrix
before synthesis, then updates every result. Empty, silent, nonfinite, truncated,
failed and pending generations are not dropped. CUDA errors stop generation while
preserving unattempted entries. Timing is synchronized **full-utterance** wall time,
including the first call's warmup; it is not time-to-first-audio. CUDA allocated-memory
peaks and clipping fractions are diagnostics, not complete device-VRAM measurements.

WER/CER include every reference in their denominator. Generation failures and ASR
errors count as full deletions and produce a nonzero exit status with a written
`wer.json`. Inspect failure rate alongside error rate; an interrupted run is not a
valid completed benchmark. Scoring retains accents, normalizes case/punctuation,
and counts spaces in CER; it does not reconcile arbitrary written/spoken number
formats. Use spoken-form references when appropriate. Whisper runs locally but may
need its model weights downloaded first. Accent/naturalness require blinded listening.

## 6. Explicit merge and independent deployment qualification

```bash
python tools/finetune_indextts25/merge_lora.py \
  --model-dir "$CKPT" --config "$CKPT/config.yaml" \
  --adapter "$WORK/runs/lr1e-5/adapter-step-000200.pt" \
  --output "$WORK/releases/gpt-step200.pth"
```

Merge runs in CPU FP32, disables adapter dropout and verifies strict tensor reload.
It refuses to overwrite files or write inside the base-model directory. Keep the
original config intact. For independent merged evaluation, copy the config outside
`CKPT` and set its `gpt_checkpoint` to the **absolute path** of this merged file.
Pass that copied config to `eval_suite.py` without `--adapter`, retaining `--model-dir`
for the unchanged auxiliary assets. Compare with the unmerged adapter and stock
on the same suite before any quantization or deployment. Neither occurs automatically.
