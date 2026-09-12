# IndexTTS 2.5 LATAM fine-tune pilot

This directory contains a local-first, reproducible pilot for adapting the **IndexTTS 2.5 autoregressive GPT** to LATAM/Argentine Spanish without changing the codec, semantic encoder, speaker encoder, S2Mel model, or vocoder.

The workflow intentionally starts with a small single-GPU LoRA experiment. It is designed for an RTX 3090-class machine and does not require GitHub Actions.

## Design

1. `prepare_manifest.py` validates metadata, hashes audio, records duration, and creates speaker/session-grouped train/dev/test splits.
2. `precompute_features.py` loads the same `IndexTTS2` stack used by inference and caches:
   - normalized/tokenized text
   - target semantic codec codes
   - CAMPPlus speaker embedding from a different same-speaker reference clip when possible
   - Wav2Vec2-BERT emotion conditioning features
3. `train_lora.py` loads only the IndexTTS GPT checkpoint and trains low-rank parametrizations on selected GPT-2 projection weights. Base weights remain frozen.
4. `eval_suite.py` synthesizes a fixed JSONL evaluation suite with stock or LoRA-adapted GPT weights and writes timing/RTF receipts.
5. `score_wer.py` transcribes generated WAVs with the already-declared `openai-whisper` dependency and computes corpus WER/CER.
6. `merge_lora.py` folds an adapter into the GPT weights and writes a normal checkpoint for deployment qualification.

The scripts are deliberately explicit about artifact provenance and never promote the last training step automatically.

## Input manifest

CSV or JSONL rows must contain:

```text
audio,text,speaker
```

Recommended optional fields:

```text
lang,gender,dialect,session,ref_audio
```

For this experiment use `lang=es`. `ref_audio` should ideally point to a *different* utterance from the same speaker. If omitted, `prepare_manifest.py` deterministically assigns another clip from the same speaker when available.

Example:

```csv
audio,text,speaker,lang,gender,dialect,session
/data/ana/s01_001.wav,"Che, ¿querés venir mañana?",ana,es,f,es-AR,s01
/data/ana/s01_002.wav,"Dale, nos vemos a las ocho.",ana,es,f,es-AR,s01
```

## 1. Validate and split

```bash
python tools/finetune_indextts25/prepare_manifest.py \
  --input /data/indextts/raw.csv \
  --output-dir /data/indextts/manifests \
  --seed 20260912
```

Splitting is grouped by `speaker/session` when `session` exists, otherwise by speaker. This prevents slices from the same recording session leaking across train/dev/test. The script also writes `dataset_receipt.json` with hashes, durations, counts, and split statistics.

## 2. Precompute model-native features

Download the official IndexTTS 2.5 checkpoints as usual, then:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/finetune_indextts25/precompute_features.py \
  --manifest /data/indextts/manifests/train.jsonl \
  --output-dir /data/indextts/features/train \
  --model-dir checkpoints \
  --config checkpoints/config.yaml \
  --device cuda:0 \
  --bf16
```

Repeat for `dev.jsonl`. Cache files are self-describing `.pt` dictionaries. The script refuses text that exceeds the model's text-position capacity and semantic-code sequences that exceed the configured mel-token capacity.

## 3. Trainer qualification

First overfit a tiny slice:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/finetune_indextts25/train_lora.py \
  --features /data/indextts/features/train \
  --config checkpoints/config.yaml \
  --model-dir checkpoints \
  --output /data/indextts/runs/smoke \
  --max-steps 30 \
  --grad-accum 1 \
  --learning-rate 3e-5 \
  --save-every 10 \
  --seed 20260912 \
  --max-items 24
```

Expected qualification gates:

- finite forward loss
- non-zero LoRA gradients
- loss decreases on the tiny subset
- checkpoint save and fresh reload succeed
- only LoRA parameters are trainable

Do not start a long run until these gates pass.

## 4. Pilot A/B

The proposed first comparison is identical initialization/data/order with only learning rate changed:

```bash
# arm A
CUDA_VISIBLE_DEVICES=0 python tools/finetune_indextts25/train_lora.py \
  --features /data/indextts/features/train \
  --dev-features /data/indextts/features/dev \
  --config checkpoints/config.yaml \
  --model-dir checkpoints \
  --output /data/indextts/runs/lr1e-5 \
  --learning-rate 1e-5 \
  --max-steps 500 \
  --grad-accum 8 \
  --warmup-steps 25 \
  --save-every 50 \
  --eval-every 50 \
  --seed 20260912

# arm B
CUDA_VISIBLE_DEVICES=1 python tools/finetune_indextts25/train_lora.py \
  --features /data/indextts/features/train \
  --dev-features /data/indextts/features/dev \
  --config checkpoints/config.yaml \
  --model-dir checkpoints \
  --output /data/indextts/runs/lr3e-5 \
  --learning-rate 3e-5 \
  --max-steps 500 \
  --grad-accum 8 \
  --warmup-steps 25 \
  --save-every 50 \
  --eval-every 50 \
  --seed 20260912
```

Defaults: rank 8, alpha 16, dropout 0.05, BF16 autocast, gradient clip 0.5. Target weights are GPT attention/MLP projections matched by:

```text
^gpt\.h\.\d+\.(attn\.(c_attn|c_proj)|mlp\.(c_fc|c_proj))$
```

The implementation uses `torch.nn.utils.parametrize`, so it works directly on the repository's GPT-2 weight tensors without adding PEFT or changing the lockfile.

## 5. Evaluation

Example eval JSONL:

```json
{"id":"ar_001","text":"Che, ¿querés venir mañana?","lang":"es","speaker_ref":"/data/eval/ana_ref.wav"}
{"id":"ar_002","text":"El total es de mil novecientos noventa y nueve pesos.","lang":"es","speaker_ref":"/data/eval/ana_ref.wav"}
```

Run stock and candidate adapters with the same suite and seeds:

```bash
python tools/finetune_indextts25/eval_suite.py \
  --suite /data/indextts/eval/dev.jsonl \
  --output /data/indextts/eval/stock \
  --model-dir checkpoints \
  --config checkpoints/config.yaml \
  --seeds 42,43,44

python tools/finetune_indextts25/eval_suite.py \
  --suite /data/indextts/eval/dev.jsonl \
  --output /data/indextts/eval/lr1e-5-step200 \
  --model-dir checkpoints \
  --config checkpoints/config.yaml \
  --adapter /data/indextts/runs/lr1e-5/adapter-step-000200.pt \
  --seeds 42,43,44

python tools/finetune_indextts25/score_wer.py \
  --receipt /data/indextts/eval/lr1e-5-step200/receipt.json \
  --model medium \
  --language es
```

Keep final-test prompts out of checkpoint selection. Select checkpoints by held-out intelligibility plus blinded accent/naturalness review, not by training loss alone.

## 6. Merge only after selecting a winner

```bash
python tools/finetune_indextts25/merge_lora.py \
  --config checkpoints/config.yaml \
  --model-dir checkpoints \
  --adapter /data/indextts/runs/lr1e-5/adapter-step-000200.pt \
  --output /data/indextts/releases/gpt-latam-step200.pth
```

Then evaluate the merged checkpoint independently before quantization or deployment.

## Multi-GPU guidance

Start single-GPU. Use the other 3090s for feature extraction/evaluation or a second controlled arm. Do not introduce DDP until the single-GPU trainer is verified; DDP changes throughput and global-batch semantics but does not pool three 24 GB GPUs into a single 72 GB memory space.
