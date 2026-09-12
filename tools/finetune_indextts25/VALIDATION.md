# Review and validation scope

This follow-up reviews the toolkit merged by PR #1 at
`a9d03519292c3994bef6bbc606e76264ba3e38d8`. Runtime model implementation files,
production services, model weights, datasets and Actions workflows are unchanged.

## Executed locally

```text
python -m pytest -q tools/finetune_indextts25/tests --disable-warnings
54 passed
```

The review sandbox used Python 3.13.5 and PyTorch 2.10.0+cpu. All toolkit/test sources
also parsed with Python 3.10 grammar. These results do not validate the repository's
complete Python 3.10/3.11, PyTorch 2.8 and Transformers 4.52.1 dependency environment.
No dependencies were upgraded in the repository.

The CPU model fixture exercises actual PyTorch autograd, parametrizations, AdamW,
serialization and merge operations. Its prefix builder is the real
`UnifiedVoice.prepare_gpt_inputs` method, extracted from repository source using AST
so optional inference dependencies need not be imported. In the review sandbox,
that method was materialized as a source slice from the pinned repository revision;
the test uses the full checkout's method when run on the workstation. The remaining
model is an intentionally small causal fixture, not the pretrained IndexTTS network.

Coverage includes:

- language-prefix preservation, exact prefix embeddings/masking and EOS gradients;
- causal target alignment and combined context-capacity checks;
- stale/invalid feature formats, shapes, vocabularies and base identity;
- FP32 adapters on the target device (CPU and meta-device placement checks);
- frozen base parameters and rejection of corrupt/missing/extra adapter state;
- inference-model aliases, JSON-safe metadata and reload/merge logit parity;
- nonfinite/zero-gradient rejection before optimizer updates;
- train/eval mode restoration even when validation throws;
- interrupted/resumed versus uninterrupted training, bit-equal in the CPU fixture;
- changed-data/settings rejection and refusal to resume weights without optimizer state;
- relative audio paths, grouped splits, duplicate text/audio, safe IDs and split references;
- fraction roundoff, insufficient groups and refusal to silently truncate audio;
- incremental evaluation receipts, synthesis errors and truncation reporting;
- failure/pending/ASR-error inclusion in WER/CER denominators;
- atomic JSON writes and nonempty-output protection;
- direct and module CLI help for all six entry points without model imports.

## Not executed here

No actual IndexTTS checkpoint, Wav2Vec2-BERT, codec, CAMPPlus, Whisper model or
workstation dataset was available in the review sandbox. Full feature extraction,
CUDA/BF16 training, GPU memory/throughput measurements, real-checkpoint reload,
ASR accuracy, acoustic quality, zero-shot speaker retention and deployment were
not executed. The meta-device test is not a CUDA execution test.

## Required workstation acceptance

Run the same CPU suite in the pinned repository environment first. Then qualify
24 examples / 30 optimizer updates, confirm the loss trend and nonzero adapter
gradients, evaluate generated speech with the fixed suite, and independently evaluate
the exported merged checkpoint. Compare stock/adapter/merged outputs across seeds
and speakers. Preserve the known-good release; do not deploy or select the final
step automatically. All configuration starting points remain experimental.
