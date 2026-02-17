# Training Guide

## Overview

Training Swahili-100 follows a strict pipeline. Each phase must complete before the next begins.

```
Data Collection → Cleaning → Dedup → Filtering → Tokenizer Training → Tokenization → Model Training → Evaluation → Alignment → Export
```

## Phase 1: Data Pipeline

### Download raw data

```bash
python data/download.py --source all --output data/raw/
```

### Clean text

```bash
python data/clean.py --input data/raw/ --output data/cleaned/
```

### Deduplicate

```bash
python data/dedup.py --input data/cleaned/ --output data/deduped/ --threshold 0.8
```

### Quality filter

```bash
python data/filter.py --input data/deduped/ --output data/filtered/
```

## Phase 2: Tokenizer

### Train BPE tokenizer

```bash
python scripts/train_tokenizer.py --input data/filtered/ --vocab_size 16384 --output tokenizer/
```

### Tokenize corpus to binary

```bash
python data/loader.py --input data/filtered/ --output data/processed/train --tokenizer tokenizer/swahili_bpe.json
```

## Phase 3: Debug Run

Always run a debug configuration first to verify the pipeline works end-to-end.

```bash
python train.py --config configs/debug.yaml
```

This should:
- Complete in under 5 minutes on CPU
- Show decreasing loss
- Save a checkpoint

If it fails, fix the issue before proceeding.

## Phase 4: Ablations

Run hyperparameter searches on free/cheap GPUs:

```bash
# Learning rate sweep (modify config per run)
python train.py --config configs/debug.yaml
```

Key hyperparameters to ablate:
- Learning rate: [1e-4, 3e-4, 1e-3]
- Batch size: [64, 128, 256]
- Architecture: GQA groups, FFN size

## Phase 5: Full Training

```bash
python train.py --config configs/swahili_base.yaml
```

Monitor training via logs:
```bash
tail -f logs/training.jsonl | python -m json.tool
```

### Resuming from checkpoint

```bash
python train.py --config configs/swahili_base.yaml --resume checkpoints/step_10000.pt
```

## Phase 6: Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/latest.pt --metric all
```

## Phase 7: Alignment

```bash
# SFT
python scripts/align.py --stage sft --checkpoint checkpoints/latest.pt --data data/sft/

# DPO
python scripts/align.py --stage dpo --checkpoint checkpoints/sft_latest.pt --data data/dpo/
```

## Phase 8: Export

```bash
# Quantize
python scripts/quantize.py --checkpoint checkpoints/latest.pt --precision int4

# ONNX
python scripts/export.py --checkpoint checkpoints/latest.pt --format onnx

# HuggingFace
python scripts/export.py --checkpoint checkpoints/latest.pt --format huggingface --repo marcoharuni95/swahili-100
```

## Tips

- **Always check loss curves.** Loss should decrease monotonically during early training.
- **Monitor GPU memory.** If OOM, reduce micro_batch_size or enable gradient checkpointing.
- **Save checkpoints frequently.** GPU instances can be preempted.
- **Log everything.** Every decision goes in docs/RESEARCH.md.
