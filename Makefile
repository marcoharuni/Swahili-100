.PHONY: help setup test lint clean data tokenizer train eval generate quantize export

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies
	pip install -r requirements.txt

setup-dev:  ## Install dev dependencies
	pip install -e ".[dev]"

test:  ## Run test suite
	python -m pytest tests/ -v

test-cov:  ## Run tests with coverage
	python -m pytest tests/ -v --cov=model --cov=data --cov=training

clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/

# --- Data Pipeline ---

data-download:  ## Download raw Swahili data
	python data/download.py --source all --output data/raw/

data-clean:  ## Clean raw text
	python data/clean.py --input data/raw/ --output data/cleaned/

data-dedup:  ## Deduplicate cleaned text
	python data/dedup.py --input data/cleaned/ --output data/deduped/

data-filter:  ## Quality filter
	python data/filter.py --input data/deduped/ --output data/filtered/

data: data-download data-clean data-dedup data-filter  ## Run full data pipeline

# --- Tokenizer ---

tokenizer:  ## Train BPE tokenizer
	python scripts/train_tokenizer.py --input data/filtered/ --vocab_size 16384 --output tokenizer/

tokenize:  ## Tokenize corpus to binary
	python data/loader.py --input data/filtered/ --output data/processed/train --tokenizer tokenizer/swahili_bpe.json

# --- Training ---

train-debug:  ## Run debug training (CPU, fast)
	python train.py --config configs/debug.yaml

train:  ## Run full training
	python train.py --config configs/swahili_base.yaml

# --- Evaluation ---

eval:  ## Run all evaluations
	python scripts/evaluate.py --checkpoint checkpoints/latest.pt --metric all

# --- Generation ---

generate:  ## Interactive text generation
	python scripts/generate.py --checkpoint checkpoints/latest.pt --interactive

# --- Post-training ---

align-sft:  ## Run supervised fine-tuning
	python scripts/align.py --stage sft --checkpoint checkpoints/latest.pt --data data/sft/

align-dpo:  ## Run DPO alignment
	python scripts/align.py --stage dpo --checkpoint checkpoints/sft_latest.pt --data data/dpo/

# --- Export ---

quantize-int8:  ## Quantize model to INT8
	python scripts/quantize.py --checkpoint checkpoints/latest.pt --precision int8

quantize-int4:  ## Quantize model to INT4
	python scripts/quantize.py --checkpoint checkpoints/latest.pt --precision int4

export-onnx:  ## Export to ONNX
	python scripts/export.py --checkpoint checkpoints/latest.pt --format onnx

export-hf:  ## Upload to HuggingFace
	python scripts/export.py --checkpoint checkpoints/latest.pt --format huggingface
