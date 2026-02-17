# Environment Setup

## Requirements

- Python 3.10+
- PyTorch 2.2+ (for tensor ops and autograd only — no nn.Module usage)
- CUDA 12.0+ (for GPU training, optional)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/marcoharuni/swahili-100.git
cd swahili-100
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 5. (Optional) Install Triton for fused kernels

```bash
pip install triton
```

## GPU Providers

For the final training run, you will need a rented GPU. Recommended providers:

| Provider | GPU | Cost/hr | Notes |
|---|---|---|---|
| Lambda Cloud | H100 | ~$2.50 | Best for training |
| RunPod | A100/H100 | ~$2-3 | Good availability |
| Vast.ai | Various | Variable | Cheapest |
| Google Colab | T4/A100 | Free-$10 | Good for ablations |

## Directory Setup

After cloning, create the data directories:

```bash
mkdir -p data/raw data/cleaned data/deduped data/filtered data/processed/train data/processed/val
```

## Weights & Biases (Optional)

For experiment tracking:

```bash
pip install wandb
wandb login
```

Then set `logging.wandb_project` in your config YAML.
