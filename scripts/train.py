"""
Main Training Script

Entry point for model training. Parses command-line arguments,
loads configuration, and launches the training loop.

Usage:
    # Debug run
    python scripts/train.py --config configs/debug.yaml

    # Full training
    python scripts/train.py --config configs/swahili_base.yaml

    # Resume from checkpoint
    python scripts/train.py --config configs/swahili_base.yaml --resume checkpoints/step_10000.pt

    # Multi-GPU (single node)
    torchrun --nproc_per_node=2 scripts/train.py --config configs/swahili_base.yaml
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Swahili-100 model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    trainer = Trainer(config_path=args.config, resume_from=args.resume)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
