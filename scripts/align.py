"""
Alignment Script

Run post-training alignment (SFT and/or DPO).

Usage:
    # Supervised Fine-Tuning
    python scripts/align.py --stage sft --checkpoint checkpoints/latest.pt --data data/sft/

    # Direct Preference Optimization
    python scripts/align.py --stage dpo --checkpoint checkpoints/sft_latest.pt --data data/dpo/
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training alignment")
    parser.add_argument("--stage", type=str, required=True, choices=["sft", "dpo"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Alignment data path")
    parser.add_argument("--config", type=str, default=None, help="Config file")
    parser.add_argument("--output", type=str, default="checkpoints/", help="Output directory")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    args = parser.parse_args()

    # TODO: Implement alignment dispatch
    # 1. Load model from checkpoint
    # 2. Based on stage, create SFTTrainer or DPOTrainer
    # 3. Run training
    # 4. Save aligned model
    print(f"Running {args.stage} alignment")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    raise NotImplementedError("Alignment script not yet implemented")


if __name__ == "__main__":
    main()
