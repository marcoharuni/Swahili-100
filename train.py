"""
Swahili-100 — Top-Level Training Entry Point

Convenience wrapper that delegates to scripts/train.py.

Usage:
    python train.py --config configs/debug.yaml
    python train.py --config configs/swahili_base.yaml
    python train.py --config configs/swahili_base.yaml --resume checkpoints/step_10000.pt
"""

from scripts.train import main

if __name__ == "__main__":
    main()
