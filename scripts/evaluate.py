"""
Evaluation Script

Run evaluation benchmarks on a trained model.

Usage:
    # Perplexity
    python scripts/evaluate.py --checkpoint checkpoints/latest.pt --metric perplexity

    # All benchmarks
    python scripts/evaluate.py --checkpoint checkpoints/latest.pt --metric all

    # NER only
    python scripts/evaluate.py --checkpoint checkpoints/latest.pt --metric ner
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Swahili-100 model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default=None, help="Config file (optional)")
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=["perplexity", "bpc", "ner", "sentiment", "translation", "generation", "all"],
        help="Evaluation metric to compute",
    )
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    # TODO: Implement evaluation dispatch
    # 1. Load model from checkpoint
    # 2. Load tokenizer
    # 3. Run selected evaluation(s)
    # 4. Print and optionally save results
    print(f"Evaluating {args.checkpoint} on metric={args.metric}")
    raise NotImplementedError("Evaluation script not yet implemented")


if __name__ == "__main__":
    main()
