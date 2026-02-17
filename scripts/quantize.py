"""
Quantization Script

Quantize a trained model for efficient inference.

Usage:
    # INT8
    python scripts/quantize.py --checkpoint checkpoints/latest.pt --precision int8

    # INT4
    python scripts/quantize.py --checkpoint checkpoints/latest.pt --precision int4 --group_size 128
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.quantize import quantize_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize Swahili-100 model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--precision", type=str, default="int8", choices=["int8", "int4"])
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--output", type=str, default="checkpoints/quantized/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f"model_{args.precision}.pt")

    print(f"Quantizing to {args.precision}...")
    stats = quantize_model(
        args.checkpoint,
        output_path,
        precision=args.precision,
        group_size=args.group_size,
    )
    print(f"Done. Saved to {output_path}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
