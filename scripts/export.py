"""
Model Export Script

Export trained model to various formats for deployment.

Usage:
    # ONNX
    python scripts/export.py --checkpoint checkpoints/latest.pt --format onnx

    # HuggingFace upload
    python scripts/export.py --checkpoint checkpoints/latest.pt --format huggingface --repo marcoharuni95/swahili-100
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Swahili-100 model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--format", type=str, required=True, choices=["onnx", "huggingface"])
    parser.add_argument("--output", type=str, default="exports/")
    parser.add_argument("--repo", type=str, default="marcoharuni95/swahili-100",
                        help="HuggingFace repo ID (for huggingface format)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.format == "onnx":
        # TODO: Implement ONNX export
        print("Exporting to ONNX...")
        raise NotImplementedError
    elif args.format == "huggingface":
        # TODO: Implement HuggingFace upload
        print(f"Uploading to HuggingFace: {args.repo}")
        raise NotImplementedError


if __name__ == "__main__":
    main()
