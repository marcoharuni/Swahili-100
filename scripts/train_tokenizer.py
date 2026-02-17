"""
Tokenizer Training Script

Trains a BPE tokenizer on the cleaned Swahili corpus.

Usage:
    python scripts/train_tokenizer.py \
        --input data/filtered/ \
        --vocab_size 16384 \
        --output tokenizer/
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tokenizer import BPETokenizer, train_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Swahili BPE tokenizer")
    parser.add_argument("--input", type=str, required=True, help="Input text directory")
    parser.add_argument("--vocab_size", type=int, default=16384, help="Target vocab size")
    parser.add_argument("--output", type=str, default="tokenizer/", help="Output directory")
    args = parser.parse_args()

    print(f"Training BPE tokenizer: vocab_size={args.vocab_size}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    train_tokenizer(args.input, args.vocab_size, args.output)

    # Validate
    tok = BPETokenizer.load(os.path.join(args.output, "swahili_bpe.json"))
    test_text = "Habari yako, jina langu ni Marco."
    encoded = tok.encode(test_text)
    decoded = tok.decode(encoded)
    print(f"\nValidation:")
    print(f"  Text:    {test_text}")
    print(f"  Tokens:  {encoded}")
    print(f"  Decoded: {decoded}")
    print(f"  Tokens/word: {tok.tokens_per_word(test_text):.2f}")
    print(f"  Vocab size:  {tok.vocab_size}")


if __name__ == "__main__":
    main()
