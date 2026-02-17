"""
Byte-Pair Encoding Tokenizer — From Scratch

Implements a complete BPE tokenizer without importing any tokenizer library:
1. Byte-level base vocabulary (256 byte tokens)
2. BPE merge learning from training corpus
3. Encoding (text -> token IDs)
4. Decoding (token IDs -> text)
5. Special token handling
6. Serialization (save/load to JSON)

The tokenizer is trained on Swahili text and optimized for Swahili morphology,
targeting < 1.5 tokens per word (vs 2-3x for English-centric tokenizers).

References:
    - Sennrich et al., 2016 — Neural Machine Translation of Rare Words with Subword Units
    - Karpathy's minbpe — https://github.com/karpathy/minbpe

Usage:
    # Train
    python data/tokenizer.py train --input data/filtered/ --vocab_size 16384 --output tokenizer/

    # Encode
    python data/tokenizer.py encode --tokenizer tokenizer/swahili_bpe.json --text "Habari yako"

    # Decode
    python data/tokenizer.py decode --tokenizer tokenizer/swahili_bpe.json --ids 100,200,300
"""

import argparse
import json
import os
from typing import Optional


# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer built from scratch.

    The base vocabulary consists of 256 byte values (0-255).
    BPE merges are learned from the training corpus and applied
    greedily during encoding.

    Attributes:
        merges: List of (pair, new_token_id) learned merge rules.
        vocab: Dict mapping token_id -> bytes.
        special_tokens: Dict mapping special token string -> token_id.
    """

    # Special token definitions
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"
    PAD_TOKEN = "<|pad|>"
    UNK_TOKEN = "<|unk|>"

    def __init__(self):
        self.merges: list[tuple[int, int]] = []
        self.vocab: dict[int, bytes] = {}
        self.special_tokens: dict[str, int] = {}
        self._build_base_vocab()

    def _build_base_vocab(self) -> None:
        """Initialize the 256 byte-level base vocabulary."""
        # TODO: Build base vocab — map each byte value (0-255) to itself
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """
        Train BPE merges from a text corpus.

        Steps:
            1. Encode text to bytes (UTF-8)
            2. Count all adjacent byte-pair frequencies
            3. Merge the most frequent pair, create new token
            4. Repeat until vocab_size is reached
            5. Register special tokens

        Args:
            text: Training text corpus (raw string).
            vocab_size: Target vocabulary size (including base 256 + special tokens).
            verbose: Print progress during training.
        """
        # TODO: Implement BPE training loop
        raise NotImplementedError

    def _count_pairs(self, token_ids: list[int]) -> dict:
        """Count frequency of all adjacent pairs in a token sequence."""
        # TODO: Implement pair counting
        raise NotImplementedError

    def _merge_pair(self, token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace all occurrences of `pair` in `token_ids` with `new_id`."""
        # TODO: Implement pair merging
        raise NotImplementedError

    def _register_special_tokens(self) -> None:
        """Add special tokens to vocabulary after BPE training."""
        # TODO: Register BOS, EOS, PAD, UNK tokens
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """
        Encode a string into a list of token IDs.

        Steps:
            1. Convert text to UTF-8 bytes
            2. Apply learned BPE merges greedily (in order of training)
            3. Optionally prepend BOS and append EOS

        Args:
            text: Input text string.
            add_bos: Whether to prepend BOS token.
            add_eos: Whether to append EOS token.

        Returns:
            List of integer token IDs.
        """
        # TODO: Implement encoding
        raise NotImplementedError

    def encode_batch(self, texts: list[str], **kwargs) -> list[list[int]]:
        """Encode a batch of strings."""
        return [self.encode(t, **kwargs) for t in texts]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back to a string.

        Steps:
            1. Map each token ID to its byte sequence via vocab
            2. Concatenate all byte sequences
            3. Decode bytes to UTF-8 string (with error handling)

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        # TODO: Implement decoding
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save tokenizer to a JSON file.

        Saves: merges, vocab (as hex-encoded bytes), special_tokens.
        """
        # TODO: Implement save
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from a JSON file."""
        # TODO: Implement load
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self.vocab) + len(self.special_tokens)

    def __len__(self) -> int:
        return self.vocab_size

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def compression_ratio(self, text: str) -> float:
        """
        Compute compression ratio: (num characters) / (num tokens).
        Higher is better — means fewer tokens per character.
        """
        tokens = self.encode(text, add_bos=False, add_eos=False)
        return len(text) / len(tokens) if tokens else 0.0

    def tokens_per_word(self, text: str) -> float:
        """
        Compute average tokens per whitespace-delimited word.
        Target: < 1.5 for Swahili.
        """
        words = text.split()
        if not words:
            return 0.0
        tokens = self.encode(text, add_bos=False, add_eos=False)
        return len(tokens) / len(words)


# ---------------------------------------------------------------------------
# Training script
# ---------------------------------------------------------------------------

def train_tokenizer(input_dir: str, vocab_size: int, output_dir: str) -> None:
    """
    Train a BPE tokenizer from text files in input_dir.

    Reads all .txt files, concatenates text (up to a limit),
    trains BPE, and saves the tokenizer.
    """
    # TODO: Implement training script
    #   1. Read text from input files
    #   2. Create BPETokenizer instance
    #   3. Call tokenizer.train(text, vocab_size)
    #   4. Save to output_dir
    raise NotImplementedError


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Swahili BPE Tokenizer")
    subparsers = parser.add_subparsers(dest="command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new tokenizer")
    train_parser.add_argument("--input", type=str, required=True)
    train_parser.add_argument("--vocab_size", type=int, default=16384)
    train_parser.add_argument("--output", type=str, default="tokenizer/")

    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode text")
    encode_parser.add_argument("--tokenizer", type=str, required=True)
    encode_parser.add_argument("--text", type=str, required=True)

    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Decode token IDs")
    decode_parser.add_argument("--tokenizer", type=str, required=True)
    decode_parser.add_argument("--ids", type=str, required=True, help="Comma-separated token IDs")

    args = parser.parse_args()

    if args.command == "train":
        train_tokenizer(args.input, args.vocab_size, args.output)
    elif args.command == "encode":
        tok = BPETokenizer.load(args.tokenizer)
        ids = tok.encode(args.text)
        print(f"Tokens: {ids}")
        print(f"Length: {len(ids)}")
        print(f"Tokens/word: {tok.tokens_per_word(args.text):.2f}")
    elif args.command == "decode":
        tok = BPETokenizer.load(args.tokenizer)
        ids = [int(x) for x in args.ids.split(",")]
        text = tok.decode(ids)
        print(f"Text: {text}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
