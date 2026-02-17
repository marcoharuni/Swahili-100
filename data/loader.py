"""
Data Loader

Loads tokenized data into batches for training. Implements:
- Memory-mapped file reading for large datasets
- Dynamic batching with sequence packing
- Shuffling across epochs
- Distributed data parallel support (shard assignment)

No dependency on torch.utils.data.DataLoader — built from scratch.

Usage:
    loader = DataLoader(
        data_path="data/processed/train",
        tokenizer_path="tokenizer/swahili_bpe.json",
        batch_size=8,
        max_seq_len=2048,
    )
    for batch in loader:
        # batch is a dict with 'input_ids' and 'labels' as tensors
        ...
"""

import os
from typing import Iterator, Optional


class DataLoader:
    """
    Custom data loader for pre-tokenized training data.

    Reads tokenized sequences from binary files, packs them into
    fixed-length sequences, and yields batches as tensors.

    Supports:
        - Sequence packing (concatenate short docs, split with EOS)
        - Epoch shuffling (file-level and chunk-level)
        - Distributed sharding (each rank reads a disjoint subset)
    """

    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 2048,
        batch_size: int = 8,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Args:
            data_path: Path to directory with tokenized .bin files.
            max_seq_len: Maximum sequence length per sample.
            batch_size: Number of sequences per batch.
            shuffle: Whether to shuffle data each epoch.
            seed: Random seed for shuffling.
            rank: Current process rank (for distributed training).
            world_size: Total number of processes.
        """
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        self._file_list: list[str] = []
        self._total_tokens: int = 0
        # TODO: Scan data_path, build file list, compute total tokens

    def _load_shard(self, filepath: str):
        """
        Memory-map a tokenized binary file and return token IDs.

        File format: flat array of uint16 token IDs.
        """
        # TODO: Implement memory-mapped loading with numpy or manual mmap
        raise NotImplementedError

    def _pack_sequences(self, token_ids) -> list:
        """
        Pack a stream of token IDs into fixed-length sequences.

        Concatenates documents separated by EOS tokens, then splits
        into chunks of max_seq_len.
        """
        # TODO: Implement sequence packing
        raise NotImplementedError

    def __iter__(self) -> Iterator[dict]:
        """
        Yield batches of {input_ids, labels} for training.

        input_ids: [batch_size, max_seq_len] — input token IDs
        labels:    [batch_size, max_seq_len] — shifted by 1 for next-token prediction
        """
        # TODO: Implement batch iteration
        raise NotImplementedError

    def __len__(self) -> int:
        """Estimated number of batches per epoch."""
        tokens_per_batch = self.batch_size * self.max_seq_len
        return self._total_tokens // (tokens_per_batch * self.world_size)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens


# ---------------------------------------------------------------------------
# Tokenization script (text -> binary)
# ---------------------------------------------------------------------------

def tokenize_corpus(
    input_dir: str,
    output_dir: str,
    tokenizer_path: str,
    max_shard_size: int = 100_000_000,
) -> None:
    """
    Tokenize a cleaned text corpus and save as binary shards.

    Reads .txt files from input_dir, encodes with the BPE tokenizer,
    and writes uint16 .bin files to output_dir.
    """
    # TODO: Implement corpus tokenization
    raise NotImplementedError


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Tokenize corpus to binary")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    args = parser.parse_args()

    tokenize_corpus(args.input, args.output, args.tokenizer)


if __name__ == "__main__":
    main()
