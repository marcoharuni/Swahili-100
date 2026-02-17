"""
Supervised Fine-Tuning (SFT)

Fine-tunes the pretrained model on instruction-response pairs
to produce a helpful assistant.

Data format:
    Each example is a dict with:
        "instruction": str — the user prompt
        "response": str — the desired model response

    The model is trained to predict the response tokens given
    the instruction, with loss computed only on response tokens.

Usage:
    python scripts/align.py --stage sft --config configs/swahili_base.yaml --data data/sft/
"""

import torch
from typing import Optional

from model.transformer import Transformer
from data.tokenizer import BPETokenizer
from training.adamw import AdamW
from training.scheduler import CosineScheduler


class SFTTrainer:
    """
    Supervised fine-tuning trainer.

    Args:
        model: Pretrained transformer model.
        tokenizer: Trained BPE tokenizer.
        lr: Learning rate (typically lower than pretraining).
        epochs: Number of SFT epochs.
    """

    def __init__(
        self,
        model: Transformer,
        tokenizer: BPETokenizer,
        lr: float = 1e-5,
        epochs: int = 3,
        max_seq_len: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.epochs = epochs
        self.max_seq_len = max_seq_len

    def format_example(self, instruction: str, response: str) -> dict:
        """
        Format an instruction-response pair into training tokens.

        Returns dict with 'input_ids' and 'labels' where labels
        are -100 (ignore) for instruction tokens and actual IDs
        for response tokens.
        """
        # TODO: Implement example formatting
        raise NotImplementedError

    def load_data(self, data_path: str) -> list[dict]:
        """Load SFT data from JSONL file."""
        # TODO: Implement data loading
        raise NotImplementedError

    def train(self, data_path: str) -> None:
        """Run supervised fine-tuning."""
        # TODO: Implement SFT training loop
        raise NotImplementedError
