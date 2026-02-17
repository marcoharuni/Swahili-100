"""
Perplexity and Bits-Per-Character Evaluation

Intrinsic evaluation metrics for language models:

1. Perplexity (PPL): exp(average cross-entropy loss)
   - Lower is better
   - Depends on tokenizer (not comparable across different tokenizers)

2. Bits-Per-Character (BPC): total_bits / total_characters
   - Lower is better
   - Tokenizer-agnostic (comparable across models)
   - BPC = (total_loss * total_tokens * log2(e)) / total_characters

Usage:
    python scripts/evaluate.py --metric perplexity --data data/processed/val/
"""

import math
import torch
from typing import Optional

from model.transformer import Transformer
from data.loader import DataLoader
from data.tokenizer import BPETokenizer


@torch.no_grad()
def compute_perplexity(
    model: Transformer,
    data_loader: DataLoader,
    max_steps: Optional[int] = None,
) -> dict:
    """
    Compute perplexity on a dataset.

    Args:
        model: Trained transformer model.
        data_loader: Validation data loader.
        max_steps: Maximum evaluation steps (None = full dataset).

    Returns:
        Dict with 'perplexity', 'avg_loss', 'total_tokens'.
    """
    # TODO: Implement perplexity computation
    # 1. Iterate over batches
    # 2. Forward pass, compute cross-entropy loss
    # 3. Accumulate total loss and token count
    # 4. PPL = exp(total_loss / total_tokens)
    raise NotImplementedError


@torch.no_grad()
def compute_bpc(
    model: Transformer,
    tokenizer: BPETokenizer,
    text: str,
    max_seq_len: int = 2048,
) -> dict:
    """
    Compute bits-per-character on a text string.

    BPC is tokenizer-agnostic — it measures how many bits the model
    needs per character, making it comparable across different
    tokenizer configurations.

    Args:
        model: Trained transformer model.
        tokenizer: BPE tokenizer.
        text: Raw text string to evaluate.
        max_seq_len: Maximum sequence length for chunking.

    Returns:
        Dict with 'bpc', 'total_bits', 'total_chars'.
    """
    # TODO: Implement BPC computation
    # 1. Encode text to tokens
    # 2. Split into chunks of max_seq_len
    # 3. For each chunk: forward pass, compute loss per token
    # 4. total_bits = sum(loss_per_token) * log2(e)
    # 5. BPC = total_bits / len(text)
    raise NotImplementedError
