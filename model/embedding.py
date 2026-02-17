"""
Token Embeddings

Implements token embedding lookup from scratch.
No nn.Embedding — just a raw weight matrix and indexing.

Features:
    - Learned embedding matrix [vocab_size, d_model]
    - Xavier/Kaiming initialization
    - Optional scaling by sqrt(d_model)
    - Weight tying support (shared with LM head)
"""

import math

# NOTE: We use torch only for tensor operations and autograd.
#       No nn.Module, no nn.Embedding — everything is manual.
import torch


class TokenEmbedding:
    """
    Token embedding layer.

    Maps integer token IDs to dense vectors via a learned embedding matrix.

    Args:
        vocab_size: Number of tokens in vocabulary.
        d_model: Embedding dimension.
        scale: Whether to scale embeddings by sqrt(d_model).
    """

    def __init__(self, vocab_size: int, d_model: int, scale: bool = True):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.scale = scale

        # Embedding weight matrix — manually initialized
        # Shape: [vocab_size, d_model]
        self.weight: torch.Tensor = None  # type: ignore
        # TODO: Initialize weight with appropriate distribution

    def _init_weights(self) -> None:
        """Initialize embedding weights."""
        # TODO: Implement weight initialization
        # Standard practice: normal(0, 1/sqrt(d_model)) or uniform
        raise NotImplementedError

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for token IDs.

        Args:
            token_ids: [batch_size, seq_len] integer tensor.

        Returns:
            [batch_size, seq_len, d_model] embedding tensor.
        """
        # TODO: Implement embedding lookup
        # 1. Index into self.weight using token_ids
        # 2. Optionally scale by sqrt(d_model)
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        """Return list of learnable parameters."""
        return [self.weight]
