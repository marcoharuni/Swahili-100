"""
Language Model Head

Maps transformer hidden states to vocabulary logits.

Features:
    - Weight tying with token embeddings (optional)
    - Multi-Token Prediction (MTP) auxiliary heads

With weight tying, the LM head shares the embedding matrix:
    logits = hidden @ embedding_weight^T

This reduces parameter count and often improves performance.

Multi-Token Prediction (MTP):
    Additional lightweight heads that predict tokens at positions +2, +3, etc.
    This provides auxiliary training signal and can improve representation quality.

Reference:
    Gloeckle et al., 2024 — Better & Faster Large Language Models via Multi-token Prediction
"""

import torch
from typing import Optional


class LMHead:
    """
    Language model output head.

    Projects hidden states [batch, seq_len, d_model] to logits [batch, seq_len, vocab_size].

    Args:
        d_model: Hidden dimension.
        vocab_size: Vocabulary size.
        tie_weights: If True, share weights with token embedding.
        embedding_weight: Embedding weight tensor (required if tie_weights=True).
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tie_weights: bool = True,
        embedding_weight: Optional[torch.Tensor] = None,
    ):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights

        if tie_weights:
            assert embedding_weight is not None, \
                "Must provide embedding_weight when tie_weights=True"
            self.weight = embedding_weight  # Shared reference
        else:
            self.weight: torch.Tensor = None  # type: ignore
            # TODO: Initialize separate LM head weight [vocab_size, d_model]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to vocabulary logits.

        Args:
            hidden: [batch, seq_len, d_model]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # TODO: Implement: logits = hidden @ weight^T
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        if self.tie_weights:
            return []  # Weight is shared with embedding
        return [self.weight]


class MultiTokenPredictionHead:
    """
    Auxiliary head for predicting future tokens.

    For predicting token at position +k (k > 1), this head takes the
    hidden state at position i and predicts the token at position i+k.

    Architecture:
        A lightweight projection: hidden -> intermediate -> logits
        (or shared LM head weight with a learned offset)

    Args:
        d_model: Hidden dimension.
        vocab_size: Vocabulary size.
        lm_head_weight: Optional shared weight from main LM head.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        lm_head_weight: Optional[torch.Tensor] = None,
    ):
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Projection to transform hidden state for this prediction offset
        self.w_proj: torch.Tensor = None  # [d_model, d_model]
        # TODO: Initialize projection weight

        # Share or create LM head weight
        self.lm_weight = lm_head_weight  # Can be shared
        if lm_head_weight is None:
            self.lm_weight: torch.Tensor = None  # type: ignore
            # TODO: Initialize separate weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Predict future token logits from current hidden states.

        Args:
            hidden: [batch, seq_len, d_model]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # TODO: Implement MTP forward
        # projected = hidden @ self.w_proj (transform for this offset)
        # logits = projected @ self.lm_weight^T
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        params = [self.w_proj]
        if self.lm_weight is not None:
            params.append(self.lm_weight)
        return params
