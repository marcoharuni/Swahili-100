"""
Transformer Block

A single transformer decoder block combining:
    1. Pre-norm (RMSNorm)
    2. Attention (GQA or MLA)
    3. Residual connection
    4. Pre-norm (RMSNorm)
    5. Feedforward (SwiGLU or MoE)
    6. Residual connection

This is the fundamental repeating unit of the model.
N blocks are stacked to form the full transformer.
"""

import torch
from typing import Optional

from model.config import ModelConfig
from model.norm import RMSNorm
from model.attention import Attention, MultiHeadLatentAttention
from model.feedforward import SwiGLU
from model.moe import MoELayer


class TransformerBlock:
    """
    Single transformer decoder block.

    Pre-norm architecture:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Args:
        config: Model configuration.
        layer_idx: Index of this block in the stack (for debugging/logging).
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        self.config = config
        self.layer_idx = layer_idx

        # Pre-attention norm
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)

        # Attention
        self.attention = Attention(config)

        # Pre-FFN norm
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)

        # Feedforward (SwiGLU or MoE)
        if config.use_moe:
            self.feedforward = MoELayer(config)
        else:
            self.feedforward = SwiGLU(config.d_model, config.d_ff)

        self.use_moe = config.use_moe

    def forward(
        self,
        x: torch.Tensor,
        cos_freqs: torch.Tensor,
        sin_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, float]:
        """
        Forward pass for one transformer block.

        Args:
            x: [batch, seq_len, d_model]
            cos_freqs, sin_freqs: RoPE frequency tensors.
            mask: Causal mask.
            start_pos: Position offset for KV-cache.
            use_cache: Enable KV-cache.

        Returns:
            output: [batch, seq_len, d_model]
            aux_loss: MoE auxiliary loss (0.0 if not using MoE)
        """
        # TODO: Implement transformer block forward pass
        # 1. attn_out = attention(attn_norm(x), ...)
        # 2. x = x + attn_out  (residual)
        # 3. ffn_out = feedforward(ffn_norm(x))
        # 4. x = x + ffn_out   (residual)
        # 5. Return x, aux_loss
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        params = []
        params.extend(self.attn_norm.parameters())
        params.extend(self.attention.parameters())
        params.extend(self.ffn_norm.parameters())
        params.extend(self.feedforward.parameters())
        return params
