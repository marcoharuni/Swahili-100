"""
Attention Mechanisms

Implements multiple attention variants from scratch:

1. Multi-Head Attention (MHA) — standard scaled dot-product attention
2. Grouped-Query Attention (GQA) — fewer KV heads, shared across query groups
3. Multi-Head Latent Attention (MLA) — DeepSeek-style compressed KV

All implementations support:
    - Causal masking (for autoregressive generation)
    - QK-Norm (for training stability)
    - KV-Cache (for efficient inference)
    - RoPE integration (applied to Q and K)

No nn.Linear, no nn.MultiheadAttention — all projections are raw matmuls.

References:
    Vaswani et al., 2017 — Attention Is All You Need
    Ainslie et al., 2023 — GQA: Training Generalized Multi-Query Transformer Models
    DeepSeek-AI, 2024 — DeepSeek-V2: A Strong, Economical, and Efficient MoE LM
"""

import math
import torch
from typing import Optional

from model.config import ModelConfig


class Attention:
    """
    Grouped-Query Attention with RoPE and optional QK-Norm.

    Supports standard MHA (n_kv_heads == n_heads) and
    GQA (n_kv_heads < n_heads) configurations.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # --- Projection weights (no bias, following LLaMA) ---
        # W_q: [d_model, n_heads * head_dim]
        self.w_q: torch.Tensor = None  # type: ignore
        # W_k: [d_model, n_kv_heads * head_dim]
        self.w_k: torch.Tensor = None  # type: ignore
        # W_v: [d_model, n_kv_heads * head_dim]
        self.w_v: torch.Tensor = None  # type: ignore
        # W_o: [n_heads * head_dim, d_model]
        self.w_o: torch.Tensor = None  # type: ignore

        # TODO: Initialize all projection weights

        # --- Optional QK-Norm ---
        self.qk_norm = config.qk_norm
        # TODO: Initialize RMSNorm for Q and K if qk_norm is True

        # --- KV-Cache for inference ---
        self._cache_k: Optional[torch.Tensor] = None
        self._cache_v: Optional[torch.Tensor] = None

    def _init_weights(self) -> None:
        """Initialize projection weights."""
        # TODO: Implement weight initialization
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        cos_freqs: torch.Tensor,
        sin_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for grouped-query attention.

        Args:
            x: Input tensor [batch, seq_len, d_model].
            cos_freqs: RoPE cosine frequencies.
            sin_freqs: RoPE sine frequencies.
            mask: Causal attention mask [seq_len, seq_len] or None.
            start_pos: Start position for KV-cache (inference).
            use_cache: Whether to use/update KV-cache.

        Returns:
            Output tensor [batch, seq_len, d_model].
        """
        # TODO: Implement full attention forward pass:
        # 1. Project x -> Q, K, V via linear projections (matmul, no bias)
        # 2. Reshape to [batch, seq_len, n_heads/n_kv_heads, head_dim]
        # 3. Apply QK-Norm if enabled
        # 4. Apply RoPE to Q and K
        # 5. Expand KV heads for GQA (repeat n_kv_groups times)
        # 6. If use_cache: update and use KV-cache
        # 7. Compute attention: softmax(Q @ K^T / sqrt(d)) @ V
        # 8. Apply causal mask
        # 9. Project output via W_o
        raise NotImplementedError

    def _expand_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand KV heads for GQA.

        If n_kv_heads < n_heads, repeat each KV head to match
        the number of query heads in its group.

        [batch, seq, n_kv_heads, head_dim] -> [batch, seq, n_heads, head_dim]
        """
        # TODO: Implement KV head expansion
        raise NotImplementedError

    def clear_cache(self) -> None:
        """Clear KV-cache (call between sequences during inference)."""
        self._cache_k = None
        self._cache_v = None

    def parameters(self) -> list[torch.Tensor]:
        """Return all learnable parameters."""
        params = [self.w_q, self.w_k, self.w_v, self.w_o]
        # TODO: Add QK-Norm parameters if enabled
        return params


class MultiHeadLatentAttention:
    """
    Multi-Head Latent Attention (MLA) — DeepSeek-V2 style.

    Compresses KV representations through a low-rank bottleneck
    to reduce KV-cache memory during inference.

    Instead of caching full K and V:
        - Compress: c = W_dkv @ x  (low-rank bottleneck)
        - Expand:   K = W_uk @ c, V = W_uv @ c

    This dramatically reduces KV-cache size.

    Args:
        config: Model configuration.
        kv_compression_dim: Dimension of the compressed KV representation.
    """

    def __init__(self, config: ModelConfig, kv_compression_dim: int = 128):
        self.config = config
        self.kv_compression_dim = kv_compression_dim

        # Compression weights
        # W_dkv: [d_model, kv_compression_dim] — down-project
        self.w_dkv: torch.Tensor = None  # type: ignore
        # W_uk: [kv_compression_dim, n_kv_heads * head_dim] — up-project K
        self.w_uk: torch.Tensor = None  # type: ignore
        # W_uv: [kv_compression_dim, n_kv_heads * head_dim] — up-project V
        self.w_uv: torch.Tensor = None  # type: ignore

        # Standard Q projection
        self.w_q: torch.Tensor = None  # type: ignore
        # Output projection
        self.w_o: torch.Tensor = None  # type: ignore

        # TODO: Initialize all weights

    def forward(
        self,
        x: torch.Tensor,
        cos_freqs: torch.Tensor,
        sin_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        MLA forward pass.

        During inference, only the compressed KV representation (c) is cached,
        not the full K and V — reducing cache size by (d_model / kv_compression_dim)x.
        """
        # TODO: Implement MLA forward pass
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        return [self.w_dkv, self.w_uk, self.w_uv, self.w_q, self.w_o]
