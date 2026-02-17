"""
Full Transformer Model

Assembles the complete decoder-only transformer:
    1. Token embedding
    2. N transformer blocks
    3. Final RMSNorm
    4. LM head (with optional weight tying)

This is the central model class that connects all components.
"""

import torch
from typing import Optional

from model.config import ModelConfig
from model.embedding import TokenEmbedding
from model.rope import precompute_freqs
from model.norm import RMSNorm
from model.block import TransformerBlock
from model.lm_head import LMHead


class Transformer:
    """
    Decoder-only transformer language model.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig):
        config.validate()
        self.config = config

        # Token embedding
        self.embedding = TokenEmbedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.blocks: list[TransformerBlock] = [
            TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)
        ]

        # Final normalization
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)

        # Language model head
        self.lm_head = LMHead(
            config.d_model,
            config.vocab_size,
            tie_weights=config.tie_embeddings,
            embedding_weight=self.embedding.weight if config.tie_embeddings else None,
        )

        # Multi-token prediction heads (optional)
        self.mtp_heads: Optional[list] = None
        if config.use_mtp:
            # TODO: Initialize MTP heads
            pass

        # Precompute RoPE frequencies
        self.cos_freqs: torch.Tensor = None  # type: ignore
        self.sin_freqs: torch.Tensor = None  # type: ignore
        # TODO: Call precompute_freqs and store results

        # Total parameter count
        self._param_count: Optional[int] = None

    def forward(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> dict:
        """
        Full forward pass.

        Args:
            token_ids: [batch, seq_len] integer token IDs.
            start_pos: Position offset for inference (KV-cache).
            use_cache: Enable KV-cache for inference.

        Returns:
            Dict with:
                'logits': [batch, seq_len, vocab_size]
                'aux_loss': total MoE auxiliary loss (if using MoE)
                'mtp_logits': list of [batch, seq_len, vocab_size] (if using MTP)
        """
        # TODO: Implement full forward pass
        # 1. Embed tokens
        # 2. Build causal mask
        # 3. Pass through each transformer block
        # 4. Final norm
        # 5. LM head -> logits
        # 6. Optional MTP heads
        raise NotImplementedError

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build causal attention mask.

        Lower-triangular mask where position i can attend to positions [0, i].
        """
        # TODO: Implement causal mask construction
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        """Return all learnable parameters (flat list)."""
        params = []
        params.extend(self.embedding.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_norm.parameters())
        if not self.config.tie_embeddings:
            params.extend(self.lm_head.parameters())
        return params

    def param_count(self) -> int:
        """Count total learnable parameters."""
        if self._param_count is None:
            self._param_count = sum(p.numel() for p in self.parameters())
        return self._param_count

    def to(self, device: torch.device) -> "Transformer":
        """Move all parameters to device."""
        # TODO: Implement device transfer for all parameters
        raise NotImplementedError

    def train_mode(self) -> None:
        """Set model to training mode (enable dropout if any)."""
        # TODO: Implement training mode
        pass

    def eval_mode(self) -> None:
        """Set model to evaluation mode (disable dropout)."""
        # TODO: Implement eval mode
        pass

    def clear_caches(self) -> None:
        """Clear all KV-caches across blocks."""
        for block in self.blocks:
            block.attention.clear_cache()
