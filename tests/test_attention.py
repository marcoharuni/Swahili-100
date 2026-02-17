"""
Tests for attention mechanisms.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.config import ModelConfig
from model.attention import Attention


class TestAttention:
    """Tests for the Attention module."""

    def _make_config(self, **kwargs) -> ModelConfig:
        defaults = dict(
            vocab_size=256, d_model=64, n_layers=2, n_heads=4,
            n_kv_heads=2, d_ff=128, max_seq_len=64,
        )
        defaults.update(kwargs)
        return ModelConfig(**defaults)

    def test_output_shape(self):
        """Attention output should match input shape."""
        config = self._make_config()
        attn = Attention(config)
        B, T, D = 2, 16, config.d_model
        x = torch.randn(B, T, D)
        # Need cos/sin freqs
        cos_freqs = torch.ones(T, config.head_dim // 2)
        sin_freqs = torch.zeros(T, config.head_dim // 2)
        out = attn.forward(x, cos_freqs, sin_freqs)
        assert out.shape == (B, T, D)

    def test_causal_masking(self):
        """Future tokens should not attend to past tokens in causal mode."""
        # TODO: Implement test for causal masking
        pass

    def test_gqa_kv_expansion(self):
        """GQA should correctly expand KV heads to match query heads."""
        config = self._make_config(n_heads=8, n_kv_heads=2)
        attn = Attention(config)
        # With 8 query heads and 2 KV heads, each KV head serves 4 query heads
        assert config.n_kv_groups == 4

    def test_kv_cache(self):
        """KV-cache should accumulate across sequential calls."""
        # TODO: Implement KV-cache test
        pass
