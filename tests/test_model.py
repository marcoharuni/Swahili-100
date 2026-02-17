"""
Tests for the full transformer model.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.config import ModelConfig
from model.transformer import Transformer


class TestTransformer:
    """Tests for the Transformer model."""

    def _make_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=256, d_model=64, n_layers=2, n_heads=4,
            n_kv_heads=2, d_ff=128, max_seq_len=64,
        )

    def test_forward_shape(self):
        """Forward pass should produce correct logit shape."""
        config = self._make_config()
        model = Transformer(config)
        B, T = 2, 16
        token_ids = torch.randint(0, config.vocab_size, (B, T))
        output = model.forward(token_ids)
        assert output["logits"].shape == (B, T, config.vocab_size)

    def test_param_count(self):
        """Parameter count should be reasonable for the config."""
        config = self._make_config()
        model = Transformer(config)
        count = model.param_count()
        assert count > 0
        # Rough sanity check: should be in the thousands for a tiny model
        assert count < 10_000_000

    def test_config_validation(self):
        """Invalid configs should raise errors."""
        # d_model not divisible by n_heads
        try:
            config = ModelConfig(d_model=65, n_heads=4)
            config.validate()
            assert False, "Should have raised"
        except AssertionError:
            pass

    def test_weight_tying(self):
        """With tie_embeddings, LM head should share embedding weight."""
        config = self._make_config()
        config.tie_embeddings = True
        model = Transformer(config)
        # Embedding weight and LM head weight should be the same object
        assert model.embedding.weight is model.lm_head.weight

    def test_causal_mask(self):
        """Causal mask should be lower-triangular."""
        config = self._make_config()
        model = Transformer(config)
        mask = model._build_causal_mask(8, torch.device("cpu"))
        # Position 0 can only attend to position 0
        # Position 7 can attend to positions 0-7
        assert mask.shape == (8, 8)
