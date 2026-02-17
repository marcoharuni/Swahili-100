"""
Tests for text generation.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.generate import sample_top_k, sample_top_p, apply_temperature


class TestSampling:
    """Tests for sampling strategies."""

    def test_top_k_restricts_choices(self):
        """Top-k should only sample from the top k tokens."""
        logits = torch.randn(100)
        # Set one token much higher
        logits[42] = 100.0
        token = sample_top_k(logits, k=1)
        assert token.item() == 42

    def test_top_p_restricts_choices(self):
        """Top-p should sample from the nucleus."""
        logits = torch.zeros(100)
        logits[0] = 100.0  # Almost all probability mass
        token = sample_top_p(logits, p=0.9)
        assert token.item() == 0

    def test_temperature_scaling(self):
        """Temperature 0 should approach argmax behavior."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        scaled = apply_temperature(logits, temperature=0.01)
        # Very low temperature should make the max much more dominant
        probs = torch.softmax(scaled, dim=-1)
        assert probs[2] > 0.99

    def test_greedy_is_argmax(self):
        """Greedy decoding should always pick the highest logit."""
        logits = torch.tensor([1.0, 5.0, 2.0, 3.0])
        token = sample_top_k(logits, k=1)
        assert token.item() == 1  # index of max value
