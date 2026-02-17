"""
Tests for the training loop.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.scheduler import CosineScheduler
from training.grad_accumulation import GradientAccumulator
from training.loss import cross_entropy_loss

import torch


class TestCosineScheduler:
    """Tests for the cosine LR scheduler."""

    def test_warmup_start(self):
        """LR should start near 0 during warmup."""
        sched = CosineScheduler(max_lr=3e-4, min_lr=3e-5, warmup_steps=100, total_steps=1000)
        lr = sched.get_lr(0)
        assert lr < 1e-5  # Should be near 0 at step 0

    def test_warmup_end(self):
        """LR should reach max_lr at end of warmup."""
        sched = CosineScheduler(max_lr=3e-4, min_lr=3e-5, warmup_steps=100, total_steps=1000)
        lr = sched.get_lr(100)
        assert abs(lr - 3e-4) < 1e-7

    def test_end_lr(self):
        """LR should reach min_lr at end of training."""
        sched = CosineScheduler(max_lr=3e-4, min_lr=3e-5, warmup_steps=100, total_steps=1000)
        lr = sched.get_lr(1000)
        assert abs(lr - 3e-5) < 1e-7


class TestGradientAccumulator:
    """Tests for gradient accumulation."""

    def test_should_step_timing(self):
        """should_step should be True only at the last micro-step."""
        accum = GradientAccumulator(accumulation_steps=4)
        steps = []
        for i in range(8):
            steps.append(accum.should_step)
            accum.advance()
        # Should be True at indices 3, 7 (every 4th step)
        assert steps == [False, False, False, True, False, False, False, True]

    def test_loss_scale(self):
        """Loss scale should be 1/accumulation_steps."""
        accum = GradientAccumulator(accumulation_steps=16)
        assert abs(accum.loss_scale - 1.0 / 16) < 1e-7


class TestCrossEntropyLoss:
    """Tests for cross-entropy loss."""

    def test_perfect_prediction(self):
        """Loss should be near 0 for perfect predictions."""
        vocab_size = 10
        logits = torch.zeros(1, 5, vocab_size)
        targets = torch.zeros(1, 5, dtype=torch.long)
        # Make logits strongly predict the correct class
        for i in range(5):
            logits[0, i, targets[0, i]] = 100.0
        loss = cross_entropy_loss(logits, targets)
        assert loss.item() < 0.01

    def test_uniform_prediction(self):
        """Loss should be log(vocab_size) for uniform predictions."""
        import math
        vocab_size = 10
        logits = torch.zeros(1, 5, vocab_size)  # uniform
        targets = torch.zeros(1, 5, dtype=torch.long)
        loss = cross_entropy_loss(logits, targets)
        expected = math.log(vocab_size)
        assert abs(loss.item() - expected) < 0.1
