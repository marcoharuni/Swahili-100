"""
Tests for optimizers (AdamW and Muon).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from training.adamw import AdamW
from training.muon import Muon


class TestAdamW:
    """Tests for the AdamW optimizer."""

    def test_step_reduces_loss(self):
        """A few optimizer steps should reduce a simple quadratic loss."""
        w = torch.randn(10, requires_grad=True)
        target = torch.zeros(10)
        opt = AdamW([w], lr=0.1)

        initial_loss = ((w - target) ** 2).sum().item()

        for _ in range(50):
            opt.zero_grad()
            loss = ((w - target) ** 2).sum()
            loss.backward()
            opt.step()

        final_loss = ((w - target) ** 2).sum().item()
        assert final_loss < initial_loss

    def test_weight_decay(self):
        """Weight decay should shrink parameters toward zero."""
        w = torch.ones(10, requires_grad=True)
        opt = AdamW([w], lr=0.01, weight_decay=0.5)

        # Zero gradient — only weight decay should act
        opt.zero_grad()
        w.grad = torch.zeros_like(w)
        opt.step()

        # Weights should have decreased
        assert w.mean().item() < 1.0

    def test_state_dict_roundtrip(self):
        """State dict save/load should preserve optimizer state."""
        w = torch.randn(5, requires_grad=True)
        opt = AdamW([w], lr=0.01)

        # Do a step
        loss = (w ** 2).sum()
        loss.backward()
        opt.step()

        state = opt.state_dict()
        assert state["step_count"] == 1


class TestMuon:
    """Tests for the Muon optimizer."""

    def test_step_reduces_loss(self):
        """Muon should reduce loss on a simple problem."""
        w = torch.randn(4, 4, requires_grad=True)
        target = torch.zeros(4, 4)
        opt = Muon([w], lr=0.01)

        initial_loss = ((w - target) ** 2).sum().item()

        for _ in range(50):
            opt.zero_grad()
            loss = ((w - target) ** 2).sum()
            loss.backward()
            opt.step()

        final_loss = ((w - target) ** 2).sum().item()
        assert final_loss < initial_loss
