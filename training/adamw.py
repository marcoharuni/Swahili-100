"""
AdamW Optimizer — From Scratch

Implements AdamW (Adam with decoupled weight decay) without importing
torch.optim or any optimizer library.

AdamW maintains:
    - First moment (mean of gradients)
    - Second moment (mean of squared gradients)
    - Bias correction for both moments
    - Decoupled weight decay (applied to weights, not gradients)

Key difference from Adam:
    Adam:  θ = θ - lr * (m_hat / (sqrt(v_hat) + eps) + λ * θ)
    AdamW: θ = θ - lr * m_hat / (sqrt(v_hat) + eps) - lr * λ * θ

Reference:
    Loshchilov & Hutter, 2019 — Decoupled Weight Decay Regularization
"""

import math
import torch
from typing import Optional


class AdamW:
    """
    AdamW optimizer.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate.
        betas: Coefficients for computing running averages (beta1, beta2).
        eps: Term added to denominator for numerical stability.
        weight_decay: Decoupled weight decay coefficient.
    """

    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        # State: first and second moments for each parameter
        self.m: list[torch.Tensor] = [torch.zeros_like(p) for p in self.params]
        self.v: list[torch.Tensor] = [torch.zeros_like(p) for p in self.params]

    def step(self) -> None:
        """
        Perform a single optimization step.

        For each parameter:
            1. Update biased first moment:  m = β1 * m + (1 - β1) * grad
            2. Update biased second moment: v = β2 * v + (1 - β2) * grad^2
            3. Bias correction:             m_hat = m / (1 - β1^t)
                                            v_hat = v / (1 - β2^t)
            4. Parameter update:            θ = θ - lr * m_hat / (sqrt(v_hat) + eps)
            5. Weight decay:                θ = θ - lr * λ * θ
        """
        # TODO: Implement AdamW step
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Zero out all parameter gradients."""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def set_lr(self, lr: float) -> None:
        """Update learning rate (called by scheduler)."""
        self.lr = lr

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        return {
            "step_count": self.step_count,
            "lr": self.lr,
            "m": [m.clone() for m in self.m],
            "v": [v.clone() for v in self.v],
        }

    def load_state_dict(self, state: dict) -> None:
        """Load optimizer state from checkpoint."""
        self.step_count = state["step_count"]
        self.lr = state["lr"]
        for i in range(len(self.params)):
            self.m[i].copy_(state["m"][i])
            self.v[i].copy_(state["v"][i])
