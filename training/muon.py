"""
Muon Optimizer — From Scratch

Implements the Muon (Momentum + Orthogonalization) optimizer.

Muon applies Nesterov momentum followed by an orthogonalization step
that projects the update onto the Stiefel manifold (orthogonal matrices).
This is done via Newton-Schulz iterations, which approximate the matrix
square root inverse without eigendecomposition.

Key idea: force weight updates to be orthogonal, which acts as an
implicit regularizer and improves training dynamics.

Reference:
    Kodryan et al., 2025 — Muon: An optimizer for hidden layers in neural networks

Notes:
    - Muon is typically applied to hidden-layer weights (2D matrices).
    - For 1D parameters (biases, norms), fall back to standard AdamW.
    - The Newton-Schulz orthogonalization is the key differentiator.
"""

import torch
import math
from typing import Optional


class Muon:
    """
    Muon optimizer.

    Applies Nesterov momentum + Newton-Schulz orthogonalization
    to weight matrices.

    Args:
        params: List of parameter tensors.
        lr: Learning rate.
        momentum: Nesterov momentum coefficient.
        ns_steps: Number of Newton-Schulz iteration steps.
        weight_decay: Weight decay coefficient.
        adamw_params: Optional list of params to optimize with AdamW instead
                      (e.g., 1D params like norms).
        adamw_lr: Learning rate for AdamW fallback params.
        adamw_betas: Beta values for AdamW fallback.
    """

    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adamw_params: Optional[list[torch.Tensor]] = None,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
    ):
        # Muon params (2D weight matrices)
        self.params = [p for p in params if p.requires_grad and p.ndim >= 2]
        self.lr = lr
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay

        # Momentum buffers
        self.velocity: list[torch.Tensor] = [torch.zeros_like(p) for p in self.params]

        # AdamW fallback for 1D params
        self.adamw_params = adamw_params or [
            p for p in params if p.requires_grad and p.ndim < 2
        ]
        self.adamw_lr = adamw_lr
        self.adamw_betas = adamw_betas
        self.adamw_m: list[torch.Tensor] = [torch.zeros_like(p) for p in self.adamw_params]
        self.adamw_v: list[torch.Tensor] = [torch.zeros_like(p) for p in self.adamw_params]
        self.step_count = 0

    def _newton_schulz(self, G: torch.Tensor) -> torch.Tensor:
        """
        Newton-Schulz iteration to orthogonalize a matrix.

        Approximates G @ (G^T @ G)^{-1/2} without eigendecomposition.

        The iteration:
            X_0 = G / ||G||_F
            X_{k+1} = X_k @ (aI + bX_k^T X_k + cX_k^T X_k X_k^T X_k)

        with specific coefficients (a, b, c) that give cubic convergence.

        Args:
            G: Input matrix [m, n] (typically m >= n).

        Returns:
            Orthogonalized matrix, same shape as G.
        """
        # TODO: Implement Newton-Schulz orthogonalization
        raise NotImplementedError

    def step(self) -> None:
        """
        Perform one Muon optimization step.

        For 2D params (Muon):
            1. Nesterov momentum: v = mu * v + grad
            2. Lookahead: update = grad + mu * v
            3. Orthogonalize: update = newton_schulz(update)
            4. Weight decay: param -= lr * wd * param
            5. Apply: param -= lr * update

        For 1D params (AdamW fallback):
            Standard AdamW update.
        """
        # TODO: Implement Muon step
        raise NotImplementedError

    def _adamw_step(self) -> None:
        """AdamW update for 1D parameters."""
        # TODO: Implement AdamW step for fallback params
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Zero gradients for all parameters."""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
        for p in self.adamw_params:
            if p.grad is not None:
                p.grad.zero_()

    def set_lr(self, lr: float) -> None:
        """Update learning rate."""
        self.lr = lr

    def state_dict(self) -> dict:
        """Serialize optimizer state."""
        return {
            "step_count": self.step_count,
            "velocity": [v.clone() for v in self.velocity],
            "adamw_m": [m.clone() for m in self.adamw_m],
            "adamw_v": [v.clone() for v in self.adamw_v],
        }

    def load_state_dict(self, state: dict) -> None:
        """Load optimizer state."""
        self.step_count = state["step_count"]
        for i in range(len(self.params)):
            self.velocity[i].copy_(state["velocity"][i])
        for i in range(len(self.adamw_params)):
            self.adamw_m[i].copy_(state["adamw_m"][i])
            self.adamw_v[i].copy_(state["adamw_v"][i])
