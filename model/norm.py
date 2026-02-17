"""
RMSNorm (Root Mean Square Layer Normalization)

Implements RMSNorm from scratch — no nn.LayerNorm, no library imports.

RMSNorm is simpler and faster than LayerNorm:
    - No mean subtraction (no centering)
    - No bias parameter
    - Only rescales by the RMS of activations

    y = x / RMS(x) * gamma

    where RMS(x) = sqrt(mean(x^2) + eps)

Reference:
    Zhang & Sennrich, 2019 — Root Mean Square Layer Normalization
"""

import torch


class RMSNorm:
    """
    Root Mean Square Layer Normalization.

    Args:
        d_model: Feature dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps

        # Learnable scale parameter (gamma), initialized to ones
        self.weight: torch.Tensor = None  # type: ignore
        # TODO: Initialize self.weight as ones tensor of shape [d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.

        Args:
            x: Input tensor [..., d_model].

        Returns:
            Normalized tensor, same shape as input.
        """
        # TODO: Implement RMSNorm forward pass
        # 1. Compute RMS: sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
        # 2. Normalize: x / rms
        # 3. Scale: normalized * self.weight
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        return [self.weight]


class FusedRMSNorm:
    """
    Fused RMSNorm kernel for improved performance.

    Combines the normalization and scaling into a single CUDA kernel
    to reduce memory bandwidth. Falls back to standard RMSNorm on CPU.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps
        self.weight: torch.Tensor = None  # type: ignore
        # TODO: Initialize weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fused RMSNorm — single-pass normalization + scaling.

        Falls back to standard implementation if not on CUDA.
        """
        # TODO: Implement fused kernel (or fallback)
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        return [self.weight]
