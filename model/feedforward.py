"""
Feedforward Network — SwiGLU

Implements the SwiGLU feedforward layer from scratch.

Standard FFN:     FFN(x) = W2 @ ReLU(W1 @ x)
SwiGLU FFN:       FFN(x) = W_down @ (SiLU(W_gate @ x) * (W_up @ x))

SwiGLU uses a gated activation — the gate and up projections are separate,
and the gate output modulates the up output via element-wise multiplication.

SiLU(x) = x * sigmoid(x) (also called "swish")

This gives better performance than ReLU/GELU for the same parameter count.

Reference:
    Shazeer, 2020 — GLU Variants Improve Transformer
    Touvron et al., 2023 — LLaMA uses SwiGLU

No nn.Linear — raw weight matrices and matmuls.
"""

import math
import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation (Sigmoid Linear Unit, aka Swish).

    silu(x) = x * sigmoid(x)
    """
    # TODO: Implement SiLU activation
    raise NotImplementedError


class SwiGLU:
    """
    SwiGLU feedforward layer.

    Architecture:
        gate = SiLU(x @ W_gate)
        up   = x @ W_up
        out  = (gate * up) @ W_down

    Shapes:
        W_gate: [d_model, d_ff]
        W_up:   [d_model, d_ff]
        W_down: [d_ff, d_model]

    Args:
        d_model: Input/output dimension.
        d_ff: Intermediate (hidden) dimension.
    """

    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff

        # Weight matrices (no bias, following LLaMA)
        self.w_gate: torch.Tensor = None  # [d_model, d_ff]
        self.w_up: torch.Tensor = None    # [d_model, d_ff]
        self.w_down: torch.Tensor = None  # [d_ff, d_model]

        # TODO: Initialize weights

    def _init_weights(self) -> None:
        """Initialize feedforward weights."""
        # TODO: Implement weight initialization
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU forward pass.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        # TODO: Implement SwiGLU forward
        # gate = silu(x @ self.w_gate)
        # up = x @ self.w_up
        # return (gate * up) @ self.w_down
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        return [self.w_gate, self.w_up, self.w_down]


class FusedSwiGLU:
    """
    Fused SwiGLU kernel.

    Combines the gate and up projections into a single matmul
    by concatenating W_gate and W_up, then splitting the output.

    This reduces kernel launch overhead and improves memory access patterns.
    """

    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff

        # Fused gate+up weight: [d_model, 2 * d_ff]
        self.w_gate_up: torch.Tensor = None  # type: ignore
        # Down projection: [d_ff, d_model]
        self.w_down: torch.Tensor = None  # type: ignore

        # TODO: Initialize weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fused SwiGLU forward pass."""
        # TODO: Implement fused SwiGLU
        # gate_up = x @ self.w_gate_up  -> [batch, seq, 2*d_ff]
        # gate, up = split(gate_up, d_ff, dim=-1)
        # return (silu(gate) * up) @ self.w_down
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        return [self.w_gate_up, self.w_down]
