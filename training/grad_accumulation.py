"""
Gradient Accumulation

Simulates large batch sizes on limited GPU memory by accumulating
gradients over multiple micro-batches before performing an optimizer step.

Effective batch size = micro_batch_size * gradient_accumulation_steps * world_size

Example:
    micro_batch = 8, accum_steps = 16, world_size = 1
    -> effective batch = 128

This module provides utilities to:
    - Track accumulation progress
    - Scale loss by accumulation steps (for correct gradient averaging)
    - Determine when to step the optimizer
"""


class GradientAccumulator:
    """
    Gradient accumulation controller.

    Tracks the micro-batch index within an accumulation cycle and
    provides the loss scaling factor.

    Args:
        accumulation_steps: Number of micro-batches per optimizer step.
    """

    def __init__(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self._micro_step = 0

    @property
    def loss_scale(self) -> float:
        """Factor to multiply loss by for correct gradient averaging."""
        return 1.0 / self.accumulation_steps

    @property
    def should_step(self) -> bool:
        """Whether the optimizer should step after this micro-batch."""
        return self._micro_step == self.accumulation_steps - 1

    @property
    def is_first_micro_step(self) -> bool:
        """Whether this is the first micro-batch in an accumulation cycle."""
        return self._micro_step == 0

    def advance(self) -> None:
        """Advance the micro-step counter."""
        self._micro_step = (self._micro_step + 1) % self.accumulation_steps

    def reset(self) -> None:
        """Reset the micro-step counter."""
        self._micro_step = 0

    @property
    def current_micro_step(self) -> int:
        return self._micro_step
