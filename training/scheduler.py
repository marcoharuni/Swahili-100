"""
Learning Rate Schedulers — From Scratch

Implements common LR schedules without torch.optim.lr_scheduler.

Supported schedules:
    - Cosine annealing with linear warmup
    - Linear warmup + linear decay
    - Constant with warmup

The scheduler computes the learning rate for a given step and
updates the optimizer's LR directly.
"""

import math


class CosineScheduler:
    """
    Cosine annealing with linear warmup.

    LR schedule:
        - Steps [0, warmup_steps): linear warmup from 0 to max_lr
        - Steps [warmup_steps, total_steps]: cosine decay from max_lr to min_lr

    This is the standard schedule used by most modern LLMs.

    Args:
        max_lr: Peak learning rate (after warmup).
        min_lr: Minimum learning rate (at end of training).
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
    """

    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step: int) -> float:
        """
        Compute learning rate for a given step.

        Args:
            step: Current training step (0-indexed).

        Returns:
            Learning rate for this step.
        """
        # TODO: Implement cosine schedule with warmup
        # Warmup phase: linear interpolation from 0 to max_lr
        # Cosine phase: cosine decay from max_lr to min_lr
        raise NotImplementedError


class LinearScheduler:
    """
    Linear warmup + linear decay.

    Args:
        max_lr: Peak learning rate.
        min_lr: Final learning rate.
        warmup_steps: Warmup steps.
        total_steps: Total steps.
    """

    def __init__(self, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step: int) -> float:
        """Compute LR with linear warmup and linear decay."""
        # TODO: Implement linear schedule
        raise NotImplementedError


def build_scheduler(config: dict) -> object:
    """
    Factory function to create a scheduler from config.

    Args:
        config: Dict with 'type', 'max_lr', 'min_lr', 'warmup_steps', 'total_steps'.

    Returns:
        Scheduler instance.
    """
    stype = config.get("type", "cosine")
    if stype == "cosine":
        return CosineScheduler(
            max_lr=config["max_lr"],
            min_lr=config["min_lr"],
            warmup_steps=config["warmup_steps"],
            total_steps=config["total_steps"],
        )
    elif stype == "linear":
        return LinearScheduler(
            max_lr=config["max_lr"],
            min_lr=config["min_lr"],
            warmup_steps=config["warmup_steps"],
            total_steps=config["total_steps"],
        )
    else:
        raise ValueError(f"Unknown scheduler type: {stype}")
