"""
Checkpoint Management

Save and load model checkpoints including:
    - Model weights
    - Optimizer state
    - Scheduler state
    - Training metadata (step, tokens seen, etc.)

Supports:
    - Incremental saving (save_every N steps)
    - Rolling checkpoints (keep last K)
    - Best-model tracking (by validation loss)
"""

import os
import torch
from typing import Optional


def save_checkpoint(
    path: str,
    model_params: list[torch.Tensor],
    optimizer_state: dict,
    scheduler_state: dict,
    step: int,
    tokens_seen: int,
    val_loss: Optional[float] = None,
    config: Optional[dict] = None,
) -> None:
    """
    Save a full training checkpoint.

    Args:
        path: File path for the checkpoint.
        model_params: List of model parameter tensors.
        optimizer_state: Optimizer state dict.
        scheduler_state: Scheduler state (current step, etc.).
        step: Global training step.
        tokens_seen: Total tokens processed.
        val_loss: Current validation loss (optional).
        config: Model/training config (optional, for reproducibility).
    """
    # TODO: Implement checkpoint saving
    # Save as a dict with torch.save:
    #   - model_state: list of param tensors
    #   - optimizer_state: optimizer state dict
    #   - scheduler_state: scheduler state
    #   - step, tokens_seen, val_loss
    #   - config
    raise NotImplementedError


def load_checkpoint(
    path: str,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Load a training checkpoint.

    Args:
        path: Path to checkpoint file.
        device: Device to load tensors to.

    Returns:
        Dict with all checkpoint data.
    """
    # TODO: Implement checkpoint loading
    raise NotImplementedError


def manage_checkpoints(
    checkpoint_dir: str,
    keep_last: int = 3,
) -> None:
    """
    Remove old checkpoints, keeping only the last K.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        keep_last: Number of recent checkpoints to keep.
    """
    # TODO: Implement checkpoint cleanup
    raise NotImplementedError
