"""
Gradient Checkpointing (Activation Recomputation)

Trades compute for memory by not storing intermediate activations
during the forward pass. Instead, activations are recomputed during
the backward pass when needed for gradient computation.

Memory savings: O(N) -> O(sqrt(N)) for N layers.
Compute cost: ~33% increase in wall-clock time.

This is essential for training large models on limited GPU memory.

Implementation:
    Instead of storing all intermediate activations, we only store
    the input to each checkpointed segment. During backward, we
    rerun the forward pass for that segment to recompute activations.

Reference:
    Chen et al., 2016 — Training Deep Nets with Sublinear Memory Cost
"""

import torch
from typing import Callable, Any


def checkpoint(
    fn: Callable,
    *args: Any,
    use_checkpointing: bool = True,
) -> Any:
    """
    Apply gradient checkpointing to a function.

    If use_checkpointing is True:
        - Forward: run fn, discard intermediate activations
        - Backward: recompute fn to get activations for gradient computation

    If use_checkpointing is False:
        - Standard forward pass (keep all activations)

    Args:
        fn: Function to checkpoint (e.g., a transformer block's forward).
        *args: Arguments to fn.
        use_checkpointing: Whether to actually checkpoint.

    Returns:
        Output of fn(*args).
    """
    if not use_checkpointing:
        return fn(*args)

    # TODO: Implement gradient checkpointing
    # This requires custom autograd:
    # 1. In forward: run fn, save only inputs (not intermediates)
    # 2. In backward: re-run fn with torch.enable_grad() to recompute
    # 3. Compute gradients normally from recomputed activations
    raise NotImplementedError


class CheckpointFunction:
    """
    Custom autograd function for gradient checkpointing.

    This hooks into PyTorch's autograd to implement the
    save-input / recompute-forward pattern.
    """

    @staticmethod
    def forward(ctx, fn, *args):
        """
        Forward pass — run fn but don't save intermediate activations.

        ctx: Autograd context for saving tensors.
        fn: The function to checkpoint.
        *args: Inputs to fn.
        """
        # TODO: Implement checkpointed forward
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass — recompute forward, then backprop normally.
        """
        # TODO: Implement checkpointed backward
        raise NotImplementedError
