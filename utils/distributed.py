"""
Distributed Training Utilities

Helpers for multi-GPU training:
    - Process group initialization
    - Gradient all-reduce
    - Data sharding
    - Distributed checkpoint saving (rank 0 only)

Supports single-node multi-GPU via PyTorch distributed.
"""

import os
import torch
from typing import Optional


def setup_distributed() -> tuple[int, int, int]:
    """
    Initialize distributed training.

    Returns:
        (rank, local_rank, world_size)
    """
    # TODO: Implement distributed setup
    # 1. Read RANK, LOCAL_RANK, WORLD_SIZE from environment
    # 2. Initialize process group (nccl backend)
    # 3. Set CUDA device for this rank
    raise NotImplementedError


def cleanup_distributed() -> None:
    """Clean up distributed process group."""
    # TODO: Implement cleanup
    raise NotImplementedError


def all_reduce_grads(params: list[torch.Tensor], world_size: int) -> None:
    """
    All-reduce gradients across processes.

    Averages gradients from all ranks so each rank has the same update.
    """
    # TODO: Implement gradient all-reduce
    raise NotImplementedError


def is_main_process(rank: int = 0) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0
