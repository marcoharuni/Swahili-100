"""
Performance Profiling

Utilities for measuring and reporting training performance:
    - Tokens per second throughput
    - GPU memory usage
    - Time per step breakdown (forward, backward, optimizer)
    - MFU (Model FLOPs Utilization) estimation
"""

import time
import torch
from typing import Optional


class Timer:
    """Simple context-manager timer."""

    def __init__(self):
        self.start_time: float = 0
        self.elapsed: float = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time


class StepProfiler:
    """
    Per-step profiler that tracks time spent in each phase.

    Usage:
        profiler = StepProfiler()
        with profiler.track("forward"):
            output = model(batch)
        with profiler.track("backward"):
            loss.backward()
        with profiler.track("optimizer"):
            optimizer.step()
        profiler.report()
    """

    def __init__(self):
        self.timings: dict[str, list[float]] = {}

    def track(self, name: str) -> Timer:
        """Return a timer context manager for a named phase."""
        timer = Timer()
        if name not in self.timings:
            self.timings[name] = []

        class _TrackedTimer:
            def __enter__(self_inner):
                timer.__enter__()
                return timer

            def __exit__(self_inner, *args):
                timer.__exit__(*args)
                self.timings[name].append(timer.elapsed)

        return _TrackedTimer()

    def report(self, step: int, tokens_per_step: int) -> dict:
        """
        Generate a profiling report.

        Returns dict with timing breakdown and throughput metrics.
        """
        # TODO: Implement profiling report
        raise NotImplementedError

    def reset(self) -> None:
        """Clear all timing data."""
        self.timings.clear()


def estimate_mfu(
    param_count: int,
    tokens_per_sec: float,
    n_layers: int,
    d_model: int,
    max_seq_len: int,
    gpu_flops: float = 312e12,  # A100 peak bf16 FLOPS
) -> float:
    """
    Estimate Model FLOPs Utilization (MFU).

    MFU = actual_flops / peak_gpu_flops

    Approximate FLOPs per token for a transformer:
        ~6 * param_count (forward + backward ≈ 3x forward)

    Args:
        param_count: Total model parameters.
        tokens_per_sec: Training throughput.
        n_layers: Number of transformer layers.
        d_model: Hidden dimension.
        max_seq_len: Sequence length.
        gpu_flops: Peak GPU FLOPS (bf16).

    Returns:
        MFU as a fraction (0.0 to 1.0).
    """
    # TODO: Implement MFU estimation
    raise NotImplementedError


def get_gpu_memory() -> dict:
    """
    Get current GPU memory usage.

    Returns dict with 'allocated_gb', 'reserved_gb', 'max_allocated_gb'.
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0}

    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
