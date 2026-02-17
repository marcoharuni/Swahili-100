"""
Training Logger

Unified logging interface that supports:
    - Console output (always)
    - JSON log files (always)
    - Weights & Biases (optional)

Logs training metrics (loss, LR, throughput, etc.) at each step
and evaluation metrics periodically.
"""

import json
import os
import time
from typing import Optional


class TrainingLogger:
    """
    Training metrics logger.

    Args:
        log_dir: Directory to save log files.
        project_name: Project name (for W&B).
        wandb_entity: W&B entity/team.
        use_wandb: Whether to log to W&B.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        project_name: str = "swahili-100",
        wandb_entity: Optional[str] = None,
        use_wandb: bool = False,
    ):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, "training.jsonl")
        self.use_wandb = use_wandb
        self._wandb_run = None

        if use_wandb:
            # TODO: Initialize W&B run (lazy import wandb)
            pass

    def log(self, step: int, metrics: dict) -> None:
        """
        Log metrics for a training step.

        Args:
            step: Global training step.
            metrics: Dict of metric name -> value.
        """
        record = {"step": step, "timestamp": time.time(), **metrics}

        # Console
        parts = [f"step={step}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(" | ".join(parts))

        # JSON file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        # W&B
        if self.use_wandb and self._wandb_run:
            # TODO: Log to W&B
            pass

    def log_config(self, config: dict) -> None:
        """Log the training configuration."""
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def finish(self) -> None:
        """Finalize logging (close W&B run, etc.)."""
        if self.use_wandb and self._wandb_run:
            # TODO: Finish W&B run
            pass
