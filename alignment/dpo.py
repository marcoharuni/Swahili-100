"""
Direct Preference Optimization (DPO)

Aligns the model to human preferences without a separate reward model.

DPO directly optimizes the policy to prefer chosen responses over rejected
ones, using the implicit reward formulation from the RL objective.

Loss:
    L_DPO = -log(sigmoid(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x))))

Where:
    y_w = chosen (preferred) response
    y_l = rejected response
    π = current policy (model being trained)
    π_ref = reference policy (frozen copy of model before DPO)
    β = temperature parameter

Reference:
    Rafailov et al., 2023 — Direct Preference Optimization

Data format:
    Each example has:
        "prompt": str
        "chosen": str — preferred response
        "rejected": str — dispreferred response

Usage:
    python scripts/align.py --stage dpo --config configs/swahili_base.yaml --data data/dpo/
"""

import torch
import math
from typing import Optional

from model.transformer import Transformer
from data.tokenizer import BPETokenizer


class DPOTrainer:
    """
    Direct Preference Optimization trainer.

    Args:
        model: SFT-trained transformer model (will be further trained).
        ref_model: Frozen reference model (copy of model before DPO).
        tokenizer: Trained BPE tokenizer.
        beta: DPO temperature parameter.
        lr: Learning rate.
        epochs: Number of DPO epochs.
    """

    def __init__(
        self,
        model: Transformer,
        ref_model: Transformer,
        tokenizer: BPETokenizer,
        beta: float = 0.1,
        lr: float = 5e-7,
        epochs: int = 1,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.lr = lr
        self.epochs = epochs

    def compute_log_probs(
        self,
        model: Transformer,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities for a sequence.

        Args:
            model: Transformer model.
            input_ids: [batch, seq_len] token IDs.
            labels: [batch, seq_len] target IDs (-100 for ignored).

        Returns:
            [batch] sum of log-probs over non-ignored positions.
        """
        # TODO: Implement log-prob computation
        raise NotImplementedError

    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DPO loss.

        L = -log(sigmoid(β * ((log π(y_w) - log π_ref(y_w)) - (log π(y_l) - log π_ref(y_l)))))
        """
        # TODO: Implement DPO loss computation
        raise NotImplementedError

    def load_data(self, data_path: str) -> list[dict]:
        """Load DPO preference data from JSONL."""
        # TODO: Implement data loading
        raise NotImplementedError

    def train(self, data_path: str) -> None:
        """Run DPO training."""
        # TODO: Implement DPO training loop
        raise NotImplementedError
