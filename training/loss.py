"""
Loss Functions — From Scratch

Implements loss functions without nn.CrossEntropyLoss.

1. Cross-Entropy Loss — standard next-token prediction loss
2. Multi-Token Prediction Loss — auxiliary loss for predicting multiple future tokens
3. MoE Auxiliary Loss — load-balancing loss for mixture-of-experts

The cross-entropy is implemented with the log-sum-exp trick
for numerical stability.
"""

import torch
import math


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Cross-entropy loss with log-sum-exp stabilization.

    Computes: -log(softmax(logits)[target]) for each position,
    averaged over non-ignored positions.

    Numerically stable via:
        log_softmax(x) = x - log(sum(exp(x)))
                       = x - max(x) - log(sum(exp(x - max(x))))

    Args:
        logits: [batch, seq_len, vocab_size] — raw model output.
        targets: [batch, seq_len] — ground truth token IDs.
        ignore_index: Target value to ignore (e.g., padding).

    Returns:
        Scalar loss tensor.
    """
    # TODO: Implement numerically stable cross-entropy
    # 1. Reshape logits to [N, vocab_size] and targets to [N]
    # 2. Compute log-softmax with the max subtraction trick
    # 3. Gather the log-probabilities at target indices
    # 4. Mask out ignore_index positions
    # 5. Return mean loss
    raise NotImplementedError


def multi_token_prediction_loss(
    mtp_logits: list[torch.Tensor],
    targets: torch.Tensor,
    mtp_weight: float = 0.1,
) -> torch.Tensor:
    """
    Multi-token prediction auxiliary loss.

    For each MTP head k, the target is the token at position (i + k + 1)
    instead of (i + 1) for the standard next-token prediction.

    The total MTP loss is the weighted average of per-head CE losses.

    Args:
        mtp_logits: List of [batch, seq_len, vocab_size] per MTP head.
        targets: [batch, seq_len] — original ground truth targets.
        mtp_weight: Weight for the MTP loss relative to main CE loss.

    Returns:
        Weighted MTP loss scalar.
    """
    # TODO: Implement MTP loss
    # For head k (0-indexed), predict token at offset (k + 2):
    #   mtp_targets_k = targets[:, k+2:]
    #   mtp_logits_k  = mtp_logits[k][:, :-(k+2)]
    #   loss_k = cross_entropy(mtp_logits_k, mtp_targets_k)
    # Return mtp_weight * mean(loss_k for all k)
    raise NotImplementedError


def compute_total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mtp_logits: list[torch.Tensor] = None,
    aux_loss: float = 0.0,
    mtp_weight: float = 0.1,
    aux_weight: float = 0.01,
    ignore_index: int = -100,
) -> dict:
    """
    Compute total training loss.

    total_loss = ce_loss + mtp_weight * mtp_loss + aux_weight * aux_loss

    Args:
        logits: Main model logits.
        targets: Ground truth targets.
        mtp_logits: Multi-token prediction logits (optional).
        aux_loss: MoE auxiliary load-balancing loss.
        mtp_weight: Weight for MTP loss.
        aux_weight: Weight for auxiliary loss.
        ignore_index: Padding token to ignore.

    Returns:
        Dict with 'total', 'ce', 'mtp', 'aux' loss values.
    """
    ce = cross_entropy_loss(logits, targets, ignore_index)

    mtp = torch.tensor(0.0)
    if mtp_logits:
        mtp = multi_token_prediction_loss(mtp_logits, targets, mtp_weight)

    total = ce + mtp + aux_weight * aux_loss

    return {
        "total": total,
        "ce": ce.item(),
        "mtp": mtp.item() if isinstance(mtp, torch.Tensor) else mtp,
        "aux": aux_loss if isinstance(aux_loss, float) else aux_loss.item(),
    }
