"""
Post-Training Quantization — From Scratch

Implements weight quantization without bitsandbytes or any quantization library.

Supported formats:
    - INT8: 8-bit symmetric quantization
    - INT4: 4-bit asymmetric quantization (group-wise)

Quantization reduces model size and enables CPU/edge inference:
    - BF16:  ~300MB for 150M params
    - INT8:  ~150MB
    - INT4:  ~75MB

Method:
    For each weight tensor (or group of weights):
        1. Compute scale = max(|w|) / max_int_value
        2. Quantize: w_q = round(w / scale)
        3. Store w_q (int8/int4) and scale (fp16)
        4. Dequantize: w_approx = w_q * scale

Group-wise quantization (for INT4):
    Split weight columns into groups of 128, quantize each group
    with its own scale factor. This preserves more precision.
"""

import torch
import math
from typing import Optional


# ---------------------------------------------------------------------------
# INT8 Quantization
# ---------------------------------------------------------------------------

def quantize_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric INT8 quantization of a weight tensor.

    Args:
        weight: FP32/BF16 weight tensor of any shape.

    Returns:
        (quantized_weight, scale)
        quantized_weight: int8 tensor, same shape.
        scale: FP16 scalar scale factor.
    """
    # TODO: Implement INT8 quantization
    raise NotImplementedError


def dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 tensor back to floating point."""
    # TODO: Implement INT8 dequantization
    raise NotImplementedError


# ---------------------------------------------------------------------------
# INT4 Quantization (group-wise)
# ---------------------------------------------------------------------------

def quantize_int4(
    weight: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric INT4 group-wise quantization.

    Packs two 4-bit values into one uint8 byte.

    Args:
        weight: FP32/BF16 weight tensor [out_features, in_features].
        group_size: Number of values per quantization group.

    Returns:
        (packed_weight, scales, zeros)
        packed_weight: uint8 tensor (two int4 values per byte).
        scales: FP16 scale per group.
        zeros: FP16 zero-point per group.
    """
    # TODO: Implement INT4 group-wise quantization
    raise NotImplementedError


def dequantize_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize INT4 packed tensor back to floating point."""
    # TODO: Implement INT4 dequantization
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Model-level quantization
# ---------------------------------------------------------------------------

def quantize_model(
    model_path: str,
    output_path: str,
    precision: str = "int8",
    group_size: int = 128,
) -> dict:
    """
    Quantize an entire model checkpoint.

    Args:
        model_path: Path to FP32/BF16 checkpoint.
        output_path: Path to save quantized checkpoint.
        precision: "int8" or "int4".
        group_size: Group size for INT4.

    Returns:
        Dict with quantization statistics (original size, quantized size, etc.).
    """
    # TODO: Implement full model quantization
    raise NotImplementedError
