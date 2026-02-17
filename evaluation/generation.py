"""
Generation Quality Evaluation

Evaluates the quality of generated Swahili text through:
1. Automated metrics (repetition, coherence heuristics)
2. Human evaluation framework (generates samples for human raters)

This module generates text samples from various prompts and
computes automated quality indicators.
"""

import torch
from typing import Optional

from model.transformer import Transformer
from model.generate import Generator
from data.tokenizer import BPETokenizer


# ---------------------------------------------------------------------------
# Evaluation prompts — diverse Swahili generation scenarios
# ---------------------------------------------------------------------------

EVAL_PROMPTS = [
    "Tanzania ni nchi",
    "Habari za asubuhi. Leo",
    "Elimu ni muhimu kwa sababu",
    "Historia ya Afrika Mashariki",
    "Mwalimu Julius Nyerere alikuwa",
    "Kilimanjaro ni mlima",
    "Lugha ya Kiswahili ina",
    "Biashara ya kimataifa",
]


def evaluate_generation(
    model: Transformer,
    tokenizer: BPETokenizer,
    prompts: Optional[list[str]] = None,
    max_tokens: int = 200,
    temperature: float = 0.8,
    num_samples: int = 3,
) -> dict:
    """
    Generate text samples and compute quality metrics.

    Args:
        model: Trained model.
        tokenizer: BPE tokenizer.
        prompts: List of prompts (uses EVAL_PROMPTS if None).
        max_tokens: Max tokens per generation.
        temperature: Sampling temperature.
        num_samples: Samples per prompt.

    Returns:
        Dict with samples and per-sample metrics.
    """
    # TODO: Implement generation evaluation
    raise NotImplementedError


def repetition_score(text: str, n: int = 4) -> float:
    """
    Measure n-gram repetition rate in generated text.

    Returns fraction of repeated n-grams (0 = no repetition, 1 = all repeated).
    """
    # TODO: Implement repetition scoring
    raise NotImplementedError


def distinct_ngrams(text: str, n: int = 2) -> float:
    """
    Compute distinct-n metric (diversity of n-grams).

    distinct-n = unique_ngrams / total_ngrams
    Higher is better (more diverse generation).
    """
    # TODO: Implement distinct-n computation
    raise NotImplementedError
