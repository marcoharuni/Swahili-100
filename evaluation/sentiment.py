"""
Sentiment Analysis Evaluation — AfriSenti

Evaluates Swahili sentiment classification using the AfriSenti benchmark.

Classes: positive, negative, neutral

Approach: Few-shot in-context classification.

Metrics: Weighted F1, accuracy.

Reference:
    Muhammad et al., 2023 — AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages
"""

import torch
from typing import Optional

from model.transformer import Transformer
from model.generate import Generator
from data.tokenizer import BPETokenizer


def evaluate_sentiment(
    model: Transformer,
    tokenizer: BPETokenizer,
    data_path: str,
    num_shots: int = 5,
    max_examples: Optional[int] = None,
) -> dict:
    """
    Evaluate sentiment classification via few-shot prompting.

    Args:
        model: Trained model.
        tokenizer: BPE tokenizer.
        data_path: Path to AfriSenti Swahili test set.
        num_shots: Number of few-shot examples.
        max_examples: Max test examples.

    Returns:
        Dict with 'accuracy', 'weighted_f1', per-class metrics.
    """
    # TODO: Implement sentiment evaluation
    raise NotImplementedError
