"""
Translation Evaluation

Evaluates Swahili-English and English-Swahili translation quality.

Metrics:
    - BLEU (from scratch — no sacrebleu import)
    - chrF (character-level F-score)

BLEU is computed as the geometric mean of n-gram precision scores
with a brevity penalty.

Reference:
    Papineni et al., 2002 — BLEU: a Method for Automatic Evaluation of Machine Translation
"""

import math
from collections import Counter
from typing import Optional

from model.transformer import Transformer
from model.generate import Generator
from data.tokenizer import BPETokenizer


# ---------------------------------------------------------------------------
# BLEU score — from scratch
# ---------------------------------------------------------------------------

def compute_ngrams(tokens: list[str], n: int) -> Counter:
    """Count n-grams in a token list."""
    # TODO: Implement n-gram counting
    raise NotImplementedError


def bleu_score(
    hypothesis: str,
    references: list[str],
    max_n: int = 4,
) -> dict:
    """
    Compute BLEU score for a single hypothesis against reference(s).

    Args:
        hypothesis: Model-generated translation.
        references: One or more reference translations.
        max_n: Maximum n-gram order (typically 4).

    Returns:
        Dict with 'bleu', 'brevity_penalty', per-n-gram precisions.
    """
    # TODO: Implement BLEU computation
    # 1. Tokenize hypothesis and references (whitespace split)
    # 2. For each n in [1, max_n]:
    #    - Count n-grams in hypothesis
    #    - Clip counts by max reference n-gram counts
    #    - Precision_n = clipped_count / total_count
    # 3. Brevity penalty: BP = exp(1 - ref_len/hyp_len) if hyp_len < ref_len else 1
    # 4. BLEU = BP * exp(mean(log(precision_n) for n in 1..max_n))
    raise NotImplementedError


def evaluate_translation(
    model: Transformer,
    tokenizer: BPETokenizer,
    data_path: str,
    direction: str = "sw-en",
    num_shots: int = 5,
    max_examples: Optional[int] = None,
) -> dict:
    """
    Evaluate translation quality.

    Args:
        model: Trained model.
        tokenizer: BPE tokenizer.
        data_path: Path to parallel test data.
        direction: "sw-en" or "en-sw".
        num_shots: Few-shot examples.
        max_examples: Max test examples.

    Returns:
        Dict with 'bleu', per-example scores.
    """
    # TODO: Implement translation evaluation
    raise NotImplementedError
