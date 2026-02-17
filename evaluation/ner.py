"""
Named Entity Recognition Evaluation — MasakhaNER

Evaluates the model's ability to recognize named entities in Swahili
using the MasakhaNER benchmark.

Entity types: PER (person), ORG (organization), LOC (location), DATE, etc.

Approach:
    Since this is a generative model (not a token classifier), we use
    in-context learning: provide a few NER examples in the prompt,
    then ask the model to tag new sentences.

Metrics: F1, Precision, Recall per entity type.

Reference:
    Adelani et al., 2021 — MasakhaNER: Named Entity Recognition for African Languages
"""

import torch
from typing import Optional

from model.transformer import Transformer
from model.generate import Generator
from data.tokenizer import BPETokenizer


def evaluate_ner(
    model: Transformer,
    tokenizer: BPETokenizer,
    data_path: str,
    num_shots: int = 5,
    max_examples: Optional[int] = None,
) -> dict:
    """
    Evaluate NER via few-shot in-context learning.

    Args:
        model: Trained model.
        tokenizer: BPE tokenizer.
        data_path: Path to MasakhaNER Swahili test set.
        num_shots: Number of few-shot examples in prompt.
        max_examples: Maximum test examples to evaluate.

    Returns:
        Dict with per-entity-type and overall F1, precision, recall.
    """
    # TODO: Implement NER evaluation
    # 1. Load MasakhaNER test data
    # 2. Construct few-shot prompt with NER examples
    # 3. For each test sentence, generate model prediction
    # 4. Parse predicted entities from generation
    # 5. Compute F1, precision, recall
    raise NotImplementedError


def _parse_entities(text: str) -> list[dict]:
    """Parse named entities from model-generated text."""
    # TODO: Implement entity parsing from generation
    raise NotImplementedError


def _compute_f1(predicted: list[dict], gold: list[dict]) -> dict:
    """Compute F1, precision, recall from predicted vs gold entities."""
    # TODO: Implement F1 computation
    raise NotImplementedError
