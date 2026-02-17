"""
Quality Filtering

Filters cleaned text based on quality heuristics:
- Document length (min/max tokens)
- Special character ratio
- Alphabetic ratio (Swahili uses Latin script)
- Repetition detection (repeated n-grams)
- Language detection confidence (fastText lid)
- Perplexity filtering (using a small n-gram LM)

Usage:
    python data/filter.py --input data/deduped/ --output data/filtered/
"""

import argparse
import os


# ---------------------------------------------------------------------------
# Filter functions — each returns True if the document passes
# ---------------------------------------------------------------------------

def filter_length(text: str, min_chars: int = 100, max_chars: int = 100000) -> bool:
    """Reject documents that are too short or too long."""
    # TODO: Implement length filter
    raise NotImplementedError


def filter_alpha_ratio(text: str, min_ratio: float = 0.7) -> bool:
    """Reject documents with too few alphabetic characters."""
    # TODO: Implement alphabetic ratio filter
    raise NotImplementedError


def filter_special_chars(text: str, max_ratio: float = 0.1) -> bool:
    """Reject documents with too many special characters."""
    # TODO: Implement special character ratio filter
    raise NotImplementedError


def filter_repetition(text: str, max_repeat_ratio: float = 0.3) -> bool:
    """Reject documents with excessive n-gram repetition."""
    # TODO: Implement repetition detection
    raise NotImplementedError


def filter_language(text: str, target_lang: str = "sw", min_confidence: float = 0.8) -> bool:
    """
    Reject documents not classified as Swahili.
    Uses fastText language identification model (lid.176.bin).
    """
    # TODO: Implement language detection via fastText
    raise NotImplementedError


def passes_all_filters(text: str) -> bool:
    """Apply all quality filters to a document."""
    filters = [
        filter_length,
        filter_alpha_ratio,
        filter_special_chars,
        filter_repetition,
        filter_language,
    ]
    return all(f(text) for f in filters)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def filter_directory(input_dir: str, output_dir: str) -> dict:
    """Filter all documents in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    stats = {"total": 0, "passed": 0, "rejected": 0}

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".txt"):
            continue
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)
        print(f"[filter] Processing {fname}...")

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for doc in fin:  # one document per line
                stats["total"] += 1
                if passes_all_filters(doc.strip()):
                    fout.write(doc)
                    stats["passed"] += 1
                else:
                    stats["rejected"] += 1

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Quality-filter Swahili text")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    stats = filter_directory(args.input, args.output)
    print(f"[filter] {stats}")


if __name__ == "__main__":
    main()
