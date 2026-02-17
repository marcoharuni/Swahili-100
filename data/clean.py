"""
Text Cleaning and Normalization

Processes raw text through a cleaning pipeline:
1. Unicode normalization (NFC)
2. Encoding repair (mojibake detection and fix)
3. Whitespace normalization
4. Control character removal
5. URL and email redaction
6. Number normalization (optional)

All operations are pure Python — no external NLP libraries.

Usage:
    python data/clean.py --input data/raw/ --output data/cleaned/
"""

import argparse
import os
import re
import unicodedata


# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Apply NFC unicode normalization."""
    return unicodedata.normalize("NFC", text)


def remove_control_chars(text: str) -> str:
    """Remove control characters except newline and tab."""
    # TODO: Implement control character filtering
    raise NotImplementedError


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single space, preserve newlines."""
    # TODO: Implement whitespace normalization
    raise NotImplementedError


def redact_urls(text: str) -> str:
    """Replace URLs with a placeholder token."""
    # TODO: Implement URL detection and redaction
    raise NotImplementedError


def redact_emails(text: str) -> str:
    """Replace email addresses with a placeholder token."""
    # TODO: Implement email detection and redaction
    raise NotImplementedError


def fix_encoding(text: str) -> str:
    """Attempt to repair common mojibake patterns."""
    # TODO: Implement encoding repair heuristics
    raise NotImplementedError


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline applied to a single document.
    Returns cleaned text or empty string if document should be discarded.
    """
    text = normalize_unicode(text)
    text = fix_encoding(text)
    text = remove_control_chars(text)
    text = redact_urls(text)
    text = redact_emails(text)
    text = normalize_whitespace(text)
    return text.strip()


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def clean_file(input_path: str, output_path: str) -> dict:
    """
    Clean a single text file. Returns statistics dict.
    """
    stats = {"input_lines": 0, "output_lines": 0, "discarded": 0}

    with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            stats["input_lines"] += 1
            cleaned = clean_text(line)
            if cleaned:
                fout.write(cleaned + "\n")
                stats["output_lines"] += 1
            else:
                stats["discarded"] += 1

    return stats


def clean_directory(input_dir: str, output_dir: str) -> None:
    """Clean all text files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    total_stats = {"input_lines": 0, "output_lines": 0, "discarded": 0}

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".txt"):
            continue
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)
        print(f"[clean] Processing {fname}...")
        stats = clean_file(input_path, output_path)
        for k in total_stats:
            total_stats[k] += stats[k]

    print(f"[clean] Done. {total_stats}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw Swahili text")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    clean_directory(args.input, args.output)


if __name__ == "__main__":
    main()
