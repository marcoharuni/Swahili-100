"""
Data Acquisition

Downloads and organizes raw Swahili text data from multiple sources:
- OSCAR (Open Super-large Crawled Aggregated coRpus)
- CC-100 (CommonCrawl monolingual dataset)
- Wikipedia Swahili dump
- News sources (web scraping pipeline)

Each source is downloaded, verified (checksum), and stored in a
standardized format under data/raw/<source_name>/.

Usage:
    python data/download.py --source oscar --output data/raw/oscar/
    python data/download.py --source all --output data/raw/
"""

import argparse
import os


# ---------------------------------------------------------------------------
# Source registry — add new data sources here
# ---------------------------------------------------------------------------
SOURCES = {
    "oscar": {
        "description": "OSCAR Swahili subset from HuggingFace",
        "url": "https://huggingface.co/datasets/oscar-corpus/OSCAR-2301",
        "language_code": "sw",
        "estimated_tokens": "500M",
    },
    "cc100": {
        "description": "CC-100 Swahili monolingual data",
        "url": "https://data.statmt.org/cc-100/",
        "language_code": "sw",
        "estimated_tokens": "300M",
    },
    "wikipedia": {
        "description": "Swahili Wikipedia dump",
        "url": "https://dumps.wikimedia.org/swwiki/",
        "language_code": "sw",
        "estimated_tokens": "50M",
    },
}


def download_oscar(output_dir: str) -> None:
    """Download OSCAR Swahili subset."""
    # TODO: Implement OSCAR download via HuggingFace datasets API
    #       or direct streaming download.
    raise NotImplementedError("OSCAR download not yet implemented")


def download_cc100(output_dir: str) -> None:
    """Download CC-100 Swahili data."""
    # TODO: Implement CC-100 download. File: sw.txt.xz
    raise NotImplementedError("CC-100 download not yet implemented")


def download_wikipedia(output_dir: str) -> None:
    """Download and extract Swahili Wikipedia dump."""
    # TODO: Implement Wikipedia dump download and wikitext extraction.
    raise NotImplementedError("Wikipedia download not yet implemented")


def download_all(output_dir: str) -> None:
    """Download all registered sources."""
    for source_name, meta in SOURCES.items():
        source_dir = os.path.join(output_dir, source_name)
        os.makedirs(source_dir, exist_ok=True)
        print(f"[download] Fetching {source_name}: {meta['description']}")
        # Dispatch to source-specific downloader
        downloader = {
            "oscar": download_oscar,
            "cc100": download_cc100,
            "wikipedia": download_wikipedia,
        }
        downloader[source_name](source_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Download Swahili text data")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=list(SOURCES.keys()) + ["all"],
        help="Data source to download",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory for raw data",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.source == "all":
        download_all(args.output)
    else:
        source_dir = os.path.join(args.output, args.source)
        os.makedirs(source_dir, exist_ok=True)
        downloader = {
            "oscar": download_oscar,
            "cc100": download_cc100,
            "wikipedia": download_wikipedia,
        }
        downloader[args.source](source_dir)


if __name__ == "__main__":
    main()
