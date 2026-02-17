"""
Swahili-100 Data Pipeline

Modules:
    download    — Data acquisition from OSCAR, CC-100, Wikipedia, etc.
    clean       — Text normalization, encoding fixes, unicode cleanup
    dedup       — MinHash LSH near-duplicate removal
    filter      — Quality filtering (perplexity, length, char ratios)
    tokenizer   — Byte-Pair Encoding tokenizer (from scratch)
    loader      — Batched data loading for training
"""
