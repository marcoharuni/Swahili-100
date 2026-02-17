"""
Swahili-100 Model Architecture

A decoder-only transformer built entirely from scratch.

Modules:
    config          — Model configuration dataclass
    embedding       — Token embeddings
    rope            — Rotary Position Encoding (RoPE)
    norm            — RMSNorm
    attention       — Multi-head attention (GQA, MLA, flash attention)
    flash_attention — Fused flash attention kernel
    feedforward     — SwiGLU feedforward network
    moe             — Mixture of Experts
    block           — Transformer block
    transformer     — Full transformer model
    lm_head         — Language model head with multi-token prediction
    generate        — Inference engine (KV-cache, speculative decoding)
"""
