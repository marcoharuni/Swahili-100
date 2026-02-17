"""
Swahili-100 Training Infrastructure

All training components built from scratch.

Modules:
    trainer             — Main training loop
    adamw               — AdamW optimizer
    muon                — Muon optimizer
    scheduler           — Learning rate schedulers
    precision           — FP8/BF16 mixed-precision training
    grad_accumulation   — Gradient accumulation
    grad_checkpoint     — Gradient checkpointing (activation recomputation)
    loss                — Loss functions (cross-entropy + MTP)
"""
