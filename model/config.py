"""
Model Configuration

Defines the ModelConfig dataclass that parameterizes every aspect
of the transformer architecture. Configs are loaded from YAML files
(see configs/) and passed to the model constructor.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuration for the Swahili-100 transformer model.

    All architectural decisions are controlled here — no magic numbers
    scattered through the codebase.
    """

    # --- Vocabulary ---
    vocab_size: int = 16384

    # --- Transformer dimensions ---
    d_model: int = 768           # Hidden size / embedding dimension
    n_layers: int = 12           # Number of transformer blocks
    n_heads: int = 12            # Number of attention heads
    n_kv_heads: int = 4          # Number of KV heads (GQA). If == n_heads, standard MHA.
    d_ff: int = 2048             # Feedforward intermediate size (SwiGLU)

    # --- Sequence ---
    max_seq_len: int = 2048      # Maximum sequence length

    # --- Positional encoding ---
    rope_theta: float = 10000.0  # RoPE base frequency

    # --- Normalization ---
    norm_eps: float = 1e-6       # RMSNorm epsilon

    # --- Regularization ---
    dropout: float = 0.0         # Dropout rate (0 = no dropout, modern practice)

    # --- Weight tying ---
    tie_embeddings: bool = True  # Tie input embedding and output LM head weights

    # --- Stability ---
    qk_norm: bool = True         # Apply RMSNorm to Q and K before attention

    # --- Mixture of Experts (optional) ---
    use_moe: bool = False
    moe_num_experts: int = 8     # Total number of experts
    moe_top_k: int = 2           # Number of experts activated per token
    moe_capacity_factor: float = 1.25  # Capacity factor for load balancing

    # --- Multi-Token Prediction (optional) ---
    use_mtp: bool = False
    mtp_num_heads: int = 2       # Number of future tokens to predict

    # --- Derived properties (computed, not set directly) ---
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        return self.d_model // self.n_heads

    @property
    def n_kv_groups(self) -> int:
        """Number of query heads per KV head (for GQA)."""
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        return self.n_heads // self.n_kv_heads

    def validate(self) -> None:
        """Validate configuration consistency."""
        assert self.vocab_size > 0
        assert self.d_model > 0
        assert self.n_layers > 0
        assert self.n_heads > 0
        assert self.n_kv_heads > 0
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        assert self.d_ff > 0
        assert self.max_seq_len > 0
        assert self.rope_theta > 0
        assert self.norm_eps > 0
        assert 0.0 <= self.dropout < 1.0

        if self.use_moe:
            assert self.moe_num_experts > 0
            assert self.moe_top_k > 0
            assert self.moe_top_k <= self.moe_num_experts

        if self.use_mtp:
            assert self.mtp_num_heads > 0

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Create config from a dictionary (e.g., parsed from YAML)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        """Serialize config to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def param_count_estimate(self) -> int:
        """
        Rough parameter count estimate.

        Embedding:      vocab_size * d_model
        Per layer:      4 * d_model^2 (attn) + 3 * d_model * d_ff (SwiGLU) + norms
        LM head:        (tied with embedding or vocab_size * d_model)
        """
        embed = self.vocab_size * self.d_model
        attn_per_layer = (
            self.d_model * self.head_dim * self.n_heads        # W_q
            + self.d_model * self.head_dim * self.n_kv_heads   # W_k
            + self.d_model * self.head_dim * self.n_kv_heads   # W_v
            + self.head_dim * self.n_heads * self.d_model       # W_o
        )
        ff_per_layer = 3 * self.d_model * self.d_ff  # SwiGLU: gate, up, down
        norm_per_layer = 2 * self.d_model  # 2x RMSNorm
        per_layer = attn_per_layer + ff_per_layer + norm_per_layer

        total = embed + self.n_layers * per_layer + self.d_model  # final norm
        if not self.tie_embeddings:
            total += self.vocab_size * self.d_model  # separate LM head

        return total
