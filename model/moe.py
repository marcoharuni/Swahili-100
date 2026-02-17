"""
Mixture of Experts (MoE)

Implements a sparse MoE layer from scratch.

Instead of a single feedforward network, MoE uses multiple "expert" FFNs.
A learned router selects the top-k experts for each token, and only those
experts process that token. This allows scaling model capacity without
proportionally scaling compute.

Architecture:
    router_logits = x @ W_router          [batch, seq, n_experts]
    top_k_indices = topk(router_logits)   [batch, seq, top_k]
    top_k_weights = softmax(top_k_logits) [batch, seq, top_k]
    output = sum(weight_i * expert_i(x))  for each selected expert

Features:
    - Top-k routing with softmax normalization
    - Auxiliary load-balancing loss (prevent expert collapse)
    - Capacity factor for bounded compute
    - Expert parallelism support (future)

References:
    Shazeer et al., 2017 — Outrageously Large Neural Networks
    Fedus et al., 2022 — Switch Transformers
    Jiang et al., 2024 — Mixtral of Experts
"""

import torch
import math
from typing import Optional

from model.config import ModelConfig
from model.feedforward import SwiGLU


class MoERouter:
    """
    Token-level expert router.

    Computes routing probabilities and selects top-k experts per token.

    Args:
        d_model: Input dimension.
        n_experts: Total number of experts.
        top_k: Number of experts to route each token to.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        # Router weight: [d_model, n_experts]
        self.w_router: torch.Tensor = None  # type: ignore
        # TODO: Initialize router weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            expert_indices: [batch, seq_len, top_k] — selected expert IDs
            expert_weights: [batch, seq_len, top_k] — routing weights (softmax)
            aux_loss: scalar — load-balancing auxiliary loss
        """
        # TODO: Implement routing
        # 1. Compute router logits: x @ W_router
        # 2. Select top-k experts per token
        # 3. Softmax over selected logits for weights
        # 4. Compute auxiliary load-balancing loss
        raise NotImplementedError

    def _load_balancing_loss(
        self, router_probs: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss to encourage balanced expert utilization.

        Without this loss, the router may collapse to always selecting
        the same expert(s), wasting capacity.

        L_aux = n_experts * sum(f_i * p_i)
        where f_i = fraction of tokens routed to expert i
              p_i = mean router probability for expert i
        """
        # TODO: Implement load-balancing loss
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        return [self.w_router]


class MoELayer:
    """
    Mixture of Experts layer.

    Replaces a standard FFN with multiple expert FFNs + a router.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.n_experts = config.moe_num_experts
        self.top_k = config.moe_top_k

        # Router
        self.router = MoERouter(config.d_model, self.n_experts, self.top_k)

        # Expert FFNs (each is a SwiGLU)
        self.experts: list[SwiGLU] = []
        # TODO: Initialize n_experts SwiGLU instances

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        MoE forward pass.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
            aux_loss: scalar load-balancing loss
        """
        # TODO: Implement MoE forward pass
        # 1. Route tokens to experts
        # 2. For each selected expert, process the routed tokens
        # 3. Combine expert outputs weighted by routing weights
        # 4. Return combined output + aux_loss
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        params = self.router.parameters()
        for expert in self.experts:
            params.extend(expert.parameters())
        return params
