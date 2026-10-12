"""Slot attention (Locatello et al., 2020) with learned per-slot queries.

Used inside the EB Adapter to compress the N visual tokens of one frame into K
entity slots. Besides the refined slots, the module exposes the attention map
of the last iteration (Eq. 3 of the paper):

    A = softmax_over_slots(K Q^T / sqrt(d))      # each token distributes over slots
    Â = A / sum_over_tokens(A)                   # each slot is a weighted mean of tokens

Â is used both for the slot update and for the token reconstruction / EBD loss.
"""
import torch
import torch.nn as nn
from torch.nn import init


class SlotAttention(nn.Module):
    def __init__(self, num_slots: int, dim: int, iters: int = 3, eps: float = 1e-8, mlp_hidden_dim: int = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        # Learned slot queries S^(0).
        self.latents = nn.Parameter(torch.randn(1, num_slots, dim))
        init.xavier_uniform_(self.latents)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)

        mlp_hidden_dim = max(dim, mlp_hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: [B, N, D] tokens of one frame (B may fold batch and frames).
        Returns:
            slots: [B, K, D]
            attn:  [B, K, N] token-normalized attention Â of the last iteration.
        """
        b, _, d = inputs.shape
        slots = self.latents.expand(b, -1, -1).to(device=inputs.device, dtype=inputs.dtype)

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        attn = None
        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale  # [B, K, N]
            attn = dots.softmax(dim=1) + self.eps                   # A: softmax over slots
            attn = attn / attn.sum(dim=-1, keepdim=True)            # Â: normalize over tokens

            updates = torch.einsum("bjd,bij->bid", v, attn)
            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn
