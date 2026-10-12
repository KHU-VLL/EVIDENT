"""Entity Bottleneck (EB) Adapter with Entity-to-eVidence (E2V) gating.

One adapter sits in parallel to an LLM decoder layer (see `inject.py`):

    X_out = DecoderLayer(X) + EBAdapter(X)

Inside the adapter (Sec. 4.2 / 4.4 of the paper):

    h          = LayerNorm(X W_down)                         # [B, L, d]
    per frame t (visual tokens f'_t of h):
        S, Â   = SlotAttention(f'_t)                        # K entity slots
        f̃'_t  = (Â^T S) W_rec                               # token reconstruction
    g_t        = norm(cos(mean(f'_t), q_sub)) * norm(cos(mean(f'_t), q_obj))
    out        = concat(f̃' ⊙ g, h_txt) W_up                 # W_up is zero-initialized

Text tokens skip the slot path but still receive the down/up bottleneck residual.
"""
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .slot_attention import SlotAttention


@dataclass
class EBContext:
    """Per-forward inputs shared by every EB Adapter of a model.

    Set once per batch (training: by the trainer, inference: by the caller) and
    read by all adapter layers. `input_ids` is refreshed on every forward by a
    pre-hook so that KV-cache decoding steps are recognized as text-only.
    """
    input_ids: Optional[torch.Tensor] = None
    subject_token_indices: Optional[List[List[int]]] = None  # per sample
    object_token_indices: Optional[List[List[int]]] = None   # per sample
    concept_valid_mask: Optional[torch.Tensor] = None         # [B, 2] bool (subject, object)


class EBAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int = 896,
        num_slots: int = 4,
        num_iterations: int = 3,
        tokens_per_frame: int = 64,
        use_e2v_gate: bool = True,
    ):
        super().__init__()
        self.tokens_per_frame = tokens_per_frame
        self.use_e2v_gate = use_e2v_gate

        # Module creation order is kept identical to the original implementation so
        # that a given seed yields the same initialization.
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.norm = nn.LayerNorm(bottleneck_dim)
        self.slot_attn = SlotAttention(num_slots=num_slots, dim=bottleneck_dim, iters=num_iterations)
        self.recon_out_proj = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        # Filled during forward; read by the EBD loss (with grad) and for analysis.
        self.last_attn: Optional[torch.Tensor] = None  # [B*T, K, N] Â
        self.last_gate: Optional[torch.Tensor] = None  # [B, T]

    @torch.autocast(device_type="cuda", enabled=False)
    def forward_text_only(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Bottleneck residual for forwards without visual tokens (KV-cache decoding).

        During training every token, including the teacher-forced answer tokens,
        receives the adapter residual, so decoding must apply it as well.
        """
        h = self.norm(self.down_proj(hidden_states.to(self.down_proj.weight.dtype)))
        return self.up_proj(h)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, hidden_states: torch.Tensor, vision_mask: torch.Tensor, ctx: EBContext) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, D] input of the decoder layer.
            vision_mask:   [B, L] True at visual tokens; every sample must hold the
                           same number of frames x tokens_per_frame visual tokens.
        """
        hidden_states = hidden_states.to(self.down_proj.weight.dtype)
        B, L, _ = hidden_states.shape
        h = self.norm(self.down_proj(hidden_states))  # [B, L, d]
        d = h.shape[-1]

        num_vis = int(vision_mask[0].sum())
        if num_vis % self.tokens_per_frame != 0:
            raise ValueError(f"{num_vis} visual tokens is not a multiple of tokens_per_frame={self.tokens_per_frame}.")
        T, N = num_vis // self.tokens_per_frame, self.tokens_per_frame
        frames = h[vision_mask].view(B, T, N, d)

        # Entity-to-eVidence gate from the down-projected visual tokens (before slot attention).
        gate = self._e2v_gate(h, frames, ctx) if self.use_e2v_gate else None
        self.last_gate = gate

        # Slot attention per frame. Frames are independent, but they are run one at a
        # time (not folded into the batch) so that bf16 results match the released checkpoints.
        recon, attns = [], []
        for t in range(T):
            slots, attn = self.slot_attn(frames[:, t])
            recon.append(self.recon_out_proj(torch.einsum("bnk,bkd->bnd", attn.transpose(1, 2), slots)))  # Â^T S
            attns.append(attn)
        recon = torch.stack(recon, dim=1)  # [B, T, N, d]
        self.last_attn = torch.stack(attns, dim=1).flatten(0, 1)  # [B*T, K, N]

        if gate is not None:
            recon = recon * gate[:, :, None, None].to(recon.dtype)

        out = h.masked_scatter(vision_mask.unsqueeze(-1), recon)
        return self.up_proj(out)

    def _e2v_gate(self, h: torch.Tensor, frames: torch.Tensor, ctx: EBContext) -> torch.Tensor:
        """g_t = ŝ_t^sub · ŝ_t^obj with per-sample min-max normalization over frames.

        A concept that is missing for a sample (no subject/object, or its tokens
        were not found in the prompt) contributes a factor of 1.
        """
        B, T = frames.shape[:2]
        if ctx.concept_valid_mask is None:
            return torch.ones(B, T, device=h.device, dtype=h.dtype)

        concepts = torch.zeros(B, 2, h.shape[-1], device=h.device, dtype=h.dtype)
        for b in range(B):
            for k, indices in enumerate((ctx.subject_token_indices, ctx.object_token_indices)):
                idx = indices[b] if indices is not None and b < len(indices) else None
                if idx and max(idx) < h.shape[1]:
                    concepts[b, k] = h[b, idx].mean(dim=0)

        frame_avg = F.normalize(frames.mean(dim=2), dim=-1)                          # [B, T, d]
        scores = torch.einsum("btd,bkd->btk", frame_avg, F.normalize(concepts, dim=-1))  # [B, T, 2]
        s_min = scores.amin(dim=1, keepdim=True)
        s_max = scores.amax(dim=1, keepdim=True)
        scores = (scores - s_min) / (s_max - s_min + 1e-6)

        valid = ctx.concept_valid_mask.to(scores.device)[:, None, :].expand_as(scores)
        scores = torch.where(valid, scores, torch.ones_like(scores))
        return scores[..., 0] * scores[..., 1]
