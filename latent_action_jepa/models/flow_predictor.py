"""Flow-matching predictor for V-JEPA-style pretraining.

Replaces the deterministic V-JEPA-2 predictor head with a flow-matching denoising
predictor: given visible-token encoder output (context) plus noisy target
latents `z_tau` at the prediction positions, predicts the velocity field
v(z_tau, tau, context) = h_target - epsilon.

Standard V-JEPA-2 protocol otherwise: online encoder forward on VISIBLE TOKENS
ONLY, EMA target encoder forward on the full clip (no grad). The target
is the EMA-encoded latent at each predicted position.

Architecture: DiT-style transformer with adaLN-Zero modulation by tau, self-
attention over (visible context + z_tau at target positions). Zero-init final
head so initial output velocity is zero (standard flow-matching init).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit3d import get_3d_sincos_pos_embed


def sinusoidal_time_embed(tau, dim, max_period=10000.0, scale=1000.0):
    """tau in [0, 1] -> [B, dim]. Accepts [B], [B, 1], or [B, 1, 1]."""
    while tau.dim() > 1:
        tau = tau.squeeze(-1)
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(half, device=tau.device).float() / half)
    args = tau[:, None] * freqs[None, :] * scale
    out = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2 == 1:
        out = torch.cat([out, torch.zeros_like(out[:, :1])], dim=-1)
    return out


class DiTSelfBlock(nn.Module):
    """DiT block: self-attention + MLP with adaLN-Zero modulation by conditioning c."""

    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(h, dim), nn.Dropout(drop))
        # 6 params: (shift, scale, gate) for attn and mlp branches
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = [
            p.unsqueeze(1) for p in self.ada(c).chunk(6, dim=-1)
        ]
        h = self.norm1(x) * (1 + scale_a) + shift_a
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a * a
        h2 = self.norm2(x) * (1 + scale_m) + shift_m
        x = x + gate_m * self.mlp(h2)
        return x


class FlowVJEPAPredictor(nn.Module):
    """Flow-matching predictor for V-JEPA.

    Inputs:
      visible  : [B, N_vis, embed_dim]   ONLINE encoder output (visible only).
      vis_idx  : [B, N_vis]               token indices into flat grid.
      pred_idx : [B, N_pred]              indices of target positions.
      z_tau    : [B, N_pred, embed_dim]   noisy target latents.
      tau      : [B] or [B,1] or [B,1,1]  in [0, 1].
    Returns:
      v_pred   : [B, N_pred, embed_dim]   predicted velocity at target positions.

    Velocity-matching loss: v_target = h_target.detach() - epsilon.
    """

    def __init__(self, embed_dim=384, pred_embed_dim=384, depth=8, heads=6,
                 grid_depth=8, grid_size=14, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.pred_embed_dim = pred_embed_dim
        self.in_proj_visible = nn.Linear(embed_dim, pred_embed_dim)
        self.in_proj_z = nn.Linear(embed_dim, pred_embed_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, pred_embed_dim) * 0.02)
        # 3D sin-cos pos embed for the FULL grid (shared with encoder)
        pos = get_3d_sincos_pos_embed(pred_embed_dim, grid_size, grid_depth)
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)
        # tau MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(pred_embed_dim, 4 * pred_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * pred_embed_dim, pred_embed_dim),
        )
        self.blocks = nn.ModuleList([
            DiTSelfBlock(pred_embed_dim, heads, mlp_ratio, drop) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(pred_embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(pred_embed_dim, 2 * pred_embed_dim))
        nn.init.zeros_(self.final_mod[-1].weight); nn.init.zeros_(self.final_mod[-1].bias)
        self.head = nn.Linear(pred_embed_dim, embed_dim)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, visible, vis_idx, pred_idx, z_tau, tau):
        B, N_vis, _ = visible.shape
        _, N_pred, _ = z_tau.shape
        pe = self.pos_embed.expand(B, -1, -1)                              # [B, n_tokens, pred_dim]
        # visible branch
        v = self.in_proj_visible(visible)
        v = v + pe.gather(1, vis_idx.unsqueeze(-1).expand(-1, -1, self.pred_embed_dim))
        # z branch
        z = self.in_proj_z(z_tau) + self.mask_token
        z = z + pe.gather(1, pred_idx.unsqueeze(-1).expand(-1, -1, self.pred_embed_dim))
        x = torch.cat([v, z], dim=1)
        # tau conditioning
        t_emb = sinusoidal_time_embed(tau, self.pred_embed_dim)
        c = self.t_mlp(t_emb)
        for blk in self.blocks:
            x = blk(x, c)
        z_out = x[:, N_vis:]
        shift, scale = self.final_mod(c).chunk(2, dim=-1)
        z_out = self.final_norm(z_out) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.head(z_out)
