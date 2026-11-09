"""V-JEPA-style predictor.

Input:
  ctx_tokens:   [B, T_ctx, N_patch, D]   (online encoder output for context frames)
  target_pos:   [B, N_mask, 2]           (which (row, col) patch positions to predict)
Output:
  pred_target_tokens: [B, N_mask, D]     (predictions at masked target positions)

Uses 2D positional embeddings (sin-cos) for spatial positions, plus per-frame
(time) embeddings for context vs target.
"""
import math
import torch
import torch.nn as nn

from .blocks import SelfAttnBlock


def sincos_2d_pos_embed(h, w, dim, device):
    """Standard 2D sin-cos position embedding [h*w, dim]."""
    half = dim // 2
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    pos_y = grid_y.flatten().float()  # [h*w]
    pos_x = grid_x.flatten().float()

    def _embed(pos, d):
        # pos: [N], d half-dim
        omega = torch.arange(d // 2, device=device).float() / (d // 2)
        omega = 1.0 / (10000 ** omega)  # [d//2]
        out = pos.unsqueeze(-1) * omega.unsqueeze(0)  # [N, d//2]
        return torch.cat([out.sin(), out.cos()], dim=-1)  # [N, d]

    emb_y = _embed(pos_y, half)
    emb_x = _embed(pos_x, half)
    return torch.cat([emb_y, emb_x], dim=-1)  # [h*w, dim]


class VJEPAPredictor(nn.Module):
    def __init__(self, dim=384, depth=6, heads=6, drop=0.0, n_ctx_frames=3, grid_h=14, grid_w=14):
        super().__init__()
        self.dim = dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_patch = grid_h * grid_w
        self.n_ctx_frames = n_ctx_frames
        # learned frame-id embeddings: 0..T_ctx-1 (context) and T_ctx (target)
        self.frame_embed = nn.Parameter(torch.randn(n_ctx_frames + 1, 1, dim) * 0.02)
        # learned mask token for target positions
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        # 2D positional embeddings (registered, fixed sin-cos)
        self.register_buffer("pos_embed_2d", sincos_2d_pos_embed(grid_h, grid_w, dim, device="cpu"))
        self.blocks = nn.ModuleList([SelfAttnBlock(dim, heads=heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, ctx_tokens, mask_indices):
        """
        ctx_tokens: [B, T_ctx, N_patch, D]
        mask_indices: [B, N_mask] integers in [0, N_patch)
        Returns predictions at masked positions: [B, N_mask, D]
        """
        B, T_ctx, N_patch, D = ctx_tokens.shape
        device = ctx_tokens.device
        pos = self.pos_embed_2d.to(device)                    # [N_patch, D]

        # context tokens with frame + spatial pos embeds
        ctx = ctx_tokens + pos.view(1, 1, N_patch, D) + self.frame_embed[:T_ctx].view(1, T_ctx, 1, D)
        ctx = ctx.reshape(B, T_ctx * N_patch, D)

        # target query tokens: mask_token + spatial pos at requested positions + target-frame embed
        N_mask = mask_indices.shape[1]
        mt = self.mask_token.expand(B, N_mask, D)
        spatial_at_mask = pos[mask_indices]                   # [B, N_mask, D]
        target_frame_emb = self.frame_embed[-1].view(1, 1, D)
        tgt_q = mt + spatial_at_mask + target_frame_emb

        x = torch.cat([ctx, tgt_q], dim=1)                    # [B, T_ctx*N_patch + N_mask, D]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, T_ctx * N_patch:])                  # take target queries only
        return self.head(x)                                   # [B, N_mask, D]
