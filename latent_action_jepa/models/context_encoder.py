"""ContextReducer: condenses 4 past state-token grids into compact context tokens.

Input:  H_ctx [B, 4, 64, state_dim]  (frames t-32, t-16, t-8, t)
Output: C_t   [B, out_tokens, action_dim]
"""
import math
import torch
import torch.nn as nn
from .blocks import CrossAttnBlock


class ContextReducer(nn.Module):
    def __init__(self, state_dim=384, action_dim=128, out_tokens=16, depth=2, heads=4, drop=0.0,
                 n_history_frames=4):
        super().__init__()
        self.proj = nn.Linear(state_dim, action_dim)
        self.queries = nn.Parameter(torch.randn(1, out_tokens, action_dim) * 0.02)
        # time embeddings: one per history frame index (0..n_history_frames-1)
        self.time_embed = nn.Parameter(torch.zeros(1, n_history_frames, 1, action_dim))
        # token (spatial) embedding: applied across the 64 patch tokens
        self.token_embed = nn.Parameter(torch.zeros(1, 1, 64, action_dim))
        self.blocks = nn.ModuleList([CrossAttnBlock(action_dim, heads=heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(action_dim)

    def forward(self, H_ctx):
        # H_ctx: [B, T_frames, 64, state_dim]
        B, T, S, D = H_ctx.shape
        x = self.proj(H_ctx)                            # [B, T, S, action_dim]
        x = x + self.time_embed[:, :T] + self.token_embed[:, :, :S]
        x = x.reshape(B, T * S, -1)                      # [B, T*S, action_dim]
        q = self.queries.expand(B, -1, -1)
        for blk in self.blocks:
            q = blk(q, x)
        return self.norm(q)
