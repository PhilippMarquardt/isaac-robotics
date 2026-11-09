import torch
import torch.nn as nn
from .blocks import CrossAttnBlock


class TransitionModel(nn.Module):
    """H_t [B,Ns,Ds], Z [B,Ka,Da] -> H_pred [B,Kf,Ds]."""

    def __init__(self, state_dim, action_dim, future_tokens, depth=6, heads=8, drop=0.0):
        super().__init__()
        self.z_proj = nn.Linear(action_dim, state_dim)
        self.h_marker = nn.Parameter(torch.zeros(1, 1, state_dim))
        self.z_marker = nn.Parameter(torch.zeros(1, 1, state_dim))
        self.future_q = nn.Parameter(torch.randn(1, future_tokens, state_dim) * 0.02)
        self.blocks = nn.ModuleList([
            CrossAttnBlock(state_dim, heads=heads, drop=drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, h_t, z):
        b = h_t.shape[0]
        z_emb = self.z_proj(z) + self.z_marker
        h = h_t + self.h_marker
        mem = torch.cat([h, z_emb], dim=1)
        q = self.future_q.expand(b, -1, -1)
        for blk in self.blocks:
            q = blk(q, mem)
        return self.norm(q)
