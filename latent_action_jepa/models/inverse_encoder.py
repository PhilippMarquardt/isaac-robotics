import torch
import torch.nn as nn
from .blocks import CrossAttnBlock


class InverseActionEncoder(nn.Module):
    """H_t [B,Ns,Ds], H_tp [B,Nf,Ds] -> Z* [B,Ka,Da]."""

    def __init__(self, state_dim, action_dim, action_tokens, depth=4, heads=8, drop=0.0):
        super().__init__()
        self.proj_t = nn.Linear(state_dim, action_dim)
        self.proj_tp = nn.Linear(state_dim, action_dim)
        self.t_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.tp_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.queries = nn.Parameter(torch.randn(1, action_tokens, action_dim) * 0.02)
        self.blocks = nn.ModuleList([
            CrossAttnBlock(action_dim, heads=heads, drop=drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(action_dim)

    def forward(self, h_t, h_tp):
        b = h_t.shape[0]
        a = self.proj_t(h_t) + self.t_marker
        c = self.proj_tp(h_tp) + self.tp_marker
        kv = torch.cat([a, c], dim=1)
        q = self.queries.expand(b, -1, -1)
        for blk in self.blocks:
            q = blk(q, kv)
        return self.norm(q)
