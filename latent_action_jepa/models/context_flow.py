"""ContextFlowPrior: velocity prediction for z_tau given context tokens C_t.

Input:
  z_tau:  [B, Ka, action_dim]
  tau:    [B, 1, 1]
  C_t:    [B, C_tokens, action_dim]
Output:
  velocity: [B, Ka, action_dim]
"""
import torch
import torch.nn as nn
from .blocks import CrossAttnBlock
from .flow_prior import sinusoidal_time_embed


class ContextFlowPrior(nn.Module):
    def __init__(self, action_dim=128, depth=6, heads=4, drop=0.0):
        super().__init__()
        self.time_dim = action_dim
        self.t_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.blocks = nn.ModuleList([CrossAttnBlock(action_dim, heads=heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(action_dim)
        self.head = nn.Linear(action_dim, action_dim)

    def forward(self, z_tau, tau, C_t):
        if tau.ndim == 1:
            tau = tau.view(-1, 1, 1)
        t_emb = sinusoidal_time_embed(tau, self.time_dim)  # [B,1,D]
        x = z_tau + t_emb
        ctx = torch.cat([C_t, t_emb + self.t_marker], dim=1)
        for blk in self.blocks:
            x = blk(x, ctx)
        return self.head(self.norm(x))
