import torch
import torch.nn as nn
from .blocks import CrossAttnBlock
from .flow_prior import sinusoidal_time_embed


class GoalConditionedFlowPrior(nn.Module):
    """Predict velocity for Z_tau conditioned on (H_t, H_goal, tau)."""

    def __init__(self, state_dim, action_dim, depth=6, heads=8, drop=0.0):
        super().__init__()
        self.ctx_t_proj = nn.Linear(state_dim, action_dim)
        self.ctx_g_proj = nn.Linear(state_dim, action_dim)
        self.t_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.g_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.time_dim = action_dim
        self.blocks = nn.ModuleList([
            CrossAttnBlock(action_dim, heads=heads, drop=drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(action_dim)
        self.head = nn.Linear(action_dim, action_dim)

    def forward(self, z_tau, tau, h_t, h_goal):
        if tau.ndim == 1:
            tau = tau.view(-1, 1, 1)
        t_emb = sinusoidal_time_embed(tau, self.time_dim)
        x = z_tau + t_emb
        ctx_t = self.ctx_t_proj(h_t) + self.t_marker
        ctx_g = self.ctx_g_proj(h_goal) + self.g_marker
        ctx = torch.cat([ctx_t, ctx_g, t_emb], dim=1)
        for blk in self.blocks:
            x = blk(x, ctx)
        x = self.norm(x)
        return self.head(x)
