"""MaskedGoalFlowPrior: velocity for z_tau given (H_t, partial H_goal, visible_mask).

Partial goal: a subset of the 16 future tokens are visible; the rest are replaced
with a learned [MASK] token plus a goal-hidden marker.
"""
import torch
import torch.nn as nn
from .blocks import CrossAttnBlock
from .flow_prior import sinusoidal_time_embed


class MaskedGoalFlowPrior(nn.Module):
    def __init__(self, state_dim=384, action_dim=128, future_tokens=16,
                 depth=6, heads=4, drop=0.0):
        super().__init__()
        self.future_tokens = future_tokens
        self.proj_h = nn.Linear(state_dim, action_dim)
        self.proj_g = nn.Linear(state_dim, action_dim)
        self.h_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.goal_visible_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.goal_hidden_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        # learned [MASK] token in raw state-space (so projection still has signal)
        self.mask_token = nn.Parameter(torch.randn(1, 1, state_dim) * 0.02)
        self.time_dim = action_dim
        self.blocks = nn.ModuleList([CrossAttnBlock(action_dim, heads=heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(action_dim)
        self.head = nn.Linear(action_dim, action_dim)

    def forward(self, z_tau, tau, H_t, H_goal, visible_mask):
        """
        z_tau: [B, Ka, action_dim]
        tau:   [B, 1, 1]
        H_t:   [B, 64, state_dim]
        H_goal:[B, 16, state_dim]
        visible_mask: [B, 16]  (True where goal token is revealed)
        """
        if tau.ndim == 1:
            tau = tau.view(-1, 1, 1)
        m = visible_mask.float().unsqueeze(-1)                         # [B,16,1]
        G_in = m * H_goal + (1.0 - m) * self.mask_token                 # broadcast learned mask
        g_proj = self.proj_g(G_in)                                       # [B,16,action_dim]
        markers = m * self.goal_visible_marker + (1.0 - m) * self.goal_hidden_marker  # [B,16,action_dim]
        g_proj = g_proj + markers
        h_proj = self.proj_h(H_t) + self.h_marker                       # [B,64,action_dim]
        t_emb = sinusoidal_time_embed(tau, self.time_dim)               # [B,1,action_dim]
        ctx = torch.cat([h_proj, g_proj, t_emb], dim=1)                  # [B, 64+16+1, action_dim]
        x = z_tau + t_emb
        for blk in self.blocks:
            x = blk(x, ctx)
        return self.head(self.norm(x))
