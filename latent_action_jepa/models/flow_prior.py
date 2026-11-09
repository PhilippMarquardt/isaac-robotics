import math
import torch
import torch.nn as nn
from .blocks import CrossAttnBlock


def sinusoidal_time_embed(tau, dim):
    # tau: [B,1,1] -> [B,1,dim]
    device = tau.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
    args = tau.view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], dim - emb.shape[-1], device=device)], dim=-1)
    return emb.unsqueeze(1)  # [B,1,dim]


class FlowPrior(nn.Module):
    """Predict velocity v_pred for Z_tau conditioned on H_t and tau."""

    def __init__(self, state_dim, action_dim, depth=6, heads=8, drop=0.0):
        super().__init__()
        self.ctx_proj = nn.Linear(state_dim, action_dim)
        self.time_dim = action_dim
        self.blocks = nn.ModuleList([
            CrossAttnBlock(action_dim, heads=heads, drop=drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(action_dim)
        self.head = nn.Linear(action_dim, action_dim)

    def forward(self, z_tau, tau, h_t):
        # z_tau [B,Ka,Da], tau [B,1,1] or [B], h_t [B,Ns,Ds]
        if tau.ndim == 1:
            tau = tau.view(-1, 1, 1)
        t_emb = sinusoidal_time_embed(tau, self.time_dim)  # [B,1,Da]
        x = z_tau + t_emb  # broadcast on tokens
        ctx = self.ctx_proj(h_t)
        ctx = torch.cat([ctx, t_emb], dim=1)
        for blk in self.blocks:
            x = blk(x, ctx)
        x = self.norm(x)
        return self.head(x)
