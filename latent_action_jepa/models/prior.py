import torch
import torch.nn as nn
from .blocks import CrossAttnBlock


class PooledMLPPrior(nn.Module):
    """Mean-pool H_t, then MLP -> reshape to [B, Ka, Da]."""

    def __init__(self, state_dim, action_tokens, action_dim, hidden=1024, depth=3, drop=0.0):
        super().__init__()
        self.action_tokens = action_tokens
        self.action_dim = action_dim
        layers = []
        in_d = state_dim
        for _ in range(depth):
            layers += [nn.Linear(in_d, hidden), nn.GELU(), nn.Dropout(drop)]
            in_d = hidden
        layers += [nn.Linear(in_d, action_tokens * action_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, h_t):
        pooled = h_t.mean(dim=1)
        z = self.mlp(pooled)
        return z.view(z.shape[0], self.action_tokens, self.action_dim)


class CrossAttnPrior(nn.Module):
    """Learned action queries cross-attend to projected H_t."""

    def __init__(self, state_dim, action_tokens, action_dim, depth=4, heads=8, drop=0.0):
        super().__init__()
        self.proj = nn.Linear(state_dim, action_dim)
        self.queries = nn.Parameter(torch.randn(1, action_tokens, action_dim) * 0.02)
        self.blocks = nn.ModuleList([
            CrossAttnBlock(action_dim, heads=heads, drop=drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(action_dim)
        self.head = nn.Linear(action_dim, action_dim)

    def forward(self, h_t):
        b = h_t.shape[0]
        kv = self.proj(h_t)
        q = self.queries.expand(b, -1, -1)
        for blk in self.blocks:
            q = blk(q, kv)
        return self.head(self.norm(q))
