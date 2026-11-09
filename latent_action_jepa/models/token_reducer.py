import torch
import torch.nn as nn
from .blocks import CrossAttnBlock


class TokenReducer(nn.Module):
    """Learned-query cross-attention reducer: [B,N,D_in] -> [B,K,D_out]."""

    def __init__(self, in_dim, out_dim, out_tokens, depth=2, heads=8, drop=0.0):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.queries = nn.Parameter(torch.randn(1, out_tokens, out_dim) * 0.02)
        self.blocks = nn.ModuleList([
            CrossAttnBlock(out_dim, heads=heads, drop=drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, tokens):
        b = tokens.shape[0]
        kv = self.proj_in(tokens)
        q = self.queries.expand(b, -1, -1)
        for blk in self.blocks:
            q = blk(q, kv)
        return self.norm(q)
