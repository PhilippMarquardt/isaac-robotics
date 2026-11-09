import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim, mult=4, drop=0.0):
        super().__init__()
        h = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttnBlock(nn.Module):
    def __init__(self, dim, heads=8, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, drop=drop)

    def forward(self, x):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, heads=8, drop=0.0):
        super().__init__()
        self.nq = nn.LayerNorm(dim)
        self.nk = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.ns = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.nm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, drop=drop)

    def forward(self, q, kv):
        a, _ = self.attn(self.nq(q), self.nk(kv), self.nk(kv), need_weights=False)
        q = q + a
        s = self.ns(q)
        sa, _ = self.self_attn(s, s, s, need_weights=False)
        q = q + sa
        q = q + self.mlp(self.nm(q))
        return q
