import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedSpatialReducer(nn.Module):
    """Parameter-free reducer: patch tokens -> spatial grid -> adaptive avg pool -> tokens.

    Input:  [B, N, D] where N is a perfect square (e.g., 196 for 14x14)
    Output: [B, K, D] where K = out_h * out_w
    Dim D is unchanged.
    """

    def __init__(self, out_tokens):
        super().__init__()
        side = int(math.isqrt(out_tokens))
        assert side * side == out_tokens, f"out_tokens must be perfect square, got {out_tokens}"
        self.out_h = side
        self.out_w = side

    def forward(self, tokens):
        b, n, d = tokens.shape
        side = int(math.isqrt(n))
        assert side * side == n, f"input tokens N={n} not perfect square"
        x = tokens.transpose(1, 2).reshape(b, d, side, side)
        x = F.adaptive_avg_pool2d(x, (self.out_h, self.out_w))
        x = x.reshape(b, d, self.out_h * self.out_w).transpose(1, 2)
        return x
