import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transition import TransitionModel
from .fixed_reducer import FixedSpatialReducer


class ResidualTransitionModel(nn.Module):
    """H_pred = base(H_t) + alpha * delta(H_t, Z).

    base is parameter-free spatial-pool reduction of H_t to future_tokens.
    delta is learned (small transformer).
    alpha = alpha_min + softplus(raw_alpha), raw_alpha init so effective alpha ~= alpha_init.
    Floor prevents Z-collapse from dead-gate basin observed when alpha=0 exactly.
    """

    def __init__(self, state_dim, action_dim, future_tokens, depth=2, heads=8, drop=0.0,
                 alpha_min=0.02, alpha_init=0.05):
        super().__init__()
        self.base = FixedSpatialReducer(future_tokens)
        self.delta = TransitionModel(state_dim, action_dim, future_tokens,
                                     depth=depth, heads=heads, drop=drop)
        self.alpha_min = float(alpha_min)
        self.raw_alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    @property
    def alpha(self):
        return self.raw_alpha.clamp(min=self.alpha_min)

    def forward(self, h_t, z):
        base = self.base(h_t)
        delta = self.delta(h_t, z)
        return base + self.alpha * delta
