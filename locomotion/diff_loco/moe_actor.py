"""Mixture-of-Experts actor model for rsl_rl PPO.

Drop-in replacement for `rsl_rl.models.MLPModel`. Swaps the backbone MLP with
K experts under a soft-gated router: out(x) = sum_k softmax(gate(x))_k * E_k(x).

Why here (not in rsl_rl itself): stock rsl_rl has no MoE. Keeping this in
diff_loco means no rsl_rl fork — the `class_name` string in the runner cfg
resolves via `rsl_rl.utils.resolve_callable("diff_loco.moe_actor:MoEActorModel")`.

Design notes:
  * Soft gating (all experts active). Cost is K× a single expert forward.
    No top-k routing + no load-balance aux loss — simpler, avoids expert
    collapse in the common case of K <= 4.
  * Output shape, obs-group handling, normalizer, distribution, JIT/ONNX
    export all delegated to the base MLPModel. We only override the
    backbone MLP (self.mlp).
  * Init of distribution-specific weights is re-run after the swap, and is
    a no-op for the default (scalar/log) GaussianDistribution used by V28.
"""
from __future__ import annotations

from dataclasses import field

import torch
import torch.nn as nn

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP


class _SoftMoEBlock(nn.Module):
    """Soft-gated mixture of K MLP experts sharing an input latent."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        expert_hidden_dims: list[int],
        num_experts: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [MLP(input_dim, output_dim, expert_hidden_dims, activation) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_w = torch.softmax(self.gate(x), dim=-1)                # (B, K)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, K, D)
        return (expert_outs * gate_w.unsqueeze(-1)).sum(dim=1)      # (B, D)


class MoEActorModel(MLPModel):
    """MLPModel whose backbone MLP is replaced by a soft-gated MoE."""

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(32,),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_experts: int = 4,
        expert_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    ) -> None:
        # Build the base MLPModel with a throwaway tiny backbone; we overwrite
        # self.mlp below. Passing a (32,) placeholder keeps the placeholder
        # Linear(s) small so the wasted init cost is negligible.
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=list(hidden_dims),
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        last_linear = None
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                last_linear = m
        assert last_linear is not None, "placeholder MLP has no nn.Linear"
        mlp_output_dim = last_linear.out_features
        input_dim = self._get_latent_dim()

        self.mlp = _SoftMoEBlock(
            input_dim=input_dim,
            output_dim=mlp_output_dim,
            expert_hidden_dims=list(expert_hidden_dims),
            num_experts=num_experts,
            activation=activation,
        )

        if self.distribution is not None:
            try:
                self.distribution.init_mlp_weights(self.mlp)
            except Exception:
                # Heteroscedastic std-head init assumes nn.Sequential[-2] — not
                # applicable to an MoE backbone. Skip silently; GaussianDistribution
                # (scalar/log, state-independent) doesn't touch backbone weights.
                pass


@configclass
class RslRlMoEModelCfg(RslRlMLPModelCfg):
    """Configuration for the MoE actor model.

    `class_name` must resolve to the MoEActorModel symbol — use the colon
    separator so rsl_rl's resolve_callable imports from our module:

        class_name="diff_loco.moe_actor:MoEActorModel"
    """

    class_name: str = "diff_loco.moe_actor:MoEActorModel"

    num_experts: int = 4
    """Number of expert MLPs. All experts are active each forward pass (soft gating)."""

    expert_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    """Hidden dims of each expert MLP."""
