"""V17 Single-Pass Transformer locomotion policy for G1 parkour.

Same multi-modal context encoder as V16 (height + depth + proprio + state
tokens -> 54x128), but replaces the iterative flow-matching DiT actor with a
single forward pass: learnable action query tokens cross-attend to context
through N decoder blocks, then a linear head emits the Gaussian mean.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories

from diff_loco.v16_flow import (
    ACT_DIM,
    ANG_DIM,
    CMD_DIM,
    DEPTH_DIM,
    DEPTH_H,
    DEPTH_W,
    EXPECTED_OBS_DIM,
    GRAV_DIM,
    HEIGHT_DIM,
    HEIGHT_H,
    HEIGHT_W,
    JPOS_DIM,
    JVEL_DIM,
    LAST_ACT_DIM,
    LIN_VEL_DIM,
    NON_VIS_DIM,
    PROPRIO_FLAT,
    PROPRIO_HIST,
    PROPRIO_PER_STEP,
    _ContextEncoder,
)


class _XAttnBlock(nn.Module):
    """Self-attn over queries + cross-attn to context + FFN. Pre-LN."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_q2 = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_q3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, q: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        y = self.norm_q1(q)
        a, _ = self.self_attn(y, y, y, need_weights=False)
        q = q + a
        y = self.norm_q2(q)
        k = self.norm_kv(ctx)
        a, _ = self.cross_attn(y, k, k, need_weights=False)
        q = q + a
        q = q + self.ffn(self.norm_q3(q))
        return q


class _ActionTransformer(nn.Module):
    """Learnable action queries -> N x (self+cross+ffn) -> Linear -> action mean."""

    def __init__(self, d_model: int, action_dim: int, n_queries: int = 4,
                 n_layers: int = 3, n_heads: int = 4):
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.blocks = nn.ModuleList([_XAttnBlock(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model * n_queries, action_dim)
        nn.init.orthogonal_(self.out_proj.weight, gain=0.01)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        B = context.shape[0]
        q = self.queries.expand(B, -1, -1)
        for blk in self.blocks:
            q = blk(q, context)
        q = self.final_norm(q).flatten(1)  # (B, n_q*d)
        return self.out_proj(q)            # (B, action_dim)


class V17TransformerModel(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        d_model: int = 128,
        n_heads: int = 4,
        n_dec_layers: int = 3,
        n_action_queries: int = 4,
        action_scale: float = 0.25,
    ) -> None:
        super().__init__()
        del activation

        self.obs_groups = list(obs_groups[obs_set])
        if len(self.obs_groups) != 1:
            raise ValueError(f"V17 expects one obs group, got {self.obs_groups}")
        self.obs_group = self.obs_groups[0]

        flat_dim = int(obs[self.obs_group].shape[-1])
        if flat_dim != EXPECTED_OBS_DIM:
            raise ValueError(
                f"V17 expects obs['{self.obs_group}'] dim {EXPECTED_OBS_DIM}, got {flat_dim}"
            )

        self.obs_normalization = obs_normalization
        self.non_vis_dim = NON_VIS_DIM
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.non_vis_dim)
        else:
            self.obs_normalizer = nn.Identity()

        self.context_enc = _ContextEncoder(d_model)

        self.is_actor = distribution_cfg is not None
        self.action_scale = float(action_scale)
        self.output_dim = int(output_dim)

        if self.is_actor:
            self.actor_head = _ActionTransformer(
                d_model, output_dim, n_action_queries, n_dec_layers, n_heads
            )
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution: Distribution = dist_class(output_dim, **distribution_cfg)
            if self.distribution.input_dim != output_dim:
                raise ValueError("V17 distribution input_dim must equal output_dim.")
            self.mlp = nn.Sequential(nn.Linear(output_dim, output_dim))
            self.distribution.init_mlp_weights(self.mlp)
        else:
            self.distribution = None  # type: ignore[assignment]
            head_dims = [d_model] + list(hidden_dims) + [output_dim]
            layers: list[nn.Module] = []
            for i in range(len(head_dims) - 1):
                layers.append(nn.Linear(head_dims[i], head_dims[i + 1]))
                if i < len(head_dims) - 2:
                    layers.append(nn.ELU())
            self.critic_head = nn.Sequential(*layers)

    def _split_obs(self, flat: torch.Tensor):
        B = flat.shape[0]
        off = 0
        proprio_flat = flat[:, off:off + PROPRIO_FLAT]; off += PROPRIO_FLAT
        cmd = flat[:, off:off + CMD_DIM]; off += CMD_DIM
        lin_vel = flat[:, off:off + LIN_VEL_DIM]; off += LIN_VEL_DIM
        height_flat = flat[:, off:off + HEIGHT_DIM]; off += HEIGHT_DIM
        depth_flat = flat[:, off:off + DEPTH_DIM]; off += DEPTH_DIM

        po = 0
        ang = proprio_flat[:, po:po + PROPRIO_HIST * ANG_DIM].view(B, PROPRIO_HIST, ANG_DIM); po += PROPRIO_HIST * ANG_DIM
        grv = proprio_flat[:, po:po + PROPRIO_HIST * GRAV_DIM].view(B, PROPRIO_HIST, GRAV_DIM); po += PROPRIO_HIST * GRAV_DIM
        jp = proprio_flat[:, po:po + PROPRIO_HIST * JPOS_DIM].view(B, PROPRIO_HIST, JPOS_DIM); po += PROPRIO_HIST * JPOS_DIM
        jv = proprio_flat[:, po:po + PROPRIO_HIST * JVEL_DIM].view(B, PROPRIO_HIST, JVEL_DIM); po += PROPRIO_HIST * JVEL_DIM
        la = proprio_flat[:, po:po + PROPRIO_HIST * LAST_ACT_DIM].view(B, PROPRIO_HIST, LAST_ACT_DIM)
        proprio_hist = torch.cat([ang, grv, jp, jv, la], dim=-1)

        height = height_flat.view(B, HEIGHT_H, HEIGHT_W)
        depth = depth_flat.view(B, DEPTH_H, DEPTH_W)
        return proprio_hist, cmd, lin_vel, height, depth

    def _encode(self, obs: TensorDict) -> torch.Tensor:
        flat = obs[self.obs_group]
        if self.obs_normalization:
            non_vis = flat[:, :self.non_vis_dim]
            vis = flat[:, self.non_vis_dim:]
            non_vis = self.obs_normalizer(non_vis)
            flat = torch.cat([non_vis, vis], dim=-1)
        proprio_hist, cmd, lin_vel, height, depth = self._split_obs(flat)
        return self.context_enc(proprio_hist, cmd, lin_vel, height, depth)

    def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None,
                   hidden_state: HiddenState = None) -> torch.Tensor:
        return self._encode(obs).mean(dim=1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            flat = obs[self.obs_group]
            self.obs_normalizer.update(flat[:, :self.non_vis_dim])  # type: ignore

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        context = self._encode(obs)
        if self.is_actor:
            mean = self.actor_head(context)                      # (B, ACT_DIM) raw mean
            mean = torch.nan_to_num(mean, nan=0.0, posinf=5.0, neginf=-5.0)
            if hasattr(self.distribution, "log_std_param"):
                with torch.no_grad():
                    p = self.distribution.log_std_param
                    bad = torch.isnan(p.data) | torch.isinf(p.data)
                    if bad.any():
                        p.data[bad] = 0.0
                    p.data.clamp_(min=-5.0, max=2.0)
            if stochastic_output:
                self.distribution.update(mean)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mean)
        pooled = context.mean(dim=1)
        return self.critic_head(pooled)

    def reset(self, dones=None, hidden_state=None) -> None: pass
    def get_hidden_state(self) -> HiddenState: return None
    def detach_hidden_state(self, dones=None) -> None: pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean if self.is_actor else torch.zeros(1)

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std if self.is_actor else torch.zeros(1)

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy if self.is_actor else torch.zeros(1)

    @property
    def output_distribution_params(self):
        return self.distribution.params if self.is_actor else ()

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(self, old_params, new_params) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self): raise NotImplementedError
    def as_onnx(self, verbose: bool): raise NotImplementedError
