"""V18 Heightmap-only single-pass transformer locomotion policy for G1 parkour.

Identical to V17 except: no depth camera. Context = heightmap(16) + proprio(1) + state(1)
= 18 tokens x 128 dim. Encoders reused from v16_flow.
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
    GRAV_DIM,
    HEIGHT_DIM,
    HEIGHT_H,
    HEIGHT_W,
    JPOS_DIM,
    JVEL_DIM,
    LAST_ACT_DIM,
    LIN_VEL_DIM,
    PROPRIO_FLAT,
    PROPRIO_HIST,
    _HeightMapEncoder,
    _ProprioEncoder,
    _StateTokenEncoder,
)
from diff_loco.v17_transformer import _ActionTransformer, _XAttnBlock  # noqa: F401


# V18-specific obs layout: proprio + cmd + lin_vel + heightmap (no depth)
EXPECTED_OBS_DIM_V18 = PROPRIO_FLAT + CMD_DIM + LIN_VEL_DIM + HEIGHT_DIM  # 778
NON_VIS_DIM_V18 = PROPRIO_FLAT + CMD_DIM + LIN_VEL_DIM  # 591


class _ContextEncoderV18(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.height_enc = _HeightMapEncoder(d_model)
        self.proprio_enc = _ProprioEncoder(d_model)
        self.state_enc = _StateTokenEncoder(d_model)

    def forward(self, proprio_hist, cmd, lin_vel, height) -> torch.Tensor:
        h = self.height_enc(height)                   # (B, 16, d)
        p = self.proprio_enc(proprio_hist)            # (B, 1, d)
        s = self.state_enc(cmd, lin_vel)              # (B, 1, d)
        return torch.cat([h, p, s], dim=1)            # (B, 18, d)


class V18HeightOnlyModel(nn.Module):
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
    ) -> None:
        super().__init__()
        del activation

        self.obs_groups = list(obs_groups[obs_set])
        if len(self.obs_groups) != 1:
            raise ValueError(f"V18 expects one obs group, got {self.obs_groups}")
        self.obs_group = self.obs_groups[0]

        flat_dim = int(obs[self.obs_group].shape[-1])
        if flat_dim != EXPECTED_OBS_DIM_V18:
            raise ValueError(
                f"V18 expects obs['{self.obs_group}'] dim {EXPECTED_OBS_DIM_V18}, got {flat_dim}"
            )

        self.obs_normalization = obs_normalization
        self.non_vis_dim = NON_VIS_DIM_V18
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.non_vis_dim)
        else:
            self.obs_normalizer = nn.Identity()

        self.context_enc = _ContextEncoderV18(d_model)

        self.is_actor = distribution_cfg is not None
        self.output_dim = int(output_dim)

        if self.is_actor:
            self.actor_head = _ActionTransformer(
                d_model, output_dim, n_action_queries, n_dec_layers, n_heads
            )
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution: Distribution = dist_class(output_dim, **distribution_cfg)
            if self.distribution.input_dim != output_dim:
                raise ValueError("V18 distribution input_dim must equal output_dim.")
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

        po = 0
        ang = proprio_flat[:, po:po + PROPRIO_HIST * ANG_DIM].view(B, PROPRIO_HIST, ANG_DIM); po += PROPRIO_HIST * ANG_DIM
        grv = proprio_flat[:, po:po + PROPRIO_HIST * GRAV_DIM].view(B, PROPRIO_HIST, GRAV_DIM); po += PROPRIO_HIST * GRAV_DIM
        jp = proprio_flat[:, po:po + PROPRIO_HIST * JPOS_DIM].view(B, PROPRIO_HIST, JPOS_DIM); po += PROPRIO_HIST * JPOS_DIM
        jv = proprio_flat[:, po:po + PROPRIO_HIST * JVEL_DIM].view(B, PROPRIO_HIST, JVEL_DIM); po += PROPRIO_HIST * JVEL_DIM
        la = proprio_flat[:, po:po + PROPRIO_HIST * LAST_ACT_DIM].view(B, PROPRIO_HIST, LAST_ACT_DIM)
        proprio_hist = torch.cat([ang, grv, jp, jv, la], dim=-1)

        height = height_flat.view(B, HEIGHT_H, HEIGHT_W)
        return proprio_hist, cmd, lin_vel, height

    def _encode(self, obs: TensorDict) -> torch.Tensor:
        flat = obs[self.obs_group]
        if self.obs_normalization:
            non_vis = flat[:, :self.non_vis_dim]
            vis = flat[:, self.non_vis_dim:]
            non_vis = self.obs_normalizer(non_vis)
            flat = torch.cat([non_vis, vis], dim=-1)
        proprio_hist, cmd, lin_vel, height = self._split_obs(flat)
        return self.context_enc(proprio_hist, cmd, lin_vel, height)

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
            mean = self.actor_head(context)
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
