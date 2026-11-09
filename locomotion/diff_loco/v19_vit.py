"""V19 small ViT locomotion policy for G1 parkour.

Heightmap only (no depth). ALL 25 heightmap patch tokens + CLS + proprio +
state = 28 tokens. Pure self-attn ViT, CLS readout -> Linear -> action mean.

Tiny: d_model=64, 2 layers, FFN 2x. ~110K params (was 790K).

Reward fix is in tasks.py (forward-only cmd, stalled penalty, feet_air_time
boost). Out_proj init gain=0.5 so mean isn't crushed by Gaussian noise std=1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    PROPRIO_PER_STEP,
)


EXPECTED_OBS_DIM_V19 = PROPRIO_FLAT + CMD_DIM + LIN_VEL_DIM + HEIGHT_DIM  # 778
NON_VIS_DIM_V19 = PROPRIO_FLAT + CMD_DIM + LIN_VEL_DIM  # 591
N_HEIGHT_TOKENS = 25


class _HeightEncV19(nn.Module):
    """(B, 11, 17) -> (B, 25, d). All tokens kept."""
    def __init__(self, d_model: int):
        super().__init__()
        self.patch = nn.Conv2d(1, d_model, kernel_size=3, stride=(2, 3))
        self.pos_embed = nn.Parameter(torch.randn(1, N_HEIGHT_TOKENS, d_model) * 0.02)

    def forward(self, height: torch.Tensor) -> torch.Tensor:
        x = self.patch(height.unsqueeze(1)).flatten(2).transpose(1, 2)
        return x + self.pos_embed


class _ProprioEncV19(nn.Module):
    """(B, 5, 117) -> (B, 1, d). Conv1d + avg-pool over hist + small proj."""
    def __init__(self, d_model: int, per_step: int = PROPRIO_PER_STEP):
        super().__init__()
        self.conv = nn.Conv1d(per_step, d_model, kernel_size=3, padding=1)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, proprio_hist: torch.Tensor) -> torch.Tensor:
        x = proprio_hist.transpose(1, 2)           # (B, 117, 5)
        x = F.elu(self.conv(x))                    # (B, d, 5)
        x = x.mean(dim=2)                          # (B, d)
        return self.norm(self.proj(x)).unsqueeze(1)


class _StateEncV19(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(CMD_DIM + LIN_VEL_DIM, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cmd: torch.Tensor, lin_vel: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(torch.cat([cmd, lin_vel], dim=-1))).unsqueeze(1)


class _TinyBlock(nn.Module):
    """Pre-LN self-attn + FFN. Default ratio 2x (V19); V20 bumps to 4x."""
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult), nn.GELU(), nn.Linear(d_model * ffn_mult, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        a, _ = self.attn(y, y, y, need_weights=False)
        x = x + a
        return x + self.ffn(self.norm2(x))


class _ViT(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.token_type_embed = nn.Parameter(torch.randn(1, 28, d_model) * 0.02)
        self.blocks = nn.ModuleList([_TinyBlock(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, h, p, s) -> torch.Tensor:
        B = h.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1), h, p, s], dim=1)  # (B, 28, d)
        x = x + self.token_type_embed
        for blk in self.blocks:
            x = blk(x)
        return self.final_norm(x)[:, 0]  # CLS


class V19ViTModel(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        out_proj_gain: float = 0.5,
    ) -> None:
        super().__init__()
        del activation

        self.obs_groups = list(obs_groups[obs_set])
        if len(self.obs_groups) != 1:
            raise ValueError(f"V19 expects one obs group, got {self.obs_groups}")
        self.obs_group = self.obs_groups[0]

        flat_dim = int(obs[self.obs_group].shape[-1])
        if flat_dim != EXPECTED_OBS_DIM_V19:
            raise ValueError(
                f"V19 expects obs['{self.obs_group}'] dim {EXPECTED_OBS_DIM_V19}, got {flat_dim}"
            )

        self.obs_normalization = obs_normalization
        self.non_vis_dim = NON_VIS_DIM_V19
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.non_vis_dim)
        else:
            self.obs_normalizer = nn.Identity()

        self.height_enc = _HeightEncV19(d_model)
        self.proprio_enc = _ProprioEncV19(d_model)
        self.state_enc = _StateEncV19(d_model)
        self.vit = _ViT(d_model, n_layers, n_heads)

        self.is_actor = distribution_cfg is not None
        self.output_dim = int(output_dim)

        if self.is_actor:
            self.out_proj = nn.Linear(d_model, output_dim)
            nn.init.orthogonal_(self.out_proj.weight, gain=out_proj_gain)
            nn.init.zeros_(self.out_proj.bias)
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution: Distribution = dist_class(output_dim, **distribution_cfg)
            if self.distribution.input_dim != output_dim:
                raise ValueError("V19 distribution input_dim must equal output_dim.")
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
        h = self.height_enc(height)
        p = self.proprio_enc(proprio_hist)
        s = self.state_enc(cmd, lin_vel)
        return self.vit(h, p, s)

    def get_latent(self, obs, masks=None, hidden_state=None):
        return self._encode(obs)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            flat = obs[self.obs_group]
            self.obs_normalizer.update(flat[:, :self.non_vis_dim])  # type: ignore

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self._encode(obs)
        if self.is_actor:
            mean = self.out_proj(latent)
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
        return self.critic_head(latent)

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
