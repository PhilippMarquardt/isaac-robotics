"""V23 height-only cross-attn: V22 + ByteDance-style forward-biased larger heightmap.

Heightmap: 2.0m forward x 1.0m lateral at 0.1m, forward-biased (offset +0.5m).
-> 21x11 = 231 cells, covers -0.5m behind to +1.5m ahead.

Patchify: Conv2d k=3 stride=(2,3) on 11x21 -> 5x7 = 35 tokens (vs V22's 25).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories

from diff_loco.v16_flow import (
    ANG_DIM,
    CMD_DIM,
    GRAV_DIM,
    JPOS_DIM,
    JVEL_DIM,
    LAST_ACT_DIM,
    LIN_VEL_DIM,
    PROPRIO_FLAT,
    PROPRIO_HIST,
    PROPRIO_PER_STEP,
)
from diff_loco.v19_vit import _StateEncV19, _TinyBlock
from diff_loco.v20_vit import _CrossAttnBlock, _PrivEncV20, _ProprioPerStepEnc


HEIGHT_H_V23 = 11
HEIGHT_W_V23 = 21
HEIGHT_DIM_V23 = HEIGHT_H_V23 * HEIGHT_W_V23  # 231

N_HEIGHT_PATCH_H = (HEIGHT_H_V23 - 3) // 2 + 1  # 5
N_HEIGHT_PATCH_W = (HEIGHT_W_V23 - 3) // 3 + 1  # 7
N_HEIGHT_TOKENS = N_HEIGHT_PATCH_H * N_HEIGHT_PATCH_W  # 35
N_LOCO_TOKENS_ACTOR = PROPRIO_HIST + 1
N_LOCO_TOKENS_CRITIC = PROPRIO_HIST + 2

PRIV_DIM = 15

EXPECTED_OBS_DIM_V23 = PROPRIO_FLAT + CMD_DIM + LIN_VEL_DIM + HEIGHT_DIM_V23
EXPECTED_OBS_DIM_V23_CRITIC = EXPECTED_OBS_DIM_V23 + PRIV_DIM
NON_VIS_DIM_V23 = PROPRIO_FLAT + CMD_DIM + LIN_VEL_DIM


class _HeightEncV23(nn.Module):
    """(B, 11, 21) -> (B, 35, d)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.patch = nn.Conv2d(1, d_model, kernel_size=3, stride=(2, 3))
        self.pos_embed = nn.Parameter(torch.randn(1, N_HEIGHT_TOKENS, d_model) * 0.02)

    def forward(self, height: torch.Tensor) -> torch.Tensor:
        x = self.patch(height.unsqueeze(1)).flatten(2).transpose(1, 2)
        return x + self.pos_embed


class _VisionEncoderV23(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, ffn_mult: int = 2):
        super().__init__()
        self.height_enc = _HeightEncV23(d_model)
        self.type_embed = nn.Parameter(torch.randn(1, N_HEIGHT_TOKENS, d_model) * 0.02)
        self.blocks = nn.ModuleList([_TinyBlock(d_model, n_heads, ffn_mult=ffn_mult) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, height: torch.Tensor) -> torch.Tensor:
        x = self.height_enc(height)
        x = x + self.type_embed
        for blk in self.blocks:
            x = blk(x)
        return self.final_norm(x)


class _LocoTransformerV23(nn.Module):
    def __init__(self, d_model: int, n_self_layers: int, n_heads: int, with_priv: bool, ffn_mult: int = 4):
        super().__init__()
        self.with_priv = with_priv
        self.proprio_enc = _ProprioPerStepEnc(d_model)
        self.state_enc = _StateEncV19(d_model)
        self.priv_enc = _PrivEncV20(d_model) if with_priv else None
        n_tokens = N_LOCO_TOKENS_CRITIC if with_priv else N_LOCO_TOKENS_ACTOR
        self.type_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        self.self_blocks = nn.ModuleList(
            [_TinyBlock(d_model, n_heads, ffn_mult=ffn_mult) for _ in range(n_self_layers)]
        )
        self.cross_block = _CrossAttnBlock(d_model, n_heads, ffn_mult=ffn_mult)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, proprio_hist, cmd, lin_vel, visual_mem, priv=None) -> torch.Tensor:
        p = self.proprio_enc(proprio_hist)
        s = self.state_enc(cmd, lin_vel)
        toks = [p, s]
        if self.with_priv:
            toks.append(self.priv_enc(priv))
        x = torch.cat(toks, dim=1)
        x = x + self.type_embed
        for blk in self.self_blocks:
            x = blk(x)
        x = self.cross_block(x, visual_mem)
        x = self.final_norm(x)
        return x.mean(dim=1)


class V23HeightAttnModel(nn.Module):
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
        d_model: int = 96,
        n_heads: int = 4,
        n_vision_layers: int = 2,
        n_loco_layers: int = 2,
        vision_ffn_mult: int = 2,
        loco_ffn_mult: int = 4,
        out_proj_gain: float = 0.5,
    ) -> None:
        super().__init__()
        del activation

        self.obs_groups = list(obs_groups[obs_set])
        if len(self.obs_groups) != 1:
            raise ValueError(f"V23 expects one obs group, got {self.obs_groups}")
        self.obs_group = self.obs_groups[0]

        self.is_actor = distribution_cfg is not None
        expected_dim = EXPECTED_OBS_DIM_V23 if self.is_actor else EXPECTED_OBS_DIM_V23_CRITIC
        flat_dim = int(obs[self.obs_group].shape[-1])
        if flat_dim != expected_dim:
            raise ValueError(
                f"V23 expects obs['{self.obs_group}'] dim {expected_dim} "
                f"(actor={self.is_actor}), got {flat_dim}"
            )

        self.obs_normalization = obs_normalization
        self.non_vis_dim = NON_VIS_DIM_V23
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.non_vis_dim)
            if not self.is_actor:
                self.priv_normalizer = EmpiricalNormalization(PRIV_DIM)
            else:
                self.priv_normalizer = nn.Identity()
        else:
            self.obs_normalizer = nn.Identity()
            self.priv_normalizer = nn.Identity()

        self.vision = _VisionEncoderV23(d_model, n_vision_layers, n_heads, ffn_mult=vision_ffn_mult)
        self.loco = _LocoTransformerV23(d_model, n_loco_layers, n_heads, with_priv=not self.is_actor, ffn_mult=loco_ffn_mult)
        self.output_dim = int(output_dim)

        if self.is_actor:
            self.out_proj = nn.Linear(d_model, output_dim)
            nn.init.orthogonal_(self.out_proj.weight, gain=out_proj_gain)
            nn.init.zeros_(self.out_proj.bias)
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution: Distribution = dist_class(output_dim, **distribution_cfg)
            if self.distribution.input_dim != output_dim:
                raise ValueError("V23 distribution input_dim must equal output_dim.")
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
        height_flat = flat[:, off:off + HEIGHT_DIM_V23]; off += HEIGHT_DIM_V23
        priv = flat[:, off:off + PRIV_DIM] if (not self.is_actor) else None

        po = 0
        ang = proprio_flat[:, po:po + PROPRIO_HIST * ANG_DIM].view(B, PROPRIO_HIST, ANG_DIM); po += PROPRIO_HIST * ANG_DIM
        grv = proprio_flat[:, po:po + PROPRIO_HIST * GRAV_DIM].view(B, PROPRIO_HIST, GRAV_DIM); po += PROPRIO_HIST * GRAV_DIM
        jp = proprio_flat[:, po:po + PROPRIO_HIST * JPOS_DIM].view(B, PROPRIO_HIST, JPOS_DIM); po += PROPRIO_HIST * JPOS_DIM
        jv = proprio_flat[:, po:po + PROPRIO_HIST * JVEL_DIM].view(B, PROPRIO_HIST, JVEL_DIM); po += PROPRIO_HIST * JVEL_DIM
        la = proprio_flat[:, po:po + PROPRIO_HIST * LAST_ACT_DIM].view(B, PROPRIO_HIST, LAST_ACT_DIM)
        proprio_hist = torch.cat([ang, grv, jp, jv, la], dim=-1)

        height = height_flat.view(B, HEIGHT_H_V23, HEIGHT_W_V23)
        return proprio_hist, cmd, lin_vel, height, priv

    def _encode(self, obs: TensorDict) -> torch.Tensor:
        flat = obs[self.obs_group]
        if self.obs_normalization:
            non_vis = flat[:, :self.non_vis_dim]
            vis_end = NON_VIS_DIM_V23 + HEIGHT_DIM_V23
            vis = flat[:, self.non_vis_dim:vis_end]
            tail = flat[:, vis_end:]
            non_vis = self.obs_normalizer(non_vis)
            if (not self.is_actor) and tail.shape[-1] == PRIV_DIM:
                tail = self.priv_normalizer(tail)
            flat = torch.cat([non_vis, vis, tail], dim=-1)
        proprio_hist, cmd, lin_vel, height, priv = self._split_obs(flat)
        visual_mem = self.vision(height)
        return self.loco(proprio_hist, cmd, lin_vel, visual_mem, priv=priv)

    def get_latent(self, obs, masks=None, hidden_state=None):
        return self._encode(obs)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            flat = obs[self.obs_group]
            self.obs_normalizer.update(flat[:, :self.non_vis_dim])  # type: ignore
            if not self.is_actor:
                vis_end = NON_VIS_DIM_V23 + HEIGHT_DIM_V23
                self.priv_normalizer.update(flat[:, vis_end:vis_end + PRIV_DIM])  # type: ignore

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
