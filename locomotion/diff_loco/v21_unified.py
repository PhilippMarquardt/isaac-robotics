"""V21 unified transformer: heightmap + depth in a single self-attn stack.

Diagnosis from V20 failure: separate vision encoder + cross-attn decoder
creates a chicken-and-egg loop. Depth encoder outputs noise early -> cross-attn
pollutes loco queries -> agent blind -> curriculum stuck at 0 -> no diverse
data for depth encoder to learn from.

V21 fix: all modalities are peer tokens in one transformer. Gradients flow
directly from action loss to every input token. No modality gets gated behind
a half-trained intermediate.

Tokens (44 total): CLS + height(25) + depth(16) + proprio(1) + state(1).
Same _TinyBlock, same d_model=64, 3 layers (vs V19's 2 to handle more tokens).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories
from tensordict import TensorDict

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
from diff_loco.v19_vit import _HeightEncV19, _ProprioEncV19, _StateEncV19, _TinyBlock


HEIGHT_H_V21 = 11
HEIGHT_W_V21 = 17
HEIGHT_DIM_V21 = HEIGHT_H_V21 * HEIGHT_W_V21  # 187

DEPTH_H = 48
DEPTH_W = 48
DEPTH_DIM = DEPTH_H * DEPTH_W  # 2304
DEPTH_PATCH = 12

N_HEIGHT_TOKENS = 25        # from V19 patchify ((11-3)/2+1)*((17-3)/3+1)
N_DEPTH_TOKENS = (DEPTH_H // DEPTH_PATCH) * (DEPTH_W // DEPTH_PATCH)  # 16
N_TOKENS_ACTOR = 1 + N_HEIGHT_TOKENS + N_DEPTH_TOKENS + 1 + 1  # CLS + h + d + proprio + state = 44
N_TOKENS_CRITIC = N_TOKENS_ACTOR + 1                           # + priv token = 45

PRIV_DIM = 15  # critic group has extra priv obs appended

EXPECTED_OBS_DIM_V21 = PROPRIO_FLAT + CMD_DIM + LIN_VEL_DIM + HEIGHT_DIM_V21 + DEPTH_DIM
EXPECTED_OBS_DIM_V21_CRITIC = EXPECTED_OBS_DIM_V21 + PRIV_DIM
NON_VIS_DIM_V21 = PROPRIO_FLAT + CMD_DIM + LIN_VEL_DIM


class _DepthEncV21(nn.Module):
    """(B, 48, 48) -> (B, 16, d). 12x12 conv patchify."""
    def __init__(self, d_model: int):
        super().__init__()
        self.patch = nn.Conv2d(1, d_model, kernel_size=DEPTH_PATCH, stride=DEPTH_PATCH)
        self.pos_embed = nn.Parameter(torch.randn(1, N_DEPTH_TOKENS, d_model) * 0.02)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        x = self.patch(depth.unsqueeze(1)).flatten(2).transpose(1, 2)
        return x + self.pos_embed


class _PrivEncV21(nn.Module):
    """(B, priv_dim) -> (B, 1, d). Critic-only privileged features token."""
    def __init__(self, d_model: int, priv_dim: int = PRIV_DIM):
        super().__init__()
        self.proj = nn.Linear(priv_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, priv: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(priv)).unsqueeze(1)


class _UnifiedViT(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, with_priv: bool):
        super().__init__()
        self.with_priv = with_priv
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        n_tok = N_TOKENS_CRITIC if with_priv else N_TOKENS_ACTOR
        self.token_type_embed = nn.Parameter(torch.randn(1, n_tok, d_model) * 0.02)
        self.blocks = nn.ModuleList([_TinyBlock(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, h, d, p, s, priv_tok=None) -> torch.Tensor:
        B = h.shape[0]
        toks = [self.cls.expand(B, -1, -1), h, d, p, s]
        if self.with_priv:
            assert priv_tok is not None
            toks.append(priv_tok)
        x = torch.cat(toks, dim=1)
        x = x + self.token_type_embed
        for blk in self.blocks:
            x = blk(x)
        return self.final_norm(x)[:, 0]  # CLS


class V21UnifiedModel(nn.Module):
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
        n_layers: int = 3,
        out_proj_gain: float = 0.5,
    ) -> None:
        super().__init__()
        del activation

        self.obs_groups = list(obs_groups[obs_set])
        if len(self.obs_groups) != 1:
            raise ValueError(f"V21 expects one obs group, got {self.obs_groups}")
        self.obs_group = self.obs_groups[0]

        flat_dim = int(obs[self.obs_group].shape[-1])
        if flat_dim not in (EXPECTED_OBS_DIM_V21, EXPECTED_OBS_DIM_V21_CRITIC):
            raise ValueError(
                f"V21 expects obs['{self.obs_group}'] dim {EXPECTED_OBS_DIM_V21} or {EXPECTED_OBS_DIM_V21_CRITIC}, got {flat_dim}"
            )

        self.obs_normalization = obs_normalization
        self.non_vis_dim = NON_VIS_DIM_V21
        self.is_critic_obs = flat_dim == EXPECTED_OBS_DIM_V21_CRITIC
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.non_vis_dim)
            if self.is_critic_obs:
                self.priv_normalizer = EmpiricalNormalization(PRIV_DIM)
            else:
                self.priv_normalizer = nn.Identity()
        else:
            self.obs_normalizer = nn.Identity()
            self.priv_normalizer = nn.Identity()

        self.height_enc = _HeightEncV19(d_model)
        self.depth_enc = _DepthEncV21(d_model)
        self.proprio_enc = _ProprioEncV19(d_model)
        self.state_enc = _StateEncV19(d_model)
        self.priv_enc = _PrivEncV21(d_model) if self.is_critic_obs else None
        self.vit = _UnifiedViT(d_model, n_layers, n_heads, with_priv=self.is_critic_obs)

        self.is_actor = distribution_cfg is not None
        self.output_dim = int(output_dim)

        if self.is_actor:
            self.out_proj = nn.Linear(d_model, output_dim)
            nn.init.orthogonal_(self.out_proj.weight, gain=out_proj_gain)
            nn.init.zeros_(self.out_proj.bias)
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution: Distribution = dist_class(output_dim, **distribution_cfg)
            if self.distribution.input_dim != output_dim:
                raise ValueError("V21 distribution input_dim must equal output_dim.")
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
        height_flat = flat[:, off:off + HEIGHT_DIM_V21]; off += HEIGHT_DIM_V21
        depth_flat = flat[:, off:off + DEPTH_DIM]; off += DEPTH_DIM
        priv = flat[:, off:off + PRIV_DIM] if self.is_critic_obs else None

        po = 0
        ang = proprio_flat[:, po:po + PROPRIO_HIST * ANG_DIM].view(B, PROPRIO_HIST, ANG_DIM); po += PROPRIO_HIST * ANG_DIM
        grv = proprio_flat[:, po:po + PROPRIO_HIST * GRAV_DIM].view(B, PROPRIO_HIST, GRAV_DIM); po += PROPRIO_HIST * GRAV_DIM
        jp = proprio_flat[:, po:po + PROPRIO_HIST * JPOS_DIM].view(B, PROPRIO_HIST, JPOS_DIM); po += PROPRIO_HIST * JPOS_DIM
        jv = proprio_flat[:, po:po + PROPRIO_HIST * JVEL_DIM].view(B, PROPRIO_HIST, JVEL_DIM); po += PROPRIO_HIST * JVEL_DIM
        la = proprio_flat[:, po:po + PROPRIO_HIST * LAST_ACT_DIM].view(B, PROPRIO_HIST, LAST_ACT_DIM)
        proprio_hist = torch.cat([ang, grv, jp, jv, la], dim=-1)
        height = height_flat.view(B, HEIGHT_H_V21, HEIGHT_W_V21)
        depth = depth_flat.view(B, DEPTH_H, DEPTH_W)
        return proprio_hist, cmd, lin_vel, height, depth, priv

    def _encode(self, obs: TensorDict) -> torch.Tensor:
        flat = obs[self.obs_group]
        if self.obs_normalization:
            non_vis = flat[:, :self.non_vis_dim]
            rest = flat[:, self.non_vis_dim:]
            non_vis = self.obs_normalizer(non_vis)
            if self.is_critic_obs:
                vis = rest[:, :HEIGHT_DIM_V21 + DEPTH_DIM]
                priv = rest[:, HEIGHT_DIM_V21 + DEPTH_DIM:]
                priv = self.priv_normalizer(priv)
                flat = torch.cat([non_vis, vis, priv], dim=-1)
            else:
                flat = torch.cat([non_vis, rest], dim=-1)
        proprio_hist, cmd, lin_vel, height, depth, priv = self._split_obs(flat)
        h = self.height_enc(height)
        d = self.depth_enc(depth)
        p = self.proprio_enc(proprio_hist)
        s = self.state_enc(cmd, lin_vel)
        priv_tok = self.priv_enc(priv) if self.is_critic_obs else None
        return self.vit(h, d, p, s, priv_tok=priv_tok)

    def get_latent(self, obs, masks=None, hidden_state=None):
        return self._encode(obs)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            flat = obs[self.obs_group]
            self.obs_normalizer.update(flat[:, :self.non_vis_dim])  # type: ignore
            if self.is_critic_obs:
                priv = flat[:, self.non_vis_dim + HEIGHT_DIM_V21 + DEPTH_DIM:]
                self.priv_normalizer.update(priv)  # type: ignore

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
