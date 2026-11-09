"""V52 — V51 history obs + transformer policy with ViT heightmap (machine E).

Single change vs V51: replace stock G1 MLP actor [512,256,128] with a small
transformer that:
  * tokenizes 5 past proprio steps (joint_pos+vel+action) as 5 tokens
  * tokenizes current state (lin_vel+ang_vel+grav+cmd) as 1 token
  * tokenizes V50/V51 stock heightmap (res=0.1 size=[1.6,1.0] -> 17x11 = 187
    cells) with a tiny ViT, kernel (2,4) stride (2,4) -> 5*4 = 20 patches
  * main transformer 2 layers / 2 heads, each block does self-attn over
    the 6 state tokens then cross-attends to the 20 ViT tokens

Heightmap stays at stock V50 resolution. The earlier attempt to bump to
res=0.05 size=[2.0,1.55] yielded 1521 cells at runtime instead of the
1312 the cfg suggested (something inside IsaacLab's grid_pattern produced
39x39 — unresolved). Sticking with stock keeps obs dim deterministic
across V50/V51/V52.

V51 obs layout (single 'policy' group), G1_29DOF has 37 actuated joints
(IsaacLab g1.usd full body):
  lin_vel(3) + ang_vel(3) + grav(3) + cmd(3)        =  12
  + jpos x5 (185) + jvel x5 (185) + act x5 (185)    = 555
  + height_scan (187)                               = 187
  TOTAL = 754 dim/step

ObservationTermCfg(history_length=5, flatten_history_dim=True) flattens
[B, 5, 37] -> [B, 185]. JOINT_DIM is derived from runtime obs dim because
G1 USD variants differ (g1_minimal=23, g1_29dof=37, +inspire=more).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


# ============================================================================
# Obs layout constants. State + heightmap are fixed; joint_dim is derived at
# runtime from total obs dim because the actuated-joint count of G1 depends
# on the chosen USD (g1_minimal vs g1_29dof).
# ============================================================================
LIN_VEL_DIM = 3
ANG_VEL_DIM = 3
GRAV_DIM = 3
# IsaacLab generated_commands for G1 base_velocity returns (lin_x, lin_y, ang_z) = 3,
# heading-following turns yaw cmd into ang_z internally.
CMD_DIM = 3
STATE_DIM = LIN_VEL_DIM + ANG_VEL_DIM + GRAV_DIM + CMD_DIM  # 12

N_HIST = 5

# Stock V50 heightmap: GridPatternCfg(res=0.1, size=[1.6, 1.0]). With
# torch.meshgrid(x, y, indexing="xy") and ordering="xy", the flattened layout
# is row-major over (Hy, Hx) = (11, 17).
HEIGHT_X = 17
HEIGHT_Y = 11
HEIGHT_DIM = HEIGHT_X * HEIGHT_Y  # 187
# Asymmetric patch (2,4): kernel/stride along (Hy, Hx). Patches: 11//2 = 5,
# 17//4 = 4 (last column of 1 dropped). 5 * 4 = 20 ViT tokens.
PATCH_Y = 2
PATCH_X = 4
N_PATCH_Y = (HEIGHT_Y - PATCH_Y) // PATCH_Y + 1  # 5
N_PATCH_X = (HEIGHT_X - PATCH_X) // PATCH_X + 1  # 4
N_PATCHES = N_PATCH_Y * N_PATCH_X                # 20

N_QUERY_TOKENS = 1 + N_HIST  # state + per-step proprio = 6


# ============================================================================
# Sinusoidal positional encoding (1D, fixed)
# ============================================================================
def _sinusoidal_pe(num_tokens: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(num_tokens, d_model)
    pos = torch.arange(0, num_tokens, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, num_tokens, d_model)


# ============================================================================
# Building blocks
# ============================================================================
class _SelfAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class _CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 2):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        a, _ = self.cross(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)
        q = q + a
        q = q + self.ffn(self.norm_ffn(q))
        return q


class _HeightViT(nn.Module):
    """Conv2d patch embed + 1 self-attn block. Output (B, 20, d_model)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # Conv2d kernel=(kY, kX) stride=(kY, kX). Input shape (B, 1, 11, 17).
        # Output: (B, d, 5, 4) -> 20 patches after flatten.
        self.patch = nn.Conv2d(
            1, d_model, kernel_size=(PATCH_Y, PATCH_X), stride=(PATCH_Y, PATCH_X)
        )
        self.register_buffer("pos_embed", _sinusoidal_pe(N_PATCHES, d_model), persistent=False)
        self.block = _SelfAttnBlock(d_model, n_heads, ffn_mult=2)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, height: torch.Tensor) -> torch.Tensor:
        # height: (B, HEIGHT_Y, HEIGHT_X). Add channel dim.
        x = self.patch(height.unsqueeze(1))                  # (B, d, P_y, P_x)
        x = x.flatten(2).transpose(1, 2).contiguous()        # (B, P, d)
        x = x + self.pos_embed
        x = self.block(x)
        return self.norm(x)


# ============================================================================
# Main model
# ============================================================================
class V52TransformerModel(nn.Module):
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
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 2,
        ffn_mult: int = 2,
        out_proj_gain: float = 0.01,
    ) -> None:
        super().__init__()
        del activation

        self.obs_groups = list(obs_groups[obs_set])
        if len(self.obs_groups) != 1:
            raise ValueError(f"V52 expects one obs group, got {self.obs_groups}")
        self.obs_group = self.obs_groups[0]

        self.is_actor = distribution_cfg is not None
        flat_dim = int(obs[self.obs_group].shape[-1])
        # Derive JOINT_DIM from runtime if EXPECTED_OBS_DIM doesn't match.
        # Stock V51 heightmap is HEIGHT_DIM = 187, but G1 actuated-joint count
        # depends on the USD (g1_minimal vs g1_29dof vs g1+inspire). Compute
        # JOINT_DIM = (flat - state - heightmap) / (3*N_HIST) and bail if
        # non-integer or if any component is wrong.
        history_dim = flat_dim - STATE_DIM - HEIGHT_DIM
        if history_dim <= 0 or history_dim % (3 * N_HIST) != 0:
            raise ValueError(
                f"V52 obs[{self.obs_group}] flat dim {flat_dim} doesn't factor "
                f"as STATE({STATE_DIM}) + 3*N_HIST*JOINT + HEIGHT({HEIGHT_DIM}). "
                f"history_dim = {history_dim}, not divisible by {3 * N_HIST}."
            )
        self.joint_dim = history_dim // (3 * N_HIST)
        self.flat_dim = flat_dim

        self.obs_normalization = obs_normalization
        self.obs_normalizer = (
            EmpiricalNormalization(flat_dim) if obs_normalization else nn.Identity()
        )

        # Token embedders. Proprio per-step = 3*joint_dim (jpos+jvel+act).
        self.proprio_per_step = 3 * self.joint_dim
        self.state_proj = nn.Linear(STATE_DIM, d_model)
        self.proprio_proj = nn.Linear(self.proprio_per_step, d_model)
        self.register_buffer("query_type_embed", _sinusoidal_pe(N_QUERY_TOKENS, d_model), persistent=False)
        self.vit = _HeightViT(d_model, n_heads)

        # Main transformer: 2 layers of (self-attn on queries + cross-attn to ViT)
        self.self_blocks = nn.ModuleList(
            [_SelfAttnBlock(d_model, n_heads, ffn_mult=ffn_mult) for _ in range(n_layers)]
        )
        self.cross_blocks = nn.ModuleList(
            [_CrossAttnBlock(d_model, n_heads, ffn_mult=ffn_mult) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self.output_dim = int(output_dim)
        if self.is_actor:
            self.out_proj = nn.Linear(d_model, output_dim)
            nn.init.orthogonal_(self.out_proj.weight, gain=out_proj_gain)
            nn.init.zeros_(self.out_proj.bias)
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution: Distribution = dist_class(output_dim, **distribution_cfg)
            if self.distribution.input_dim != output_dim:
                raise ValueError("V52 distribution input_dim must equal output_dim.")
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

    # ---- obs splitting ----
    def _split_obs(self, flat: torch.Tensor):
        B = flat.shape[0]
        J = self.joint_dim
        off = 0
        lin_vel = flat[:, off:off + LIN_VEL_DIM]; off += LIN_VEL_DIM
        ang_vel = flat[:, off:off + ANG_VEL_DIM]; off += ANG_VEL_DIM
        grav = flat[:, off:off + GRAV_DIM]; off += GRAV_DIM
        cmd = flat[:, off:off + CMD_DIM]; off += CMD_DIM

        jpos_flat = flat[:, off:off + N_HIST * J]; off += N_HIST * J
        jvel_flat = flat[:, off:off + N_HIST * J]; off += N_HIST * J
        act_flat = flat[:, off:off + N_HIST * J]; off += N_HIST * J

        height_flat = flat[:, off:off + HEIGHT_DIM]; off += HEIGHT_DIM

        state = torch.cat([lin_vel, ang_vel, grav, cmd], dim=-1)         # (B, 13)
        jpos = jpos_flat.view(B, N_HIST, J)
        jvel = jvel_flat.view(B, N_HIST, J)
        act = act_flat.view(B, N_HIST, J)
        proprio_steps = torch.cat([jpos, jvel, act], dim=-1)             # (B, 5, 3J)
        height = height_flat.view(B, HEIGHT_Y, HEIGHT_X)                 # (B, 11, 17)
        return state, proprio_steps, height

    def _encode(self, obs: TensorDict) -> torch.Tensor:
        flat = obs[self.obs_group]
        if self.obs_normalization:
            flat = self.obs_normalizer(flat)
        state, proprio_steps, height = self._split_obs(flat)

        s_tok = self.state_proj(state).unsqueeze(1)                      # (B, 1, d)
        p_tok = self.proprio_proj(proprio_steps)                         # (B, 5, d)
        q = torch.cat([s_tok, p_tok], dim=1) + self.query_type_embed     # (B, 6, d)

        kv = self.vit(height)                                            # (B, 20, d)

        for s_blk, c_blk in zip(self.self_blocks, self.cross_blocks):
            q = s_blk(q)
            q = c_blk(q, kv)
        q = self.final_norm(q)
        return q.mean(dim=1)                                             # (B, d)

    # ---- rsl_rl interface ----
    def get_latent(self, obs, masks=None, hidden_state=None):
        return self._encode(obs)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.obs_normalizer.update(obs[self.obs_group])  # type: ignore

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self._encode(obs)
        latent = torch.nan_to_num(latent, nan=0.0, posinf=5.0, neginf=-5.0)
        if self.is_actor:
            mean = self.out_proj(latent)
            mean = torch.nan_to_num(mean, nan=0.0, posinf=3.0, neginf=-3.0)
            if hasattr(self.distribution, "log_std_param"):
                with torch.no_grad():
                    p = self.distribution.log_std_param
                    bad = torch.isnan(p.data) | torch.isinf(p.data)
                    if bad.any():
                        p.data[bad] = -1.0
                    p.data.clamp_(min=-5.0, max=1.0)
            if stochastic_output:
                self.distribution.update(mean)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mean)
        val = self.critic_head(latent)
        return torch.nan_to_num(val, nan=0.0, posinf=50.0, neginf=-50.0)

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


# ============================================================================
# Env cfg — V51 obs verbatim. No heightmap override (stock 11x17 = 187).
# ============================================================================
from isaaclab.utils import configclass

from diff_loco.v51_parkour import (
    G1V51ParkourEnvCfg,
    G1V51ParkourEnvCfg_PLAY,
)


@configclass
class G1V52ParkourEnvCfg(G1V51ParkourEnvCfg):
    pass


@configclass
class G1V52ParkourEnvCfg_PLAY(G1V51ParkourEnvCfg_PLAY):
    pass


# ============================================================================
# PPO runner — wires the transformer model
# ============================================================================
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def _make_gaussian_cfg() -> "RslRlMLPModelCfg.GaussianDistributionCfg":
    return RslRlMLPModelCfg.GaussianDistributionCfg(
        class_name="GaussianDistribution",
        init_std=0.5,
        std_type="log",
    )


@configclass
class G1V52ParkourPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_v52_parkour"
    empirical_normalization = False

    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        class_name="diff_loco.v52_transformer:V52TransformerModel",
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=0.5,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = RslRlMLPModelCfg(
        class_name="diff_loco.v52_transformer:V52TransformerModel",
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ============================================================================
# Gym registration
# ============================================================================
def register_v52_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V52-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V52-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v52_transformer:G1V52ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v52_transformer:G1V52ParkourPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V52-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v52_transformer:G1V52ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v52_transformer:G1V52ParkourPPORunnerCfg",
        },
    )
