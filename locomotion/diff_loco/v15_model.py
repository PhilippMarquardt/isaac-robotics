"""V15 actor/critic: small MLP + cross-attention heightmap encoder.

Architecture (per design spec, 2026-04-20):
  proprio history (5 × 93) --Conv1D(93->64,k=3)-->ELU-->Conv1D(64->64,k=3)-->ELU
                           --flatten-->Linear(320->128)-->LayerNorm
                           --cat[base_lin_vel(3), cmd(3)]-->Linear(134->128)-->ELU  = proprio_embed(128)
  height_scan (17x11=187) --patchify (2x3 patches of 6x6 cells, pad 1 each dim)
                          --Linear(36->64) + learned pos_embed        = 6 tokens (B,6,64)
  Q=proprio_embed->64, K,V=tokens --MultiHeadAttn(4 heads, 1 layer)--> terrain_embed(64)
  cat[proprio_embed(128), terrain_embed(64)] = 192
    --Linear(192->512)ELU-->Linear(512->256)ELU-->Linear(256->128)ELU-->Linear(128->29)
    --tanh * 0.25 (actor only; critic outputs 1 dim, no tanh)

Obs layout (single group 'policy', term order MUST match env cfg):
  [0:15]    base_ang_vel hist5  (5*3)
  [15:30]   projected_gravity hist5
  [30:175]  joint_pos hist5  (5*29)
  [175:320] joint_vel hist5
  [320:465] last_action hist5
  [465:468] velocity_commands (3)
  [468:471] base_lin_vel (3)
  [471:658] height_scan (187)

Same interface as DiffusionModel / MLPModel (rsl-rl 5.x): __init__ takes
(obs, obs_groups, obs_set, output_dim, **cfg) and exposes forward,
get_latent, output_mean/std/entropy, log_prob, KL, normalization, jit/onnx.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


ANG_DIM = 3
GRAV_DIM = 3
# G1 parkour env has 37 actuated DOF (full body incl. fingers).
JPOS_DIM = 37
JVEL_DIM = 37
ACT_DIM = 37
PROPRIO_PER_STEP = ANG_DIM + GRAV_DIM + JPOS_DIM + JVEL_DIM + ACT_DIM  # 117
PROPRIO_HIST = 5
PROPRIO_FLAT = PROPRIO_PER_STEP * PROPRIO_HIST  # 585

CMD_DIM = 3
LIN_VEL_DIM = 3
HEIGHT_DIM = 187  # 17 cols x 11 rows
HEIGHT_H, HEIGHT_W = 11, 17

EXPECTED_OBS_DIM = PROPRIO_FLAT + CMD_DIM + LIN_VEL_DIM + HEIGHT_DIM  # 778


def _elu_mlp(dims: list[int], last_act: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class _ProprioEncoder(nn.Module):
    """Temporal conv over 5-frame proprio history + fuse with base_lin_vel + cmd."""

    def __init__(self, proprio_per_step: int = PROPRIO_PER_STEP, hist: int = PROPRIO_HIST,
                 embed_dim: int = 128, aux_dim: int = CMD_DIM + LIN_VEL_DIM):
        super().__init__()
        self.hist = hist
        self.proprio_per_step = proprio_per_step
        self.conv = nn.Sequential(
            nn.Conv1d(proprio_per_step, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ELU(),
        )
        self.temporal_proj = nn.Sequential(
            nn.Linear(64 * hist, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim + aux_dim, embed_dim),
            nn.ELU(),
        )

    def forward(self, proprio_seq: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        # proprio_seq: (B, hist, per_step) -> conv wants (B, per_step, hist)
        x = proprio_seq.transpose(1, 2)  # (B, 93, 5)
        x = self.conv(x)                 # (B, 64, 5)
        x = x.flatten(1)                 # (B, 320)
        x = self.temporal_proj(x)        # (B, 128)
        x = torch.cat([x, aux], dim=-1)  # (B, 128 + aux)
        return self.fuse(x)              # (B, 128)


class _HeightCrossAttn(nn.Module):
    """Patchify 11x17 grid into 2x3 = 6 patches of 6x6 cells (pad 1 to 12x18),
    linear-project each patch to 64-dim token, add learned pos embed, then
    cross-attend with proprio_embed as query."""

    CELL_H = 6
    CELL_W = 6
    PATCH_H = 2
    PATCH_W = 3
    N_PATCHES = PATCH_H * PATCH_W  # 6
    PATCH_FLAT = CELL_H * CELL_W   # 36

    def __init__(self, proprio_dim: int = 128, embed_dim: int = 64, n_heads: int = 4,
                 grid_hw: tuple[int, int] = (HEIGHT_H, HEIGHT_W)):
        super().__init__()
        self.grid_h, self.grid_w = grid_hw
        self.pad_h = self.PATCH_H * self.CELL_H - self.grid_h  # 12 - 11 = 1
        self.pad_w = self.PATCH_W * self.CELL_W - self.grid_w  # 18 - 17 = 1
        self.patch_proj = nn.Linear(self.PATCH_FLAT, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.N_PATCHES, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.q_proj = nn.Linear(proprio_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, height_flat: torch.Tensor, proprio_embed: torch.Tensor) -> torch.Tensor:
        B = height_flat.shape[0]
        h = height_flat.view(B, self.grid_h, self.grid_w)
        if self.pad_h or self.pad_w:
            h = torch.nn.functional.pad(h, (0, self.pad_w, 0, self.pad_h))
        # (B, PH*CH, PW*CW) -> (B, PH, CH, PW, CW) -> (B, PH, PW, CH, CW)
        h = h.view(B, self.PATCH_H, self.CELL_H, self.PATCH_W, self.CELL_W)
        h = h.permute(0, 1, 3, 2, 4).contiguous()
        h = h.view(B, self.N_PATCHES, self.PATCH_FLAT)  # (B, 6, 36)
        tokens = self.patch_proj(h) + self.pos_embed     # (B, 6, 64)
        q = self.q_proj(proprio_embed).unsqueeze(1)      # (B, 1, 64)
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        return out.squeeze(1)  # (B, 64)


class V15Model(nn.Module):
    """V15 actor or critic. Same signature as rsl-rl MLPModel so
    construct_algorithm can instantiate via class_name.

    For actor: distribution_cfg provided, output_dim=29 (joint targets).
    For critic: distribution_cfg=None, output_dim=1.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",  # fixed ELU per spec, accepted for MLPModel parity
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        proprio_embed_dim: int = 128,
        height_embed_dim: int = 64,
        n_heads: int = 4,
        action_scale: float = 0.25,
    ) -> None:
        super().__init__()
        del activation  # spec: always ELU

        self.obs_groups = list(obs_groups[obs_set])
        if len(self.obs_groups) != 1:
            raise ValueError(
                f"V15Model expects exactly one obs group for '{obs_set}', got {self.obs_groups}. "
                "Config the runner obs_groups = {'actor': ['policy'], 'critic': ['critic']}."
            )
        self.obs_group = self.obs_groups[0]

        flat_dim = int(obs[self.obs_group].shape[-1])
        if flat_dim != EXPECTED_OBS_DIM:
            raise ValueError(
                f"V15Model expects obs['{self.obs_group}'] of dim {EXPECTED_OBS_DIM} "
                f"(proprio_hist({PROPRIO_FLAT}) + cmd({CMD_DIM}) + base_lin_vel({LIN_VEL_DIM}) "
                f"+ height({HEIGHT_DIM})), got {flat_dim}. "
                "Check env obs term order in tasks.py matches v15_model.py layout."
            )

        # Normalize proprio + cmd + base_lin_vel (but NOT heightmap — already clipped).
        self.obs_normalization = obs_normalization
        self.non_height_dim = PROPRIO_FLAT + CMD_DIM + LIN_VEL_DIM  # 471
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.non_height_dim)
        else:
            self.obs_normalizer = nn.Identity()

        self.proprio_enc = _ProprioEncoder(embed_dim=proprio_embed_dim)
        self.height_enc = _HeightCrossAttn(
            proprio_dim=proprio_embed_dim, embed_dim=height_embed_dim, n_heads=n_heads,
        )

        head_in = proprio_embed_dim + height_embed_dim  # 192
        head_dims = [head_in] + list(hidden_dims) + [output_dim]
        self.head = _elu_mlp(head_dims, last_act=False)

        self.action_scale = float(action_scale)
        self.output_dim = int(output_dim)
        self.is_actor = distribution_cfg is not None

        # --- distribution (actor only) -----------------------------------
        if self.is_actor:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution = dist_class(output_dim, **distribution_cfg)
            if self.distribution.input_dim != output_dim:
                raise ValueError("V15Model distribution input_dim must equal output_dim.")
            # rsl-rl's Distribution.init_mlp_weights expects a .mlp attribute.
            self.mlp = self.head
            self.distribution.init_mlp_weights(self.mlp)
        else:
            self.distribution = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # obs splitting
    # ------------------------------------------------------------------
    def _split_obs(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = flat.shape[0]
        proprio_flat = flat[:, :PROPRIO_FLAT]           # (B, 465)
        rest = flat[:, PROPRIO_FLAT:]
        cmd = rest[:, :CMD_DIM]                          # (B, 3)
        lin_vel = rest[:, CMD_DIM:CMD_DIM + LIN_VEL_DIM] # (B, 3)
        height = rest[:, CMD_DIM + LIN_VEL_DIM:]         # (B, 187)
        # Per-term history layout: [ang(5*3), grav(5*3), jpos(5*37), jvel(5*37), act(5*37)]
        off = 0
        ang = proprio_flat[:, off:off + PROPRIO_HIST * ANG_DIM].view(B, PROPRIO_HIST, ANG_DIM)
        off += PROPRIO_HIST * ANG_DIM
        grav = proprio_flat[:, off:off + PROPRIO_HIST * GRAV_DIM].view(B, PROPRIO_HIST, GRAV_DIM)
        off += PROPRIO_HIST * GRAV_DIM
        jpos = proprio_flat[:, off:off + PROPRIO_HIST * JPOS_DIM].view(B, PROPRIO_HIST, JPOS_DIM)
        off += PROPRIO_HIST * JPOS_DIM
        jvel = proprio_flat[:, off:off + PROPRIO_HIST * JVEL_DIM].view(B, PROPRIO_HIST, JVEL_DIM)
        off += PROPRIO_HIST * JVEL_DIM
        act = proprio_flat[:, off:off + PROPRIO_HIST * ACT_DIM].view(B, PROPRIO_HIST, ACT_DIM)
        proprio_seq = torch.cat([ang, grav, jpos, jvel, act], dim=-1)  # (B, 5, 117)
        return proprio_seq, cmd, lin_vel, height

    def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None,
                   hidden_state: HiddenState = None) -> torch.Tensor:
        flat = obs[self.obs_group]
        if self.obs_normalization:
            non_height = flat[:, :self.non_height_dim]
            height = flat[:, self.non_height_dim:]
            non_height = self.obs_normalizer(non_height)
            flat = torch.cat([non_height, height], dim=-1)
        proprio_seq, cmd, lin_vel, height = self._split_obs(flat)
        aux = torch.cat([lin_vel, cmd], dim=-1)  # (B, 6)
        proprio_embed = self.proprio_enc(proprio_seq, aux)        # (B, 128)
        terrain_embed = self.height_enc(height, proprio_embed)    # (B, 64)
        return torch.cat([proprio_embed, terrain_embed], dim=-1)  # (B, 192)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            flat = obs[self.obs_group]
            non_height = flat[:, :self.non_height_dim]
            self.obs_normalizer.update(non_height)  # type: ignore

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        raw = self.head(latent)
        if self.is_actor:
            mean = torch.tanh(raw) * self.action_scale
            mean = torch.nan_to_num(mean, nan=0.0, posinf=self.action_scale, neginf=-self.action_scale)
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
        return raw  # critic: raw scalar

    # ------------------------------------------------------------------
    # rsl-rl hooks
    # ------------------------------------------------------------------
    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

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
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params if self.is_actor else ()

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(self, old_params, new_params) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("V15Model JIT export not implemented (attention + conv).")

    def as_onnx(self, verbose: bool) -> nn.Module:
        raise NotImplementedError("V15Model ONNX export not implemented.")
