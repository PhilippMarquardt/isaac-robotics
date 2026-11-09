"""V16 Flow-Matching DiT locomotion policy for G1 parkour.

Architecture:
  HeightMapEncoder: (B,17,11) -> 16 tokens x d=128 (Conv2d stride 3 patchify + learned pos)
  DepthEncoder: (B,48,48) -> 36 tokens x d=128 (Conv2d 8x8 patch + 2 transformer blocks)
  ProprioEncoder: (B,5,proprio_step) -> 1 token  (temporal Conv1d x2 -> proj)
  StateTokenEncoder: cat[command(3), base_lin_vel(3)] -> 1 token
  Context: concat -> (B, 54, 128)

  ActionDiT: 4 layers, d=128, 4 heads. Each block = self-attn over 8 action tokens
             + cross-attn to context + AdaLN(time_embed) + FFN.
             Input: noisy chunk (B,8,ACT_DIM), tau (B,), context (B,54,128).
             Output: velocity (B,8,ACT_DIM).

Rollout: eps ~ N(0,I) shape (B,8,ACT_DIM); 4 Euler ODE steps -> action_chunk.
         Execute chunk[:,0] (replan-every-step). Chunk[:,1:] discarded.

PPO integration (rsl-rl MLPModel interface):
  - Actor: deterministic flow-decode -> Gaussian mean (B, ACT_DIM) with learnable log_std
  - Critic: same encoders + MLP over pooled tokens -> scalar
  - Standard PPO surrogate on Gaussian over chunk[:,0]. No flow-matching aux loss
    (would need rsl-rl extension); DiT learns via surrogate gradient only.
  - Actions scaled tanh * 0.25 before distribution mean (matches V15 gait bounds).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


# --- env obs layout (matches tasks.py V16 cfg; single 'policy' group) -------
ANG_DIM = 3
GRAV_DIM = 3
JPOS_DIM = 37
JVEL_DIM = 37
LAST_ACT_DIM = 37
PROPRIO_PER_STEP = ANG_DIM + GRAV_DIM + JPOS_DIM + JVEL_DIM + LAST_ACT_DIM  # 117
PROPRIO_HIST = 5
PROPRIO_FLAT = PROPRIO_PER_STEP * PROPRIO_HIST  # 585

CMD_DIM = 3
LIN_VEL_DIM = 3
HEIGHT_H, HEIGHT_W = 11, 17
HEIGHT_DIM = HEIGHT_H * HEIGHT_W  # 187
DEPTH_H, DEPTH_W = 32, 32
DEPTH_DIM = DEPTH_H * DEPTH_W  # 1024

EXPECTED_OBS_DIM = PROPRIO_FLAT + CMD_DIM + LIN_VEL_DIM + HEIGHT_DIM + DEPTH_DIM  # 1802 (was 3082)
NON_VIS_DIM = PROPRIO_FLAT + CMD_DIM + LIN_VEL_DIM  # normalize only proprio+cmd+lin_vel

# --- action chunk -----------------------------------------------------------
ACT_DIM = 37
CHUNK_SIZE = 8
N_FLOW_STEPS = 4


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
class _HeightMapEncoder(nn.Module):
    """(B, 11, 17) -> (B, 16, d)."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        # Conv stride (2,3): ceil((11-3+1)/2)=5, ceil((17-3+1)/3)=5 -> 5*5=25 tokens
        self.patch = nn.Conv2d(1, d_model, kernel_size=3, stride=(2, 3))
        # We'll keep the first 16 tokens (top-left scan) for fixed 16 output.
        self.n_tokens = 16
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.02)

    def forward(self, height: torch.Tensor) -> torch.Tensor:
        x = height.unsqueeze(1)  # (B, 1, 11, 17)
        x = self.patch(x)        # (B, d, ~5, ~5)
        x = x.flatten(2).transpose(1, 2)  # (B, N, d)
        if x.shape[1] >= self.n_tokens:
            x = x[:, : self.n_tokens]
        else:
            pad = self.n_tokens - x.shape[1]
            x = F.pad(x, (0, 0, 0, pad))
        return x + self.pos_embed


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        a, _ = self.attn(y, y, y, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class _DepthEncoder(nn.Module):
    """(B, 48, 48) -> (B, 36, d)."""

    def __init__(self, d_model: int = 128, patch_size: int = 8, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.patch = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
        n_tokens = (DEPTH_H // patch_size) * (DEPTH_W // patch_size)  # 36
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        self.blocks = nn.Sequential(*[_TransformerBlock(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        x = depth.unsqueeze(1)   # (B, 1, 48, 48)
        x = self.patch(x)        # (B, d, 6, 6)
        x = x.flatten(2).transpose(1, 2)  # (B, 36, d)
        x = x + self.pos_embed
        return self.blocks(x)


class _ProprioEncoder(nn.Module):
    """(B, 5, 117) -> (B, 1, d). Temporal Conv1d x2 + proj + LN."""

    def __init__(self, d_model: int = 128, hist: int = PROPRIO_HIST, per_step: int = PROPRIO_PER_STEP):
        super().__init__()
        self.hist = hist
        self.per_step = per_step
        self.conv = nn.Sequential(
            nn.Conv1d(per_step, 128, kernel_size=3, padding=1), nn.ELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.ELU(),
        )
        self.proj = nn.Linear(128 * hist, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, proprio_hist: torch.Tensor) -> torch.Tensor:
        x = proprio_hist.transpose(1, 2)  # (B, 117, 5)
        x = self.conv(x)                  # (B, 128, 5)
        x = x.flatten(1)                  # (B, 640)
        x = self.proj(x)                  # (B, d)
        return self.norm(x).unsqueeze(1)  # (B, 1, d)


class _StateTokenEncoder(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.proj = nn.Linear(CMD_DIM + LIN_VEL_DIM, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cmd: torch.Tensor, lin_vel: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cmd, lin_vel], dim=-1)
        return self.norm(self.proj(x)).unsqueeze(1)  # (B, 1, d)


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (for flow matching tau in [0,1])
# ---------------------------------------------------------------------------
class _SinusoidalTimeEmbed(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        half = d_model // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1))
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        # tau: (B,)
        args = tau.unsqueeze(-1) * self.freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, d)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# DiT block with AdaLN modulation
# ---------------------------------------------------------------------------
class _DiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        # AdaLN: produce (scale1, shift1, gate1, scale2, shift2, gate2, scale3, shift3, gate3)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 9 * d_model))
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

    def _modulate(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, context: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        params = self.adaln(t_emb)  # (B, 9*d)
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = params.chunk(9, dim=-1)
        # self-attn
        y = self._modulate(self.norm1(x), s1, b1)
        a, _ = self.self_attn(y, y, y, need_weights=False)
        x = x + g1.unsqueeze(1) * a
        # cross-attn
        y = self._modulate(self.norm2(x), s2, b2)
        a, _ = self.cross_attn(y, context, context, need_weights=False)
        x = x + g2.unsqueeze(1) * a
        # ffn
        y = self._modulate(self.norm3(x), s3, b3)
        x = x + g3.unsqueeze(1) * self.ffn(y)
        return x


class _ActionDiT(nn.Module):
    def __init__(self, d_model: int = 128, action_dim: int = ACT_DIM, chunk_size: int = CHUNK_SIZE,
                 n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_embed = nn.Linear(action_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)
        self.time_embed = _SinusoidalTimeEmbed(d_model)
        self.blocks = nn.ModuleList([_DiTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.out_proj = nn.Linear(d_model, action_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, noisy: torch.Tensor, tau: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.action_embed(noisy) + self.pos_embed
        t_emb = self.time_embed(tau)
        for blk in self.blocks:
            x = blk(x, context, t_emb)
        return self.out_proj(self.final_norm(x))


# ---------------------------------------------------------------------------
# Shared context encoder module (used by both actor & critic)
# ---------------------------------------------------------------------------
class _ContextEncoder(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.height_enc = _HeightMapEncoder(d_model)
        self.depth_enc = _DepthEncoder(d_model)
        self.proprio_enc = _ProprioEncoder(d_model)
        self.state_enc = _StateTokenEncoder(d_model)

    def forward(self, proprio_hist, cmd, lin_vel, height, depth) -> torch.Tensor:
        h = self.height_enc(height)                   # (B, 16, d)
        d = self.depth_enc(depth)                     # (B, 36, d)
        p = self.proprio_enc(proprio_hist)            # (B, 1, d)
        s = self.state_enc(cmd, lin_vel)              # (B, 1, d)
        return torch.cat([h, d, p, s], dim=1)         # (B, 54, d)


# ---------------------------------------------------------------------------
# V16FlowModel — rsl-rl MLPModel-compatible actor OR critic
# ---------------------------------------------------------------------------
class V16FlowModel(nn.Module):
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
        n_dit_layers: int = 4,
        n_flow_steps: int = N_FLOW_STEPS,
        chunk_size: int = CHUNK_SIZE,
        action_scale: float = 0.25,
    ) -> None:
        super().__init__()
        del activation

        self.obs_groups = list(obs_groups[obs_set])
        if len(self.obs_groups) != 1:
            raise ValueError(f"V16FlowModel expects one obs group, got {self.obs_groups}")
        self.obs_group = self.obs_groups[0]

        flat_dim = int(obs[self.obs_group].shape[-1])
        if flat_dim != EXPECTED_OBS_DIM:
            raise ValueError(
                f"V16FlowModel expects obs['{self.obs_group}'] dim {EXPECTED_OBS_DIM}, got {flat_dim}"
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
        self.chunk_size = chunk_size
        self.n_flow_steps = n_flow_steps

        if self.is_actor:
            # DiT produces action chunks of shape (B, chunk_size, output_dim)
            self.dit = _ActionDiT(d_model, output_dim, chunk_size, n_dit_layers, n_heads)
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution: Distribution = dist_class(output_dim, **distribution_cfg)
            if self.distribution.input_dim != output_dim:
                raise ValueError("V16 distribution input_dim must equal output_dim.")
            # rsl-rl Distribution.init_mlp_weights expects self.mlp; give a small head
            self.mlp = nn.Sequential(nn.Linear(output_dim, output_dim))
            self.distribution.init_mlp_weights(self.mlp)
        else:
            self.distribution = None  # type: ignore[assignment]
            # Critic: pool context tokens, MLP -> scalar
            head_in = d_model
            head_dims = [head_in] + list(hidden_dims) + [output_dim]
            layers: list[nn.Module] = []
            for i in range(len(head_dims) - 1):
                layers.append(nn.Linear(head_dims[i], head_dims[i + 1]))
                if i < len(head_dims) - 2:
                    layers.append(nn.ELU())
            self.critic_head = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # obs parsing
    # ------------------------------------------------------------------
    def _split_obs(self, flat: torch.Tensor):
        B = flat.shape[0]
        off = 0
        proprio_flat = flat[:, off:off + PROPRIO_FLAT]; off += PROPRIO_FLAT
        cmd = flat[:, off:off + CMD_DIM]; off += CMD_DIM
        lin_vel = flat[:, off:off + LIN_VEL_DIM]; off += LIN_VEL_DIM
        height_flat = flat[:, off:off + HEIGHT_DIM]; off += HEIGHT_DIM
        depth_flat = flat[:, off:off + DEPTH_DIM]; off += DEPTH_DIM

        # Proprio term-major history: ang h5, grav h5, jpos h5, jvel h5, act h5
        po = 0
        ang = proprio_flat[:, po:po + PROPRIO_HIST * ANG_DIM].view(B, PROPRIO_HIST, ANG_DIM); po += PROPRIO_HIST * ANG_DIM
        grv = proprio_flat[:, po:po + PROPRIO_HIST * GRAV_DIM].view(B, PROPRIO_HIST, GRAV_DIM); po += PROPRIO_HIST * GRAV_DIM
        jp = proprio_flat[:, po:po + PROPRIO_HIST * JPOS_DIM].view(B, PROPRIO_HIST, JPOS_DIM); po += PROPRIO_HIST * JPOS_DIM
        jv = proprio_flat[:, po:po + PROPRIO_HIST * JVEL_DIM].view(B, PROPRIO_HIST, JVEL_DIM); po += PROPRIO_HIST * JVEL_DIM
        la = proprio_flat[:, po:po + PROPRIO_HIST * LAST_ACT_DIM].view(B, PROPRIO_HIST, LAST_ACT_DIM)
        proprio_hist = torch.cat([ang, grv, jp, jv, la], dim=-1)  # (B, 5, 117)

        height = height_flat.view(B, HEIGHT_H, HEIGHT_W)
        depth = depth_flat.view(B, DEPTH_H, DEPTH_W)
        return proprio_hist, cmd, lin_vel, height, depth

    # ------------------------------------------------------------------
    # rsl-rl interface
    # ------------------------------------------------------------------
    def _encode(self, obs: TensorDict) -> torch.Tensor:
        flat = obs[self.obs_group]
        if self.obs_normalization:
            non_vis = flat[:, :self.non_vis_dim]
            vis = flat[:, self.non_vis_dim:]
            non_vis = self.obs_normalizer(non_vis)
            flat = torch.cat([non_vis, vis], dim=-1)
        proprio_hist, cmd, lin_vel, height, depth = self._split_obs(flat)
        return self.context_enc(proprio_hist, cmd, lin_vel, height, depth)  # (B, 54, d)

    def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None,
                   hidden_state: HiddenState = None) -> torch.Tensor:
        context = self._encode(obs)
        return context.mean(dim=1)  # pooled context (B, d) for reference / critic

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            flat = obs[self.obs_group]
            self.obs_normalizer.update(flat[:, :self.non_vis_dim])  # type: ignore

    def _flow_decode(self, context: torch.Tensor) -> torch.Tensor:
        B = context.shape[0]
        dev = context.device
        x = torch.randn(B, self.chunk_size, self.output_dim, device=dev)
        dt = 1.0 / float(self.n_flow_steps)
        for i in range(self.n_flow_steps):
            tau = torch.full((B,), i * dt, device=dev)
            v = self.dit(x, tau, context)
            x = x + v * dt
        return x  # (B, chunk_size, output_dim)

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
            chunk = self._flow_decode(context)           # (B, 8, ACT_DIM)
            first = chunk[:, 0]                          # (B, ACT_DIM)
            mean = torch.tanh(first) * self.action_scale
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
        # critic
        pooled = context.mean(dim=1)
        return self.critic_head(pooled)

    # ------------------------------------------------------------------
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
