"""V25 tiny transformer: 24x24 heightmap ViT + 2-layer loco transformer w/ cross-attn.

Comparison to RobustV2 MLP policy. Design targets:
  - heightmap 24x24 high-res -> patch 6x6 stride 6 -> 4x4 = 16 tokens (<=30)
  - main transformer: 2 layers, 4 heads, d_model=64 (tiny, fast)
  - cross-attend state tokens to heightmap tokens
  - obs layout: [state_flat, height_flat(576)] -- state dim inferred at init
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


HEIGHT_H_V25 = 24
HEIGHT_W_V25 = 24
HEIGHT_DIM_V25 = HEIGHT_H_V25 * HEIGHT_W_V25  # 576
PATCH = 6
N_PATCH_H = HEIGHT_H_V25 // PATCH  # 4
N_PATCH_W = HEIGHT_W_V25 // PATCH  # 4
N_HEIGHT_TOKENS_V25 = N_PATCH_H * N_PATCH_W  # 16


class _SelfAttnBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, ffn_mult: int = 2):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * ffn_mult), nn.GELU(), nn.Linear(d * ffn_mult, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ffn(self.n2(x))
        return x


class _CrossAttnBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, ffn_mult: int = 2):
        super().__init__()
        self.nq = nn.LayerNorm(d)
        self.nkv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * ffn_mult), nn.GELU(), nn.Linear(d * ffn_mult, d))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(self.nq(q), self.nkv(kv), self.nkv(kv), need_weights=False)
        q = q + a
        q = q + self.ffn(self.n2(q))
        return q


class _HeightEncV25(nn.Module):
    """(B, 24, 24) -> (B, 16, d). Conv2d patch 6x6 stride 6."""
    def __init__(self, d: int):
        super().__init__()
        self.patch = nn.Conv2d(1, d, kernel_size=PATCH, stride=PATCH)
        self.pos = nn.Parameter(torch.randn(1, N_HEIGHT_TOKENS_V25, d) * 0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.patch(h.unsqueeze(1)).flatten(2).transpose(1, 2)
        return x + self.pos


class V25TinyTFModel(nn.Module):
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
        n_vision_layers: int = 2,
        n_loco_layers: int = 2,
        n_cross_layers: int = 1,
        ffn_mult: int = 2,
        out_proj_gain: float = 0.5,
    ) -> None:
        super().__init__()
        del activation

        self.obs_groups = list(obs_groups[obs_set])
        if len(self.obs_groups) != 1:
            raise ValueError(f"V25 expects one obs group, got {self.obs_groups}")
        self.obs_group = self.obs_groups[0]

        self.is_actor = distribution_cfg is not None
        flat_dim = int(obs[self.obs_group].shape[-1])
        if flat_dim <= HEIGHT_DIM_V25:
            raise ValueError(f"V25 obs dim {flat_dim} must exceed height dim {HEIGHT_DIM_V25}")

        self.state_dim = flat_dim - HEIGHT_DIM_V25
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.state_dim)
        else:
            self.obs_normalizer = nn.Identity()

        # State -> 1 token. Simple and fast.
        self.state_proj = nn.Linear(self.state_dim, d_model)
        self.state_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.height_enc = _HeightEncV25(d_model)
        self.vision_blocks = nn.ModuleList(
            [_SelfAttnBlock(d_model, n_heads, ffn_mult=ffn_mult) for _ in range(n_vision_layers)]
        )
        self.vision_norm = nn.LayerNorm(d_model)

        self.loco_blocks = nn.ModuleList(
            [_SelfAttnBlock(d_model, n_heads, ffn_mult=ffn_mult) for _ in range(n_loco_layers)]
        )
        self.cross_blocks = nn.ModuleList(
            [_CrossAttnBlock(d_model, n_heads, ffn_mult=ffn_mult) for _ in range(n_cross_layers)]
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
                raise ValueError("V25 distribution input_dim must equal output_dim.")
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

    def _split(self, flat: torch.Tensor):
        state = flat[:, :self.state_dim]
        height = flat[:, self.state_dim:].view(-1, HEIGHT_H_V25, HEIGHT_W_V25)
        return state, height

    def _encode(self, obs: TensorDict) -> torch.Tensor:
        flat = obs[self.obs_group]
        state, height = self._split(flat)
        if self.obs_normalization:
            state = self.obs_normalizer(state)
        # Vision path
        v = self.height_enc(height)
        for blk in self.vision_blocks:
            v = blk(v)
        v = self.vision_norm(v)
        # Loco path
        s = self.state_proj(state).unsqueeze(1) + self.state_type  # (B, 1, d)
        for blk in self.loco_blocks:
            s = blk(s)
        for blk in self.cross_blocks:
            s = blk(s, v)
        s = self.final_norm(s)
        return s.mean(dim=1)

    def get_latent(self, obs, masks=None, hidden_state=None):
        return self._encode(obs)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            flat = obs[self.obs_group]
            self.obs_normalizer.update(flat[:, :self.state_dim])  # type: ignore

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
