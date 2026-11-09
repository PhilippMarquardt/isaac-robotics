"""Transformer-denoiser diffusion actor for rsl-rl 5.x PPO.

Drop-in replacement for `DiffusionModel` using a transformer encoder over the
proprioceptive history as denoiser context. Inspired by BeyondMimic (transformer
decoder DDPM) but kept single-step (no action chunking) so it reuses the PPO
ratio / Gaussian log-prob path unchanged.

Token layout per forward:
    [hist_1, hist_2, ..., hist_H, action_t, time_t]   (H+2 tokens, d_model)

Self-attention over all tokens; action token's output is projected to
action_dim as noise prediction.
"""
from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


def _make_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps)


class _SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


class _TransformerDenoiser(nn.Module):
    """Transformer-encoder denoiser.

    Input:
        obs_latent: (B, H * per_step_dim) flat proprio history
        x_t:        (B, action_dim)
        t:          (B,) int timestep
    Output:
        noise pred: (B, action_dim)
    """

    def __init__(
        self,
        history_length: int,
        per_step_dim: int,
        action_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        time_embed_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.history_length = history_length
        self.per_step_dim = per_step_dim
        self.d_model = d_model

        self.hist_proj = nn.Linear(per_step_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.time_embed = _SinusoidalTimeEmbed(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, d_model)

        # Learned positional embeddings for H history tokens + action + time (H+2 total).
        self.pos_embed = nn.Parameter(torch.zeros(history_length + 2, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Linear(d_model, action_dim)
        # Zero-init the head so initial noise prediction is 0 -> stable start.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs_latent: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = obs_latent.shape[0]
        # Reshape flat history to (B, H, per_step_dim).
        hist = obs_latent.view(B, self.history_length, self.per_step_dim)
        hist_tok = self.hist_proj(hist)  # (B, H, d)
        act_tok = self.action_proj(x_t).unsqueeze(1)  # (B, 1, d)
        time_tok = self.time_proj(self.time_embed(t)).unsqueeze(1)  # (B, 1, d)
        tokens = torch.cat([hist_tok, act_tok, time_tok], dim=1)  # (B, H+2, d)
        tokens = tokens + self.pos_embed.unsqueeze(0)
        out = self.encoder(tokens)
        # Noise prediction from the action-token position (index H).
        action_out = out[:, self.history_length, :]  # (B, d)
        return self.head(action_out)


class DiffusionTransformerModel(nn.Module):
    """Transformer-encoder diffusion actor matching rsl-rl MLPModel interface.

    `hidden_dims` in runner cfg is reinterpreted: [d_model, n_heads, n_layers, dim_ff].
    `history_length` and `per_step_dim` must match the policy obs group.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 4, 3, 256),
        activation: str = "gelu",  # unused (transformer uses gelu internally)
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_diffusion_steps: int = 4,
        time_embed_dim: int = 64,
        history_length: int = 10,
        per_step_dim: int = 78,
    ) -> None:
        super().__init__()

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

        if self.obs_dim != history_length * per_step_dim:
            raise ValueError(
                f"DiffusionTransformerModel obs_dim {self.obs_dim} != "
                f"history_length {history_length} * per_step_dim {per_step_dim}"
            )

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        if distribution_cfg is None:
            raise ValueError("DiffusionTransformerModel requires distribution_cfg.")
        dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
        self.distribution: Distribution = dist_class(output_dim, **distribution_cfg)
        if self.distribution.input_dim != output_dim:
            raise ValueError("Distribution input_dim must equal action dim.")

        self.action_dim = output_dim
        self.num_diffusion_steps = int(num_diffusion_steps)

        # Reinterpret hidden_dims tuple: (d_model, n_heads, n_layers, dim_ff).
        d_model, n_heads, n_layers, dim_ff = tuple(hidden_dims)
        self.denoiser = _TransformerDenoiser(
            history_length=history_length,
            per_step_dim=per_step_dim,
            action_dim=output_dim,
            d_model=int(d_model),
            n_heads=int(n_heads),
            n_layers=int(n_layers),
            dim_feedforward=int(dim_ff),
            time_embed_dim=int(time_embed_dim),
        )

        # rsl-rl's distribution.init_mlp_weights expects a `.mlp`; give it the head.
        self.mlp = self.denoiser.head
        self.distribution.init_mlp_weights(self.mlp)

        betas = _make_beta_schedule(self.num_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)

    def _denoise(self, obs_latent: torch.Tensor) -> torch.Tensor:
        batch = obs_latent.shape[0]
        device = obs_latent.device
        x = torch.zeros(batch, self.action_dim, device=device)
        for t in reversed(range(self.num_diffusion_steps)):
            t_tensor = torch.full((batch,), float(t), device=device)
            noise_pred = self.denoiser(obs_latent, x, t_tensor)
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            coef = (1.0 - alpha_t) / torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))
            x = (x - coef * noise_pred) / torch.sqrt(alpha_t)
        return x

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        mean = self._denoise(latent)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        mean = mean.clamp(-10.0, 10.0)
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

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        obs_list = [obs[g] for g in self.obs_groups]
        latent = torch.cat(obs_list, dim=-1)
        return self.obs_normalizer(latent)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            obs_list = [obs[g] for g in self.obs_groups]
            self.obs_normalizer.update(torch.cat(obs_list, dim=-1))  # type: ignore

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        return _JITExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _ONNXExport(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active = obs_groups[obs_set]
        dim = 0
        for g in active:
            if len(obs[g].shape) != 2:
                raise ValueError(f"DiffusionTransformerModel expects 1D obs, got {obs[g].shape} for '{g}'.")
            dim += obs[g].shape[-1]
        return active, dim


class _JITExport(nn.Module):
    def __init__(self, model: DiffusionTransformerModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.denoiser = copy.deepcopy(model.denoiser)
        self.action_dim = model.action_dim
        self.num_diffusion_steps = model.num_diffusion_steps
        self.register_buffer("betas", model.betas.clone())
        self.register_buffer("alphas", model.alphas.clone())
        self.register_buffer("alpha_bars", model.alpha_bars.clone())
        self.deterministic_output = model.distribution.as_deterministic_output_module()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.obs_normalizer(x)
        batch = latent.shape[0]
        a = torch.zeros(batch, self.action_dim, device=latent.device)
        for t in range(self.num_diffusion_steps - 1, -1, -1):
            t_tensor = torch.full((batch,), float(t), device=latent.device)
            noise_pred = self.denoiser(latent, a, t_tensor)
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            coef = (1.0 - alpha_t) / torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))
            a = (a - coef * noise_pred) / torch.sqrt(alpha_t)
        return self.deterministic_output(a)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _ONNXExport(_JITExport):
    is_recurrent: bool = False

    def __init__(self, model: DiffusionTransformerModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
