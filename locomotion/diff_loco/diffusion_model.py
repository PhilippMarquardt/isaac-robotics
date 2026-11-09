"""DDPM-based actor that plugs into rsl-rl 5.x PPO as a drop-in for MLPModel.

The denoiser is a small MLP conditioned on (obs_latent, noisy_action, t_embed).
At rollout time it runs K reverse diffusion steps from pure Gaussian noise and
returns the denoised action. A learnable Gaussian distribution wraps that
output so PPO can compute a tractable log-prob / KL for the ratio term.

Interface mirrors `rsl_rl.models.MLPModel` exactly so the OnPolicyRunner sees
it as an MLPModel.
"""
from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


def _make_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps)


def _build_depth_cnn(in_ch: int, hw: tuple[int, int], out_dim: int) -> nn.Module:
    # ~9.6k params. Gradient flows from PPO through actor.
    return nn.Sequential(
        nn.Conv2d(in_ch, 8, kernel_size=5, stride=2, padding=2),
        nn.ELU(),
        nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
        nn.ELU(),
        nn.AdaptiveAvgPool2d((2, 4)),
        nn.Flatten(),
        nn.Linear(16 * 2 * 4, out_dim),
        nn.ELU(),
    )


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


class _DenoiserNet(nn.Module):
    """Predicts noise given (obs_latent, x_t, t). Plain MLP over concatenated inputs."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        activation: str = "elu",
        time_embed_dim: int = 32,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_embed = _SinusoidalTimeEmbed(time_embed_dim)
        self.mlp = MLP(
            input_dim=obs_dim + action_dim + time_embed_dim,
            output_dim=action_dim,
            hidden_dims=list(hidden_dims),
            activation=activation,
        )

    def forward(self, obs_latent: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        z = torch.cat([obs_latent, x_t, t_emb], dim=-1)
        return self.mlp(z)


class DiffusionModel(nn.Module):
    """Diffusion-based actor matching the rsl-rl MLPModel interface.

    Signature of __init__ matches MLPModel so rsl-rl's construct_algorithm can
    instantiate it uniformly:  Model(obs, obs_groups, obs_set, output_dim, **cfg).
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_diffusion_steps: int = 4,
        time_embed_dim: int = 32,
        depth_encoder_cfg: dict | None = None,
    ) -> None:
        super().__init__()

        # --- observation plumbing ------------------------------------------
        # Split obs groups into depth (CNN path) and non-depth (flat path).
        self.obs_groups = list(obs_groups[obs_set])
        self.depth_group_name: str | None = None
        self.depth_shape: tuple[int, int, int] | None = None
        self.depth_cnn: nn.Module | None = None
        depth_feat_dim = 0
        if depth_encoder_cfg is not None:
            self.depth_group_name = depth_encoder_cfg["group"]
            C, H, W = depth_encoder_cfg["shape"]
            self.depth_shape = (int(C), int(H), int(W))
            depth_feat_dim = int(depth_encoder_cfg.get("out_dim", 64))
            if self.depth_group_name not in self.obs_groups:
                raise ValueError(
                    f"depth group '{self.depth_group_name}' not in obs set '{obs_set}' "
                    f"(active groups: {self.obs_groups})"
                )
            raw_dim = int(obs[self.depth_group_name].shape[-1])
            if raw_dim != C * H * W:
                raise ValueError(
                    f"depth group '{self.depth_group_name}' flat dim {raw_dim} != {C}*{H}*{W}={C*H*W}"
                )
            self.depth_cnn = _build_depth_cnn(self.depth_shape[0], (self.depth_shape[1], self.depth_shape[2]), depth_feat_dim)

        self.non_depth_groups = [g for g in self.obs_groups if g != self.depth_group_name]
        non_depth_dim = 0
        for g in self.non_depth_groups:
            if obs[g].ndim != 2:
                raise ValueError(f"DiffusionModel expects 1D obs, got {obs[g].shape} for '{g}'.")
            non_depth_dim += int(obs[g].shape[-1])
        self.non_depth_dim = non_depth_dim
        self.obs_dim = non_depth_dim + depth_feat_dim

        self.obs_normalization = obs_normalization
        if obs_normalization:
            # depth stays unnormalized (CNN handles its own scale).
            self.obs_normalizer = EmpiricalNormalization(non_depth_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        # --- output distribution (Gaussian head on top of the denoiser) ----
        if distribution_cfg is None:
            raise ValueError(
                "DiffusionModel requires a distribution_cfg (e.g. GaussianDistribution) "
                "to expose a tractable log_prob for PPO."
            )
        dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
        self.distribution: Distribution = dist_class(output_dim, **distribution_cfg)
        if self.distribution.input_dim != output_dim:
            raise ValueError(
                "DiffusionModel expects a Gaussian-like distribution whose input_dim equals the action dim."
            )

        self.action_dim = output_dim
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.time_embed_dim = int(time_embed_dim)

        # --- denoiser network ---------------------------------------------
        self.denoiser = _DenoiserNet(
            obs_dim=self.obs_dim,
            action_dim=output_dim,
            hidden_dims=tuple(hidden_dims),
            activation=activation,
            time_embed_dim=self.time_embed_dim,
        )

        # Alias so rsl-rl's distribution.init_mlp_weights sees an `.mlp`.
        self.mlp = self.denoiser.mlp
        self.distribution.init_mlp_weights(self.mlp)

        # --- DDPM noise schedule (registered as buffers for device mgmt) ---
        betas = _make_beta_schedule(self.num_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)

    # ------------------------------------------------------------------
    # forward / sampling
    # ------------------------------------------------------------------
    def _denoise(self, obs_latent: torch.Tensor) -> torch.Tensor:
        """DDIM-style deterministic reverse diffusion from x_T = 0.

        Starting from zeros (not randn) makes the action mean a deterministic
        function of obs — required so PPO's old/new log-prob ratio measures the
        policy change, not noise resampling. Exploration lives in the
        Gaussian head wrapped around this mean.
        """
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
        # Numerical defense: DDPM actor + PPO tends to NaN log_std or produce
        # inf action means under rare grad spikes. Recover at forward time
        # rather than crashing update().
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        mean = mean.clamp(-10.0, 10.0)
        if hasattr(self.distribution, "log_std_param"):
            with torch.no_grad():
                p = self.distribution.log_std_param
                bad = torch.isnan(p.data) | torch.isinf(p.data)
                if bad.any():
                    p.data[bad] = 0.0  # log(1) = 0 → std=1
                p.data.clamp_(min=-5.0, max=2.0)  # std in [0.0067, 7.4]
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mean)

    # ------------------------------------------------------------------
    # obs wiring (identical to MLPModel)
    # ------------------------------------------------------------------
    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        non_depth = torch.cat([obs[g] for g in self.non_depth_groups], dim=-1)
        non_depth = self.obs_normalizer(non_depth)
        if self.depth_cnn is None:
            return non_depth
        raw_depth = obs[self.depth_group_name]  # type: ignore[index]
        C, H, W = self.depth_shape  # type: ignore[misc]
        depth_img = raw_depth.view(raw_depth.shape[0], C, H, W)
        depth_feat = self.depth_cnn(depth_img)
        return torch.cat([non_depth, depth_feat], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            non_depth = torch.cat([obs[g] for g in self.non_depth_groups], dim=-1)
            self.obs_normalizer.update(non_depth)  # type: ignore

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    # ------------------------------------------------------------------
    # distribution plumbing (identical to MLPModel)
    # ------------------------------------------------------------------
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
        if self.depth_cnn is not None:
            raise NotImplementedError("JIT export not supported for depth-CNN DiffusionModel yet.")
        return _JITExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        if self.depth_cnn is not None:
            raise NotImplementedError("ONNX export not supported for depth-CNN DiffusionModel yet.")
        return _ONNXExport(self, verbose)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active = obs_groups[obs_set]
        dim = 0
        for g in active:
            if len(obs[g].shape) != 2:
                raise ValueError(f"DiffusionModel expects 1D obs, got {obs[g].shape} for '{g}'.")
            dim += obs[g].shape[-1]
        return active, dim


class _JITExport(nn.Module):
    def __init__(self, model: DiffusionModel) -> None:
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

    def __init__(self, model: DiffusionModel, verbose: bool) -> None:
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
