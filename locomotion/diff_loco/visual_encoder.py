"""Hiking-in-the-Wild-style raw depth observation for V14.

Pipeline per env step, mirroring the paper:
  1. RayCasterCamera emits (B, 72, 128, 1) depth.
  2. nan/inf ray misses -> max range (4m).
  3. D435i sensor-sim noise (1% rel Gaussian, 3mm abs, ~10% dropout).
  4. Clamp to D435i operating range (0.28-4m), normalize to [0, 1].
  5. Area-downsample 72x128 -> 36x64 (paper downsamples 270x480 -> 36x64).
  6. Center-crop 36x64 -> 18x36 (paper's final network input).
  7. Flatten to (B, 648) — fed directly to the actor MLP, no CNN.

Class name `dinov3_depth_features` kept for tasks.py backwards compat.
"""
from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import RayCasterCamera


class dinov3_depth_features(ManagerTermBase):
    """Paper-style depth obs: preprocess, downsample, center-crop, flatten."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.add_noise: bool = cfg.params.get("add_noise", True)
        self.depth_min: float = cfg.params.get("depth_min", 0.28)
        self.depth_max: float = cfg.params.get("depth_max", 4.0)
        self.noise_rel: float = cfg.params.get("noise_rel", 0.01)
        self.noise_abs: float = cfg.params.get("noise_abs", 0.003)
        self.dropout_p: float = cfg.params.get("dropout_p", 0.10)
        self.downsample_hw: tuple[int, int] = tuple(cfg.params.get("downsample_hw", (36, 64)))
        self.crop_hw: tuple[int, int] = tuple(cfg.params.get("crop_hw", (18, 36)))

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
        data_type: str = "distance_to_image_plane",
        add_noise: bool | None = None,
        hf_name: str | None = None,  # ignored, cfg backwards compat
        model_device: str | None = None,  # ignored, cfg backwards compat
        feat_dim: int | None = None,  # ignored, cfg backwards compat
    ) -> torch.Tensor:
        sensor: RayCasterCamera = env.scene.sensors[sensor_cfg.name]
        depth = sensor.data.output[data_type]
        add_noise = self.add_noise if add_noise is None else add_noise
        dmin, dmax = self.depth_min, self.depth_max
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2).contiguous()
        depth = depth.float()
        depth = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, dmax))
        depth = torch.clamp(depth, dmin, dmax)
        if add_noise:
            if self.noise_rel > 0.0:
                depth = depth * (1.0 + self.noise_rel * torch.randn_like(depth))
            if self.noise_abs > 0.0:
                depth = depth + self.noise_abs * torch.randn_like(depth)
            if self.dropout_p > 0.0:
                mask = torch.rand_like(depth) < self.dropout_p
                depth = torch.where(mask, torch.full_like(depth, dmax), depth)
            depth = torch.clamp(depth, dmin, dmax)
        depth = (depth - dmin) / (dmax - dmin)
        # camera renders directly at downsample_hw (no intermediate interp).
        # if resolution ever diverges from cfg, area-downsample as fallback.
        if depth.shape[-2:] != self.downsample_hw:
            depth = torch.nn.functional.interpolate(depth, size=self.downsample_hw, mode="area")
        _, _, H, W = depth.shape
        ch, cw = self.crop_hw
        top = (H - ch) // 2
        left = (W - cw) // 2
        depth = depth[:, :, top : top + ch, left : left + cw]
        return depth.flatten(1)
