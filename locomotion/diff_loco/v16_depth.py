"""V16 raw 48x48 depth observation term (flattened to 2304).

RayCasterCamera on head_link emits a (B, H, W, 1) depth. We:
  1. Mask inf/nan -> max range.
  2. Apply D435i-style noise (relative + absolute + dropout).
  3. Clamp to [depth_min, depth_max] and normalize to [0, 1].
  4. Resize to 48x48 via area interpolation.
  5. Flatten to (B, 2304).
"""
from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import RayCasterCamera


class raw_depth_48(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.add_noise: bool = cfg.params.get("add_noise", True)
        self.depth_min: float = cfg.params.get("depth_min", 0.28)
        self.depth_max: float = cfg.params.get("depth_max", 4.0)
        self.noise_rel: float = cfg.params.get("noise_rel", 0.01)
        self.noise_abs: float = cfg.params.get("noise_abs", 0.003)
        self.dropout_p: float = cfg.params.get("dropout_p", 0.10)
        self.out_h: int = int(cfg.params.get("out_h", 48))
        self.out_w: int = int(cfg.params.get("out_w", 48))

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
        data_type: str = "distance_to_image_plane",
        add_noise: bool | None = None,
        out_h: int | None = None,
        out_w: int | None = None,
    ) -> torch.Tensor:
        sensor: RayCasterCamera = env.scene.sensors[sensor_cfg.name]
        depth = sensor.data.output[data_type]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2).contiguous()
        depth = depth.float()
        dmin, dmax = self.depth_min, self.depth_max
        depth = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, dmax))
        depth = torch.clamp(depth, dmin, dmax)
        an = self.add_noise if add_noise is None else add_noise
        if an:
            if self.noise_rel > 0.0:
                depth = depth * (1.0 + self.noise_rel * torch.randn_like(depth))
            if self.noise_abs > 0.0:
                depth = depth + self.noise_abs * torch.randn_like(depth)
            if self.dropout_p > 0.0:
                mask = torch.rand_like(depth) < self.dropout_p
                depth = torch.where(mask, torch.full_like(depth, dmax), depth)
            depth = torch.clamp(depth, dmin, dmax)
        depth = (depth - dmin) / (dmax - dmin)
        th = self.out_h if out_h is None else int(out_h)
        tw = self.out_w if out_w is None else int(out_w)
        if depth.shape[-2:] != (th, tw):
            depth = torch.nn.functional.interpolate(depth, size=(th, tw), mode="area")
        return depth.flatten(1)  # (B, th*tw)
