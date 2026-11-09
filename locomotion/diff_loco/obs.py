"""Custom observation functions for diff_loco."""
from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster


def lidar_distance(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    max_distance: float = 10.0,
) -> torch.Tensor:
    """Per-ray distance from a ray-caster sensor, clipped to max_distance.

    Returns a (num_envs, num_rays) tensor of Euclidean distances from the
    sensor origin to each ray hit point, clamped so misses (rays hitting
    nothing, reported as +inf or far plane) don't blow up normalization.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # sensor.data.pos_w: (N, 3), sensor.data.ray_hits_w: (N, num_rays, 3)
    origin = sensor.data.pos_w.unsqueeze(1)
    d = torch.linalg.norm(sensor.data.ray_hits_w - origin, dim=-1)
    return torch.clamp(d, max=max_distance)


def noisy_height_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    noise_std: float = 0.08,
    dropout_p: float = 0.15,
    dropout_fill: float = 1.0,
) -> torch.Tensor:
    """Height scan with Gaussian noise + per-ray dropout + inf-safe fill.

    Models real onboard elevation-map pipelines (depth-to-height / LiDAR):
      - Gaussian noise per ray (noise_std, ~8cm default).
      - Bernoulli dropout: ~dropout_p rays clamped to +dropout_fill
        (represents occluded / no-return cells).
      - Non-finite hits (pits/gaps) clamped to +dropout_fill.

    Shape: (N, R). Caller should still apply clip=(-1,1) to bound outputs.
    V13.5 uses this in actor obs; critic keeps clean height_scan.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits_z = sensor.data.ray_hits_w[..., 2]
    sensor_z = sensor.data.pos_w[:, 2].unsqueeze(1)
    raw = sensor_z - hits_z - offset
    raw = torch.where(torch.isfinite(raw), raw, torch.full_like(raw, dropout_fill))
    if noise_std > 0.0:
        raw = raw + torch.randn_like(raw) * noise_std
    if dropout_p > 0.0:
        drop_mask = torch.rand_like(raw) < dropout_p
        raw = torch.where(drop_mask, torch.full_like(raw, dropout_fill), raw)
    return raw
