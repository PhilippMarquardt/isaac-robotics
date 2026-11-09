"""Lidar input augmentation for sim-to-real robustness.

Based on the MID-360 datasheet noise figures and common LiDAR augmentation practice:

  1. Gaussian range jitter (σ_r = 0.02 m, MID-360 spec)
  2. Per-ray dropout (rate 5–15 %)
  3. Outlier injection (0.5 % rays replaced w/ random along-ray hit)
  4. Pose jitter (lidar origin ±2 cm)
  5. Sector dropout (random 30° azimuth window zeroed, prob 10 %)

GT is NEVER noised. Apply only to lidar input pcl + mask + origin.

All ops are vectorised over the batch dim.
"""

from __future__ import annotations

import torch


def quat_xyzw_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply unit xyzw quaternion to a (..., 3) vector."""
    qx, qy, qz, qw = q.unbind(-1)
    t = 2 * torch.cross(torch.stack([qx, qy, qz], -1), v, dim=-1)
    return v + qw.unsqueeze(-1) * t + torch.cross(torch.stack([qx, qy, qz], -1), t, dim=-1)


def augment_lidar(
    pcl: torch.Tensor,           # (B, R, 3) world frame
    mask: torch.Tensor,          # (B, R) bool
    origin: torch.Tensor,        # (B, 3)
    quat_xyzw: torch.Tensor,     # (B, 4)
    *,
    sigma_range: float = 0.02,
    dropout_rate: float = 0.10,
    outlier_rate: float = 0.005,
    pose_jitter: float = 0.02,
    sector_drop_prob: float = 0.10,
    sector_width_deg: float = 30.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (pcl_aug, mask_aug, origin_aug). All tensors on input device."""
    device = pcl.device
    B, R, _ = pcl.shape
    pcl_aug = pcl.clone()
    mask_aug = mask.clone()
    origin_aug = origin.clone()

    # Pose jitter — adds noise to lidar origin (sim'd state-estimation drift)
    if pose_jitter > 0.0:
        origin_aug = origin_aug + torch.randn_like(origin_aug) * pose_jitter

    # Per-ray gaussian range jitter — perturb along the (hit - origin) direction
    if sigma_range > 0.0:
        delta = pcl_aug - origin.unsqueeze(1)  # (B, R, 3) ray vector
        d = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        dir_hat = delta / d
        d_noise = d + torch.randn_like(d) * sigma_range
        d_noise = d_noise.clamp_min(0.05)
        pcl_aug = origin.unsqueeze(1) + dir_hat * d_noise

    # Per-ray dropout — flip mask off
    if dropout_rate > 0.0:
        keep = torch.rand(B, R, device=device) > dropout_rate
        mask_aug = mask_aug & keep

    # Outlier injection — perturb a small fraction of points to nonsense distance
    if outlier_rate > 0.0:
        outlier = torch.rand(B, R, device=device) < outlier_rate
        # outlier_rate applied AFTER dropout so we only perturb still-valid rays
        delta = pcl_aug - origin.unsqueeze(1)
        d = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        dir_hat = delta / d
        # random distance 0.1 m – 30 m
        rand_d = torch.rand_like(d) * 30.0 + 0.1
        new_pts = origin.unsqueeze(1) + dir_hat * rand_d
        outl_b = outlier.unsqueeze(-1)
        pcl_aug = torch.where(outl_b, new_pts, pcl_aug)

    # Sector dropout — kill a 30° azimuth window with given probability per env
    if sector_drop_prob > 0.0 and sector_width_deg > 0.0:
        rel = pcl_aug - origin_aug.unsqueeze(1)
        # local rel rotated into lidar frame for proper azimuth (use quat_inv)
        # but world-frame azimuth is fine for symmetric removal
        az = torch.atan2(rel[..., 1], rel[..., 0])  # (B, R) in radians, (-π, π)
        do_drop = torch.rand(B, device=device) < sector_drop_prob  # (B,)
        sector_w = torch.deg2rad(torch.tensor(sector_width_deg, device=device))
        sector_center = (torch.rand(B, device=device) * 2 - 1) * 3.14159265
        # angular distance to sector center
        diff = (az - sector_center.unsqueeze(1) + 3.14159265).fmod(6.2831853) - 3.14159265
        in_sector = diff.abs() < (sector_w / 2)
        kill = in_sector & do_drop.unsqueeze(1)
        mask_aug = mask_aug & ~kill

    return pcl_aug, mask_aug, origin_aug
