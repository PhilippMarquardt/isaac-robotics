"""Early-fusion temporal dense 3D UNet for lidar reconstruction.

All K+1 history frames are transformed into the CURRENT base frame at time t,
voxelized at 5 cm in a 64³ cube, and aggregated into a small set of dense
channels. The aggregated input is then passed through a standard 3D UNet
with skip connections, producing 64³ occupancy logits.

Why early fusion at 64³ (and not late fusion at 4³ as the GRU model did):
  - the GRU at 4³ has to compress fine geometry before fusing → information
    loss before the cross-time signal is exploited;
  - skip connections from the 64³ / 32³ / 16³ encoder levels carry fine
    geometry directly to the decoder.

Input channels (default — simple, model-friendly):
  0 — current_occ           any lidar hit at this voxel in the CURRENT frame
  1 — history_any_occ       any lidar hit at this voxel in ANY past slot
  2 — history_count_norm    fraction of past slots that have a hit here
  3 — newest_age_s_norm     age of the most recent past observation (≥−1)
  4 — oldest_age_s_norm     age of the earliest past observation
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


GRID = 64
VOXEL_SIZE = 0.05
HALF = GRID * VOXEL_SIZE / 2  # 1.6 m
N_INPUT_CHANNELS = 5


def _conv_bn_relu(in_c: int, out_c: int, k: int = 3, p: int = 1) -> nn.Sequential:
    g = max(1, min(8, out_c // 4))
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=k, padding=p, bias=False),
        nn.GroupNorm(g, out_c),
        nn.ReLU(inplace=True),
    )


class _DownBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            _conv_bn_relu(in_c, out_c),
            _conv_bn_relu(out_c, out_c),
        )
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.block(x)          # carries skip
        return self.down(feat), feat  # downsampled, skip


class _UpBlock(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, in_c // 2, kernel_size=2, stride=2, bias=False)
        self.block = nn.Sequential(
            _conv_bn_relu(in_c // 2 + skip_c, out_c),
            _conv_bn_relu(out_c, out_c),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class TemporalUNet(nn.Module):
    """Standard 3D UNet on aggregated temporal input."""

    def __init__(self, in_c: int = N_INPUT_CHANNELS, base_c: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_c, base_c * 2, base_c * 4, base_c * 8
        self.stem = _conv_bn_relu(in_c, c1)
        self.d1 = _DownBlock(c1, c2)
        self.d2 = _DownBlock(c2, c3)
        self.d3 = _DownBlock(c3, c4)
        self.bottleneck = nn.Sequential(
            _conv_bn_relu(c4, c4),
            _conv_bn_relu(c4, c4),
        )
        self.u3 = _UpBlock(c4, c4, c3)
        self.u2 = _UpBlock(c3, c3, c2)
        self.u1 = _UpBlock(c2, c2, c1)
        self.head = nn.Conv3d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_c, 64, 64, 64)
        s0 = self.stem(x)             # (B, c1, 64)
        x1, s1 = self.d1(s0)          # x1: (B, c2, 32) — pre-down skip s1: (B, c2, 64)
        x2, s2 = self.d2(x1)          # s2: (B, c3, 32)
        x3, s3 = self.d3(x2)          # s3: (B, c4, 16); x3: (B, c4, 8)
        b = self.bottleneck(x3)       # (B, c4, 8)
        u3 = self.u3(b, s3)           # back to 16
        u2 = self.u2(u3, s2)          # back to 32
        u1 = self.u1(u2, s1)          # back to 64
        return self.head(u1)          # (B, 1, 64)


# ============================ voxelization (early-fusion aggregation) ===================


def quat_xyzw_to_R(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def aggregate_history_to_dense(
    lidar_pcl: torch.Tensor,         # (B, T, N, 3) WORLD
    lidar_mask: torch.Tensor,        # (B, T, N) bool
    base_pos_t: torch.Tensor,        # (B, 3)
    base_quat_t: torch.Tensor,       # (B, 4) xyzw
    age_s: torch.Tensor,             # (B, T) seconds, ≤ 0
    valid_history: torch.Tensor,     # (B, T) bool
) -> torch.Tensor:
    """Return (B, 5, 64, 64, 64) early-fused input.

    Channels:
      0 — current_occ
      1 — history_any_occ
      2 — history_count_norm
      3 — newest_age_s_norm  (0 if no past obs)
      4 — oldest_age_s_norm  (0 if no past obs)
    """
    B, T, N, _ = lidar_pcl.shape
    device = lidar_pcl.device
    R_t = quat_xyzw_to_R(base_quat_t)  # (B, 3, 3)

    out = torch.zeros((B, 5, GRID, GRID, GRID), device=device, dtype=torch.float32)
    cur_idx = T - 1
    age_norm = (age_s / 1.5).clamp_(-3.0, 0.0)  # (B, T)

    for b in range(B):
        Rb = R_t[b].T  # world -> local
        bp = base_pos_t[b]
        history_count = torch.zeros(GRID, GRID, GRID, device=device)
        newest_age = torch.zeros(GRID, GRID, GRID, device=device)
        oldest_age = torch.zeros(GRID, GRID, GRID, device=device)
        any_hist = torch.zeros(GRID, GRID, GRID, device=device, dtype=torch.bool)
        n_valid_past = 0

        for t in range(T):
            if not bool(valid_history[b, t]):
                continue
            msk = lidar_mask[b, t]
            pts_w = lidar_pcl[b, t][msk]
            if pts_w.shape[0] == 0:
                continue
            pts_l = (Rb @ (pts_w - bp).T).T
            in_box = ((pts_l > -HALF) & (pts_l < HALF)).all(-1)
            pts_l = pts_l[in_box]
            if pts_l.shape[0] == 0:
                continue
            idx = ((pts_l + HALF) / VOXEL_SIZE).long().clamp_(0, GRID - 1)
            ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]

            if t == cur_idx:
                out[b, 0, ix, iy, iz] = 1.0  # current_occ
            else:
                # past slot
                any_hist[ix, iy, iz] = True
                history_count[ix, iy, iz] = history_count[ix, iy, iz] + 1.0
                age_t = float(age_norm[b, t])  # ≤ 0
                # "newest" = max (closest to 0, since ages are ≤0)
                # "oldest" = min (most negative)
                cur_new = newest_age[ix, iy, iz]
                cur_old = oldest_age[ix, iy, iz]
                empty = (cur_new == 0) & ~any_hist[ix, iy, iz].clone()
                # simpler: set newest = max, oldest = min (init to 0 means "no past obs")
                # use torch.maximum / minimum with broadcasting
                newest_age[ix, iy, iz] = torch.maximum(cur_new, torch.full_like(cur_new, age_t))
                # oldest: we want the most negative; init at 0 means we need a guard
                # mask where we already have past obs
                old_mask = oldest_age[ix, iy, iz] < 0
                # if we already have an old obs at this voxel, take min(old, age_t); else age_t
                new_old = torch.where(old_mask, torch.minimum(cur_old, torch.full_like(cur_old, age_t)),
                                       torch.full_like(cur_old, age_t))
                oldest_age[ix, iy, iz] = new_old
                n_valid_past += 1

        out[b, 1] = any_hist.float()
        denom = max(1, T - 1)  # past slots = T-1
        out[b, 2] = history_count / denom
        out[b, 3] = newest_age
        out[b, 4] = oldest_age

    return out


def gt_to_dense(
    gt_pcl: torch.Tensor,
    gt_n: torch.Tensor,
    base_pos_t: torch.Tensor,
    base_quat_t: torch.Tensor,
) -> torch.Tensor:
    B = gt_pcl.shape[0]
    R_t = quat_xyzw_to_R(base_quat_t)
    out = torch.zeros((B, 1, GRID, GRID, GRID), device=gt_pcl.device, dtype=torch.float32)
    for b in range(B):
        n = int(gt_n[b].item())
        if n <= 0:
            continue
        Rb = R_t[b].T
        bp = base_pos_t[b]
        gtw = gt_pcl[b, :n]
        gtl = (Rb @ (gtw - bp).T).T
        in_box = ((gtl > -HALF) & (gtl < HALF)).all(-1)
        gtl = gtl[in_box]
        if gtl.shape[0] == 0:
            continue
        idx = ((gtl + HALF) / VOXEL_SIZE).long().clamp_(0, GRID - 1)
        out[b, 0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return out
