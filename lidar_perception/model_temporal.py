"""Temporal LiDAR reconstruction with a per-spatial-cell GRU.

Pipeline per training step:

  1. For each of K+1 history slots, transform that slot's lidar points + base
     pose into the CURRENT base frame at time t (slot K). Voxelize at 5 cm in
     a 64³ cube. Each voxel carries (occ, age, ...sub-voxel offsets).
  2. A shared 3D CNN encoder downsamples each frame (B, C, 64, 64, 64) →
     (B, hidden_c, 4, 4, 4). Weights are shared across slots.
  3. Reshape to (K+1, B*4³, hidden_c) — one independent GRU per spatial cell.
     The GRU fuses the K+1 frames temporally per location, yielding a single
     (B*4³, hidden_c) feature.
  4. A dense 3D transposed-conv decoder upsamples (B, hidden_c, 4, 4, 4) back
     to (B, 1, 64, 64, 64) — 5 cm occupancy logits over the whole cube.

Why per-cell GRU instead of one big GRU on flattened features:
  - per-cell weights are shared across the 4³=64 cells, so model is small
    and translation-equivariant in the bottleneck grid.
  - GRU sequence length is small (K+1=11), so even a small hidden_c works.

Why dense (not sparse) encoder/decoder:
  - 64³ = 262k voxels is small; dense Conv3d is fast and gradient-friendly.
  - The single-frame `dense_head` experiment already showed dense decoding
    breaks the sparse-decoder support cap.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


GRID = 64
VOXEL_SIZE = 0.05
HALF = GRID * VOXEL_SIZE / 2  # 1.6 m


def _conv_bn_relu(in_c, out_c, k=3, s=1, p=1):
    g = max(1, min(8, out_c // 4))
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
        nn.GroupNorm(g, out_c),
        nn.ReLU(inplace=True),
    )


class PerFrameEncoder(nn.Module):
    """Dense 3D conv encoder, 64³ → 4³ × hidden_c, shared across time slots."""

    def __init__(self, in_c: int = 3, base_c: int = 32, hidden_c: int = 256):
        super().__init__()
        # 64 -> 32 -> 16 -> 8 -> 4
        c1, c2, c3 = base_c, base_c * 2, base_c * 4
        self.stem = _conv_bn_relu(in_c, c1)
        self.d1 = nn.Sequential(_conv_bn_relu(c1, c1, s=2), _conv_bn_relu(c1, c2))
        self.d2 = nn.Sequential(_conv_bn_relu(c2, c2, s=2), _conv_bn_relu(c2, c3))
        self.d3 = nn.Sequential(_conv_bn_relu(c3, c3, s=2), _conv_bn_relu(c3, hidden_c))
        self.d4 = nn.Sequential(_conv_bn_relu(hidden_c, hidden_c, s=2),
                                 _conv_bn_relu(hidden_c, hidden_c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_c, 64, 64, 64)
        h = self.stem(x)
        h = self.d1(h)
        h = self.d2(h)
        h = self.d3(h)
        h = self.d4(h)  # (B, hidden_c, 4, 4, 4)
        return h


class SpatialGRU(nn.Module):
    """Independent GRU per spatial cell at the 4³ bottleneck."""

    def __init__(self, hidden_c: int = 256, gru_hidden: int = 128):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.input_proj = nn.Linear(hidden_c, gru_hidden)
        self.gru = nn.GRU(input_size=gru_hidden, hidden_size=gru_hidden,
                          num_layers=1, batch_first=False)
        self.output_proj = nn.Linear(gru_hidden, hidden_c)

    def forward(self, feats_seq: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        feats_seq: (T, B, hidden_c, 4, 4, 4)   T = K+1 (current is last)
        valid_mask: (B, T) bool — False slots get current-frame features instead
                    (handles cold-start padding).
        Returns: (B, hidden_c, 4, 4, 4)
        """
        T, B, C, D, H, W = feats_seq.shape
        S = D * H * W  # 64 spatial cells
        x = feats_seq.permute(0, 1, 3, 4, 5, 2).reshape(T, B * S, C)  # (T, B*S, C)
        x = self.input_proj(x)  # (T, B*S, gru_hidden)

        # GRU expects (T, N, F). Mask handling: if any slot is invalid in a sample
        # we still feed the (padded-with-current) data the dataset provided, but
        # we additionally repeat the CURRENT slot for each invalid slot so the
        # GRU never sees a "fake other" frame from cold-start padding.
        # Concretely: for each batch b, take features[T-1, b] and broadcast to
        # invalid slots.
        if valid_mask is not None:
            valid_T = valid_mask.transpose(0, 1)  # (T, B)
            cur = x[T - 1].unsqueeze(0).expand(T, -1, -1)  # (T, B*S, h)
            valid_T_full = valid_T.unsqueeze(-1).repeat_interleave(S, dim=1)  # (T, B*S, 1)
            x = torch.where(valid_T_full, x, cur)

        out, _ = self.gru(x)  # (T, B*S, gru_hidden)
        last = out[T - 1]  # (B*S, gru_hidden) — current-frame fused state
        last = self.output_proj(last)  # (B*S, hidden_c)
        last = last.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        return last  # (B, hidden_c, 4, 4, 4)


class DenseDecoder(nn.Module):
    """4³ × hidden_c → 64³ × 1 logits via transposed convs."""

    def __init__(self, hidden_c: int = 256, base_c: int = 32):
        super().__init__()
        c3, c2, c1 = base_c * 4, base_c * 2, base_c
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(hidden_c, c3, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(8, c3), nn.ReLU(inplace=True),
            _conv_bn_relu(c3, c3),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(8, c2), nn.ReLU(inplace=True),
            _conv_bn_relu(c2, c2),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(8, c1), nn.ReLU(inplace=True),
            _conv_bn_relu(c1, c1),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(c1, c1, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(8, c1), nn.ReLU(inplace=True),
            _conv_bn_relu(c1, c1),
        )
        self.head = nn.Conv3d(c1, 1, kernel_size=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        return self.head(h)


class TemporalGRUModel(nn.Module):
    """Full pipeline: per-frame encode → spatial GRU → dense decode."""

    def __init__(self, in_c: int = 3, base_c: int = 32, hidden_c: int = 256,
                 gru_hidden: int = 128):
        super().__init__()
        self.encoder = PerFrameEncoder(in_c=in_c, base_c=base_c, hidden_c=hidden_c)
        self.gru = SpatialGRU(hidden_c=hidden_c, gru_hidden=gru_hidden)
        self.decoder = DenseDecoder(hidden_c=hidden_c, base_c=base_c)

    def forward(self, x_seq: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_seq: (B, T, in_c, 64, 64, 64)   T = K+1
        valid_mask: (B, T) bool
        returns: (B, 1, 64, 64, 64) logits
        """
        B, T, C, D, H, W = x_seq.shape
        x = x_seq.reshape(B * T, C, D, H, W)
        feats = self.encoder(x)  # (B*T, hidden_c, 4, 4, 4)
        feats = feats.reshape(B, T, *feats.shape[1:]).permute(1, 0, 2, 3, 4, 5)  # (T, B, ...)
        fused = self.gru(feats, valid_mask)  # (B, hidden_c, 4, 4, 4)
        return self.decoder(fused)  # (B, 1, 64, 64, 64)


# ============================ voxelization helper ====================================


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


def voxelize_history_to_dense(
    lidar_pcl: torch.Tensor,        # (B, T, N, 3) WORLD
    lidar_mask: torch.Tensor,       # (B, T, N) bool
    base_pos_t: torch.Tensor,       # (B, 3) WORLD pos at current frame
    base_quat_t: torch.Tensor,      # (B, 4) xyzw at current frame
    age_s: torch.Tensor,            # (B, T) age in seconds (≤0)
    valid_history: torch.Tensor,    # (B, T) bool
) -> torch.Tensor:
    """Transform every history frame's lidar into the current base frame and
    voxelize each into its own dense (B, T, 3, 64, 64, 64) tensor.

    Channels (per voxel):
      0 — occupancy (0/1)
      1 — normalized age (age_s / 1.5; ≈ 0..-1 over 1.5 s history)
      2 — slot-index normalized (slot / T; helps order signal even if all ages 0)
    """
    B, T, N, _ = lidar_pcl.shape
    device = lidar_pcl.device
    R_t = quat_xyzw_to_R(base_quat_t)  # (B, 3, 3)

    out = torch.zeros((B, T, 3, GRID, GRID, GRID), device=device, dtype=torch.float32)
    age_norm = (age_s / 1.5).clamp_(-3.0, 0.0)  # (B, T)

    for b in range(B):
        Rb = R_t[b].T  # world -> local
        bp = base_pos_t[b]
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
            # occupancy + age + slot
            out[b, t, 0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
            out[b, t, 1, idx[:, 0], idx[:, 1], idx[:, 2]] = float(age_norm[b, t])
            out[b, t, 2, idx[:, 0], idx[:, 1], idx[:, 2]] = float(t) / max(1, T - 1)
    return out


def gt_to_dense(
    gt_pcl: torch.Tensor,           # (B, M, 3) WORLD
    gt_n: torch.Tensor,             # (B,) valid count
    base_pos_t: torch.Tensor,       # (B, 3)
    base_quat_t: torch.Tensor,      # (B, 4) xyzw
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
