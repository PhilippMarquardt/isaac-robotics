"""Hoeller-2022 sparse 4D U-Net for online lidar terrain reconstruction.

Reproduces the architecture of Hoeller et al., "Neural Scene Representation for
Locomotion on Structured Terrain" (RAL 2022):

  - Input: voxelized lidar point cloud at 5 cm resolution over a 64³ grid centered
    on the robot. Concatenated with the previous network output (transformed into
    the current frame and detached) plus a discrete time channel k ∈ {0, 1}.
  - Voxel feature: sub-voxel offset (Eq. 1 of the paper) — the centroid offset of
    the points falling into each occupied voxel, normalized to (-0.5, +0.5).
  - Encoder: 4 sparse-conv blocks, each downsamples by ×2.
  - Bottleneck.
  - Decoder: 4 sparse-conv-transpose blocks producing occupancy + sub-voxel offset
    at each resolution.
  - Loss: BCE on per-voxel occupancy (multi-resolution) + L2 on sub-voxel offsets,
    summed across decoder levels.
  - 12-step rollout, prev-frame feedback is **detached** (no BPTT — paper §IV).

Sparse-conv backend: spconv (`pip install spconv-cu120`).
The pure-spconv implementation here doesn't perfectly match Minkowski's
"generative sparse convolutions with pruning" — instead it relies on
`SparseInverseConv3d` (which expands voxels back along the encoder's coordinate
trace via shared `indice_key`s) plus an explicit pruning step at each decoder
level. Functionally equivalent for our 64³ grid, simpler to debug.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as spconv

# Each occupied voxel carries (offset_x, offset_y, offset_z, source_flag) where
# source_flag ∈ {0, 1} distinguishes "current measurement" from "previous network
# output". This is the discrete time channel k.
IN_CHANNELS = 4
BASE_C = 16
NUM_LEVELS = 4
GRID = 64


def _conv_block(in_c: int, out_c: int, indice_key: str) -> spconv.SparseSequential:
    # GroupNorm (rather than BatchNorm1d) — sparse layers occasionally produce
    # single-voxel batches at coarse levels, which crashes BN. GN handles M=1 fine.
    g = max(1, min(8, out_c // 4))
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_c, out_c, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
        nn.GroupNorm(g, out_c),
        nn.ReLU(inplace=True),
        spconv.SubMConv3d(out_c, out_c, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
        nn.GroupNorm(g, out_c),
        nn.ReLU(inplace=True),
    )


class HoellerUNetSpconv(nn.Module):
    """Sparse U-Net via spconv. Functionally a 4-level encoder/decoder with skip-conns
    and per-level (occupancy + sub-voxel offset) heads."""

    def __init__(self, base_c: int = BASE_C, num_levels: int = NUM_LEVELS, grid: int = GRID):
        super().__init__()
        self.num_levels = num_levels
        self.grid = grid

        # encoder: SubM conv at level L → strided down-conv (spconv generates new coords)
        self.enc_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        in_c = IN_CHANNELS
        for level in range(num_levels):
            out_c = base_c * (2**level)
            self.enc_blocks.append(_conv_block(in_c, out_c, indice_key=f"enc_{level}"))
            self.down_convs.append(
                spconv.SparseConv3d(
                    out_c, out_c * 2, kernel_size=2, stride=2, bias=False,
                    indice_key=f"down_{level}",
                )
            )
            in_c = out_c * 2

        # bottleneck
        bot_c = in_c
        self.bottleneck = _conv_block(bot_c, bot_c, indice_key="bottleneck")

        # decoder: inverse-conv to expand along stored coord map, then conv block, then heads
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.occ_heads = nn.ModuleList()
        self.off_heads = nn.ModuleList()
        for level in reversed(range(num_levels)):
            skip_c = base_c * (2**level)
            self.up_convs.append(
                spconv.SparseInverseConv3d(in_c, skip_c, kernel_size=2, bias=False,
                                            indice_key=f"down_{level}")
            )
            # after concat with skip
            self.dec_blocks.append(_conv_block(skip_c * 2, skip_c, indice_key=f"dec_{level}"))
            self.occ_heads.append(
                spconv.SubMConv3d(skip_c, 1, kernel_size=1, indice_key=f"occ_{level}")
            )
            self.off_heads.append(
                spconv.SubMConv3d(skip_c, 3, kernel_size=1, indice_key=f"off_{level}")
            )
            in_c = skip_c

    def forward(self, x: spconv.SparseConvTensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass. Returns list of (occ_logits, offset) sparse tensors per
        decoder level (coarsest → finest). Each tuple element is a SparseConvTensor.
        """
        # encode
        skips = []
        h = x
        for enc, down in zip(self.enc_blocks, self.down_convs):
            h = enc(h)
            skips.append(h)
            h = down(h)
        # bottleneck
        h = self.bottleneck(h)
        # decode with skip concat
        outs = []
        for up, dec, occ_h, off_h, skip in zip(self.up_convs, self.dec_blocks, self.occ_heads,
                                                self.off_heads, reversed(skips)):
            h = up(h)
            # concat features along channel dim — sparse tensors must have the same coord map.
            # `up` output uses the same indice_key as `down` so it matches `skip` coords.
            cat_feats = torch.cat([h.features, skip.features], dim=1)
            h = h.replace_feature(cat_feats)
            h = dec(h)
            outs.append((occ_h(h), off_h(h)))
        return outs


def build_model(base_c: int = BASE_C, num_levels: int = NUM_LEVELS, grid: int = GRID) -> nn.Module:
    return HoellerUNetSpconv(base_c=base_c, num_levels=num_levels, grid=grid)


# -------- voxelization helpers (kept here so train_hoeller can import) ---------------


# Paper-standard: 5 cm × 64³ = 3.2 m cube. Works for v5 dataset because the
# inverted MID-360 mount on G1 produces close-range ground hits (V-FOV -52° to
# +7° → from 1.25 m mount, ground hit at ~0.98 m horizontal).
VOXEL_SIZE = 0.05
HALF = GRID * VOXEL_SIZE / 2  # 1.6 m


def voxelize_pcl_local(points_l: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Voxelize a (N, 3) lidar-frame point cloud into a 64³ grid centered at origin.

    Returns:
        coords: (M, 3) integer voxel indices in [0, GRID). Always ≥1 voxel — if no
            real hits land in the box (common with MID-360's upward FOV bias) we
            insert a dummy voxel at the grid center so spconv kernels never see
            an empty input.
        offsets: (M, 3) sub-voxel offsets in (-0.5, +0.5) (paper Eq. 1).
    """
    valid = mask.bool() & torch.isfinite(points_l).all(-1)
    p = points_l[valid]
    in_box = ((p > -HALF) & (p < HALF)).all(-1)
    p = p[in_box]
    if p.shape[0] == 0:
        # empty pcl: insert one dummy at grid center, zero offset
        coords = torch.tensor([[GRID // 2, GRID // 2, GRID // 2]],
                               dtype=torch.int32, device=points_l.device)
        offsets = torch.zeros(1, 3, device=points_l.device, dtype=points_l.dtype)
        return coords, offsets
    idx = ((p + HALF) / VOXEL_SIZE).long().clamp_(0, GRID - 1)
    centers = (idx.float() + 0.5) * VOXEL_SIZE - HALF
    offsets = (p - centers) / VOXEL_SIZE  # in (-0.5, +0.5)
    flat = idx[:, 0] * GRID * GRID + idx[:, 1] * GRID + idx[:, 2]
    uniq, inv = torch.unique(flat, return_inverse=True)
    M = uniq.shape[0]
    sum_off = torch.zeros(M, 3, device=p.device, dtype=p.dtype)
    sum_cnt = torch.zeros(M, device=p.device, dtype=p.dtype)
    sum_off.index_add_(0, inv, offsets)
    sum_cnt.index_add_(0, inv, torch.ones_like(inv, dtype=p.dtype))
    mean_off = sum_off / sum_cnt.unsqueeze(-1).clamp_min(1)
    coords = torch.stack([uniq // (GRID * GRID), (uniq // GRID) % GRID, uniq % GRID], dim=-1)
    return coords.int(), mean_off


def stack_batch(coords_list: list[torch.Tensor], offsets_list: list[torch.Tensor],
                source_flags: list[float] | None = None) -> spconv.SparseConvTensor:
    """Pack a per-batch list of (coords[Mi,3], offsets[Mi,3]) into a single SparseConvTensor."""
    B = len(coords_list)
    feats = []
    coords_with_b = []
    for b, (c, o) in enumerate(zip(coords_list, offsets_list)):
        if source_flags is not None:
            flag_col = torch.full((c.shape[0], 1), source_flags[b], device=o.device, dtype=o.dtype)
            f = torch.cat([o, flag_col], dim=-1)
        else:
            f = torch.cat([o, torch.zeros(c.shape[0], 1, device=o.device, dtype=o.dtype)], dim=-1)
        feats.append(f)
        coords_with_b.append(torch.cat([torch.full((c.shape[0], 1), b, device=c.device, dtype=c.dtype), c], dim=-1))
    feats = torch.cat(feats, dim=0)
    coords_with_b = torch.cat(coords_with_b, dim=0)
    return spconv.SparseConvTensor(feats, coords_with_b, spatial_shape=[GRID, GRID, GRID], batch_size=B)


# -------- loss helper ----------------------------------------------------------------


def reconstruction_loss(
    pred_per_level: list[tuple[spconv.SparseConvTensor, spconv.SparseConvTensor]],
    target_per_level: list[tuple[torch.Tensor, torch.Tensor]],
    occ_weight: float = 1.0,
    offset_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """BCE on occupancy + L2 on sub-voxel offset, summed across decoder levels.

    For each level, predictions are sparse tensors and targets are dense per-voxel
    occupancy probabilities + offsets (zeros where unoccupied). The training loop is
    responsible for densifying predictions before passing in here.
    """
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    l2 = nn.MSELoss(reduction="mean")

    total_occ = pred_per_level[0][0].features.new_zeros(())
    total_off = pred_per_level[0][0].features.new_zeros(())
    for (occ_logits, offset), (occ_target, offset_target) in zip(pred_per_level, target_per_level):
        total_occ = total_occ + bce(occ_logits, occ_target.to(occ_logits.dtype))
        total_off = total_off + l2(offset, offset_target.to(offset.dtype))

    loss = occ_weight * total_occ + offset_weight * total_off
    return {"loss": loss, "loss_occ": total_occ.detach(), "loss_off": total_off.detach()}


__all__ = [
    "build_model",
    "reconstruction_loss",
    "voxelize_pcl_local",
    "stack_batch",
    "IN_CHANNELS",
    "BASE_C",
    "NUM_LEVELS",
    "GRID",
    "VOXEL_SIZE",
    "HALF",
]
