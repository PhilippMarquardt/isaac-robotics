"""Unified training script for v6 dataset.

Three encoder architectures, same decoder, same data, same loss for direct ablation:
  - hoeller : sparse 4D U-Net (Hoeller 2022 RAL baseline)
  - pointnet : PointNet++ encoder + sparse-conv decoder
  - ptv3 : Point Transformer v3 encoder + sparse-conv decoder (CVPR'24 SOTA)

All work on the close-range 3.2 m cube (paper-faithful 64³ × 5 cm).
Lidar input clipped to cube + augmented; GT also clipped to same cube.

Usage:
    python train_v6.py --encoder hoeller --steps 5000 --out_dir runs/hoeller_v6
    python train_v6.py --encoder pointnet --steps 5000 --out_dir runs/pointnet_v6
    python train_v6.py --encoder ptv3 --steps 5000 --out_dir runs/ptv3_v6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as spconv

# local
from dataset import LidarFrameDataset, build_split, save_split, load_split
from augment import augment_lidar


GRID = 64
VOXEL_SIZE = 0.05
HALF = GRID * VOXEL_SIZE / 2  # 1.6 m close-range cube


# ============================ encoders =================================================


def _conv_block_sparse(in_c, out_c, indice_key):
    g = max(1, min(8, out_c // 4))
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_c, out_c, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
        nn.GroupNorm(g, out_c), nn.ReLU(inplace=True),
        spconv.SubMConv3d(out_c, out_c, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
        nn.GroupNorm(g, out_c), nn.ReLU(inplace=True),
    )


class HoellerEncoder(nn.Module):
    """Sparse 3D U-Net encoder: 4 down-conv levels, paper-style."""

    def __init__(self, in_c=4, base_c=16, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        self.enc_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        c_in = in_c
        for level in range(num_levels):
            c_out = base_c * (2 ** level)
            self.enc_blocks.append(_conv_block_sparse(c_in, c_out, f"enc_{level}"))
            self.down_convs.append(
                spconv.SparseConv3d(c_out, c_out * 2, kernel_size=2, stride=2, bias=False,
                                    indice_key=f"down_{level}")
            )
            c_in = c_out * 2
        self.bottleneck = _conv_block_sparse(c_in, c_in, "bottleneck")
        self.out_c = c_in

    def forward(self, x: spconv.SparseConvTensor) -> tuple[spconv.SparseConvTensor, list[spconv.SparseConvTensor]]:
        skips = []
        h = x
        for enc, down in zip(self.enc_blocks, self.down_convs):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        return h, skips


class PointNetEncoder(nn.Module):
    """PointNet-lite encoder: 3 mini-blocks with global max-pool features broadcast to voxel grid.

    NOT full PointNet++ (no FPS/grouping) — a lightweight, fast baseline.
    Maps voxel features → enriched voxel features by injecting per-batch global context.
    """

    def __init__(self, in_c=4, base_c=16, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        # operate on sparse tensor features as a "point cloud" — (M, in_c)
        self.mlp1 = nn.Sequential(nn.Linear(in_c, base_c), nn.ReLU(),
                                   nn.Linear(base_c, base_c * 2), nn.ReLU(),
                                   nn.Linear(base_c * 2, base_c * 4))
        # Then sparse-conv path same as Hoeller's, but feed it a richer feature
        c_in = base_c * 4 + in_c
        self.proj = spconv.SubMConv3d(c_in, base_c, kernel_size=1, indice_key="proj")
        self.enc_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        c_in = base_c
        for level in range(num_levels):
            c_out = base_c * (2 ** level)
            self.enc_blocks.append(_conv_block_sparse(c_in, c_out, f"enc_{level}"))
            self.down_convs.append(
                spconv.SparseConv3d(c_out, c_out * 2, kernel_size=2, stride=2, bias=False,
                                    indice_key=f"down_{level}")
            )
            c_in = c_out * 2
        self.bottleneck = _conv_block_sparse(c_in, c_in, "bottleneck")
        self.out_c = c_in

    def forward(self, x: spconv.SparseConvTensor) -> tuple[spconv.SparseConvTensor, list[spconv.SparseConvTensor]]:
        feats = x.features  # (M, in_c)
        # PointNet-style permutation-invariant global feature
        global_feat = self.mlp1(feats)  # (M, 4C)
        # max-pool per batch element
        batch_idx = x.indices[:, 0].long()
        B = x.batch_size
        global_max = torch.zeros(B, global_feat.shape[1], device=feats.device)
        for b in range(B):
            mask = batch_idx == b
            if mask.any():
                global_max[b] = global_feat[mask].max(dim=0).values
        # broadcast back to per-voxel
        global_per_voxel = global_max[batch_idx]
        feats_enriched = torch.cat([feats, global_per_voxel], dim=-1)
        x = x.replace_feature(feats_enriched)
        x = self.proj(x)
        skips = []
        h = x
        for enc, down in zip(self.enc_blocks, self.down_convs):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        return h, skips


class PTv3StubEncoder(nn.Module):
    """PTv3-style stub: window self-attention over voxel features at each level.

    Real PTv3 uses serialized ordering + space-filling curves. This stub uses
    nn.MultiheadAttention over voxels grouped by batch — sufficient for our small
    grids. Replace with the official PTv3 module if installed.
    """

    def __init__(self, in_c=4, base_c=16, num_levels=4, heads=4):
        super().__init__()
        self.num_levels = num_levels
        self.proj_in = spconv.SubMConv3d(in_c, base_c, kernel_size=1, indice_key="proj_in")
        self.enc_blocks = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        c_in = base_c
        for level in range(num_levels):
            c_out = base_c * (2 ** level)
            self.enc_blocks.append(_conv_block_sparse(c_in, c_out, f"enc_{level}"))
            self.attns.append(nn.MultiheadAttention(c_out, heads, dropout=0.0, batch_first=True))
            self.down_convs.append(
                spconv.SparseConv3d(c_out, c_out * 2, kernel_size=2, stride=2, bias=False,
                                    indice_key=f"down_{level}")
            )
            c_in = c_out * 2
        self.bottleneck = _conv_block_sparse(c_in, c_in, "bottleneck")
        self.out_c = c_in

    def _attend(self, x: spconv.SparseConvTensor, attn: nn.MultiheadAttention,
                 max_voxels: int = 4096) -> spconv.SparseConvTensor:
        feats = x.features  # (M, C)
        batch_idx = x.indices[:, 0].long()
        B = x.batch_size
        out = torch.zeros_like(feats)
        for b in range(B):
            sel = batch_idx == b
            n = int(sel.sum().item())
            if n == 0:
                continue
            f = feats[sel]
            if n > max_voxels:
                # random subsample to fit attention memory
                idx = torch.randperm(n, device=f.device)[:max_voxels]
                f_sub = f[idx]
                attended, _ = attn(f_sub.unsqueeze(0), f_sub.unsqueeze(0), f_sub.unsqueeze(0))
                # scatter back: replace subsampled positions, leave others as-is
                target = f.clone()
                target[idx] = attended.squeeze(0)
                out[sel] = target
            else:
                attended, _ = attn(f.unsqueeze(0), f.unsqueeze(0), f.unsqueeze(0))
                out[sel] = attended.squeeze(0)
        return x.replace_feature(out)

    def forward(self, x: spconv.SparseConvTensor):
        x = self.proj_in(x)
        skips = []
        h = x
        for enc, attn, down in zip(self.enc_blocks, self.attns, self.down_convs):
            h = enc(h)
            h = self._attend(h, attn)  # window-self-attn at this level
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        return h, skips


# ============================ shared decoder ============================================


class SparseDecoder(nn.Module):
    """Symmetric sparse-conv decoder with skip connections + per-level occ heads."""

    def __init__(self, base_c=16, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        in_c = base_c * (2 ** num_levels)
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.occ_heads = nn.ModuleList()
        for level in reversed(range(num_levels)):
            skip_c = base_c * (2 ** level)
            self.up_convs.append(
                spconv.SparseInverseConv3d(in_c, skip_c, kernel_size=2, bias=False,
                                            indice_key=f"down_{level}")
            )
            self.dec_blocks.append(_conv_block_sparse(skip_c * 2, skip_c, f"dec_{level}"))
            self.occ_heads.append(spconv.SubMConv3d(skip_c, 1, kernel_size=1, indice_key=f"occ_{level}"))
            in_c = skip_c

    def forward(self, h: spconv.SparseConvTensor, skips: list):
        outs = []
        finest_feat = None
        for i, (up, dec, head, skip) in enumerate(zip(
            self.up_convs, self.dec_blocks, self.occ_heads, reversed(skips)
        )):
            h = up(h)
            cat_feats = torch.cat([h.features, skip.features], dim=1)
            h = h.replace_feature(cat_feats)
            h = dec(h)
            outs.append(head(h))
            if i == self.num_levels - 1:  # finest level — keep features
                finest_feat = h
        return outs, finest_feat


class DenseFinalHead(nn.Module):
    """Convert finest-level sparse features to a dense 64³ occupancy logit map.

    Lets the model emit logits at voxels with no sparse support — required to
    predict GT in the lidar blind cone and to escape the sparse-decoder support
    ceiling (~24% of GT voxels are reachable by SubMConv occ heads).
    """

    def __init__(self, in_c: int):
        super().__init__()
        g = max(1, min(8, in_c // 4))
        self.net = nn.Sequential(
            nn.Conv3d(in_c, in_c, kernel_size=3, padding=1),
            nn.GroupNorm(g, in_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_c, in_c // 2, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, in_c // 8), in_c // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_c // 2, 1, kernel_size=1),
        )

    def forward(self, finest_sp: spconv.SparseConvTensor) -> torch.Tensor:
        # spconv .dense() pads absent voxels with zero features — we then run
        # dense conv, so absent voxels still produce real (nonzero) logits.
        d = finest_sp.dense()
        return self.net(d)


class FullModel(nn.Module):
    def __init__(self, encoder: nn.Module, base_c=16, num_levels=4, dense_head: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = SparseDecoder(base_c=base_c, num_levels=num_levels)
        self.dense_head = DenseFinalHead(base_c) if dense_head else None

    def forward(self, x: spconv.SparseConvTensor):
        h, skips = self.encoder(x)
        outs, finest_feat = self.decoder(h, skips)
        dense_logits = self.dense_head(finest_feat) if self.dense_head is not None else None
        return outs, dense_logits


# ============================ voxelization ==============================================


def voxelize_pcl_local(points_l: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (coords (M, 3) int32, offsets (M, 3) sub-voxel)."""
    valid = mask.bool() & torch.isfinite(points_l).all(-1)
    p = points_l[valid]
    in_box = ((p > -HALF) & (p < HALF)).all(-1)
    p = p[in_box]
    if p.shape[0] == 0:
        coords = torch.tensor([[GRID // 2, GRID // 2, GRID // 2]], dtype=torch.int32, device=points_l.device)
        offsets = torch.zeros(1, 3, device=points_l.device, dtype=points_l.dtype)
        return coords, offsets
    idx = ((p + HALF) / VOXEL_SIZE).long().clamp_(0, GRID - 1)
    centers = (idx.float() + 0.5) * VOXEL_SIZE - HALF
    offsets = (p - centers) / VOXEL_SIZE
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


def world_to_local_batch(pts_w: torch.Tensor, origin_w: torch.Tensor, quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Transform (B, N, 3) world points to (B, N, 3) local (lidar) frame."""
    R = quat_xyzw_to_R(quat_xyzw)  # (B, 3, 3)
    delta = pts_w - origin_w.unsqueeze(1)
    return torch.einsum("bij,bnj->bni", R.transpose(-2, -1), delta)


# ============================ training step =============================================


def build_input_tensor(coords_list, offsets_list, batch_size, device):
    feats = []
    coords = []
    for b, (c, o) in enumerate(zip(coords_list, offsets_list)):
        feats.append(torch.cat([o, torch.zeros(c.shape[0], 1, device=o.device, dtype=o.dtype)], dim=-1))
        coords.append(torch.cat([torch.full((c.shape[0], 1), b, device=c.device, dtype=c.dtype), c], dim=-1))
    feats = torch.cat(feats, dim=0).contiguous()
    coords = torch.cat(coords, dim=0).contiguous()
    return spconv.SparseConvTensor(feats, coords, spatial_shape=[GRID, GRID, GRID], batch_size=batch_size)


def sparse_to_dense_logits(occ_sp, fill: float = -20.0):
    """Convert SparseConvTensor to dense logits with explicit `fill` for absent voxels.
    spconv's .dense() pads with 0 → sigmoid 0.5 → IoU at low thresholds is broken.
    """
    bs = occ_sp.batch_size
    sh = occ_sp.spatial_shape
    feats = occ_sp.features
    out = torch.full((bs, 1, *sh), fill, device=feats.device, dtype=feats.dtype)
    ind = occ_sp.indices.long()
    if ind.shape[0] > 0:
        out[ind[:, 0], 0, ind[:, 1], ind[:, 2], ind[:, 3]] = feats[:, 0]
    return out


def train_step(model, batch, device, *, augment=True, pos_weight=3.0,
               mask_to_lidar=False, lidar_dilate=3, return_iou=False, iou_thresh=0.5):
    bs = batch["lidar_pcl"].shape[0]
    pcl = batch["lidar_pcl"].to(device)
    mask = batch["lidar_mask"].to(device)
    origin = batch["lidar_origin"].to(device)
    quat = batch["lidar_quat"].to(device)
    base = batch["base_pos"].to(device)
    base_quat = batch["base_quat"].to(device)
    gt_pcl = batch["gt_pcl"].to(device)
    gt_n = batch["gt_n"].to(device)

    if augment:
        pcl, mask, origin = augment_lidar(pcl, mask, origin, quat)

    # transform input + GT into base-local frame (cube centered on base)
    pcl_local = world_to_local_batch(pcl, base, base_quat)
    # GT — only valid prefix per batch
    coords_in_list, offsets_in_list = [], []
    target_dense = torch.zeros((bs, 1, GRID, GRID, GRID), device=device)
    lidar_dense = torch.zeros((bs, 1, GRID, GRID, GRID), device=device)
    for b in range(bs):
        c_in, o_in = voxelize_pcl_local(pcl_local[b], mask[b])
        coords_in_list.append(c_in)
        offsets_in_list.append(o_in)
        if c_in.shape[0] > 0:
            lidar_dense[b, 0, c_in[:, 0], c_in[:, 1], c_in[:, 2]] = 1.0
        n = int(gt_n[b].item())
        if n > 0:
            gt_b = gt_pcl[b, :n]
            gt_b_local = (gt_b - base[b]).unsqueeze(0)  # ignore base orientation for cube alignment
            gt_b_local = quat_xyzw_to_R(base_quat[b]).T @ gt_b_local.squeeze(0).T
            gt_b_local = gt_b_local.T
            in_box = ((gt_b_local > -HALF) & (gt_b_local < HALF)).all(-1)
            gt_b_local = gt_b_local[in_box]
            if gt_b_local.shape[0] > 0:
                gi = ((gt_b_local + HALF) / VOXEL_SIZE).long().clamp_(0, GRID - 1)
                target_dense[b, 0, gi[:, 0], gi[:, 1], gi[:, 2]] = 1.0

    # reachable = lidar voxels dilated by lidar_dilate (in voxels) — supervise
    # only inside this region so model is not penalized in the blind cone.
    if mask_to_lidar:
        k = lidar_dilate * 2 + 1
        reachable = F.max_pool3d(lidar_dense, kernel_size=k, stride=1, padding=lidar_dilate) > 0.5

    sp_in = build_input_tensor(coords_in_list, offsets_in_list, bs, device)

    outs, dense_logits = model(sp_in)  # list of sparse occ logits + optional dense
    # use only finest-resolution head for primary loss; coarser as aux
    pw = torch.tensor([pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pw)

    loss = torch.zeros((), device=device)
    for i, occ_sp in enumerate(outs):
        # for each level, build a dense mask of pred coords + target
        spatial = occ_sp.spatial_shape
        Bsz = occ_sp.batch_size
        occ_dense = occ_sp.dense()
        # downsample target_dense to this level via avg pool
        s = GRID // spatial[0]
        if s > 1:
            tgt = F.avg_pool3d(target_dense, kernel_size=s, stride=s)
            tgt = (tgt > 0.0).float()
        else:
            tgt = target_dense
        ind = occ_sp.indices.long()
        pred_mask = torch.zeros((Bsz, 1, *spatial), dtype=torch.bool, device=device)
        if ind.shape[0] > 0:
            pred_mask[ind[:, 0], 0, ind[:, 1], ind[:, 2], ind[:, 3]] = True
        valid = pred_mask | (tgt > 0.5)
        if mask_to_lidar:
            r = F.max_pool3d(reachable.float(), kernel_size=s, stride=s) > 0.5 if s > 1 else reachable
            valid = valid & r
        if valid.any():
            loss = loss + bce(occ_dense[valid], tgt[valid])

    # Dense head loss — every voxel in the cube (or reachable region) gets gradient
    if dense_logits is not None:
        if mask_to_lidar:
            valid_d = reachable
            if valid_d.any():
                loss = loss + bce(dense_logits[valid_d], target_dense[valid_d])
        else:
            loss = loss + bce(dense_logits, target_dense)

    if return_iou:
        # IoU: prefer dense_logits if available, else use sparse (fill=-20)
        with torch.no_grad():
            if dense_logits is not None:
                finest = dense_logits
            else:
                finest = sparse_to_dense_logits(outs[-1], fill=-20.0)
            pm = finest.sigmoid() > iou_thresh
            tm = target_dense > 0.5
            if mask_to_lidar:
                pm = pm & reachable
                tm = tm & reachable
            inter = (pm & tm).sum().float()
            union = (pm | tm).sum().float()
            iou = (inter / union).item() if union.item() > 0 else float("nan")
        return loss, iou
    return loss


def val_iou_pass(model, val_loader, device, n_batches, *, mask_to_lidar=False,
                 lidar_dilate=3, iou_thresh=0.5):
    """Compute mean IoU on val batches (no augment, no grad)."""
    model.eval()
    ious = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            _, iou = train_step(model, batch, device, augment=False,
                                mask_to_lidar=mask_to_lidar, lidar_dilate=lidar_dilate,
                                return_iou=True, iou_thresh=iou_thresh)
            if iou == iou:
                ious.append(iou)
    model.train()
    return float(np.mean(ious)) if ious else float("nan")


# ============================ main ====================================================


def make_encoder(name: str):
    if name == "hoeller":
        return HoellerEncoder()
    if name == "pointnet":
        return PointNetEncoder()
    if name == "ptv3":
        return PTv3StubEncoder()
    raise ValueError(f"unknown encoder {name}")


def collate(batch_list):
    # custom collate — stack tensors, keep gt_pcl as-is (already padded)
    keys = batch_list[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch_list], dim=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="/root/lidar_dataset_v6")
    ap.add_argument("--encoder", choices=["hoeller", "pointnet", "ptv3"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--dense_head", action="store_true",
                    help="add dense 64³ final head — lets model emit logits in blind cone")
    ap.add_argument("--mask_to_lidar", action="store_true",
                    help="restrict loss to voxels near lidar hits (skip blind cone)")
    ap.add_argument("--lidar_dilate", type=int, default=3,
                    help="dilation radius in voxels for lidar reachability mask")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=100,
                    help="run val IoU every N steps")
    ap.add_argument("--eval_batches", type=int, default=8,
                    help="number of val batches to average IoU over")
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    split_path = os.path.join(args.dataset_dir, "split.json")
    if os.path.exists(split_path):
        train_set, val_set = load_split(split_path)
        print(f"[INFO] loaded existing split: train={len(train_set)} val={len(val_set)} tiles", flush=True)
    else:
        train_set, val_set = build_split(args.dataset_dir, val_frac=0.2, seed=0)
        save_split(train_set, val_set, split_path)
        print(f"[INFO] new split: train={len(train_set)} val={len(val_set)} tiles → {split_path}", flush=True)

    train_ds = LidarFrameDataset(args.dataset_dir, train_set)
    val_ds = LidarFrameDataset(args.dataset_dir, val_set)
    print(f"[INFO] train frames: {len(train_ds)}  val frames: {len(val_ds)}", flush=True)

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate,
    )

    device = torch.device(args.device)
    encoder = make_encoder(args.encoder)
    model = FullModel(encoder, dense_head=args.dense_head).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] {args.encoder} model params: {n_params/1e6:.2f} M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    log_path = os.path.join(args.out_dir, "train.log")
    with open(log_path, "w") as f:
        f.write("step,loss,iou_train,iou_val,lr,elapsed_s\n")
    with open(os.path.join(args.out_dir, "hparams.json"), "w") as f:
        json.dump({**vars(args), "n_params": n_params}, f, indent=2)

    t0 = time.time()
    step = 0
    iterator = iter(loader)
    while step < args.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        opt.zero_grad()
        do_iou = (step + 1) % args.log_every == 0
        out = train_step(model, batch, device, augment=not args.no_aug,
                         mask_to_lidar=args.mask_to_lidar,
                         lidar_dilate=args.lidar_dilate,
                         return_iou=do_iou)
        loss, iou_train = (out if do_iou else (out, float("nan")))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        iou_val = float("nan")
        if step % args.eval_every == 0:
            iou_val = val_iou_pass(model, val_loader, device, args.eval_batches,
                                   mask_to_lidar=args.mask_to_lidar,
                                   lidar_dilate=args.lidar_dilate)
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            line = f"{step},{loss.item():.4f},{iou_train:.4f},{iou_val:.4f},{sched.get_last_lr()[0]:.6e},{elapsed:.1f}"
            print(f"[step {step:5d}/{args.steps}] loss={loss.item():.4f} iou_tr={iou_train:.3f} iou_val={iou_val:.3f} lr={sched.get_last_lr()[0]:.2e} elapsed={elapsed:.1f}s", flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        if step % args.save_every == 0:
            ck = os.path.join(args.out_dir, f"ckpt_{step:06d}.pt")
            torch.save({"step": step, "model": model.state_dict(), "loss": loss.item()}, ck)
            print(f"[CKPT] {ck}", flush=True)

    print(f"[DONE] {args.steps} steps in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
