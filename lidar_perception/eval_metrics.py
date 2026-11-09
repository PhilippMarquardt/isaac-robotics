"""Quantitative evaluation of a trained Hoeller reconstruction model.

Computes per-trajectory:
  - voxel IoU (predicted occupancy vs GT occupancy at finest resolution)
  - chamfer distance (between predicted occupied voxel centers and GT pcl)

Aggregates over a held-out validation split and writes a results JSON.

Usage:
    python eval_metrics.py \\
        --ckpt /workspace/isaac-robotics/evals/lidar_recon/baseline_hoeller/ckpt_002000.pt \\
        --dataset_dir /workspace/lidar_dataset_v2 \\
        --val_split 0.1 \\
        --rollout 12 \\
        --out /workspace/isaac-robotics/evals/lidar_recon/baseline_hoeller/eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch

import hoeller_unet
from hoeller_unet import GRID, HALF, VOXEL_SIZE, build_model

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from train_hoeller import (LidarChunkIndex, _voxelize_step, _voxelize_gt_step,
                           world_to_local, local_to_world)


def _voxel_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """IoU on dense (D, H, W) boolean masks."""
    inter = (pred_mask & gt_mask).sum().float()
    union = (pred_mask | gt_mask).sum().float()
    if union == 0:
        return float("nan")
    return float(inter / union)


def _multi_threshold_iou(pred_dense: torch.Tensor, gt_mask: torch.Tensor,
                         thresholds: tuple = (0.1, 0.2, 0.3, 0.5, 0.7)) -> dict:
    """IoU at multiple sigmoid thresholds — useful early in training when
    most logits are still negative."""
    out = {}
    for th in thresholds:
        out[f"iou_t{th:.2f}"] = _voxel_iou(pred_dense > th, gt_mask)
    return out


def _chamfer_l2(pred_pts: torch.Tensor, gt_pts: torch.Tensor, max_pts: int = 4096) -> float:
    """Mean of (forward + backward) nearest-neighbour distance, L2."""
    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return float("nan")
    if pred_pts.shape[0] > max_pts:
        idx = torch.randperm(pred_pts.shape[0])[:max_pts]
        pred_pts = pred_pts[idx]
    if gt_pts.shape[0] > max_pts:
        idx = torch.randperm(gt_pts.shape[0])[:max_pts]
        gt_pts = gt_pts[idx]
    d = torch.cdist(pred_pts.unsqueeze(0), gt_pts.unsqueeze(0)).squeeze(0)
    fwd = d.min(dim=1).values.mean()
    bwd = d.min(dim=0).values.mean()
    return float(fwd + bwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--rollout", type=int, default=12)
    ap.add_argument("--max_trajs", type=int, default=50, help="cap how many val trajs to score")
    ap.add_argument("--downsample", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="eval.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"[INFO] indexing dataset...", flush=True)
    idx = LidarChunkIndex(args.dataset_dir)
    n = len(idx)
    n_val = max(1, int(n * args.val_split))
    val_idxs = list(range(n - n_val, n))[: args.max_trajs]
    print(f"[INFO] {n} trajectories, {len(val_idxs)} held-out for val", flush=True)

    device = torch.device(args.device)
    model = build_model().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] loaded ckpt step={ckpt.get('step', '?')}", flush=True)

    ious = []
    chamfers = []
    multi_iou = {f"iou_t{th:.2f}": [] for th in (0.1, 0.2, 0.3, 0.5, 0.7)}
    sigmoid_max_per_step = []
    n_valid_steps = 0

    with torch.no_grad():
        for ti in val_idxs:
            d = idx.sample(ti, args.rollout, downsample=args.downsample)
            prev_pts_local = None
            prev_origin = None
            prev_quat = None
            for t in range(args.rollout):
                cc, oo = _voxelize_step(d, t, device)
                if prev_pts_local is not None:
                    cur_origin = d["lidar_origin_world"][t].to(device)
                    cur_quat = d["lidar_quat_world"][t].to(device)
                    pts_w = local_to_world(prev_pts_local.unsqueeze(0), prev_origin.unsqueeze(0),
                                           prev_quat.unsqueeze(0)).squeeze(0)
                    pts_l_cur = world_to_local(pts_w.unsqueeze(0), cur_origin.unsqueeze(0),
                                               cur_quat.unsqueeze(0)).squeeze(0)
                    cc_prev, oo_prev = hoeller_unet.voxelize_pcl_local(
                        pts_l_cur, torch.ones(pts_l_cur.shape[0], dtype=torch.bool, device=device)
                    )
                    cc_all = torch.cat([cc, cc_prev], dim=0)
                    oo_all = torch.cat([oo, oo_prev], dim=0)
                    flags = torch.cat([torch.zeros(cc.shape[0], 1, device=device),
                                       torch.ones(cc_prev.shape[0], 1, device=device)], dim=0)
                else:
                    cc_all, oo_all = cc, oo
                    flags = torch.zeros(cc.shape[0], 1, device=device)

                feats = torch.cat([oo_all, flags], dim=-1)
                coords = torch.cat([torch.zeros(cc_all.shape[0], 1, device=device, dtype=cc_all.dtype),
                                    cc_all], dim=-1)
                sp_in = hoeller_unet.spconv.SparseConvTensor(
                    feats, coords, spatial_shape=[GRID, GRID, GRID], batch_size=1
                )
                outs = model(sp_in)
                pred_dense = outs[-1][0].dense().sigmoid().squeeze()  # (D, H, W)
                pred_mask = pred_dense > 0.5

                gc, _ = _voxelize_gt_step(d, t, device)
                gt_mask = torch.zeros((GRID, GRID, GRID), dtype=torch.bool, device=device)
                gt_mask[gc[:, 0].long(), gc[:, 1].long(), gc[:, 2].long()] = True

                ious.append(_voxel_iou(pred_mask, gt_mask))
                multi = _multi_threshold_iou(pred_dense, gt_mask)
                for k, v in multi.items():
                    multi_iou[k].append(v)
                sigmoid_max_per_step.append(float(pred_dense.max()))

                pred_coords = torch.nonzero(pred_mask, as_tuple=False).float()
                pred_pts = (pred_coords + 0.5) * VOXEL_SIZE - HALF
                gt_pts = (gc.float() + 0.5) * VOXEL_SIZE - HALF
                chamfers.append(_chamfer_l2(pred_pts, gt_pts))
                n_valid_steps += 1

                # autoregressive feedback
                prev_pts_local = pred_pts
                prev_origin = d["lidar_origin_world"][t].to(device)
                prev_quat = d["lidar_quat_world"][t].to(device)

    iou_arr = torch.tensor([x for x in ious if x == x])  # filter nan
    chamfer_arr = torch.tensor([x for x in chamfers if x == x])
    results = {
        "ckpt": args.ckpt,
        "n_trajs": len(val_idxs),
        "n_steps": n_valid_steps,
        "voxel_iou_mean": float(iou_arr.mean()) if iou_arr.numel() > 0 else float("nan"),
        "voxel_iou_std": float(iou_arr.std()) if iou_arr.numel() > 1 else 0.0,
        "chamfer_l2_mean": float(chamfer_arr.mean()) if chamfer_arr.numel() > 0 else float("nan"),
        "chamfer_l2_std": float(chamfer_arr.std()) if chamfer_arr.numel() > 1 else 0.0,
        "sigmoid_max_mean": (sum(sigmoid_max_per_step) / len(sigmoid_max_per_step)
                              if sigmoid_max_per_step else float("nan")),
    }
    for k, vs in multi_iou.items():
        vs_clean = [v for v in vs if v == v]
        results[k] = (sum(vs_clean) / len(vs_clean)) if vs_clean else float("nan")

    print(f"[RESULTS] {json.dumps(results, indent=2)}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] {args.out}", flush=True)


if __name__ == "__main__":
    main()
