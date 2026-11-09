"""Trivial baselines for temporal lidar reconstruction — no model, just IoU
between simple input-derived voxel masks and GT.

Measures:
  - copy_current:    IoU(current_lidar voxels, GT) — the "model just copies" floor
  - copy_any_hist:   IoU(any-frame lidar voxels, GT) — naive temporal union
  - copy_current_dilate1: IoU(current_lidar dilated ±1, GT) — soft floor
  - copy_any_hist_dilate1: same for history union

Reports both strict IoU and ±1-voxel tolerant. Run on val split (trajectory-level).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_temporal import TemporalLidarDataset, temporal_collate
from model_temporal_unet import aggregate_history_to_dense, gt_to_dense, GRID, HALF
from train_temporal_unet import split_entries_by_trajectory, voxel_iou, blind_recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="/root/lidar_dataset_fwd")
    ap.add_argument("--history_n", type=int, default=10)
    ap.add_argument("--history_stride", type=int, default=5)
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--lidar_dilate", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    full = TemporalLidarDataset(args.dataset_dir, tiles=None,
                                history_n=args.history_n, history_stride=args.history_stride,
                                max_lidar_pts=20000, max_gt_pts=200000)
    train_idx, val_idx = split_entries_by_trajectory(full.entries, val_frac=0.2, seed=args.seed)
    val_ds = TemporalLidarDataset.__new__(TemporalLidarDataset)
    val_ds.__dict__.update(full.__dict__)
    val_ds.entries = [full.entries[i] for i in val_idx]
    print(f"[INFO] val entries: {len(val_ds)}", flush=True)

    loader = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False,
                                          num_workers=0, collate_fn=temporal_collate)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics = {k: [] for k in [
        "copy_current_iou", "copy_anyhist_iou",
        "copy_current_iou_tol1", "copy_anyhist_iou_tol1",
        "copy_current_iou_reach", "copy_anyhist_iou_reach",
        "copy_current_blind_recall", "copy_anyhist_blind_recall",
    ]}

    n_done = 0
    with torch.no_grad():
        for batch in loader:
            if n_done >= args.n_eval:
                break
            pcl = batch["lidar_pcl"].to(device)
            msk = batch["lidar_mask"].to(device)
            base_pos = batch["base_pos"].to(device)
            base_quat = batch["base_quat"].to(device)
            age_s = batch["frame_age_s"].to(device)
            valid_history = batch["valid_history"].to(device)
            gt_pcl = batch["gt_pcl"].to(device)
            gt_n = batch["gt_n"].to(device)
            cur_idx = pcl.shape[1] - 1

            x = aggregate_history_to_dense(pcl, msk, base_pos[:, cur_idx], base_quat[:, cur_idx],
                                            age_s, valid_history)
            target = gt_to_dense(gt_pcl, gt_n, base_pos[:, cur_idx], base_quat[:, cur_idx])
            current_occ = x[:, 0:1]
            any_hist = x[:, 1:2]
            cur_or_hist = ((current_occ > 0.5) | (any_hist > 0.5)).float()

            # convert "occ mask" to fake-logit (large +/-) so voxel_iou's sigmoid > 0.5 trick works
            def mask_as_logits(m):
                return (m * 40.0 - 20.0)  # 1.0->+20, 0.0->-20

            cur_logits = mask_as_logits(current_occ)
            any_logits = mask_as_logits(cur_or_hist)

            k = args.lidar_dilate * 2 + 1
            reach = F.max_pool3d(current_occ, kernel_size=k, stride=1, padding=args.lidar_dilate) > 0.5

            metrics["copy_current_iou"].append(voxel_iou(cur_logits, target, 0.5))
            metrics["copy_anyhist_iou"].append(voxel_iou(any_logits, target, 0.5))
            metrics["copy_current_iou_tol1"].append(voxel_iou(cur_logits, target, 0.5, tolerant=1))
            metrics["copy_anyhist_iou_tol1"].append(voxel_iou(any_logits, target, 0.5, tolerant=1))
            metrics["copy_current_iou_reach"].append(voxel_iou(cur_logits, target, 0.5, region=reach))
            metrics["copy_anyhist_iou_reach"].append(voxel_iou(any_logits, target, 0.5, region=reach))
            metrics["copy_current_blind_recall"].append(blind_recall(cur_logits, target, reach, 0.5))
            metrics["copy_anyhist_blind_recall"].append(blind_recall(any_logits, target, reach, 0.5))
            n_done += pcl.shape[0]

    print(f"[INFO] evaluated {n_done} val frames", flush=True)
    print()
    print("Trivial-baseline metrics on val (no model):")
    print(f"  {'metric':<30} {'mean':>8}")
    for k, vs in metrics.items():
        vs = [v for v in vs if v == v]
        if vs:
            print(f"  {k:<30} {np.mean(vs):>8.4f}")
        else:
            print(f"  {k:<30} {'nan':>8}")


if __name__ == "__main__":
    main()
