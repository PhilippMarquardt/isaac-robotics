"""Export model predictions vs GT vs lidar input as a colored PLY for CloudCompare review.

Loads a trained checkpoint, runs inference on N val frames, writes one PLY per frame
containing:
  - Blue (60,120,255): lidar input pcl (in world frame)
  - Orange (220,170,90): GT terrain pcl (clipped to 3.2 m cube around base)
  - Green (40,200,80): model prediction voxels >0.5 sigmoid (in world frame)
  - Pale green (180,230,180): model prediction voxels 0.3-0.5 sigmoid

Usage:
    python export_pred_ply.py --ckpt /path/to/ckpt.pt --encoder pointnet \\
        --dataset_dir /root/lidar_dataset_v6 --n_samples 3 --out_dir /tmp/preds
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import torch

import spconv.pytorch as spconv

from dataset import LidarFrameDataset, load_split
from train_v6 import (FullModel, make_encoder,
                       voxelize_pcl_local, world_to_local_batch, build_input_tensor,
                       quat_xyzw_to_R, GRID, VOXEL_SIZE, HALF, collate)


def write_ply(path: str, points: np.ndarray, colors: np.ndarray):
    n = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    body = np.empty(n, dtype=[("x","<f4"),("y","<f4"),("z","<f4"),
                               ("r","u1"),("g","u1"),("b","u1")])
    body["x"] = points[:, 0].astype(np.float32)
    body["y"] = points[:, 1].astype(np.float32)
    body["z"] = points[:, 2].astype(np.float32)
    body["r"] = colors[:, 0]; body["g"] = colors[:, 1]; body["b"] = colors[:, 2]
    with open(path, "wb") as f:
        f.write(header); f.write(body.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoder", required=True, choices=["hoeller", "pointnet", "ptv3"])
    ap.add_argument("--dataset_dir", default="/root/lidar_dataset_v6")
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    encoder = make_encoder(args.encoder)
    model = FullModel(encoder).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[INFO] loaded {args.ckpt} step={ck.get('step', '?')}", flush=True)

    _, val_tiles = load_split(os.path.join(args.dataset_dir, "split.json"))
    val_ds = LidarFrameDataset(args.dataset_dir, val_tiles)
    print(f"[INFO] val frames: {len(val_ds)}", flush=True)
    sample_indices = np.random.permutation(len(val_ds))[:args.n_samples]

    with torch.no_grad():
        for k, idx in enumerate(sample_indices):
            batch = collate([val_ds[int(idx)]])
            pcl = batch["lidar_pcl"].to(device)
            mask = batch["lidar_mask"].to(device)
            origin = batch["lidar_origin"].to(device)
            quat = batch["lidar_quat"].to(device)
            base = batch["base_pos"].to(device)
            base_quat = batch["base_quat"].to(device)
            gt_pcl = batch["gt_pcl"].to(device)
            gt_n = batch["gt_n"].to(device)

            # voxelize input
            pcl_local = world_to_local_batch(pcl, base, base_quat)
            c, o = voxelize_pcl_local(pcl_local[0], mask[0])
            sp_in = build_input_tensor([c], [o], 1, device)
            outs, dense_logits = model(sp_in)
            if dense_logits is not None:
                finest = dense_logits.sigmoid().squeeze()
            else:
                finest = outs[-1].dense().sigmoid().squeeze()  # (D, H, W)

            # extract predictions in world frame
            R = quat_xyzw_to_R(base_quat[0])
            base_w = base[0]

            pts_list = []
            cols_list = []

            # lidar input (world frame, blue)
            lid_w = pcl[0][mask[0]].cpu().numpy()
            pts_list.append(lid_w)
            cols_list.append(np.tile(np.array([60,120,255], np.uint8), (lid_w.shape[0], 1)))

            # GT (clip to 3.2m cube around base, orange)
            n = int(gt_n[0].item())
            if n > 0:
                gt_w = gt_pcl[0, :n]
                gt_local = (R.T @ (gt_w - base_w).T).T
                in_box = ((gt_local > -HALF) & (gt_local < HALF)).all(-1)
                gt_box = gt_w[in_box].cpu().numpy()
                pts_list.append(gt_box)
                cols_list.append(np.tile(np.array([220,170,90], np.uint8), (gt_box.shape[0], 1)))

            # predictions (only confident >0.5, green)
            high = (finest > 0.5).nonzero(as_tuple=False)
            if high.shape[0] > 0:
                local_pts = (high.float() + 0.5) * VOXEL_SIZE - HALF
                world_pts = (R @ local_pts.T).T + base_w
                world_pts = world_pts.cpu().numpy()
                pts_list.append(world_pts)
                cols_list.append(np.tile(np.array([40,200,80], np.uint8), (world_pts.shape[0], 1)))

            all_pts = np.concatenate(pts_list, axis=0)
            all_cols = np.concatenate(cols_list, axis=0)
            out = os.path.join(args.out_dir, f"pred_{args.encoder}_frame{int(idx):05d}.ply")
            write_ply(out, all_pts, all_cols)
            n_high = int(high.shape[0])
            print(f"[SAMPLE {k+1}/{args.n_samples}] frame_idx={idx} lidar={lid_w.shape[0]} "
                  f"gt_in_cube={pts_list[1].shape[0] if len(pts_list)>1 else 0} "
                  f"pred>0.5={n_high} → {out}", flush=True)


if __name__ == "__main__":
    main()
