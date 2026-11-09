"""Export PLY of temporal-UNet PREDICTIONS vs lidar input vs GT.

For each picked val sample: run inference once, voxelize, write a single PLY
in the BASE-LOCAL frame at time t (everything centered at the red pelvis sphere).

Voxel colors (5 cm grid inside 3.2 m cube):
  blue  (60,120,255)   current lidar (input channel 0)
  cyan  (140,200,255)  history union (any past lidar) — input channel 1
  orange(220,170,90)   GT clipped to cube
  green (40,200,80)    pred > 0.5
  palegrn(180,230,180) pred 0.3-0.5

Markers:
  red sphere     pelvis at time t (origin of local frame)
  yellow sphere  lidar mount (+0.65 m above pelvis)
  green wire     3.2 m training cube outline

Usage:
    python viz_temporal_pred.py --run_dir /workspace/.../temporal_diverse_big_s15 \\
        --dataset_dir /root/lidar_dataset_diverse_big \\
        --indices 100 5000 30000 --out_dir /tmp/pred_viz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_temporal import TemporalLidarDataset, temporal_collate
from model_temporal_unet import (TemporalUNet, aggregate_history_to_dense, gt_to_dense,
                                  GRID, VOXEL_SIZE, HALF, N_INPUT_CHANNELS)
from train_temporal_unet import split_entries_by_trajectory


def write_ply(path, pts, cols):
    n = pts.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    body = np.empty(n, dtype=[("x","<f4"),("y","<f4"),("z","<f4"),
                              ("r","u1"),("g","u1"),("b","u1")])
    body["x"]=pts[:,0].astype(np.float32); body["y"]=pts[:,1].astype(np.float32); body["z"]=pts[:,2].astype(np.float32)
    body["r"]=cols[:,0]; body["g"]=cols[:,1]; body["b"]=cols[:,2]
    with open(path, "wb") as f:
        f.write(header); f.write(body.tobytes())


def cube_wire(half, n_per_edge=120):
    h = half
    c = np.array([[-h,-h,-h],[h,-h,-h],[h,h,-h],[-h,h,-h],
                  [-h,-h, h],[h,-h, h],[h,h, h],[-h,h, h]], np.float32)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    out=[]
    for a,b in edges:
        t = np.linspace(0,1,n_per_edge).reshape(-1,1)
        out.append(c[a] + t*(c[b]-c[a]))
    return np.concatenate(out, axis=0)


def sphere_pts(center, r, n, rng):
    d = rng.normal(size=(n,3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    rr = r * (rng.random(n)**(1/3)).reshape(-1,1)
    return center + d*rr


def voxel_centers(mask: torch.Tensor) -> np.ndarray:
    """(D,H,W) bool → (M,3) numpy local-frame voxel centers."""
    idx = torch.nonzero(mask, as_tuple=False).cpu().numpy()
    if idx.size == 0:
        return np.zeros((0, 3), np.float32)
    return (idx.astype(np.float32) + 0.5) * VOXEL_SIZE - HALF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="dir containing hparams.json + ckpt_*.pt")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--indices", type=int, nargs="+", required=True)
    ap.add_argument("--ckpt", default=None, help="path to ckpt; default = latest in run_dir")
    ap.add_argument("--use_val_split", action="store_true",
                    help="restrict indices to val-split entries (default: full dataset)")
    ap.add_argument("--mask_observed", action="store_true",
                    help="hide predictions outside the observed region "
                         "(observed = current+history lidar voxels dilated)")
    ap.add_argument("--lidar_dilate", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    h = json.load(open(os.path.join(args.run_dir, "hparams.json")))
    full = TemporalLidarDataset(args.dataset_dir, tiles=None,
                                 history_n=h["history_n"], history_stride=h["history_stride"],
                                 max_lidar_pts=20000, max_gt_pts=200000)
    if args.use_val_split:
        _, val_idx = split_entries_by_trajectory(full.entries, val_frac=0.2, seed=h["seed"])
        full.entries = [full.entries[i] for i in val_idx]
        print(f"[INFO] restricted to {len(full.entries)} val entries", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalUNet(in_c=N_INPUT_CHANNELS, base_c=h["base_c"]).to(device)
    if args.ckpt is None:
        ckpts = sorted([f for f in os.listdir(args.run_dir) if f.startswith("ckpt_")])
        ck_path = os.path.join(args.run_dir, ckpts[-1])
    else:
        ck_path = args.ckpt
    state = torch.load(ck_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[INFO] loaded {ck_path}  (step={state.get('step','?')})", flush=True)

    rng = np.random.default_rng(0)

    for i in args.indices:
        if i < 0 or i >= len(full):
            print(f"[WARN] idx {i} out of range")
            continue
        s = full[i]
        batch = temporal_collate([s])
        with torch.no_grad():
            pcl = batch["lidar_pcl"].to(device)
            msk = batch["lidar_mask"].to(device)
            base_pos = batch["base_pos"].to(device)
            base_quat = batch["base_quat"].to(device)
            age_s = batch["frame_age_s"].to(device)
            valid_history = batch["valid_history"].to(device)
            gt_pcl = batch["gt_pcl"].to(device)
            gt_n = batch["gt_n"].to(device)
            cur_idx = pcl.shape[1] - 1
            base_pos_t = base_pos[:, cur_idx]
            base_quat_t = base_quat[:, cur_idx]
            x = aggregate_history_to_dense(pcl, msk, base_pos_t, base_quat_t, age_s, valid_history)
            target = gt_to_dense(gt_pcl, gt_n, base_pos_t, base_quat_t)
            logits = model(x)
            probs = logits.sigmoid()[0, 0]  # (D, H, W)

        cur_occ_d = (x[0, 0] > 0.5)             # device
        any_hist_d = (x[0, 1] > 0.5)            # device
        observed_d = cur_occ_d | any_hist_d     # observed surface voxels
        if args.mask_observed:
            k = args.lidar_dilate * 2 + 1
            reach = F.max_pool3d(observed_d.unsqueeze(0).unsqueeze(0).float(),
                                  kernel_size=k, stride=1, padding=args.lidar_dilate) > 0.5
            reach_mask = reach.squeeze().bool()
        else:
            reach_mask = torch.ones_like(observed_d, dtype=torch.bool)

        cur_occ = cur_occ_d.cpu()
        any_hist = any_hist_d.cpu()
        any_hist_only = any_hist & ~cur_occ     # past-only voxels
        gt = (target[0, 0] > 0.5).cpu()
        pred_hi_d = probs > 0.5
        pred_lo_d = (probs > 0.3) & (probs <= 0.5)
        if args.mask_observed:
            pred_hi_d = pred_hi_d & reach_mask
            pred_lo_d = pred_lo_d & reach_mask
        pred_hi = pred_hi_d.cpu()
        pred_lo = pred_lo_d.cpu()

        cur_pts = voxel_centers(cur_occ)
        any_pts = voxel_centers(any_hist_only)
        gt_pts = voxel_centers(gt)
        pred_hi_pts = voxel_centers(pred_hi)
        pred_lo_pts = voxel_centers(pred_lo)

        n_lid_in = int(cur_occ.sum())
        n_hist_only = int(any_hist_only.sum())
        n_gt = int(gt.sum())
        n_hi = int(pred_hi.sum())
        n_lo = int(pred_lo.sum())

        # IoU stats for the file's own context
        inter = (pred_hi & gt).sum().item()
        union = (pred_hi | gt).sum().item()
        iou = inter / max(1, union)

        pcs = [
            (cur_pts,   np.array([60,120,255], np.uint8)),
            (any_pts,   np.array([140,200,255], np.uint8)),
            (gt_pts,    np.array([220,170,90],  np.uint8)),
            (pred_lo_pts, np.array([180,230,180], np.uint8)),
            (pred_hi_pts, np.array([40,200,80],   np.uint8)),
            # markers
            (cube_wire(HALF),                       np.array([0,255,0], np.uint8)),
            (sphere_pts(np.zeros(3), 0.08, 400, rng),       np.array([255,0,0], np.uint8)),
            (sphere_pts(np.array([0,0,0.65]), 0.05, 200, rng), np.array([255,255,0], np.uint8)),
        ]
        pts = np.concatenate([p for p,_ in pcs], axis=0)
        cols = np.concatenate([np.tile(c,(p.shape[0],1)) for p,c in pcs], axis=0)
        out_path = os.path.join(args.out_dir, f"pred_{i:05d}.ply")
        write_ply(out_path, pts, cols)
        print(f"[idx={i}] env={int(s['env_id'].item())} ep={int(s['episode_id'].item())} step={int(s['step_in_ep'].item())}  "
              f"lidar={n_lid_in}vx hist_only={n_hist_only}vx gt={n_gt}vx pred>0.5={n_hi}vx pred 0.3-0.5={n_lo}vx  "
              f"iou(@0.5)={iou:.3f}  → {out_path}", flush=True)


if __name__ == "__main__":
    main()
