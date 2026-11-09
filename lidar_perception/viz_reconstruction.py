"""Render trained Hoeller U-Net's reconstruction for qualitative inspection.

Loads a checkpoint, samples a trajectory from the dataset, runs the rollout, and
saves matplotlib 3D scatter plots of:
  - input lidar pcl (lidar frame)
  - GT pcl (lidar frame)
  - predicted occupied voxels (lidar frame, threshold 0.5)

One plot per timestep, saved as PNGs. Useful as a sanity check after training.

Usage:
    python viz_reconstruction.py \\
        --ckpt /workspace/isaac-robotics/evals/lidar_recon/baseline_hoeller/ckpt_002000.pt \\
        --dataset_dir /workspace/lidar_dataset_v2 \\
        --out_dir /workspace/isaac-robotics/evals/lidar_recon/baseline_hoeller/viz \\
        --traj 0 --rollout 6
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hoeller_unet
from hoeller_unet import GRID, HALF, VOXEL_SIZE, build_model

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from train_hoeller import (LidarChunkIndex, _voxelize_step,
                           _voxelize_gt_step, world_to_local, local_to_world)


def _scatter_voxels(ax, coords: torch.Tensor, color: str, label: str, alpha: float = 0.4):
    if coords.shape[0] == 0:
        return
    pts = coords.float().cpu() * VOXEL_SIZE - HALF
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, c=color, alpha=alpha, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--rollout", type=int, default=6)
    ap.add_argument("--downsample", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] indexing dataset...", flush=True)
    idx = LidarChunkIndex(args.dataset_dir)

    device = torch.device(args.device)
    model = build_model().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] loaded ckpt step={ckpt.get('step', '?')}", flush=True)

    d = idx.sample(args.traj, args.rollout, downsample=args.downsample)

    prev_pts_local = None
    prev_origin = None
    prev_quat = None

    for t in range(args.rollout):
        with torch.no_grad():
            cc, oo = _voxelize_step(d, t, device)

            # add prev-frame transformed if available
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
                cc_all = cc
                oo_all = oo
                flags = torch.zeros(cc.shape[0], 1, device=device)

            feats = torch.cat([oo_all, flags], dim=-1)
            coords = torch.cat([torch.zeros(cc_all.shape[0], 1, device=device, dtype=cc_all.dtype),
                                cc_all], dim=-1)
            sp_in = hoeller_unet.spconv.SparseConvTensor(
                feats, coords, spatial_shape=[GRID, GRID, GRID], batch_size=1
            )
            outs = model(sp_in)
            finest_occ = outs[-1][0].dense().sigmoid().squeeze()  # (D, H, W)

            pred_mask = finest_occ > 0.5
            pred_coords = torch.nonzero(pred_mask, as_tuple=False)

            gc, _ = _voxelize_gt_step(d, t, device)

            # render
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(1, 3, 1, projection="3d")
            _scatter_voxels(ax1, cc, "tab:blue", "lidar")
            ax1.set_title(f"t={t} input lidar")
            ax1.set_xlim([-HALF, HALF]); ax1.set_ylim([-HALF, HALF]); ax1.set_zlim([-HALF, HALF])

            ax2 = fig.add_subplot(1, 3, 2, projection="3d")
            _scatter_voxels(ax2, gc, "tab:green", "GT")
            ax2.set_title(f"t={t} GT")
            ax2.set_xlim([-HALF, HALF]); ax2.set_ylim([-HALF, HALF]); ax2.set_zlim([-HALF, HALF])

            ax3 = fig.add_subplot(1, 3, 3, projection="3d")
            _scatter_voxels(ax3, pred_coords, "tab:red", "pred")
            ax3.set_title(f"t={t} prediction")
            ax3.set_xlim([-HALF, HALF]); ax3.set_ylim([-HALF, HALF]); ax3.set_zlim([-HALF, HALF])

            out = os.path.join(args.out_dir, f"viz_traj{args.traj}_t{t}.png")
            plt.tight_layout()
            plt.savefig(out, dpi=120)
            plt.close()
            print(f"[VIZ] {out}  lidar_voxels={cc.shape[0]}  gt_voxels={gc.shape[0]}  pred_voxels={pred_coords.shape[0]}", flush=True)

            # update prev for next iter
            prev_pts_local = (pred_coords.float() + 0.5) * VOXEL_SIZE - HALF
            prev_origin = d["lidar_origin_world"][t].to(device)
            prev_quat = d["lidar_quat_world"][t].to(device)


if __name__ == "__main__":
    main()
