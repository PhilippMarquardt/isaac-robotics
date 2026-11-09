"""Hoeller-2022 Fig 6 style: isometric 3D voxel cubes around the robot.

Single big panel, clean background, no axes. Voxels rendered as Minecraft cubes via
Axes3D.voxels with proper shading and edge lines. Robot marked at center with a
larger orange marker. Paper convention: dark-grey cubes for measured points, optional
red box outline for the predicted region.

Usage:
    python viz_paper_iso.py --chunk /root/lidar_dataset_v3/chunk_00080.pt \\
        --frame 500 --window 12 --grid 32 --voxel_size 0.30 --out viz.png

For grid options:
    - 32³ at 0.30 m → 9.6 m cube (training grid; matches Hoeller-style fig)
    - 48³ at 0.40 m → 19.2 m cube (wider, sees full 360 ground ring)
    - 64³ at 0.50 m → 32 m cube (very wide; cubes get tiny)
"""

from __future__ import annotations

import argparse
import os

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def world_to_local(pts_w, origin_w, quat_xyzw):
    x, y, z, w = quat_xyzw.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ]).reshape(3, 3)
    delta = pts_w - origin_w.unsqueeze(0)
    return delta @ R


def voxelize(pts_local, grid, voxel_size):
    half = grid * voxel_size / 2
    occ = np.zeros((grid, grid, grid), dtype=bool)
    if pts_local.shape[0] == 0:
        return occ
    p = pts_local.numpy()
    in_box = ((p > -half) & (p < half)).all(-1)
    p = p[in_box]
    if p.shape[0] == 0:
        return occ
    idx = np.floor((p + half) / voxel_size).astype(np.int64)
    idx = np.clip(idx, 0, grid - 1)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def render(occ, voxel_size, half, base_local, out_path,
            color="#3b4854", figsize=(10, 10),
            elev=35, azim=-55, draw_red_box=False, marker_size=140,
            show_robot=True, occ_gt=None, gt_color="#9bb1c7"):
    """Render each occupied voxel as a 3D scatter dot at its center, paper-style.

    `occ_gt` (optional): boolean grid of GT-scanner voxels. Rendered first in a
    lighter color so the lidar voxels overlay on top.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # GT floor voxels first (background layer)
    if occ_gt is not None:
        occ_gt_only = occ_gt & (~occ)  # only render GT cells not already in lidar
        gt_idx = np.argwhere(occ_gt_only)
        if gt_idx.shape[0] > 0:
            gc = (gt_idx + 0.5) * voxel_size - half
            ax.scatter(gc[:, 0], gc[:, 1], gc[:, 2],
                        s=marker_size * 0.8, c=gt_color, marker="o",
                        edgecolors="#5a6b7c", linewidths=0.4,
                        depthshade=True, alpha=0.85, zorder=5)

    # lidar voxels (foreground)
    occupied = np.argwhere(occ)
    if occupied.shape[0] > 0:
        centers = (occupied + 0.5) * voxel_size - half
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                    s=marker_size, c=color, marker="o",
                    edgecolors="#1a1a1a", linewidths=0.5,
                    depthshade=True, alpha=0.95, zorder=10)

    if show_robot:
        # robot stand-in: a small stack of orange spheres at base position
        bx, by, bz = base_local[0], base_local[1], base_local[2]
        for dz, sz in [(-0.05, 360), (0.25, 280), (0.5, 220)]:
            ax.scatter([bx], [by], [bz + dz], s=sz, c="#ff7e1a", marker="o",
                        edgecolors="black", linewidths=1.0, depthshade=True, zorder=999)

    if draw_red_box:
        for dx in (-half, half):
            for dy in (-half, half):
                ax.plot([dx, dx], [dy, dy], [-half, half], color="#cc2222", lw=1.6, zorder=998)
        for dx in (-half, half):
            for dz in (-half, half):
                ax.plot([dx, dx], [-half, half], [dz, dz], color="#cc2222", lw=1.6, zorder=998)
        for dy in (-half, half):
            for dz in (-half, half):
                ax.plot([-half, half], [dy, dy], [dz, dz], color="#cc2222", lw=1.6, zorder=998)

    ax.set_xlim([-half, half]); ax.set_ylim([-half, half])
    ax.set_zlim([-half / 2, half * 0.7])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 0.6])

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--frame", type=int, default=500)
    ap.add_argument("--window", type=int, default=24,
                    help="0 = single frame; >0 = accumulate +/- window/2 frames")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--voxel_size", type=float, default=0.30)
    ap.add_argument("--out", default="viz_paper_iso.png")
    ap.add_argument("--elev", type=float, default=35.0)
    ap.add_argument("--azim", type=float, default=-55.0)
    ap.add_argument("--color", default="#3b4854")
    ap.add_argument("--draw_red_box", action="store_true")
    ap.add_argument("--marker_size", type=float, default=140.0)
    ap.add_argument("--show_gt", action="store_true",
                    help="overlay GT-scanner floor voxels in lighter color")
    args = ap.parse_args()

    print(f"[INFO] loading {args.chunk}")
    d = torch.load(args.chunk, map_location="cpu", weights_only=True)
    F = d["episode_id"].shape[0]
    c = max(0, min(args.frame, F - 1))

    cur_origin = d["lidar_origin_world"][c]
    cur_quat = d["lidar_quat_world"][c]
    base_local = world_to_local(d["base_pos_world"][c].unsqueeze(0), cur_origin, cur_quat).squeeze(0).numpy()

    if args.window <= 0:
        pts_lidar = d["lidar_points_world"][c][d["lidar_hit_mask"][c]]
        local = world_to_local(pts_lidar, cur_origin, cur_quat)
        gt_acc = []
        if args.show_gt:
            gpts = d["gt_points_world"][c][d["gt_hit_mask"][c]]
            gt_acc.append(world_to_local(gpts, cur_origin, cur_quat))
    else:
        same_ep = d["episode_id"][c]
        lo = max(0, c - args.window // 2)
        hi = min(F, c + args.window // 2)
        accum, gt_acc = [], []
        for t in range(lo, hi):
            if d["episode_id"][t] != same_ep:
                continue
            ptsw = d["lidar_points_world"][t][d["lidar_hit_mask"][t]]
            if ptsw.shape[0] > 0:
                accum.append(world_to_local(ptsw, cur_origin, cur_quat))
            if args.show_gt:
                gpts = d["gt_points_world"][t][d["gt_hit_mask"][t]]
                if gpts.shape[0] > 0:
                    gt_acc.append(world_to_local(gpts, cur_origin, cur_quat))
        local = torch.cat(accum, dim=0) if accum else torch.zeros(0, 3)

    gt_local = torch.cat(gt_acc, dim=0) if gt_acc else torch.zeros(0, 3)

    occ = voxelize(local, args.grid, args.voxel_size)
    occ_gt = voxelize(gt_local, args.grid, args.voxel_size) if args.show_gt else None
    half = args.grid * args.voxel_size / 2
    print(f"[INFO] frame {c}, window {args.window}, voxels occupied: {int(occ.sum())} / {args.grid**3}")
    if occ_gt is not None:
        print(f"[INFO] GT voxels: {int(occ_gt.sum())}")
    print(f"[INFO] grid {args.grid}³ × {args.voxel_size:.2f} m = ±{half:.2f} m cube")

    render(occ, args.voxel_size, half, base_local, args.out,
            color=args.color, elev=args.elev, azim=args.azim,
            draw_red_box=args.draw_red_box, marker_size=args.marker_size,
            occ_gt=occ_gt)
    print(f"[SAVED] {args.out}")


if __name__ == "__main__":
    main()
