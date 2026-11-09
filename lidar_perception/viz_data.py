"""Render lidar + GT as voxel cubes — Hoeller-2022 paper-style.

Each occupied voxel becomes a small cube via `Axes3D.voxels(...)`. Layout matches
paper Fig 6 (4-panel A/B/C/D rows, 3-column views).

Per row:
  col 1: input lidar pcl voxelized (sparse measurement, dark grey cubes)
  col 2: accumulated multi-frame voxelized (denser)
  col 3: GT terrain voxels, with the 64³ box outline highlighted (red)

Robot location marked with an orange marker at base position.

Usage:
    python viz_data.py --chunk /root/lidar_dataset_v3/chunk_00050.pt \\
        --centers 200,400,600,800 --window 12 --out viz.png
"""

from __future__ import annotations

import argparse
import os

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def world_to_local(pts_w: torch.Tensor, origin_w: torch.Tensor, quat_xyzw: torch.Tensor) -> torch.Tensor:
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


def voxelize(pts_local: torch.Tensor, grid: int, voxel_size: float) -> np.ndarray:
    """Hard-occupancy boolean grid of shape (grid, grid, grid)."""
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


def voxel_cube_render(ax, occ: np.ndarray, voxel_size: float, half: float,
                       face_color: str, edge_color: str, alpha: float = 0.9):
    """Render a 3D boolean grid as cubes via Axes3D.voxels.

    `occ` is (D, H, W). Cube centers are placed so the grid is centered at origin.
    """
    if occ.shape[0] > 64:
        # decimate — voxels() with >64³ is slow
        s = occ.shape[0] // 64
        occ = occ[::s, ::s, ::s]
    D, H, W = occ.shape
    edges = np.linspace(-half, half, D + 1)
    x, y, z = np.meshgrid(edges, edges, edges, indexing="ij")
    ax.voxels(x, y, z, occ, facecolors=face_color, edgecolors=edge_color, alpha=alpha,
              linewidth=0.1)


def render_row(fig, gs_row: int, n_rows: int, label: str, chunk: dict, c: int,
                window: int, grid_in: int, voxel_size: float, plot_half: float):
    F = chunk["episode_id"].shape[0]
    cur_origin = chunk["lidar_origin_world"][c]
    cur_quat = chunk["lidar_quat_world"][c]
    base_now = chunk["base_pos_world"][c]
    base_local = world_to_local(base_now.unsqueeze(0), cur_origin, cur_quat).squeeze(0).numpy()
    same_episode = chunk["episode_id"][c]
    half = grid_in * voxel_size / 2

    # single-frame lidar voxels
    lidar_now = chunk["lidar_points_world"][c][chunk["lidar_hit_mask"][c]]
    lidar_now_local = world_to_local(lidar_now, cur_origin, cur_quat)
    occ_lidar = voxelize(lidar_now_local, grid_in, voxel_size)

    # accumulated lidar across window
    lo = max(0, c - window // 2)
    hi = min(F, c + window // 2)
    accum = []
    for t in range(lo, hi):
        if chunk["episode_id"][t] != same_episode:
            continue
        pts = chunk["lidar_points_world"][t][chunk["lidar_hit_mask"][t]]
        if pts.shape[0] == 0:
            continue
        accum.append(world_to_local(pts, cur_origin, cur_quat))
    accum_local = torch.cat(accum, dim=0) if accum else torch.zeros(0, 3)
    occ_accum = voxelize(accum_local, grid_in, voxel_size)

    # GT terrain voxels (use the same grid so axes line up)
    gt_pts = chunk["gt_points_world"][c][chunk["gt_hit_mask"][c]]
    gt_local = world_to_local(gt_pts, cur_origin, cur_quat)
    occ_gt = voxelize(gt_local, grid_in, voxel_size)

    print(f"[ROW {label}] frame {c}: lidar voxels={occ_lidar.sum()}  "
          f"accum={occ_accum.sum()}  gt={occ_gt.sum()}")

    titles = [
        f"{label} — single-frame lidar ({int(occ_lidar.sum())} voxels)",
        f"{label} — accumulated {hi - lo} frames ({int(occ_accum.sum())} voxels)",
        f"{label} — GT terrain ({int(occ_gt.sum())} voxels)",
    ]
    occs = [occ_lidar, occ_accum, occ_gt]
    colors = ["#3a4a55", "#3a4a55", "#7d96b8"]

    for col, (occ, color, title) in enumerate(zip(occs, colors, titles)):
        ax = fig.add_subplot(n_rows, 3, gs_row * 3 + col + 1, projection="3d")
        voxel_cube_render(ax, occ, voxel_size, half, color, "#1a1a1a", alpha=0.9)
        # robot base marker
        ax.scatter([base_local[0]], [base_local[1]], [base_local[2]],
                    s=180, c="#ff8c1a", marker="o", edgecolors="black", linewidths=1.0,
                    label="robot base")
        # lidar mount at origin
        ax.scatter([0], [0], [0], s=120, c="#e64545", marker="^", edgecolors="black",
                    linewidths=1.0, label="lidar mount")
        if col == 2:
            # red box outline for the predicted region (paper convention)
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ax.plot([dx * plot_half, dx * plot_half], [dy * plot_half, dy * plot_half],
                         [-plot_half, plot_half], color="#cc2222", lw=2.0, alpha=0.9)
            for axis in [(0, plot_half), (1, plot_half)]:
                pass
        ax.set_xlim([-plot_half, plot_half])
        ax.set_ylim([-plot_half, plot_half])
        ax.set_zlim([-plot_half / 2, plot_half / 2])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.view_init(elev=28, azim=-65)
        if col == 0 and gs_row == 0:
            ax.legend(loc="upper right", fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--centers", default="100,300,500,700",
                    help="comma-separated frame indices (one per row, paper A/B/C/D)")
    ap.add_argument("--window", type=int, default=24,
                    help="frames to accumulate around each center")
    ap.add_argument("--grid", type=int, default=32,
                    help="voxel grid edge (lower = coarser cubes, faster render)")
    ap.add_argument("--voxel_size", type=float, default=0.30,
                    help="voxel edge in meters (matches training: 0.30)")
    ap.add_argument("--plot_half", type=float, default=4.8,
                    help="render half-extent in meters (= grid/2 * voxel_size for fit)")
    ap.add_argument("--out", default="viz.png")
    args = ap.parse_args()

    print(f"[INFO] loading {args.chunk}")
    d = torch.load(args.chunk, map_location="cpu", weights_only=True)
    F = d["episode_id"].shape[0]
    print(f"[INFO] {F} frames in chunk")

    centers = [int(x) for x in args.centers.split(",")]
    centers = [c for c in centers if 0 <= c < F][:4]  # cap at 4 rows
    n_rows = len(centers)

    fig = plt.figure(figsize=(15, 5 * n_rows))
    fig.suptitle(
        f"{os.path.basename(args.chunk)} — voxel cube view "
        f"({args.grid}³ grid, {args.voxel_size*100:.0f} cm voxels)",
        y=1.0, fontsize=12,
    )

    for r, c in enumerate(centers):
        label = chr(ord("A") + r)
        render_row(fig, r, n_rows, label, d, c, args.window,
                    args.grid, args.voxel_size, args.plot_half)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {args.out}")


if __name__ == "__main__":
    main()
