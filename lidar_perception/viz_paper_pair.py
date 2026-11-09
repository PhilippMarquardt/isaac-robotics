"""Hoeller-style two-column render for v4 dataset.

LEFT  = network INPUT  = lidar pcl, clipped to ±1.6 m cube, voxelized at 5 cm
RIGHT = ground TRUTH   = mesh-sampled GT, voxelized at 5 cm

Both rendered as 3D cubes (Minecraft style) on a clean white background.
Robot mesh in center (orange humanoid silhouette via Poly3DCollection).

Outputs one PNG per frame.

Usage:
    python viz_paper_pair.py --chunk /root/lidar_dataset_v4/chunk_00010.pt \\
        --frames 7 14 180 422 --out_dir /tmp/paper
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


GRID = 64
VOXEL_SIZE = 0.05
HALF = GRID * VOXEL_SIZE / 2  # 1.6 m


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def voxelize_pts(pts: np.ndarray, base_pos: np.ndarray) -> np.ndarray:
    """Boolean (GRID, GRID, GRID) of 5 cm voxels in cube ±HALF around base_pos."""
    occ = np.zeros((GRID, GRID, GRID), dtype=bool)
    if pts.shape[0] == 0:
        return occ
    rel = pts - base_pos
    in_box = ((rel > -HALF) & (rel < HALF)).all(axis=-1)
    rel = rel[in_box]
    if rel.shape[0] == 0:
        return occ
    idx = np.floor((rel + HALF) / VOXEL_SIZE).astype(np.int64)
    idx = np.clip(idx, 0, GRID - 1)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def _box_faces(center, size):
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = np.array([
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ])
    return [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]],
        [v[0], v[3], v[7], v[4]],
    ]


def build_g1_faces(base_pos, base_quat_xyzw):
    R = quat_xyzw_to_R(base_quat_xyzw)
    orange = np.array([0.95, 0.50, 0.18])
    dark_orange = np.array([0.65, 0.32, 0.10])
    darker = np.array([0.42, 0.20, 0.05])
    head = np.array([0.92, 0.55, 0.20])
    foot = np.array([0.20, 0.10, 0.04])
    parts = [
        ([0.0, 0.0, 0.30], [0.20, 0.16, 0.40], orange),
        ([0.0, 0.0, 0.0], [0.22, 0.16, 0.10], dark_orange),
        ([0.0, 0.0, 0.62], [0.16, 0.14, 0.18], head),
        ([0.0, -0.08, -0.20], [0.10, 0.10, 0.30], dark_orange),
        ([0.0, 0.08, -0.20], [0.10, 0.10, 0.30], dark_orange),
        ([0.0, -0.08, -0.50], [0.09, 0.09, 0.28], darker),
        ([0.0, 0.08, -0.50], [0.09, 0.09, 0.28], darker),
        ([0.06, -0.08, -0.66], [0.20, 0.10, 0.05], foot),
        ([0.06, 0.08, -0.66], [0.20, 0.10, 0.05], foot),
        ([0.0, -0.18, 0.45], [0.08, 0.10, 0.25], orange),
        ([0.0, 0.18, 0.45], [0.08, 0.10, 0.25], orange),
        ([0.0, -0.20, 0.20], [0.07, 0.08, 0.22], dark_orange),
        ([0.0, 0.20, 0.20], [0.07, 0.08, 0.22], dark_orange),
    ]
    light_world = np.array([0.4, -0.6, -0.7])
    light_world = light_world / np.linalg.norm(light_world)
    faces_world, colors_world = [], []
    for center, size, base_color in parts:
        for face in _box_faces(center, size):
            face_arr = np.array(face)
            face_world = (R @ face_arr.T).T + base_pos
            v = face_world
            n = np.cross(v[1] - v[0], v[2] - v[0])
            n = n / max(np.linalg.norm(n), 1e-6)
            diff = max(0.0, float(-np.dot(n, light_world)))
            intensity = 0.35 + 0.75 * diff
            c = np.clip(base_color * intensity, 0, 1)
            faces_world.append(face_world)
            colors_world.append(c)
    return faces_world, colors_world


def render_panel(ax, occ: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray,
                  title: str, voxel_color="#3b4854"):
    """Render an occupancy grid as voxel cubes + G1 humanoid centered at base."""
    # voxels via Axes3D.voxels (cubes!)
    edges = np.linspace(-HALF, HALF, GRID + 1) + 0  # local
    # build edge grids for ax.voxels — needs (GRID+1, GRID+1, GRID+1)
    x = (np.arange(GRID + 1) * VOXEL_SIZE - HALF) + base_pos[0]
    y = (np.arange(GRID + 1) * VOXEL_SIZE - HALF) + base_pos[1]
    z = (np.arange(GRID + 1) * VOXEL_SIZE - HALF) + base_pos[2]
    xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")
    if occ.sum() > 0:
        ax.voxels(xg, yg, zg, occ, facecolors=voxel_color, edgecolors="#1a1a1a",
                   alpha=0.95, linewidth=0.08)

    # G1 silhouette
    g1_faces, g1_colors = build_g1_faces(base_pos, base_quat)
    poly = Poly3DCollection(g1_faces, facecolors=g1_colors, edgecolors=(0, 0, 0, 0.4),
                             linewidths=0.4)
    ax.add_collection3d(poly)

    ax.set_xlim([base_pos[0] - HALF, base_pos[0] + HALF])
    ax.set_ylim([base_pos[1] - HALF, base_pos[1] + HALF])
    ax.set_zlim([base_pos[2] - HALF, base_pos[2] + HALF])
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(title, fontsize=12)
    ax.view_init(elev=24, azim=-55)


def render_frame(d: dict, t: int, out_path: str):
    base = d["base_pos_world"][t].numpy()
    quat = d["base_quat_world"][t].numpy()
    lidar_pts = d["lidar_points_world"][t][d["lidar_hit_mask"][t]].numpy()
    n_gt = int(d["gt_n_points"][t].item())
    gt_pts = d["gt_points_world"][t, :n_gt].numpy()

    occ_in = voxelize_pts(lidar_pts, base)
    occ_gt = voxelize_pts(gt_pts, base)

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    ax_l = fig.add_subplot(1, 2, 1, projection="3d")
    ax_r = fig.add_subplot(1, 2, 2, projection="3d")
    render_panel(ax_l, occ_in, base, quat,
                  title=f"INPUT (lidar voxelized)  occ={int(occ_in.sum())}")
    render_panel(ax_r, occ_gt, base, quat,
                  title=f"GROUND TRUTH (mesh)  occ={int(occ_gt.sum())}",
                  voxel_color="#7d96b8")
    fig.suptitle(f"frame {t}  base={tuple(round(float(x), 2) for x in base)}", y=0.92)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"frame={t}  in_vox={int(occ_in.sum())}  gt_vox={int(occ_gt.sum())}  → {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] loading {args.chunk}", flush=True)
    d = torch.load(args.chunk, map_location="cpu", weights_only=True)
    chunk_id = re.search(r"chunk_(\d+)", os.path.basename(args.chunk))
    chunk_id = chunk_id.group(1) if chunk_id else "0"

    for t in args.frames:
        out = os.path.join(args.out_dir, f"paper_pair_{chunk_id}_{t:05d}.png")
        render_frame(d, t, out)


if __name__ == "__main__":
    main()
