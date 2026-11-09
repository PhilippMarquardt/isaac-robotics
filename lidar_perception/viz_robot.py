"""Render lidar scene with a real-looking G1 humanoid (matplotlib 3D).

Uses mpl_toolkits Poly3DCollection to draw the robot from primitive boxes/spheres
with shading. Outputs a single PNG.

Usage:
    python viz_robot.py --chunk /root/lidar_dataset_v3/chunk_00080.pt \\
        --frame 500 --window 200 --out viz.png
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------- math ----------------------------------------------------------------------

def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


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


# ---------- G1 mesh primitives ---------------------------------------------------------

def _box_faces(center, size):
    """Return 6 face quads for an axis-aligned box."""
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = np.array([
        [cx - hx, cy - hy, cz - hz],  # 0
        [cx + hx, cy - hy, cz - hz],  # 1
        [cx + hx, cy + hy, cz - hz],  # 2
        [cx - hx, cy + hy, cz - hz],  # 3
        [cx - hx, cy - hy, cz + hz],  # 4
        [cx + hx, cy - hy, cz + hz],  # 5
        [cx + hx, cy + hy, cz + hz],  # 6
        [cx - hx, cy + hy, cz + hz],  # 7
    ])
    faces = [
        [v[0], v[1], v[2], v[3]],  # bottom
        [v[4], v[5], v[6], v[7]],  # top
        [v[0], v[1], v[5], v[4]],  # -y
        [v[2], v[3], v[7], v[6]],  # +y
        [v[1], v[2], v[6], v[5]],  # +x
        [v[0], v[3], v[7], v[4]],  # -x
    ]
    return faces


def _shade_faces(faces, base_color, light_dir=np.array([0.4, -0.6, -0.7])):
    """Per-face lambertian shading from a base RGB."""
    light_dir = light_dir / np.linalg.norm(light_dir)
    base = np.array(base_color)
    colors = []
    for face in faces:
        v = np.array(face)
        # face normal from first 3 vertices
        n = np.cross(v[1] - v[0], v[2] - v[0])
        n = n / max(np.linalg.norm(n), 1e-6)
        # diffuse term, clamped
        diff = max(0.0, float(-np.dot(n, light_dir)))
        ambient = 0.35
        intensity = ambient + 0.75 * diff
        c = np.clip(base * intensity, 0, 1)
        colors.append(c)
    return colors


def build_g1_faces(base_pos: np.ndarray, base_quat_xyzw: np.ndarray):
    """Return list of (faces, color_per_face) tuples in WORLD frame."""
    R = quat_xyzw_to_R(base_quat_xyzw)
    orange = np.array([0.95, 0.50, 0.18])
    dark_orange = np.array([0.65, 0.32, 0.10])
    darker = np.array([0.42, 0.20, 0.05])
    head = np.array([0.92, 0.55, 0.20])
    foot = np.array([0.20, 0.10, 0.04])

    parts = [
        ([0.0, 0.0, 0.30], [0.20, 0.16, 0.40], orange),         # torso
        ([0.0, 0.0, 0.0], [0.22, 0.16, 0.10], dark_orange),     # hips
        ([0.0, 0.0, 0.62], [0.16, 0.14, 0.18], head),           # head
        ([0.0, -0.08, -0.20], [0.10, 0.10, 0.30], dark_orange), # left thigh
        ([0.0, 0.08, -0.20], [0.10, 0.10, 0.30], dark_orange),  # right thigh
        ([0.0, -0.08, -0.50], [0.09, 0.09, 0.28], darker),      # left calf
        ([0.0, 0.08, -0.50], [0.09, 0.09, 0.28], darker),       # right calf
        ([0.06, -0.08, -0.66], [0.20, 0.10, 0.05], foot),       # left foot
        ([0.06, 0.08, -0.66], [0.20, 0.10, 0.05], foot),        # right foot
        ([0.0, -0.18, 0.45], [0.08, 0.10, 0.25], orange),       # left upper arm
        ([0.0, 0.18, 0.45], [0.08, 0.10, 0.25], orange),        # right upper arm
        ([0.0, -0.20, 0.20], [0.07, 0.08, 0.22], dark_orange),  # left forearm
        ([0.0, 0.20, 0.20], [0.07, 0.08, 0.22], dark_orange),   # right forearm
    ]

    faces_world: list[np.ndarray] = []
    colors_world: list[np.ndarray] = []
    light_world = np.array([0.4, -0.6, -0.7])
    light_world = light_world / np.linalg.norm(light_world)

    for center, size, base_color in parts:
        faces = _box_faces(center, size)
        for face in faces:
            face_arr = np.array(face)
            # rotate + translate to world
            face_world = (R @ face_arr.T).T + base_pos
            # shade in world frame
            v = face_world
            n = np.cross(v[1] - v[0], v[2] - v[0])
            n = n / max(np.linalg.norm(n), 1e-6)
            diff = max(0.0, float(-np.dot(n, light_world)))
            intensity = 0.35 + 0.75 * diff
            c = np.clip(base_color * intensity, 0, 1)
            faces_world.append(face_world)
            colors_world.append(c)

    return faces_world, colors_world


# ---------- main ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--frame", type=int, default=500)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--out", default="viz_robot.png")
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--azim", type=float, default=-55.0)
    ap.add_argument("--clip_xy", type=float, default=10.0)
    ap.add_argument("--clip_z", type=float, default=4.0)
    ap.add_argument("--marker_size", type=float, default=8.0)
    ap.add_argument("--show_gt", action="store_true", default=True)
    ap.add_argument("--center_on_base", action="store_true", default=True)
    args = ap.parse_args()

    print(f"[INFO] loading {args.chunk}")
    d = torch.load(args.chunk, map_location="cpu", weights_only=True)
    F = d["episode_id"].shape[0]
    c = max(0, min(args.frame, F - 1))

    base_pos = d["base_pos_world"][c].numpy()
    base_quat = d["base_quat_world"][c].numpy()

    # gather lidar pcl in WORLD
    if args.window <= 0:
        lidar_pts = d["lidar_points_world"][c][d["lidar_hit_mask"][c]].numpy()
    else:
        same_ep = d["episode_id"][c]
        lo = max(0, c - args.window // 2)
        hi = min(F, c + args.window // 2)
        accum = []
        for t in range(lo, hi):
            if d["episode_id"][t] != same_ep:
                continue
            pts = d["lidar_points_world"][t][d["lidar_hit_mask"][t]].numpy()
            if pts.shape[0] > 0:
                accum.append(pts)
        lidar_pts = np.concatenate(accum, axis=0) if accum else np.zeros((0, 3))

    gt_pts = d["gt_points_world"][c][d["gt_hit_mask"][c]].numpy() if args.show_gt else np.zeros((0, 3))
    print(f"[INFO] frame {c}: lidar={lidar_pts.shape[0]} gt={gt_pts.shape[0]}")

    # center coordinates on base for cleaner plot
    if args.center_on_base:
        lidar_pts = lidar_pts - base_pos
        gt_pts = gt_pts - base_pos
        robot_pos_plot = np.zeros(3)
    else:
        robot_pos_plot = base_pos

    # G1 faces — built around base, then we offset by -base_pos if centering
    g1_faces, g1_colors = build_g1_faces(robot_pos_plot, base_quat)

    # ---- figure ----
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # ground plane disc (subtle)
    theta = np.linspace(0, 2 * np.pi, 64)
    rad = args.clip_xy * 0.95
    gx = robot_pos_plot[0] + rad * np.cos(theta)
    gy = robot_pos_plot[1] + rad * np.sin(theta)
    gz = np.full_like(gx, robot_pos_plot[2] - 0.71)
    ax.plot_trisurf(gx, gy, gz, color="#f5f5f5", edgecolor="none", alpha=0.4, shade=False)

    # GT terrain pcl (warm colormap)
    if gt_pts.shape[0] > 0:
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2],
                    c=gt_pts[:, 2], cmap="copper", s=args.marker_size * 0.8,
                    alpha=0.85, edgecolors="none", zorder=5)

    # lidar pcl (cool colormap)
    if lidar_pts.shape[0] > 0:
        ax.scatter(lidar_pts[:, 0], lidar_pts[:, 1], lidar_pts[:, 2],
                    c=lidar_pts[:, 2], cmap="Blues_r", s=args.marker_size,
                    alpha=0.7, edgecolors="none", zorder=10)

    # G1 robot — Poly3DCollection with shaded faces
    poly = Poly3DCollection(g1_faces, facecolors=g1_colors, edgecolors=(0, 0, 0, 0.4),
                             linewidths=0.4)
    ax.add_collection3d(poly)

    half = args.clip_xy
    ax.set_xlim([robot_pos_plot[0] - half, robot_pos_plot[0] + half])
    ax.set_ylim([robot_pos_plot[1] - half, robot_pos_plot[1] + half])
    ax.set_zlim([robot_pos_plot[2] - 1.5, robot_pos_plot[2] + 3.0])
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 0.55])

    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {args.out}")


if __name__ == "__main__":
    main()
