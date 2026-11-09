"""Export a frame's lidar + GT + robot mesh as a single colored .ply file.

Open with CloudCompare (free, https://cloudcompare.org/) or MeshLab for proper
3D viewing with rotation/zoom/shading.

Robot is sampled as a dense surface point cloud (orange) so it renders correctly
in any point-cloud viewer.

Usage:
    python export_ply.py --chunk /root/lidar_dataset_v3/chunk_00080.pt \\
        --frame 500 --window 200 --out scene.ply
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def sample_box_surface(center, size, density=80) -> np.ndarray:
    """Return ~density² points on each face of an axis-aligned box."""
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    pts = []
    n_per_face = density
    for sgn in (-1, 1):
        # +x and -x faces
        v, w = np.meshgrid(np.linspace(-hy, hy, n_per_face), np.linspace(-hz, hz, n_per_face))
        face = np.stack([np.full_like(v, sgn * hx), v, w], axis=-1).reshape(-1, 3)
        pts.append(face + np.array([cx, cy, cz]))
        # +y and -y
        v, w = np.meshgrid(np.linspace(-hx, hx, n_per_face), np.linspace(-hz, hz, n_per_face))
        face = np.stack([v, np.full_like(v, sgn * hy), w], axis=-1).reshape(-1, 3)
        pts.append(face + np.array([cx, cy, cz]))
        # +z and -z
        v, w = np.meshgrid(np.linspace(-hx, hx, n_per_face), np.linspace(-hy, hy, n_per_face))
        face = np.stack([v, w, np.full_like(v, sgn * hz)], axis=-1).reshape(-1, 3)
        pts.append(face + np.array([cx, cy, cz]))
    return np.concatenate(pts, axis=0)


def sample_sphere(center, radius, n_pts=2000) -> np.ndarray:
    """Fibonacci sphere — uniformly distributed points on unit sphere * radius."""
    indices = np.arange(0, n_pts) + 0.5
    phi = np.arccos(1 - 2 * indices / n_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)
    return np.stack([x, y, z], axis=-1) + np.array(center)


def build_g1_pointcloud(base_pos, base_quat_xyzw):
    """Sample G1 silhouette as a dense oriented point cloud."""
    R = quat_xyzw_to_R(base_quat_xyzw)
    orange = np.array([242, 128, 46], dtype=np.uint8)
    dark_orange = np.array([166, 82, 26], dtype=np.uint8)
    darker = np.array([107, 51, 13], dtype=np.uint8)
    head = np.array([235, 140, 51], dtype=np.uint8)

    parts = []
    parts.append((sample_box_surface([0.0, 0.0, 0.30], [0.20, 0.16, 0.40], 50), orange))   # torso
    parts.append((sample_box_surface([0.0, 0.0, 0.0], [0.22, 0.16, 0.10], 50), dark_orange))  # hips
    parts.append((sample_sphere([0.0, 0.0, 0.62], 0.085, 2500), head))   # head
    for sy in (-1, 1):
        parts.append((sample_box_surface([0.0, 0.08 * sy, -0.20], [0.10, 0.10, 0.30], 35), dark_orange))
        parts.append((sample_box_surface([0.0, 0.08 * sy, -0.50], [0.09, 0.09, 0.28], 35), darker))
        parts.append((sample_box_surface([0.06, 0.08 * sy, -0.66], [0.20, 0.10, 0.05], 35), [42, 21, 8]))
        parts.append((sample_box_surface([0.0, 0.18 * sy, 0.45], [0.08, 0.10, 0.25], 35), orange))
        parts.append((sample_box_surface([0.0, 0.20 * sy, 0.20], [0.07, 0.08, 0.22], 35), dark_orange))

    pts_local = np.concatenate([p for p, _ in parts], axis=0)
    colors = np.concatenate([
        np.tile(np.array(c, dtype=np.uint8), (p.shape[0], 1)) for p, c in parts
    ], axis=0)

    pts_world = (R @ pts_local.T).T + base_pos
    return pts_world, colors


def write_ply(path: str, points: np.ndarray, colors: np.ndarray):
    """Binary PLY with vertex+color."""
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
    points = points.astype(np.float32)
    colors = colors.astype(np.uint8)
    body = np.empty(n, dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    body["x"] = points[:, 0]; body["y"] = points[:, 1]; body["z"] = points[:, 2]
    body["r"] = colors[:, 0]; body["g"] = colors[:, 1]; body["b"] = colors[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(body.tobytes())


def colorize_z(pts: np.ndarray, low_rgb=(40, 70, 110), high_rgb=(200, 230, 250)) -> np.ndarray:
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    z = pts[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    rng = max(zmax - zmin, 1e-6)
    t = ((z - zmin) / rng).reshape(-1, 1)
    return (np.array(low_rgb) + t * (np.array(high_rgb) - np.array(low_rgb))).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--frame", type=int, default=500)
    ap.add_argument("--window", type=int, default=200,
                    help="0 = single frame; >0 accumulate ±window/2 frames of lidar")
    ap.add_argument("--out", default="scene.ply")
    ap.add_argument("--include_robot", action="store_true", default=True)
    ap.add_argument("--include_gt", action="store_true", default=True)
    args = ap.parse_args()

    print(f"[INFO] loading {args.chunk}")
    d = torch.load(args.chunk, map_location="cpu", weights_only=True)
    F = d["episode_id"].shape[0]
    c = max(0, min(args.frame, F - 1))

    base_pos = d["base_pos_world"][c].numpy()
    base_quat = d["base_quat_world"][c].numpy()

    # collect lidar
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
    lidar_colors = colorize_z(lidar_pts, (40, 70, 110), (200, 230, 250))
    print(f"[INFO] lidar points: {lidar_pts.shape[0]}")

    # GT terrain
    gt_pts = np.zeros((0, 3))
    gt_colors = np.zeros((0, 3), dtype=np.uint8)
    if args.include_gt:
        gt_pts = d["gt_points_world"][c][d["gt_hit_mask"][c]].numpy()
        gt_colors = colorize_z(gt_pts, (140, 90, 50), (210, 180, 100))
    print(f"[INFO] GT points: {gt_pts.shape[0]}")

    # G1 robot
    robot_pts = np.zeros((0, 3))
    robot_colors = np.zeros((0, 3), dtype=np.uint8)
    if args.include_robot:
        robot_pts, robot_colors = build_g1_pointcloud(base_pos, base_quat)
    print(f"[INFO] robot points: {robot_pts.shape[0]}")

    all_pts = np.concatenate([lidar_pts, gt_pts, robot_pts], axis=0)
    all_colors = np.concatenate([lidar_colors, gt_colors, robot_colors], axis=0)
    print(f"[INFO] total: {all_pts.shape[0]} points → {args.out}")

    write_ply(args.out, all_pts, all_colors)
    print(f"[SAVED] {args.out}")
    print(f"[NEXT] open with CloudCompare (https://cloudcompare.org/) or MeshLab.")


if __name__ == "__main__":
    main()
