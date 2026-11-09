"""Minimal viz for v4 dataset — two color-coded clouds per frame in one .ply.

  - Lidar input (blue, 60,120,255): lidar_points_world[t, lidar_hit_mask[t]]
  - Ground truth (orange/tan, 220,170,90): gt_points_world[t, :gt_n_points[t]]

No markers, no cube, no robot. Just the two raw clouds in world frame.

Usage:
    python export_input_vs_gt.py --chunk /root/lidar_dataset_v4/chunk_00010.pt \\
        --frames 100 500 800 --out_dir /tmp
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import torch


def write_ply(path: str, pts: np.ndarray, cols: np.ndarray):
    n = pts.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    body = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                               ("r", "u1"), ("g", "u1"), ("b", "u1")])
    body["x"] = pts[:, 0].astype(np.float32)
    body["y"] = pts[:, 1].astype(np.float32)
    body["z"] = pts[:, 2].astype(np.float32)
    body["r"] = cols[:, 0]; body["g"] = cols[:, 1]; body["b"] = cols[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(body.tobytes())


def _sphere_pts(center, radius, n, rng):
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r = radius * (rng.random(n) ** (1.0 / 3.0)).reshape(-1, 1)
    return center + dirs * r


def _cube_wireframe(center, half, pts_per_edge=100):
    h = half
    corners = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h,  h], [h, -h,  h], [h, h,  h], [-h, h,  h],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    out = []
    for a, b in edges:
        t = np.linspace(0, 1, pts_per_edge).reshape(-1, 1)
        out.append(corners[a] + t * (corners[b] - corners[a]))
    return np.concatenate(out, axis=0) + center


def export_frame(d: dict, t: int, out_path: str, with_markers: bool = False):
    lidar_pts = d["lidar_points_world"][t][d["lidar_hit_mask"][t]].numpy()
    n_gt = int(d["gt_n_points"][t].item())
    gt_pts = d["gt_points_world"][t, :n_gt].numpy()

    pcs = [
        (lidar_pts, np.array([60, 120, 255], dtype=np.uint8)),
        (gt_pts,    np.array([220, 170, 90], dtype=np.uint8)),
    ]
    extras = ""
    if with_markers:
        rng = np.random.default_rng(0)
        base = d["base_pos_world"][t].numpy()
        lidar_origin = d["lidar_origin_world"][t].numpy()
        pelvis = _sphere_pts(base, 0.10, 500, rng)
        lidar_marker = _sphere_pts(lidar_origin, 0.05, 200, rng)
        cube = _cube_wireframe(base, 1.6, 100)
        pcs += [
            (pelvis,       np.array([255,   0,   0], dtype=np.uint8)),
            (lidar_marker, np.array([255, 255,   0], dtype=np.uint8)),
            (cube,         np.array([  0, 255,   0], dtype=np.uint8)),
        ]
        extras = f"  markers: pelvis=500 lidar=200 cube_wire=1200"

    pts = np.concatenate([p for p, _ in pcs], axis=0)
    cols = np.concatenate([np.tile(c, (p.shape[0], 1)) for p, c in pcs], axis=0)
    write_ply(out_path, pts, cols)
    print(
        f"frame_idx={t}  gt_n_points={n_gt}  lidar_hit_count={lidar_pts.shape[0]}{extras}  → {out_path}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--markers", action="store_true",
                    help="Add red pelvis sphere, yellow lidar sphere, green wireframe cube.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] loading {args.chunk}", flush=True)
    d = torch.load(args.chunk, map_location="cpu", weights_only=True)
    F = d["episode_id"].shape[0]
    chunk_id = re.search(r"chunk_(\d+)", os.path.basename(args.chunk))
    chunk_id = chunk_id.group(1) if chunk_id else "0"

    for t in args.frames:
        if t < 0 or t >= F:
            print(f"[WARN] frame {t} out of range [0, {F})", flush=True)
            continue
        suffix = "_markers" if args.markers else ""
        out = os.path.join(args.out_dir, f"input_vs_gt_{chunk_id}_{t:05d}{suffix}.ply")
        export_frame(d, t, out, with_markers=args.markers)


if __name__ == "__main__":
    main()
