"""Robotics-grade lidar dataset viewer via Rerun.

Streams a chunk of lidar trajectory data to a Rerun .rrd file:
  - World coordinate frame
  - G1 robot drawn as a stylized humanoid (head sphere, torso box, hip box, legs)
    placed at the actual recorded base pose each frame
  - Lidar mount frame (with FOV cone)
  - Lidar point cloud per frame, colored by elevation
  - GT terrain point cloud per frame
  - Optional: accumulated lidar history (last N frames in a fading-blue cloud)

Open the resulting .rrd file with:
    pip install rerun-sdk
    rerun viz.rrd

Usage:
    python viz_rerun.py --chunk /root/lidar_dataset_v3/chunk_00080.pt \\
        --start 400 --end 600 --downsample 5 --out viz.rrd
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import rerun as rr


def quat_xyzw_to_wxyz(q: torch.Tensor) -> np.ndarray:
    """Rerun uses (x, y, z, w). Our data IS xyzw — pass through, return ndarray."""
    return q.numpy()


def build_g1_humanoid(rr_path: str, base_pos_w: np.ndarray, base_quat_xyzw: np.ndarray,
                       lidar_offset_z: float = 0.30):
    """Draw a stylized G1 silhouette (head + torso + hips + legs) at the given pose.

    Uses Boxes3D + small spheres so it reads as a humanoid in any viewport. Sized
    roughly to the real G1: ~1.3 m tall, ~0.4 m shoulder width.
    """
    # base transform — rerun expects xyzw quaternion
    rr.log(rr_path, rr.Transform3D(translation=base_pos_w, rotation=rr.Quaternion(xyzw=base_quat_xyzw)))

    # body parts in body frame (Z up). G1 base = pelvis center.
    boxes_centers = []
    boxes_sizes = []
    box_colors = []

    # torso: 0.20 m × 0.16 m × 0.40 m, centered 0.20 m above base
    boxes_centers.append([0.0, 0.0, 0.30])
    boxes_sizes.append([0.20, 0.16, 0.40])
    box_colors.append([200, 90, 30])

    # hips: 0.22 m × 0.16 m × 0.10 m at base
    boxes_centers.append([0.0, 0.0, 0.0])
    boxes_sizes.append([0.22, 0.16, 0.10])
    box_colors.append([160, 70, 25])

    # head: 0.16 m × 0.14 m × 0.18 m at +0.62 m
    boxes_centers.append([0.0, 0.0, 0.62])
    boxes_sizes.append([0.16, 0.14, 0.18])
    box_colors.append([220, 110, 40])

    # left/right thighs: 0.10 × 0.10 × 0.30 at hip ±0.08, z=-0.20
    for sy in (-1, 1):
        boxes_centers.append([0.0, 0.08 * sy, -0.20])
        boxes_sizes.append([0.10, 0.10, 0.30])
        box_colors.append([160, 70, 25])
    # left/right calves: at z=-0.50
    for sy in (-1, 1):
        boxes_centers.append([0.0, 0.08 * sy, -0.50])
        boxes_sizes.append([0.09, 0.09, 0.30])
        box_colors.append([130, 55, 18])
    # left/right feet
    for sy in (-1, 1):
        boxes_centers.append([0.06, 0.08 * sy, -0.70])
        boxes_sizes.append([0.20, 0.10, 0.05])
        box_colors.append([90, 35, 10])

    # left/right shoulders/arms
    for sy in (-1, 1):
        # upper arm
        boxes_centers.append([0.0, 0.20 * sy, 0.45])
        boxes_sizes.append([0.08, 0.08, 0.25])
        box_colors.append([180, 80, 28])
        # forearm
        boxes_centers.append([0.0, 0.22 * sy, 0.20])
        boxes_sizes.append([0.07, 0.07, 0.22])
        box_colors.append([150, 65, 22])

    rr.log(
        rr_path + "/body",
        rr.Boxes3D(
            centers=np.array(boxes_centers, dtype=np.float32),
            sizes=np.array(boxes_sizes, dtype=np.float32),
            colors=np.array(box_colors, dtype=np.uint8),
            fill_mode=rr.components.FillMode.Solid,
        ),
    )

    # lidar mount marker (small red sphere at head height)
    rr.log(
        rr_path + "/lidar_mount",
        rr.Points3D(
            positions=np.array([[0.0, 0.0, lidar_offset_z + 0.4]], dtype=np.float32),
            colors=np.array([[230, 50, 50]], dtype=np.uint8),
            radii=np.array([0.05], dtype=np.float32),
        ),
    )


def colorize_by_z(pts: np.ndarray, cmap_low=(60, 80, 110), cmap_high=(220, 235, 250)) -> np.ndarray:
    """Linear colormap on Z elevation, returns (N, 3) uint8."""
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    z = pts[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    rng = max(zmax - zmin, 1e-6)
    t = ((z - zmin) / rng).reshape(-1, 1)
    lo = np.array(cmap_low, dtype=np.float32)
    hi = np.array(cmap_high, dtype=np.float32)
    colors = (lo + t * (hi - lo)).astype(np.uint8)
    return colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=200)
    ap.add_argument("--downsample", type=int, default=2,
                    help="stream every Nth frame (50 Hz / N effective)")
    ap.add_argument("--accumulate", type=int, default=0,
                    help="show last K frames as a faded history cloud")
    ap.add_argument("--out", default="viz.rrd")
    ap.add_argument("--app_id", default="g1_lidar")
    args = ap.parse_args()

    print(f"[INFO] loading {args.chunk}")
    d = torch.load(args.chunk, map_location="cpu", weights_only=True)
    F = d["episode_id"].shape[0]
    end = min(args.end, F)
    print(f"[INFO] {F} frames in chunk; streaming {args.start}..{end} step {args.downsample}")

    rr.init(args.app_id, spawn=False)
    rr.save(args.out)

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("/world", rr.Arrows3D(
        vectors=np.eye(3) * 0.5,
        origins=np.zeros((3, 3)),
        colors=np.array([[230, 70, 70], [70, 200, 70], [70, 130, 230]], dtype=np.uint8),
    ), static=True)

    history_buffer = []
    for t in range(args.start, end, args.downsample):
        rr.set_time("sim_time", duration=float(d["timestamp_sim"][t]))
        rr.set_time("frame", sequence=int(t))

        base_pos = d["base_pos_world"][t].numpy()
        base_quat = quat_xyzw_to_wxyz(d["base_quat_world"][t])
        lidar_origin = d["lidar_origin_world"][t].numpy()
        lidar_quat = quat_xyzw_to_wxyz(d["lidar_quat_world"][t])

        # G1 body
        build_g1_humanoid("/world/g1", base_pos, base_quat)

        # Lidar mount transform — used as parent for the lidar pcl
        rr.log("/world/lidar", rr.Transform3D(
            translation=lidar_origin,
            rotation=rr.Quaternion(xyzw=lidar_quat),
        ))

        # current-frame lidar pcl in WORLD frame (already world per dataset spec)
        lidar_pts = d["lidar_points_world"][t][d["lidar_hit_mask"][t]].numpy()
        if lidar_pts.shape[0] > 0:
            colors = colorize_by_z(lidar_pts, cmap_low=(40, 70, 110), cmap_high=(200, 230, 250))
            rr.log("/world/lidar_pcl", rr.Points3D(
                positions=lidar_pts.astype(np.float32),
                colors=colors,
                radii=np.full(lidar_pts.shape[0], 0.04, dtype=np.float32),
            ))

        # GT terrain pcl (lighter)
        gt_pts = d["gt_points_world"][t][d["gt_hit_mask"][t]].numpy()
        if gt_pts.shape[0] > 0:
            gt_colors = colorize_by_z(gt_pts, cmap_low=(150, 100, 60), cmap_high=(80, 200, 120))
            rr.log("/world/gt_pcl", rr.Points3D(
                positions=gt_pts.astype(np.float32),
                colors=gt_colors,
                radii=np.full(gt_pts.shape[0], 0.03, dtype=np.float32),
            ))

        # accumulated history
        if args.accumulate > 0:
            history_buffer.append(lidar_pts)
            if len(history_buffer) > args.accumulate:
                history_buffer.pop(0)
            if history_buffer:
                hist = np.concatenate(history_buffer, axis=0)
                rr.log("/world/lidar_history", rr.Points3D(
                    positions=hist.astype(np.float32),
                    colors=np.full((hist.shape[0], 3), [40, 60, 90], dtype=np.uint8),
                    radii=np.full(hist.shape[0], 0.025, dtype=np.float32),
                ))

    print(f"[SAVED] {args.out}  ({end - args.start} frames @ stride {args.downsample})")
    print(f"[INFO] open with: rerun {args.out}")


if __name__ == "__main__":
    main()
