"""Aggregate GT terrain across an entire trajectory and export as .ply.

Each frame's GT scanner gives a dense 3.2 m × 3.2 m patch directly under the robot.
As the robot walks, these patches stitch together into a much wider, paper-style
"complete terrain reconstruction" point cloud.

Optionally also adds the lidar pcl (accumulated) and a single G1 mesh at the
chosen end-frame position to anchor the scene visually.

Usage:
    python export_ply_aggregate.py --chunk /root/lidar_dataset_v3/chunk_00080.pt \\
        --start 0 --end 1000 --robot_at 500 --out scene_full.ply
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from export_ply import (build_g1_pointcloud, write_ply, colorize_z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", required=True,
                    help="one or more chunk paths")
    ap.add_argument("--step", type=int, default=2,
                    help="aggregate every Nth frame (avoid duplicates)")
    ap.add_argument("--ignore_episode", action="store_true",
                    help="aggregate across all episodes (useful since GT is world-frame)")
    ap.add_argument("--include_lidar", action="store_true", default=True)
    ap.add_argument("--include_robot", action="store_true", default=True)
    ap.add_argument("--out", default="scene_full.ply")
    args = ap.parse_args()

    gt_all, lidar_all = [], []
    n_frames = 0
    base_pos_for_robot = None
    base_quat_for_robot = None

    for chunk_path in args.chunks:
        print(f"[INFO] loading {chunk_path}")
        d = torch.load(chunk_path, map_location="cpu", weights_only=True)
        F = d["episode_id"].shape[0]

        # pick one frame's pose (middle of first chunk) as the robot anchor
        if base_pos_for_robot is None:
            mid = F // 2
            base_pos_for_robot = d["base_pos_world"][mid].numpy()
            base_quat_for_robot = d["base_quat_world"][mid].numpy()
            anchor_episode = d["episode_id"][mid].item()
        else:
            anchor_episode = None  # won't filter

        for t in range(0, F, args.step):
            if not args.ignore_episode and anchor_episode is not None:
                if d["episode_id"][t].item() != anchor_episode:
                    continue
            gt = d["gt_points_world"][t][d["gt_hit_mask"][t]].numpy()
            if gt.shape[0] > 0:
                gt_all.append(gt)
            if args.include_lidar:
                lid = d["lidar_points_world"][t][d["lidar_hit_mask"][t]].numpy()
                if lid.shape[0] > 0:
                    lidar_all.append(lid)
            n_frames += 1
    print(f"[INFO] aggregated {n_frames} frames")

    gt_pts = np.concatenate(gt_all, axis=0) if gt_all else np.zeros((0, 3))
    lidar_pts = np.concatenate(lidar_all, axis=0) if lidar_all else np.zeros((0, 3))
    print(f"[INFO] GT total: {gt_pts.shape[0]}  lidar total: {lidar_pts.shape[0]}")

    # color
    gt_colors = colorize_z(gt_pts, (140, 90, 50), (210, 180, 100))   # brown→tan
    lidar_colors = colorize_z(lidar_pts, (40, 70, 110), (180, 210, 240))   # blue→pale

    # robot at one pose
    robot_pts = np.zeros((0, 3))
    robot_colors = np.zeros((0, 3), dtype=np.uint8)
    if args.include_robot and base_pos_for_robot is not None:
        robot_pts, robot_colors = build_g1_pointcloud(base_pos_for_robot, base_quat_for_robot)
    print(f"[INFO] robot pts: {robot_pts.shape[0]}")

    all_pts = np.concatenate([gt_pts, lidar_pts, robot_pts], axis=0)
    all_colors = np.concatenate([gt_colors, lidar_colors, robot_colors], axis=0)
    print(f"[INFO] writing {all_pts.shape[0]} pts → {args.out}")
    write_ply(args.out, all_pts, all_colors)
    print(f"[SAVED] {args.out}")
    print(f"[NEXT] open with CloudCompare or MeshLab")


if __name__ == "__main__":
    main()
