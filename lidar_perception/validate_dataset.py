"""Validate a chunked lidar dataset on disk.

Loads every chunk, checks shapes / dtypes / NaN-free / monotonic episode counters,
prints per-chunk hit-rate stats, and runs a few sanity assertions.

Usage:
    python validate_dataset.py /workspace/lidar_dataset_v1
    python validate_dataset.py /workspace/lidar_dataset_smoke --plot
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch


EXPECTED_KEYS = {
    "env_id",
    "episode_id",
    "step_in_episode",
    "timestamp_sim",
    "lidar_points_world",
    "lidar_hit_mask",
    "lidar_origin_world",
    "lidar_quat_world",
    "base_pos_world",
    "base_quat_world",
    "gt_points_world",
    "gt_hit_mask",
}


def check_chunk(path: str) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=True)
    keys = set(d.keys())
    missing = EXPECTED_KEYS - keys
    extra = keys - EXPECTED_KEYS
    F = d["episode_id"].shape[0]
    issues = []
    if missing:
        issues.append(f"missing keys: {missing}")
    if extra:
        issues.append(f"unexpected keys: {extra}")

    # shape checks
    expected_shapes = {
        "env_id": (F,),
        "episode_id": (F,),
        "step_in_episode": (F,),
        "timestamp_sim": (F,),
        "lidar_origin_world": (F, 3),
        "lidar_quat_world": (F, 4),
        "base_pos_world": (F, 3),
        "base_quat_world": (F, 4),
    }
    for k, shape in expected_shapes.items():
        if k in d and tuple(d[k].shape) != shape:
            issues.append(f"{k} shape {tuple(d[k].shape)} != {shape}")

    # variable-but-consistent dims
    R_lidar = d["lidar_points_world"].shape[1] if "lidar_points_world" in d else None
    R_gt = d["gt_points_world"].shape[1] if "gt_points_world" in d else None
    if R_lidar is None or R_gt is None:
        issues.append("missing point cloud tensor")
        return _result(path, F, R_lidar, R_gt, 0.0, 0.0, issues)
    if d["lidar_points_world"].shape != (F, R_lidar, 3):
        issues.append(f"lidar_points_world shape {tuple(d['lidar_points_world'].shape)}")
    if d["lidar_hit_mask"].shape != (F, R_lidar):
        issues.append(f"lidar_hit_mask shape {tuple(d['lidar_hit_mask'].shape)}")
    if d["gt_points_world"].shape != (F, R_gt, 3):
        issues.append(f"gt_points_world shape {tuple(d['gt_points_world'].shape)}")
    if d["gt_hit_mask"].shape != (F, R_gt):
        issues.append(f"gt_hit_mask shape {tuple(d['gt_hit_mask'].shape)}")

    # finite-only inside masked points (zeros where masked, must be finite)
    for k in ("lidar_points_world", "gt_points_world", "lidar_origin_world",
              "lidar_quat_world", "base_pos_world", "base_quat_world", "timestamp_sim"):
        t = d[k].float()
        if not torch.isfinite(t).all():
            issues.append(f"{k} has non-finite values")

    # episode/step monotonicity per env
    ids = d["env_id"]
    eps = d["episode_id"]
    steps = d["step_in_episode"]
    for env in ids.unique().tolist():
        sel = ids == env
        e = eps[sel].tolist()
        s = steps[sel].tolist()
        for i in range(1, len(e)):
            if e[i] == e[i - 1] and s[i] != s[i - 1] + 1:
                issues.append(f"env {env}: step jump within episode {e[i]}: {s[i - 1]} -> {s[i]}")
                break
            if e[i] < e[i - 1]:
                issues.append(f"env {env}: episode regressed {e[i - 1]} -> {e[i]}")
                break

    # quaternion ~unit length
    for k in ("lidar_quat_world", "base_quat_world"):
        n = torch.linalg.norm(d[k].float(), dim=-1)
        if (n < 0.95).any() or (n > 1.05).any():
            issues.append(f"{k} norms not ~1 (range {float(n.min()):.3f}..{float(n.max()):.3f})")

    lidar_hr = float(d["lidar_hit_mask"].float().mean())
    gt_hr = float(d["gt_hit_mask"].float().mean())
    return _result(path, F, R_lidar, R_gt, lidar_hr, gt_hr, issues)


def _result(path, F, R_lidar, R_gt, lidar_hr, gt_hr, issues):
    return {
        "path": path,
        "F": F,
        "R_lidar": R_lidar,
        "R_gt": R_gt,
        "lidar_hit_rate": lidar_hr,
        "gt_hit_rate": gt_hr,
        "issues": issues,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("--plot", action="store_true",
                    help="Save matplotlib viz of one frame's lidar + GT pcl + lidar pose.")
    args = ap.parse_args()

    chunks = sorted(glob.glob(os.path.join(args.dataset_dir, "*.pt")))
    if not chunks:
        print(f"[FAIL] no .pt files in {args.dataset_dir}")
        sys.exit(1)
    print(f"[INFO] {len(chunks)} chunks in {args.dataset_dir}")

    total_frames = 0
    total_issues = 0
    sums = {"lidar_hr": 0.0, "gt_hr": 0.0}
    for c in chunks:
        r = check_chunk(c)
        total_frames += r["F"]
        total_issues += len(r["issues"])
        sums["lidar_hr"] += r["lidar_hit_rate"] * r["F"]
        sums["gt_hr"] += r["gt_hit_rate"] * r["F"]
        flag = "OK" if not r["issues"] else f"FAIL ({len(r['issues'])})"
        print(
            f"  {os.path.basename(c)}: F={r['F']:5d}  R_lidar={r['R_lidar']}  R_gt={r['R_gt']}  "
            f"lidar_hr={r['lidar_hit_rate']:.3f}  gt_hr={r['gt_hit_rate']:.3f}  [{flag}]"
        )
        for issue in r["issues"]:
            print(f"      ! {issue}")

    print(
        f"[SUMMARY] frames={total_frames}  issues={total_issues}  "
        f"avg lidar_hr={sums['lidar_hr'] / max(total_frames, 1):.3f}  "
        f"avg gt_hr={sums['gt_hr'] / max(total_frames, 1):.3f}"
    )

    if args.plot and chunks:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib not available, skipping plot")
            return

        d = torch.load(chunks[0], map_location="cpu", weights_only=True)
        f = 0
        lidar = d["lidar_points_world"][f]
        mask = d["lidar_hit_mask"][f]
        gt = d["gt_points_world"][f]
        gt_mask = d["gt_hit_mask"][f]
        lp = d["lidar_origin_world"][f]
        bp = d["base_pos_world"][f]

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        sub = lidar[mask][::10]
        if sub.shape[0] > 0:
            ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=1, c="tab:blue", alpha=0.4)
        ax.scatter([lp[0]], [lp[1]], [lp[2]], s=40, c="red", marker="^", label="lidar")
        ax.scatter([bp[0]], [bp[1]], [bp[2]], s=40, c="orange", marker="o", label="base")
        ax.set_title("lidar pcl (world)")
        ax.legend()

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        sub2 = gt[gt_mask]
        if sub2.shape[0] > 0:
            ax2.scatter(sub2[:, 0], sub2[:, 1], sub2[:, 2], s=2, c="tab:green", alpha=0.6)
        ax2.scatter([bp[0]], [bp[1]], [bp[2]], s=40, c="orange", marker="o")
        ax2.set_title("GT pcl (world)")

        out = os.path.join(args.dataset_dir, "validate_frame0.png")
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        print(f"[PLOT] {out}")


if __name__ == "__main__":
    main()
