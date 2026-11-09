"""Visualize a temporal sample as PLY for CloudCompare.

Two views per sample:
  *_world.ply  — all K+1 lidar frames + GT in WORLD frame, colored by age.
                 Reveals the robot's actual motion trajectory.
  *_local.ply  — same frames but ALL transformed into the current base frame at
                 time t. The CURRENT lidar's blind cone (gap around origin)
                 should be filled in by older frames. This is the proof that
                 motion fills the dead center.

Color legend (lidar):
  current frame (slot K, age 0)  → bright blue
  older frames                   → blue → cyan → green → orange → red
GT cloud:
  orange/tan
Markers:
  red sphere     pelvis at time t
  yellow sphere  lidar mount at time t (+0.65 m)
  green wireframe  3.2 m training cube around base at t

Usage:
    python viz_temporal.py --dataset_dir /root/lidar_dataset_fwd \\
        --history_n 10 --history_stride 5 \\
        --indices 50 100 200 --out_dir /tmp/temporal_viz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_temporal import TemporalLidarDataset


HALF = 1.6


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float32)


def write_ply(path, pts, cols):
    n = pts.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    body = np.empty(n, dtype=[("x","<f4"),("y","<f4"),("z","<f4"),
                              ("r","u1"),("g","u1"),("b","u1")])
    body["x"]=pts[:,0].astype(np.float32); body["y"]=pts[:,1].astype(np.float32); body["z"]=pts[:,2].astype(np.float32)
    body["r"]=cols[:,0]; body["g"]=cols[:,1]; body["b"]=cols[:,2]
    with open(path, "wb") as f:
        f.write(header); f.write(body.tobytes())


def color_by_age(slot: int, K: int):
    """Colormap: current (slot=K) bright blue, oldest (slot=0) red.
    Through cyan/green/orange.
    """
    t = slot / max(1, K)  # 0..1, 0=oldest, 1=current
    if t > 0.95:
        return np.array([60, 120, 255], np.uint8)   # current — strong blue
    # piecewise: red(0) -> orange(0.25) -> green(0.5) -> cyan(0.75) -> blue(1.0)
    stops = [
        (0.00, np.array([200, 30, 30])),
        (0.25, np.array([240,140, 50])),
        (0.50, np.array([100,200, 80])),
        (0.75, np.array([ 70,200,200])),
        (1.00, np.array([ 80,150,255])),
    ]
    for i in range(len(stops)-1):
        t0, c0 = stops[i]; t1, c1 = stops[i+1]
        if t0 <= t <= t1:
            a = (t - t0) / max(1e-6, t1 - t0)
            return ((1-a)*c0 + a*c1).astype(np.uint8)
    return stops[-1][1].astype(np.uint8)


def cube_wire(center, half, n_per_edge=120):
    h = half
    c = np.array([[-h,-h,-h],[h,-h,-h],[h,h,-h],[-h,h,-h],
                  [-h,-h, h],[h,-h, h],[h,h, h],[-h,h, h]], np.float32)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    out=[]
    for a,b in edges:
        t = np.linspace(0,1,n_per_edge).reshape(-1,1)
        out.append(c[a] + t*(c[b]-c[a]))
    return np.concatenate(out, axis=0) + center


def sphere_pts(center, r, n, rng):
    d = rng.normal(size=(n,3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    rr = r * (rng.random(n)**(1/3)).reshape(-1,1)
    return center + d*rr


def export_sample(s: dict, out_prefix: str, idx: int, max_per_frame: int = 5000):
    K1 = s["lidar_pcl"].shape[0]
    K = K1 - 1
    rng = np.random.default_rng(0)

    base_pos = s["base_pos"].numpy()      # (K+1, 3)
    base_quat = s["base_quat"].numpy()    # (K+1, 4) xyzw
    lidar_pcl = s["lidar_pcl"].numpy()    # (K+1, N, 3)
    lidar_mask = s["lidar_mask"].numpy()  # (K+1, N)
    lidar_origin = s["lidar_origin"].numpy()
    valid_history = s["valid_history"].numpy()
    age = s["frame_age_s"].numpy()
    gt_pts = s["gt_pcl"][:int(s["gt_n"].item())].numpy()

    base_pos_t = base_pos[K]
    base_quat_t = base_quat[K]
    R_t = quat_xyzw_to_R(base_quat_t)

    # ---------- WORLD view ----------
    pcs_w = []
    for slot in range(K1):
        if not valid_history[slot]:
            continue
        msk = lidar_mask[slot]
        pts = lidar_pcl[slot][msk]
        if pts.shape[0] > max_per_frame:
            pts = pts[rng.choice(pts.shape[0], max_per_frame, replace=False)]
        col = np.tile(color_by_age(slot, K), (pts.shape[0], 1))
        pcs_w.append((pts, col))
    # GT
    pcs_w.append((gt_pts, np.tile(np.array([220,170,90], np.uint8), (gt_pts.shape[0], 1))))
    # markers (at current pose)
    pcs_w.append((sphere_pts(base_pos_t, 0.08, 400, rng), np.tile(np.array([255,0,0], np.uint8), (400,1))))
    pcs_w.append((sphere_pts(lidar_origin[K], 0.05, 200, rng), np.tile(np.array([255,255,0], np.uint8), (200,1))))
    pcs_w.append((cube_wire(base_pos_t, HALF), np.tile(np.array([0,255,0], np.uint8), (1440,1))))

    pts_w = np.concatenate([p for p,_ in pcs_w], axis=0)
    cols_w = np.concatenate([c for _,c in pcs_w], axis=0)
    out_w = f"{out_prefix}_world.ply"
    write_ply(out_w, pts_w, cols_w)

    # ---------- LOCAL view (everything in base-at-t frame) ----------
    pcs_l = []
    counts_in_cube = []
    for slot in range(K1):
        if not valid_history[slot]:
            continue
        msk = lidar_mask[slot]
        pts = lidar_pcl[slot][msk]                              # world frame
        pts_l = (R_t.T @ (pts - base_pos_t).T).T                # to base-at-t local
        # clip view to a generous box for readability (4× cube)
        in_view = (np.abs(pts_l) < 4 * HALF).all(-1)
        pts_l = pts_l[in_view]
        in_cube = (np.abs(pts_l) < HALF).all(-1)
        counts_in_cube.append((slot, age[slot], int(in_cube.sum())))
        if pts_l.shape[0] > max_per_frame:
            pts_l = pts_l[rng.choice(pts_l.shape[0], max_per_frame, replace=False)]
        col = np.tile(color_by_age(slot, K), (pts_l.shape[0], 1))
        pcs_l.append((pts_l, col))
    # GT in local frame
    gt_l = (R_t.T @ (gt_pts - base_pos_t).T).T
    in_view = (np.abs(gt_l) < 4 * HALF).all(-1)
    gt_l = gt_l[in_view]
    pcs_l.append((gt_l, np.tile(np.array([220,170,90], np.uint8), (gt_l.shape[0], 1))))
    # markers (now centered at origin)
    pcs_l.append((sphere_pts(np.zeros(3), 0.08, 400, rng), np.tile(np.array([255,0,0], np.uint8), (400,1))))
    pcs_l.append((sphere_pts(np.array([0,0,0.65]), 0.05, 200, rng), np.tile(np.array([255,255,0], np.uint8), (200,1))))
    pcs_l.append((cube_wire(np.zeros(3), HALF), np.tile(np.array([0,255,0], np.uint8), (1440,1))))

    pts_l = np.concatenate([p for p,_ in pcs_l], axis=0)
    cols_l = np.concatenate([c for _,c in pcs_l], axis=0)
    out_l = f"{out_prefix}_local.ply"
    write_ply(out_l, pts_l, cols_l)

    print(f"[idx={idx}] env={int(s['env_id'].item())} ep={int(s['episode_id'].item())} step={int(s['step_in_ep'].item())}", flush=True)
    print(f"  valid_history: {valid_history.tolist()}", flush=True)
    print(f"  per-slot lidar voxels in 3.2m cube (after transform):", flush=True)
    for slot, a, n in counts_in_cube:
        print(f"    slot={slot:2d} age={a:+.2f}s  pts_in_cube={n}", flush=True)
    print(f"  → {out_w}", flush=True)
    print(f"  → {out_l}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--indices", type=int, nargs="+", required=True,
                    help="Sample indices to export.")
    ap.add_argument("--history_n", type=int, default=10)
    ap.add_argument("--history_stride", type=int, default=5)
    ap.add_argument("--max_per_frame", type=int, default=5000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ds = TemporalLidarDataset(args.dataset_dir, tiles=None,
                              history_n=args.history_n,
                              history_stride=args.history_stride,
                              max_lidar_pts=20000, max_gt_pts=200000)
    print(f"[INFO] dataset has {len(ds)} samples", flush=True)
    for i in args.indices:
        if i < 0 or i >= len(ds):
            print(f"[WARN] idx {i} out of range")
            continue
        s = ds[i]
        prefix = os.path.join(args.out_dir, f"temporal_{i:05d}")
        export_sample(s, prefix, i, max_per_frame=args.max_per_frame)


if __name__ == "__main__":
    main()
