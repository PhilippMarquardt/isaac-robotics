"""Export PLY of the actual training region: lidar + GT clipped to 3.2 m cube
in base-local frame, plus the dilated reachable mask used by mask_to_lidar loss.

Colors:
  blue   (60,120,255)  lidar voxels (5 cm) — input
  orange (220,170,90)  GT voxels (5 cm)    — supervision target
  cyan   (140,200,255) reachable mask voxels — loss region (lidar dilated by k)
  green  wireframe     3.2 m cube outline
  red    sphere        base/pelvis
  yellow sphere        lidar origin (+0.65 m above pelvis)

Usage:
    python export_train_region.py --chunk /root/lidar_dataset_v6/chunk_00010.pt \
        --frames 100 500 --out_dir /tmp --dilate 3
"""
import argparse, os, re
import numpy as np
import torch

GRID = 64
VOXEL_SIZE = 0.05
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


def voxelize_local(pts_l):
    """Drop pts outside cube, return unique voxel CENTERS in local frame."""
    in_box = ((pts_l > -HALF) & (pts_l < HALF)).all(-1)
    p = pts_l[in_box]
    if p.shape[0] == 0:
        return np.zeros((0,3), np.float32), np.zeros((0,3), np.int64)
    idx = np.clip(((p + HALF) / VOXEL_SIZE).astype(np.int64), 0, GRID-1)
    # unique
    flat = idx[:,0]*GRID*GRID + idx[:,1]*GRID + idx[:,2]
    _, keep = np.unique(flat, return_index=True)
    idx = idx[keep]
    centers = (idx.astype(np.float32) + 0.5) * VOXEL_SIZE - HALF
    return centers, idx


def dilate_idx(idx, k):
    """3D max-dilation: return voxel centers of all voxels within k Linf-distance of any input idx."""
    if idx.shape[0] == 0:
        return np.zeros((0,3), np.float32)
    occ = np.zeros((GRID,GRID,GRID), bool)
    occ[idx[:,0], idx[:,1], idx[:,2]] = True
    out = np.zeros_like(occ)
    for dx in range(-k, k+1):
        for dy in range(-k, k+1):
            for dz in range(-k, k+1):
                src = occ
                xs = slice(max(0,-dx), GRID-max(0,dx))
                xd = slice(max(0,dx), GRID-max(0,-dx))
                ys = slice(max(0,-dy), GRID-max(0,dy))
                yd = slice(max(0,dy), GRID-max(0,-dy))
                zs = slice(max(0,-dz), GRID-max(0,dz))
                zd = slice(max(0,dz), GRID-max(0,-dz))
                out[xd,yd,zd] |= src[xs,ys,zs]
    ii = np.argwhere(out)
    return (ii.astype(np.float32) + 0.5) * VOXEL_SIZE - HALF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--dilate", type=int, default=3, help="reachable-mask dilation radius (voxels)")
    ap.add_argument("--no_reachable", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    d = torch.load(args.chunk, map_location="cpu", weights_only=True)
    chunk_id = re.search(r"chunk_(\d+)", os.path.basename(args.chunk))
    chunk_id = chunk_id.group(1) if chunk_id else "0"

    rng = np.random.default_rng(0)
    for t in args.frames:
        base = d["base_pos_world"][t].numpy()
        bquat = d["base_quat_world"][t].numpy()
        R = quat_xyzw_to_R(bquat)

        lid_w = d["lidar_points_world"][t][d["lidar_hit_mask"][t]].numpy()
        n_gt = int(d["gt_n_points"][t].item())
        gt_w = d["gt_points_world"][t, :n_gt].numpy()
        lid_l = (R.T @ (lid_w - base).T).T
        gt_l = (R.T @ (gt_w - base).T).T

        lid_centers, lid_idx = voxelize_local(lid_l)
        gt_centers, _ = voxelize_local(gt_l)

        pcs = [
            (lid_centers, np.array([60,120,255], np.uint8)),
            (gt_centers,  np.array([220,170,90], np.uint8)),
        ]
        if not args.no_reachable:
            reach_centers = dilate_idx(lid_idx, args.dilate)
            pcs.append((reach_centers, np.array([140,200,255], np.uint8)))

        # markers
        pcs.append((cube_wire(np.zeros(3), HALF), np.array([0,255,0], np.uint8)))
        pcs.append((sphere_pts(np.zeros(3), 0.08, 400, rng), np.array([255,0,0], np.uint8)))
        pcs.append((sphere_pts(np.array([0,0,0.65]), 0.05, 200, rng), np.array([255,255,0], np.uint8)))

        pts = np.concatenate([p for p,_ in pcs], axis=0)
        cols = np.concatenate([np.tile(c,(p.shape[0],1)) for p,c in pcs], axis=0)
        out = os.path.join(args.out_dir, f"train_region_{chunk_id}_{t:05d}.ply")
        write_ply(out, pts, cols)
        print(f"[t={t}] lidar={lid_centers.shape[0]}vx gt={gt_centers.shape[0]}vx "
              f"reach={pcs[2][0].shape[0] if not args.no_reachable else 0}vx → {out}", flush=True)


if __name__ == "__main__":
    main()
