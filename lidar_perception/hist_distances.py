"""Plot histograms of lidar hit distances + GT point distances around the robot.

Helps visualise which radial shells get hit, and where lidar vs GT differ spatially.
"""
import argparse, os, glob
import torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="/root/lidar_dataset_v6")
    ap.add_argument("--n_frames", type=int, default=100, help="random frames to aggregate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    files = sorted(glob.glob(os.path.join(args.dataset_dir, "chunk_*.pt")))
    # collect from random frames across all chunks
    radii_lid_horiz, radii_lid_3d = [], []
    radii_gt_horiz, radii_gt_3d = [], []
    z_lid_rel, z_gt_rel = [], []
    fps_per_chunk = max(1, args.n_frames // len(files))
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=True)
        F = d["episode_id"].shape[0]
        sel = rng.choice(F, size=min(fps_per_chunk, F), replace=False)
        for t in sel:
            base = d["base_pos_world"][t].numpy()
            lid_o = d["lidar_origin_world"][t].numpy()
            hits = d["lidar_points_world"][t][d["lidar_hit_mask"][t]].numpy()
            n = int(d["gt_n_points"][t].item())
            gt = d["gt_points_world"][t, :n].numpy()
            # distances from lidar origin (lidar reach perspective)
            lh = np.sqrt(((hits[:, :2] - lid_o[:2])**2).sum(-1))
            l3 = np.linalg.norm(hits - lid_o, axis=-1)
            # distances from base for GT (cube reference)
            gh = np.sqrt(((gt[:, :2] - base[:2])**2).sum(-1))
            g3 = np.linalg.norm(gt - base, axis=-1)
            radii_lid_horiz.append(lh); radii_lid_3d.append(l3)
            radii_gt_horiz.append(gh); radii_gt_3d.append(g3)
            z_lid_rel.append(hits[:, 2] - base[2])
            z_gt_rel.append(gt[:, 2] - base[2])
    radii_lid_horiz = np.concatenate(radii_lid_horiz)
    radii_lid_3d = np.concatenate(radii_lid_3d)
    radii_gt_horiz = np.concatenate(radii_gt_horiz)
    radii_gt_3d = np.concatenate(radii_gt_3d)
    z_lid_rel = np.concatenate(z_lid_rel)
    z_gt_rel = np.concatenate(z_gt_rel)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")

    # 1. horizontal radius histogram, log scale
    ax = axes[0, 0]
    bins = np.linspace(0, 30, 61)
    ax.hist(radii_lid_horiz, bins=bins, alpha=0.6, color="#3b78c4", label=f"lidar (n={len(radii_lid_horiz)})")
    ax.hist(radii_gt_horiz, bins=bins, alpha=0.6, color="#dc8a3b", label=f"GT (n={len(radii_gt_horiz)})")
    ax.set_xlabel("horizontal distance from robot [m]")
    ax.set_ylabel("count")
    ax.set_title("horizontal radius histogram (XY plane)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_yscale("log")

    # 2. 3D distance histogram
    ax = axes[0, 1]
    bins = np.linspace(0, 30, 61)
    ax.hist(radii_lid_3d, bins=bins, alpha=0.6, color="#3b78c4", label="lidar")
    ax.hist(radii_gt_3d, bins=bins, alpha=0.6, color="#dc8a3b", label="GT")
    ax.set_xlabel("3D distance from robot [m]")
    ax.set_ylabel("count (log)")
    ax.set_title("3D distance histogram")
    ax.legend(); ax.grid(alpha=0.3); ax.set_yscale("log")

    # 3. z-relative-to-base histogram (vertical extent)
    ax = axes[1, 0]
    zbins = np.linspace(-3, 3, 61)
    ax.hist(z_lid_rel, bins=zbins, alpha=0.6, color="#3b78c4", label="lidar")
    ax.hist(z_gt_rel, bins=zbins, alpha=0.6, color="#dc8a3b", label="GT")
    ax.axvline(0, color="black", ls="--", alpha=0.5, label="pelvis")
    ax.axvline(0.65, color="red", ls=":", alpha=0.5, label="lidar mount (+0.65 m)")
    ax.set_xlabel("z relative to pelvis [m]")
    ax.set_ylabel("count (log)")
    ax.set_title("vertical-axis histogram")
    ax.legend(); ax.grid(alpha=0.3); ax.set_yscale("log")

    # 4. fraction of lidar hits inside cube vs cube extent
    ax = axes[1, 1]
    extents = np.linspace(0.5, 30, 60)
    frac_lid = [(radii_lid_horiz < e).mean() for e in extents]
    frac_gt = [(radii_gt_horiz < e).mean() for e in extents]
    ax.plot(extents, frac_lid, color="#3b78c4", label="lidar", lw=2)
    ax.plot(extents, frac_gt, color="#dc8a3b", label="GT", lw=2)
    ax.axvline(5, color="green", ls="--", alpha=0.6, label="GT cube half-extent (5 m)")
    ax.axvline(1.6, color="purple", ls=":", alpha=0.6, label="train cube half-extent (1.6 m)")
    ax.set_xlabel("horizontal radius [m]")
    ax.set_ylabel("fraction of points within radius")
    ax.set_title("CDF of horizontal distance")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

    plt.suptitle(f"Lidar vs GT distance distributions  ({args.dataset_dir})", y=0.995)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[SAVED] {args.out}")
    print(f"lidar hits: {len(radii_lid_horiz)}  GT pts: {len(radii_gt_horiz)}")
    print(f"lidar horiz: median={np.median(radii_lid_horiz):.2f} m, p95={np.percentile(radii_lid_horiz, 95):.2f} m")
    print(f"GT horiz:    median={np.median(radii_gt_horiz):.2f} m, p95={np.percentile(radii_gt_horiz, 95):.2f} m")


if __name__ == "__main__":
    main()
