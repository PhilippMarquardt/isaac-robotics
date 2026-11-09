"""Z transfer / cluster analysis.

Hypothesis: Z encodes a state-portable motion vector. To test:
  - Pick a Z_donor.
  - Apply it to N different H_t's.
  - The predicted delta = T(H_t, Z) - base(H_t) should be more similar across
    different H_t's than predictions made with different Z values.

Concretely we measure:
  same-Z spread: avg cos(delta_i, delta_j) for i,j with the SAME Z applied to different H_t.
  diff-Z spread: avg cos(delta_i, delta_j) for i,j with DIFFERENT Zs applied to same/different H_t.

If same-Z >> diff-Z, Z is encoding a state-portable direction.

Also: K-means cluster Z*'s and report Z self-clustering (silhouette-style).
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from models.transition_residual import ResidualTransitionModel
from models.fixed_reducer import FixedSpatialReducer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_zs", type=int, default=64, help="number of donor Zs")
    ap.add_argument("--n_ht", type=int, default=64, help="number of H_t states to apply each Z to")
    ap.add_argument("--n_clusters", type=int, default=16)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tr = ResidualTransitionModel(cfg["state_dim"], cfg["action_dim"], cfg["future_tokens"],
                                 depth=cfg["transition_depth"]).to(device)
    tr.load_state_dict(ckpt["transition_model"])
    for p in tr.parameters(): p.requires_grad_(False)
    tr.eval()
    # base reducer is the same parameter-free FixedSpatialReducer(16) used by tr.base
    base_reducer = tr.base

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t_pool = bundle["H_t_train"][:2000].to(device)  # use up to 2000 states
    Z_pool = bundle["Z_star_train"][:2000].to(device)

    # Pick n_zs donor Zs at random from train pool
    z_idx = torch.randperm(Z_pool.shape[0], device=device)[: args.n_zs]
    donor_Zs = Z_pool[z_idx]  # [n_zs, Ka, Da]
    # Pick n_ht random H_t states (different set, not paired with these Zs)
    ht_idx = torch.randperm(H_t_pool.shape[0], device=device)[: args.n_ht]
    Hts = H_t_pool[ht_idx]  # [n_ht, Ns, Ds]

    # Compute delta[z, h] = T(H_t_h, Z_z) - base(H_t_h)  -> [n_zs, n_ht, 16, Ds]
    deltas = torch.zeros(args.n_zs, args.n_ht, 16, Hts.shape[-1], device=device)
    with torch.no_grad():
        for zi in range(args.n_zs):
            z = donor_Zs[zi:zi+1].expand(args.n_ht, -1, -1)
            pred = tr(Hts, z)  # [n_ht, 16, D]
            base = base_reducer(Hts)
            deltas[zi] = pred - base

    # Flatten + normalize per (z, h)
    delta_flat = deltas.reshape(args.n_zs, args.n_ht, -1)
    delta_flat = F.normalize(delta_flat, dim=-1)  # [n_zs, n_ht, 16*D]

    # Same-Z spread: average cosine across different h, same z
    same_z_cos = 0.0; same_z_count = 0
    for zi in range(args.n_zs):
        m = delta_flat[zi]  # [n_ht, D]
        sim = m @ m.T  # [n_ht, n_ht]
        # average off-diagonal
        off = sim - torch.eye(args.n_ht, device=device)
        avg = off.sum() / (args.n_ht * (args.n_ht - 1))
        same_z_cos += avg.item(); same_z_count += 1
    same_z_cos /= same_z_count
    print(f"Same-Z avg cos(delta(z,h_i), delta(z,h_j))  = {same_z_cos:+.4f}")

    # Diff-Z spread: avg cosine across different z, same h
    diff_z_cos = 0.0; diff_z_count = 0
    for hi in range(args.n_ht):
        m = delta_flat[:, hi]  # [n_zs, D]
        sim = m @ m.T
        off = sim - torch.eye(args.n_zs, device=device)
        avg = off.sum() / (args.n_zs * (args.n_zs - 1))
        diff_z_cos += avg.item(); diff_z_count += 1
    diff_z_cos /= diff_z_count
    print(f"Diff-Z avg cos(delta(z_a,h), delta(z_b,h))  = {diff_z_cos:+.4f}")

    # All-pairs spread (different z AND different h)
    all_flat = delta_flat.reshape(-1, delta_flat.shape[-1])  # [n_zs*n_ht, D]
    sim = all_flat @ all_flat.T
    off = sim - torch.eye(all_flat.shape[0], device=device)
    avg_all = off.sum() / (all_flat.shape[0] * (all_flat.shape[0] - 1))
    print(f"All-pairs avg cos                            = {avg_all.item():+.4f}")

    # Selectivity ratio
    ratio = same_z_cos / max(diff_z_cos, 1e-6)
    print(f"\nSelectivity: same_z_cos / diff_z_cos = {ratio:.2f}")

    # Quick k-means on Z*
    print(f"\nk-means on Z* (k={args.n_clusters})...")
    Z_flat = Z_pool.flatten(1)  # [N, 2048]
    Z_flat = F.normalize(Z_flat, dim=-1)
    # init random
    g = torch.Generator(device=device).manual_seed(42)
    centers = Z_flat[torch.randperm(Z_flat.shape[0], device=device, generator=g)[: args.n_clusters]].clone()
    for it in range(20):
        sim = Z_flat @ centers.T
        assign = sim.argmax(-1)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(args.n_clusters, device=device)
        for c in range(args.n_clusters):
            mask = (assign == c)
            if mask.any():
                new_centers[c] = Z_flat[mask].mean(0)
                counts[c] = mask.sum().float()
        new_centers = F.normalize(new_centers, dim=-1)
        if torch.allclose(new_centers, centers, atol=1e-5):
            break
        centers = new_centers
    print(f"  cluster sizes: min={counts.min().item():.0f}  max={counts.max().item():.0f}  median={counts.median().item():.0f}")

    # Within-cluster vs cross-cluster Z* cosine
    intra_sum = 0.0; intra_count = 0
    for c in range(args.n_clusters):
        mask = (assign == c)
        if mask.sum() < 2: continue
        Zc = Z_flat[mask]
        s = Zc @ Zc.T
        off = s - torch.eye(s.shape[0], device=device)
        intra_sum += off.sum().item(); intra_count += s.shape[0] * (s.shape[0] - 1)
    intra_cos = intra_sum / max(intra_count, 1)
    sample_a = Z_flat[torch.randperm(Z_flat.shape[0])[:2000]]
    sample_b = Z_flat[torch.randperm(Z_flat.shape[0])[:2000]]
    overall_cos = (sample_a * sample_b).sum(-1).mean().item()
    print(f"  within-cluster Z* avg cos = {intra_cos:+.4f}")
    print(f"  overall random pair Z* cos = {overall_cos:+.4f}")

    results = {
        "same_z_cos": same_z_cos, "diff_z_cos": diff_z_cos, "all_pairs_cos": avg_all.item(),
        "selectivity_ratio": ratio,
        "kmeans_intra_cos": intra_cos, "kmeans_overall_cos": overall_cos,
        "n_clusters": args.n_clusters, "cluster_size_min": counts.min().item(), "cluster_size_max": counts.max().item(),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
