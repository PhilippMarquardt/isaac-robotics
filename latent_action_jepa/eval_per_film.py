"""Per-film diagnostics and cluster-by-film analysis on a multi-video bundle.

Splits val by `video_id_val` and runs the standard probes on each film slice.
Also fits k-means on training Z* and reports cluster purity by film.
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from models.transition_residual import ResidualTransitionModel


def retrieval(q, k, N, name=""):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    mr = (r == gt).float().argmax(-1).float().mean().item() + 1
    if name:
        print(f"  {name:30s} top1={top1:.3f}  top5={top5:.3f}  mr={mr:.2f}")
    return top1, top5, mr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_clusters", type=int, default=16)
    args = ap.parse_args()
    device = "cuda"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tr = ResidualTransitionModel(cfg["state_dim"], cfg["action_dim"], cfg["future_tokens"],
                                 depth=cfg["transition_depth"]).to(device)
    tr.load_state_dict(ckpt["transition_model"])
    for p in tr.parameters(): p.requires_grad_(False)
    tr.eval()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t = bundle["H_t_val"].to(device); H_tp = bundle["H_tp_val"].to(device)
    Z = bundle["Z_star_val"].to(device)
    vid = bundle["video_id_val"].to(device)
    H_t_train = bundle["H_t_train"].to(device)
    Z_train = bundle["Z_star_train"].to(device)
    vid_train = bundle["video_id_train"].to(device)
    film_names = bundle["film_names"]
    print(f"films: {film_names}")
    print(f"val N={H_t.shape[0]}  train N={H_t_train.shape[0]}")

    results = {"per_film": {}, "global": {}, "k_means": {}}

    print("\n--- per-film val diagnostics ---")
    for fi, fname in enumerate(film_names):
        mask = (vid == fi)
        N = mask.sum().item()
        if N < 5: continue
        Ht_f = H_t[mask]; Htp_f = H_tp[mask]; Z_f = Z[mask]
        std_z = Z_f.flatten(1).std(0).mean().item()
        with torch.no_grad():
            t1_c, _, _ = retrieval(tr(Ht_f, Z_f).mean(1), Htp_f.mean(1), N, f"[{fname[:14]}] correct_Z")
            t1_z, _, _ = retrieval(tr(Ht_f, torch.zeros_like(Z_f)).mean(1), Htp_f.mean(1), N, f"[{fname[:14]}] zero_Z")
            t1_r, _, _ = retrieval(tr(Ht_f, torch.randn_like(Z_f) * std_z).mean(1), Htp_f.mean(1), N, f"[{fname[:14]}] random_Z")
            t1_zo, _, _ = retrieval(tr(torch.zeros_like(Ht_f), Z_f).mean(1), Htp_f.mean(1), N, f"[{fname[:14]}] Z_only")
            t1_cp, _, _ = retrieval(Ht_f.mean(1), Htp_f.mean(1), N, f"[{fname[:14]}] copy_Ht")
            # wrong_H within-film
            perm = torch.randperm(N, device=device, generator=torch.Generator(device=device).manual_seed(0))
            t1_wh, _, _ = retrieval(tr(Ht_f[perm], Z_f).mean(1), Htp_f.mean(1), N, f"[{fname[:14]}] wrong_H (within-film)")
        results["per_film"][fname] = {"N": N, "correct_Z": t1_c, "zero_Z": t1_z, "random_Z": t1_r,
                                       "Z_only": t1_zo, "copy_Ht": t1_cp, "wrong_H_within": t1_wh}

    # ---- k-means cluster purity by film ----
    print(f"\n--- k-means(k={args.n_clusters}) on train Z* ---")
    Zf = F.normalize(Z_train.flatten(1), dim=-1)
    g = torch.Generator(device=device).manual_seed(0)
    init_idx = torch.randperm(Zf.shape[0], device=device, generator=g)[: args.n_clusters]
    centers = Zf[init_idx].clone()
    for _ in range(30):
        sim = Zf @ centers.T
        assign = sim.argmax(-1)
        new = torch.zeros_like(centers)
        cnt = torch.zeros(args.n_clusters, device=device)
        for c in range(args.n_clusters):
            m = (assign == c)
            if m.any():
                new[c] = Zf[m].mean(0); cnt[c] = m.sum().float()
        new = F.normalize(new, dim=-1)
        if torch.allclose(new, centers, atol=1e-5): break
        centers = new

    # Per-cluster film distribution
    print(f"\n--- cluster purity by film ---")
    n_films = len(film_names)
    cluster_purity = []
    for c in range(args.n_clusters):
        m = (assign == c)
        if m.sum() == 0: continue
        film_counts = torch.bincount(vid_train[m], minlength=n_films).float()
        total = film_counts.sum().item()
        dist = film_counts / total
        max_frac = dist.max().item()
        max_film = film_names[dist.argmax().item()]
        entropy = -(dist.clamp_min(1e-9) * dist.clamp_min(1e-9).log()).sum().item() / float(torch.log(torch.tensor(n_films)).item())  # normalized
        cluster_purity.append({"cluster": c, "size": int(total), "max_film": max_film, "max_frac": max_frac, "norm_entropy": entropy})
        print(f"  cluster {c:2d} (n={int(total):4d}): max-film={max_film[:12]} {max_frac*100:5.1f}%  norm_entropy={entropy:.3f}")

    # summary stats
    avg_max_frac = sum(c["max_frac"] for c in cluster_purity) / len(cluster_purity)
    avg_entropy = sum(c["norm_entropy"] for c in cluster_purity) / len(cluster_purity)
    print(f"\nAvg max-film fraction = {avg_max_frac:.3f}  (1.0 = perfect film-purity, 1/N_films = perfect-mixed)")
    print(f"Avg normalized entropy = {avg_entropy:.3f}  (0 = single-film cluster, 1 = uniform mix)")
    results["k_means"] = {"avg_max_film_frac": avg_max_frac, "avg_norm_entropy": avg_entropy,
                           "clusters": cluster_purity, "n_films": n_films}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
