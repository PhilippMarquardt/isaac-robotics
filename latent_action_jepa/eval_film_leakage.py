"""Film-leakage diagnostics.

Trains linear probes:
  H_t -> film_id        (baseline: H_t naturally carries film appearance)
  Z*  -> film_id        (desired LOW: Z should encode action, not film mode)

Compares H_t probe vs Z probe accuracy on held-out val.

Also reports cluster purity stats (max-film fraction, normalized entropy).
"""
import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def fit_logistic_probe(X_train, y_train, X_val, y_val, n_classes, steps=2000, lr=1e-3, wd=1e-4):
    """Linear probe with cross-entropy."""
    device = X_train.device
    d = X_train.shape[-1]
    head = nn.Linear(d, n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    bs = min(256, X_train.shape[0])
    for step in range(steps):
        idx = torch.randint(0, X_train.shape[0], (bs,), device=device)
        logits = head(X_train[idx])
        loss = F.cross_entropy(logits, y_train[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        tr_acc = (head(X_train).argmax(-1) == y_train).float().mean().item()
        va_acc = (head(X_val).argmax(-1) == y_val).float().mean().item()
    return tr_acc, va_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_clusters", type=int, default=32)
    ap.add_argument("--probe_steps", type=int, default=2000)
    args = ap.parse_args()
    device = "cuda"

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t_tr = bundle["H_t_train"].to(device); H_tp_tr = bundle["H_tp_train"].to(device)
    Z_tr = bundle["Z_star_train"].to(device)
    vid_tr = bundle["video_id_train"].to(device)
    H_t_va = bundle["H_t_val"].to(device); H_tp_va = bundle["H_tp_val"].to(device)
    Z_va = bundle["Z_star_val"].to(device)
    vid_va = bundle["video_id_val"].to(device)
    film_names = bundle["film_names"]
    n_films = len(film_names)
    print(f"films: {film_names}  n={n_films}")
    print(f"train {H_t_tr.shape[0]}  val {H_t_va.shape[0]}")
    chance = 1.0 / n_films
    print(f"film chance = {chance:.4f}")

    results = {"n_films": n_films, "film_chance": chance}

    # ---- H_t -> film_id probe ----
    print("\nH_t -> film probe")
    Ht_tr_pool = H_t_tr.mean(1)
    Ht_va_pool = H_t_va.mean(1)
    tr_acc, va_acc = fit_logistic_probe(Ht_tr_pool, vid_tr, Ht_va_pool, vid_va, n_films, steps=args.probe_steps)
    print(f"  H_t-pool probe:  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}")
    results["Ht_probe"] = {"train_acc": tr_acc, "val_acc": va_acc}

    # H_t flatten probe (richer)
    Ht_tr_flat = H_t_tr.flatten(1)
    Ht_va_flat = H_t_va.flatten(1)
    tr_acc, va_acc = fit_logistic_probe(Ht_tr_flat, vid_tr, Ht_va_flat, vid_va, n_films, steps=args.probe_steps)
    print(f"  H_t-flat probe:  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}")
    results["Ht_flat_probe"] = {"train_acc": tr_acc, "val_acc": va_acc}

    # ---- Z* -> film_id probe ----
    print("\nZ -> film probe")
    Z_tr_pool = Z_tr.mean(1)
    Z_va_pool = Z_va.mean(1)
    tr_acc, va_acc = fit_logistic_probe(Z_tr_pool, vid_tr, Z_va_pool, vid_va, n_films, steps=args.probe_steps)
    print(f"  Z-pool probe:    train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}")
    results["Z_probe"] = {"train_acc": tr_acc, "val_acc": va_acc}

    Z_tr_flat = Z_tr.flatten(1)
    Z_va_flat = Z_va.flatten(1)
    tr_acc, va_acc = fit_logistic_probe(Z_tr_flat, vid_tr, Z_va_flat, vid_va, n_films, steps=args.probe_steps)
    print(f"  Z-flat probe:    train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}")
    results["Z_flat_probe"] = {"train_acc": tr_acc, "val_acc": va_acc}

    # ---- Cluster purity by film ----
    print(f"\nk-means(k={args.n_clusters}) on train Z* (normalized)")
    Zf = F.normalize(Z_tr.flatten(1), dim=-1)
    g = torch.Generator(device=device).manual_seed(0)
    centers = Zf[torch.randperm(Zf.shape[0], device=device, generator=g)[: args.n_clusters]].clone()
    for _ in range(30):
        sim = Zf @ centers.T
        assign = sim.argmax(-1)
        new = torch.zeros_like(centers)
        for c in range(args.n_clusters):
            m = (assign == c)
            if m.any():
                new[c] = Zf[m].mean(0)
        new = F.normalize(new, dim=-1)
        if torch.allclose(new, centers, atol=1e-5): break
        centers = new

    log_n = float(torch.log(torch.tensor(n_films, dtype=torch.float)).item())
    max_fracs = []
    entropies = []
    for c in range(args.n_clusters):
        m = (assign == c)
        if m.sum() < 5: continue
        cnt = torch.bincount(vid_tr[m], minlength=n_films).float()
        total = cnt.sum().item()
        dist = cnt / total
        max_fracs.append(dist.max().item())
        ent = -(dist.clamp_min(1e-9) * dist.clamp_min(1e-9).log()).sum().item() / log_n
        entropies.append(ent)
    avg_max = sum(max_fracs) / len(max_fracs)
    avg_ent = sum(entropies) / len(entropies)
    print(f"  avg max-film fraction:  {avg_max:.3f}  (1.0 = pure film, {chance:.3f} = uniform)")
    print(f"  avg normalized entropy: {avg_ent:.3f}  (1.0 = uniform across films)")
    n_pure = sum(1 for f in max_fracs if f >= 0.95)
    n_mixed = sum(1 for f in max_fracs if f < 0.5)
    print(f"  clusters >=95% one film: {n_pure}/{len(max_fracs)}")
    print(f"  clusters <50% max-film:  {n_mixed}/{len(max_fracs)} (well-mixed)")
    results["clusters"] = {
        "n": len(max_fracs), "avg_max_film_frac": avg_max, "avg_norm_entropy": avg_ent,
        "n_pure_95": n_pure, "n_mixed_below_50": n_mixed,
    }

    print("\n=== verdict ===")
    print(f"  H_t probe val acc:  {results['Ht_probe']['val_acc']:.3f}  (high = H_t encodes film, expected)")
    print(f"  Z   probe val acc:  {results['Z_probe']['val_acc']:.3f}  (low = Z hides film, desired)")
    print(f"  Z/Ht ratio:         {results['Z_probe']['val_acc'] / max(results['Ht_probe']['val_acc'], 1e-6):.2f}")
    print(f"  cluster max-film:   {avg_max:.3f}  (target ↓; was 0.838 at 4 films)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
