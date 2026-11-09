"""Evaluate a list of film-adversarial sweep checkpoints. Prints diagnostics,
selectivity, film-probe accuracy, cluster purity for each.
"""
import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transition_residual import ResidualTransitionModel
from models.inverse_encoder import InverseActionEncoder


def retrieval(q, k, N, name=""):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    mr = (r == gt).float().argmax(-1).float().mean().item() + 1
    return top1, mr


def fit_probe(X_train, y_train, X_val, y_val, n_classes, steps=2000, lr=1e-3):
    head = nn.Linear(X_train.shape[-1], n_classes).to(X_train.device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    bs = min(256, X_train.shape[0])
    for _ in range(steps):
        idx = torch.randint(0, X_train.shape[0], (bs,), device=X_train.device)
        loss = F.cross_entropy(head(X_train[idx]), y_train[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (head(X_val).argmax(-1) == y_val).float().mean().item()


def kmeans_cluster_purity(Z, vid, n_films, n_clusters=32):
    Zf = F.normalize(Z.flatten(1), dim=-1)
    device = Zf.device
    g = torch.Generator(device=device).manual_seed(0)
    centers = Zf[torch.randperm(Zf.shape[0], device=device, generator=g)[:n_clusters]].clone()
    for _ in range(30):
        sim = Zf @ centers.T; assign = sim.argmax(-1)
        new = torch.zeros_like(centers)
        for c in range(n_clusters):
            m = (assign == c)
            if m.any(): new[c] = Zf[m].mean(0)
        new = F.normalize(new, dim=-1)
        if torch.allclose(new, centers, atol=1e-5): break
        centers = new
    log_n = float(torch.log(torch.tensor(n_films, dtype=torch.float)).item())
    fracs = []; ents = []
    for c in range(n_clusters):
        m = (assign == c)
        if m.sum() < 5: continue
        cnt = torch.bincount(vid[m], minlength=n_films).float()
        dist = cnt / cnt.sum()
        fracs.append(dist.max().item())
        ent = -(dist.clamp_min(1e-9) * dist.clamp_min(1e-9).log()).sum().item() / log_n
        ents.append(ent)
    return {"avg_max_frac": sum(fracs)/len(fracs), "avg_norm_entropy": sum(ents)/len(ents),
            "n_clusters": len(fracs)}


def selectivity_ratio(tr, H_t_val, Z_val, n_zs=64, n_ht=100):
    device = H_t_val.device
    z_idx = torch.randperm(Z_val.shape[0], device=device)[:n_zs]
    h_idx = torch.randperm(H_t_val.shape[0], device=device)[:n_ht]
    Zs = Z_val[z_idx]; Hts = H_t_val[h_idx]
    deltas = torch.zeros(n_zs, n_ht, 16, H_t_val.shape[-1], device=device)
    with torch.no_grad():
        for zi in range(n_zs):
            z = Zs[zi:zi+1].expand(n_ht, -1, -1)
            pred = tr(Hts, z); base = tr.base(Hts)
            deltas[zi] = pred - base
    df = F.normalize(deltas.reshape(n_zs, n_ht, -1), dim=-1)
    sz = 0.0
    for zi in range(n_zs):
        m = df[zi]; off = m @ m.T - torch.eye(n_ht, device=device)
        sz += (off.sum() / (n_ht*(n_ht-1))).item()
    sz /= n_zs
    dz = 0.0
    for hi in range(n_ht):
        m = df[:, hi]; off = m @ m.T - torch.eye(n_zs, device=device)
        dz += (off.sum() / (n_zs*(n_zs-1))).item()
    dz /= n_ht
    return sz, dz, sz / max(dz, 1e-6)


def evaluate_one(ckpt_path, bundle_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tr = ResidualTransitionModel(cfg["state_dim"], cfg["action_dim"], cfg["future_tokens"],
                                 depth=cfg["transition_depth"]).to(device)
    tr.load_state_dict(ckpt["transition_model"]); tr.eval()
    for p in tr.parameters(): p.requires_grad_(False)

    b = torch.load(bundle_path, map_location="cpu", weights_only=False)
    H_t = b["H_t_val"].to(device); H_tp = b["H_tp_val"].to(device); Z = b["Z_star_val"].to(device)
    vid = b["video_id_val"].to(device)
    Z_tr = b["Z_star_train"].to(device); vid_tr_full = b["video_id_train"].to(device)
    N = H_t.shape[0]
    n_films = int(vid_tr_full.max().item() + 1)

    out = {"alpha": tr.alpha.item(), "n_val": N, "n_films": n_films,
           "action_tokens": cfg["action_tokens"], "action_dim": cfg["action_dim"],
           "lambda_film_adv": ckpt["args"].get("lambda_film_adv", None)}

    with torch.no_grad():
        out["correct_Z"] = retrieval(tr(H_t, Z).mean(1), H_tp.mean(1), N)[0]
        std_z = Z.flatten(1).std(0).mean()
        out["zero_Z"] = retrieval(tr(H_t, torch.zeros_like(Z)).mean(1), H_tp.mean(1), N)[0]
        out["random_Z"] = retrieval(tr(H_t, torch.randn_like(Z) * std_z).mean(1), H_tp.mean(1), N)[0]
        out["shuffle_Z"] = retrieval(tr(H_t, torch.roll(Z, 1, 0)).mean(1), H_tp.mean(1), N)[0]
        out["copy_Ht"] = retrieval(H_t.mean(1), H_tp.mean(1), N)[0]
        out["Z_only"] = retrieval(tr(torch.zeros_like(H_t), Z).mean(1), H_tp.mean(1), N)[0]
        perm = torch.randperm(N, device=device, generator=torch.Generator(device=device).manual_seed(0))
        out["wrong_H"] = retrieval(tr(H_t[perm], Z).mean(1), H_tp.mean(1), N)[0]

    sz, dz, sel = selectivity_ratio(tr, H_t, Z)
    out["same_z_cos"] = sz; out["diff_z_cos"] = dz; out["selectivity"] = sel

    # Probe from scratch on Z and on H_t (fresh classifier each time)
    out["Z_probe_val_acc"] = fit_probe(Z_tr.mean(1), vid_tr_full, Z.mean(1), vid, n_films)
    out["Ht_probe_val_acc"] = fit_probe(b["H_t_train"].to(device).mean(1), vid_tr_full,
                                          H_t.mean(1), vid, n_films)

    cluster = kmeans_cluster_purity(Z_tr, vid_tr_full, n_films, n_clusters=32)
    out.update(cluster)
    out["chance"] = 1.0 / n_films
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="ckpt:bundle pairs separated by =")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    all_res = {}
    for spec in args.runs:
        ckpt, bundle = spec.split("=")
        name = os.path.basename(os.path.dirname(ckpt))
        print(f"\n=== {name} ===")
        r = evaluate_one(ckpt, bundle)
        all_res[name] = r
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items() if not isinstance(v, dict)}, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_res, f, indent=2)

    # Print table
    print("\n=== summary ===")
    headers = ["run", "λ", "correct", "Z_only", "shuffle", "wrong_H", "sel", "Z→film", "H_t→film", "clust_pure"]
    print("  " + "  ".join(f"{h:>14s}" for h in headers))
    for name, r in all_res.items():
        cells = [name[-14:], f"{r['lambda_film_adv']}", f"{r['correct_Z']:.3f}", f"{r['Z_only']:.3f}",
                 f"{r['shuffle_Z']:.3f}", f"{r['wrong_H']:.3f}", f"{r['selectivity']:.2f}x",
                 f"{r['Z_probe_val_acc']:.3f}", f"{r['Ht_probe_val_acc']:.3f}", f"{r['avg_max_frac']:.3f}"]
        print("  " + "  ".join(f"{c:>14s}" for c in cells))


if __name__ == "__main__":
    main()
