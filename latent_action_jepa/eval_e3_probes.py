"""E3 — Z film probe and within-film portability check.

If Z encodes film_id cleanly (high acc), Z is scene-tied and not robot-useful.
We compare Z->film probe to H_t->film probe as baseline.

Within-film portability: re-do z-swap but donor j sampled from SAME film as i.
"""
import argparse
import json
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.vjepa2_clips import IMAGENET_MEAN, IMAGENET_STD
from models.encoder import ViTEncoder
from models.fixed_reducer import FixedSpatialReducer
from models.inverse_encoder import InverseActionEncoder
from models.transition_residual import ResidualTransitionModel


def sample_pair_batch(cache, anchors, delta, device):
    xt, xtp = [], []
    for fi, a in anchors:
        frames = cache["frames"][fi]
        data = frames["data"] if isinstance(frames, dict) else frames
        xt.append(data[a]); xtp.append(data[a + delta])
    xt = torch.stack(xt, 0).to(device).float() / 255.0
    xtp = torch.stack(xtp, 0).to(device).float() / 255.0
    m = IMAGENET_MEAN.squeeze(0).to(device).view(1, 3, 1, 1)
    s = IMAGENET_STD.squeeze(0).to(device).view(1, 3, 1, 1)
    return (xt - m) / s, (xtp - m) / s


def linear_probe(X, y, n_classes, train_frac=0.7, epochs=400, lr=3e-3, wd=1e-3, device="cuda"):
    """Train a linear probe. Returns train/val accuracy."""
    N, D = X.shape
    g = torch.Generator(device=X.device).manual_seed(0)
    perm = torch.randperm(N, generator=g, device=X.device)
    n_tr = int(N * train_frac)
    tr_idx, va_idx = perm[:n_tr], perm[n_tr:]
    head = nn.Linear(D, n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        idx = tr_idx[torch.randint(0, n_tr, (min(256, n_tr),), device=device)]
        logits = head(X[idx])
        loss = F.cross_entropy(logits, y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        tr_acc = (head(X[tr_idx]).argmax(-1) == y[tr_idx]).float().mean().item()
        va_acc = (head(X[va_idx]).argmax(-1) == y[va_idx]).float().mean().item()
    return tr_acc, va_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--la_ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"

    ck = torch.load(args.la_ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    enc = ViTEncoder(a["encoder_name"], pretrained=True, layer_idx=None).to(device).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    state_dim = enc.embed_dim
    red_state = FixedSpatialReducer(a["state_tokens"]).to(device)
    red_future = FixedSpatialReducer(a["future_tokens"]).to(device)
    inv = InverseActionEncoder(state_dim, a["action_dim"], a["action_tokens"],
                                depth=a["inverse_depth"]).to(device).eval()
    tr = ResidualTransitionModel(state_dim, a["action_dim"], a["future_tokens"],
                                  depth=a["transition_depth"]).to(device).eval()
    inv.load_state_dict(ck["inv"]); tr.load_state_dict(ck["tr"])
    for p in list(inv.parameters()) + list(tr.parameters()): p.requires_grad_(False)
    print(f"loaded F7 teacher. alpha={tr.alpha.item():.3f}")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])
    val_specs = []
    fids = []
    for fi in range(n_films):
        data = cache["frames"][fi]["data"] if isinstance(cache["frames"][fi], dict) else cache["frames"][fi]
        n_frames = data.shape[0]
        last_anchor = n_frames - a["delta"] - 1
        train_end = int(last_anchor * 0.9)
        val_start = train_end + a["delta"] + 16
        for av in range(val_start, last_anchor, a["anchor_stride"]):
            val_specs.append((fi, av)); fids.append(fi)
    N = len(val_specs)
    fids = torch.tensor(fids, dtype=torch.long, device=device)
    print(f"val N={N}, films={n_films}")

    all_Ht, all_Htp, all_Z = [], [], []
    B_eval = 16
    delta_input = a.get("delta_input", False)
    if delta_input:
        print("  using delta_input mode (Z = inv(0, Htp - red_future(Ht)))")
    with torch.no_grad():
        for s in range(0, N, B_eval):
            specs = val_specs[s:s + B_eval]
            xt, xtp = sample_pair_batch(cache, specs, a["delta"], device)
            Ht = red_state(enc(xt)); Htp = red_future(enc(xtp))
            if delta_input:
                Ht_r = red_future(Ht)
                Z = inv(torch.zeros_like(Ht_r), Htp - Ht_r)
            else:
                Z = inv(Ht, Htp)
            all_Ht.append(Ht); all_Htp.append(Htp); all_Z.append(Z)
    Ht = torch.cat(all_Ht, 0); Htp = torch.cat(all_Htp, 0); Z = torch.cat(all_Z, 0)

    # ---- linear probes ----
    print("\n=== linear probes ===")
    out = {"N": N}
    Z_flat = Z.flatten(1)                         # [N, Ka*Da]
    Ht_flat = Ht.mean(1)                          # [N, D] mean-pool over tokens
    Htp_flat = Htp.mean(1)
    delta_flat = (Htp.mean(1) - red_future(red_state.forward.__self__(Ht.reshape(-1, a["state_tokens"], state_dim)) if False else Ht).mean(1)) if False else (Htp.mean(1) - red_future(Ht).mean(1))
    # simpler: delta = Htp.mean(1) - red_future(Ht).mean(1)
    delta_simple = Htp.mean(1) - red_future(Ht).mean(1)

    print("  film probe (chance = {:.3f}):".format(1.0 / n_films))
    tr_a, va_a = linear_probe(Z_flat, fids, n_films, device=device)
    print(f"    Z      -> film:   train={tr_a:.3f}  val={va_a:.3f}")
    out["probe_Z_film"] = {"train": tr_a, "val": va_a}
    tr_a, va_a = linear_probe(Ht_flat, fids, n_films, device=device)
    print(f"    H_t    -> film:   train={tr_a:.3f}  val={va_a:.3f}  (baseline)")
    out["probe_Ht_film"] = {"train": tr_a, "val": va_a}
    tr_a, va_a = linear_probe(delta_simple, fids, n_films, device=device)
    print(f"    H_tp-H_t -> film: train={tr_a:.3f}  val={va_a:.3f}  (motion only)")
    out["probe_delta_film"] = {"train": tr_a, "val": va_a}

    # ---- within-film portability ----
    print("\n=== within-film portability ===")
    # For each i, find a donor j with SAME film != i.
    perm = torch.zeros(N, dtype=torch.long, device=device)
    film_to_idx = {f: torch.where(fids == f)[0] for f in fids.unique().tolist()}
    for i in range(N):
        f = fids[i].item()
        pool = film_to_idx[f]
        if pool.numel() <= 1:
            perm[i] = i
        else:
            mask = (pool != i)
            choice = pool[mask]
            perm[i] = choice[torch.randint(0, choice.numel(), (1,), device=device).item()]
    with torch.no_grad():
        H_pred_swap = tr(Ht, Z[perm])

        def retr(q, k, name):
            qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
            sim = qn @ kn.T
            r = sim.argsort(dim=-1, descending=True)
            gt = torch.arange(q.shape[0], device=device).unsqueeze(-1)
            t1 = (r[:, :1] == gt).any(-1).float().mean().item()
            t5 = (r[:, :5] == gt).any(-1).float().mean().item()
            mr = (r == gt).float().argmax(-1).float().mean().item() + 1
            print(f"  {name:40s} top1={t1:.4f}  top5={t5:.4f}  mr={mr:.1f}")
            return {"top1": t1, "top5": t5, "mr": mr}

        out["correct"] = retr(tr(Ht, Z).mean(1), Htp.mean(1), "T(H_t, Z) vs own (reference)")
        out["sf_swap_vs_own"] = retr(H_pred_swap.mean(1), Htp.mean(1),
                                     "SAME-FILM swap T(H_t_i,Z_j) vs OWN H_tp_i")
        out["sf_swap_vs_donor"] = retr(H_pred_swap.mean(1), Htp[perm].mean(1),
                                       "SAME-FILM swap T(H_t_i,Z_j) vs DONOR H_tp_j")

    # ---- k-means cluster purity ----
    print("\n=== k-means cluster purity ===")
    from sklearn.cluster import KMeans
    Z_np = Z_flat.detach().cpu().numpy()
    y_np = fids.detach().cpu().numpy()
    for K in [8, 16, 32]:
        km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(Z_np)
        labels = km.labels_
        # cluster purity vs film
        from collections import Counter
        purity = 0
        for c in range(K):
            members = y_np[labels == c]
            if len(members) == 0: continue
            most = Counter(members.tolist()).most_common(1)[0][1]
            purity += most
        purity = purity / N
        print(f"  K={K:2d}  film_purity={purity:.3f}  (chance = {1.0/n_films:.3f})")
        out[f"kmeans{K}_film_purity"] = purity

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
