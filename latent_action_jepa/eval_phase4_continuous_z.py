"""Phase 4 — continuous-Z analysis on a flow-action JEPA ckpt.

Run on the chosen best checkpoint. Computes:

  - selectivity: same-Z/diff-H vs diff-Z/same-H delta cosine ratio
  - portability: batch swap T(H_t_i, Z_j) vs own and donor targets
  - film probes: linear probe Z -> film_id, vs H_t -> film_id, vs delta -> film_id
  - k-means cluster purity by film at k=8, 16, 32

Designed for ckpt produced by train_flow_action_jepa_v2.py.
"""
import argparse
import json
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.vjepa2_clips import sample_clip_batch
from models.vit3d import VisionTransformer3D
from models.la_modules import (
    StepBase, InverseFromDelta, InverseFromDeltaPerStep,
    MultiStepTransitionPerStep, ContextDiTFlow,
)


def linear_probe(X, y, n_classes, train_frac=0.7, epochs=400, lr=3e-3, wd=1e-3, device="cuda"):
    with torch.enable_grad():
        X = X.detach()
        N, D = X.shape
        g = torch.Generator(device=X.device).manual_seed(0)
        perm = torch.randperm(N, generator=g, device=X.device)
        n_tr = int(N * train_frac)
        tr_idx, va_idx = perm[:n_tr], perm[n_tr:]
        head = nn.Linear(D, n_classes).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            idx = tr_idx[torch.randint(0, n_tr, (min(256, n_tr),), device=device)]
            loss = F.cross_entropy(head(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            tr_acc = (head(X[tr_idx]).argmax(-1) == y[tr_idx]).float().mean().item()
            va_acc = (head(X[va_idx]).argmax(-1) == y[va_idx]).float().mean().item()
        return tr_acc, va_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_eval", type=int, default=800)
    ap.add_argument("--n_donors", type=int, default=24)
    args = ap.parse_args()
    device = "cuda"

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck["args"]

    grid_d = a["num_frames"] // a["tubelet_size"]
    grid_s = a["img_size"] // a["patch_size"]
    n_patch = grid_s * grid_s
    n_tokens = grid_d * n_patch
    n_ctx_tubs = a["n_context_tubelets"]
    n_fut_tubs = grid_d - n_ctx_tubs
    n_ctx_tokens = n_ctx_tubs * n_patch
    S = a["action_steps"]; A = a["action_dim"]
    print(f"grid_d={grid_d} P={n_patch} ctx={n_ctx_tubs} fut={n_fut_tubs} S={S} A={A}")

    online = VisionTransformer3D(
        num_frames=a["num_frames"], tubelet_size=a["tubelet_size"], patch_size=a["patch_size"],
        img_size=a["img_size"], embed_dim=a["embed_dim"], depth=a["depth"],
        num_heads=a["num_heads"], mlp_ratio=4.0,
    ).to(device).eval()
    target_enc = VisionTransformer3D(
        num_frames=a["num_frames"], tubelet_size=a["tubelet_size"], patch_size=a["patch_size"],
        img_size=a["img_size"], embed_dim=a["embed_dim"], depth=a["depth"],
        num_heads=a["num_heads"], mlp_ratio=4.0,
    ).to(device).eval()
    base = StepBase(state_dim=a["embed_dim"], n_steps=S, depth=a.get("base_depth", 2)).to(device).eval()
    inv_cls = InverseFromDeltaPerStep if a.get("per_step_inverse", False) else InverseFromDelta
    inv = inv_cls(state_dim=a["embed_dim"], action_dim=A, n_steps=S, n_patch=n_patch,
                  depth=a["inverse_depth"]).to(device).eval()
    tr = MultiStepTransitionPerStep(state_dim=a["embed_dim"], action_dim=A, n_steps=S,
                                     n_patch=n_patch, depth=a["transition_depth"],
                                     alpha_min=a.get("alpha_min", 0.02),
                                     alpha_init=a.get("alpha_init", 0.05)).to(device).eval()
    # optional effect normalizer (v3/v4 ckpts)
    effect_norm = None
    if "effect_norm" in ck and a.get("effect_norm", "none") != "none":
        from models.la_modules import EffectNormalizer
        effect_norm = EffectNormalizer(a["effect_norm"], state_dim=a["embed_dim"]).to(device).eval()
        effect_norm.load_state_dict(ck["effect_norm"])
        print(f"  effect_norm: {a['effect_norm']}")
    online.load_state_dict(ck["online"])
    target_enc.load_state_dict(ck["target"])
    base.load_state_dict(ck["base"])
    inv.load_state_dict(ck["inv"])
    tr.load_state_dict(ck["tr"])
    print(f"loaded; alpha={tr.alpha.mean().item():.3f}")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])
    span = (a["num_frames"] - 1) * a["frame_stride"] + 1
    val_specs = []
    fids = []
    for fi in range(n_films):
        n_frames = cache["frames"][fi].shape[0]
        last_anchor = n_frames - span - 1
        train_end = int(last_anchor * 0.9)
        val_start = train_end + a["num_frames"] * a["frame_stride"]
        for av in range(val_start, last_anchor, a["clip_anchor_stride"]):
            val_specs.append((fi, av)); fids.append(fi)
    val_specs = val_specs[:args.n_eval]
    fids = fids[:args.n_eval]
    fids = torch.tensor(fids, dtype=torch.long, device=device)
    N = len(val_specs)
    print(f"N_eval={N}  films={n_films}")

    flat_grid = torch.arange(n_tokens, device=device)
    vis_idx_base = flat_grid[:n_ctx_tokens]
    fut_idx_base = flat_grid[n_ctx_tokens:]

    # collect H_t, H_targets, Z, delta_mean
    all_Ht, all_Htarg, all_Z, all_delta = [], [], [], []
    B_eval = 2
    with torch.no_grad():
        for s in range(0, N, B_eval):
            specs = val_specs[s:s + B_eval]
            clip, _ = sample_clip_batch(cache, specs, a["num_frames"], a["frame_stride"], device)
            B_v = clip.shape[0]
            vis = vis_idx_base.unsqueeze(0).expand(B_v, -1)
            fut = fut_idx_base.unsqueeze(0).expand(B_v, -1)
            H_ctx = online(clip, keep_indices=vis)
            H_t = H_ctx[:, -n_patch:, :]
            H_full = target_enc(clip)
            D = H_full.shape[-1]
            H_targets = H_full.gather(1, fut.unsqueeze(-1).expand(-1, -1, D)).reshape(B_v, S, n_patch, D)
            base_p = base(H_t)
            delta = H_targets - base_p
            if effect_norm is not None:
                delta = effect_norm(delta)
            Z = inv(delta)
            all_Ht.append(H_t.cpu()); all_Htarg.append(H_targets.cpu())
            all_Z.append(Z.cpu()); all_delta.append(delta.cpu())
            del H_full, clip
        torch.cuda.empty_cache()
        H_t = torch.cat(all_Ht, 0).to(device)
        H_targets = torch.cat(all_Htarg, 0).to(device)
        Z = torch.cat(all_Z, 0).to(device)
        delta = torch.cat(all_delta, 0).to(device)
        D = H_t.shape[-1]
        results = {"N": int(N), "n_films": n_films, "chance_film": 1.0 / n_films}

        # ---- Linear probes ----
        print("\n=== linear probes ===")
        Z_flat = Z.flatten(1)
        Ht_mean = H_t.mean(1)
        delta_mean = delta.mean(2).flatten(1)         # [N, S*D]
        for name, X in [("Z", Z_flat), ("H_t_mean", Ht_mean), ("delta_mean", delta_mean)]:
            tr_a, va_a = linear_probe(X, fids, n_films, device=device)
            print(f"  probe {name:14s} -> film: train={tr_a:.3f}  val={va_a:.3f}")
            results[f"probe_{name}_film"] = {"train": tr_a, "val": va_a}

        # ---- selectivity ----
        print("\n=== selectivity ===")
        K = min(args.n_donors, N)
        n_states = min(48, N)
        g1 = torch.Generator(device=device).manual_seed(11)
        g2 = torch.Generator(device=device).manual_seed(13)
        state_idx = torch.randperm(N, generator=g1, device=device)[:n_states]
        donor_idx = torch.randperm(N, generator=g2, device=device)[:K]
        deltas_pred = torch.zeros(K, n_states, S, n_patch, D, device=device)
        with torch.no_grad():
            for di, d in enumerate(donor_idx):
                z = Z[d:d+1].expand(n_states, -1, -1)
                pred = tr(H_t[state_idx], z)
                base_b = H_t[state_idx].unsqueeze(1).expand(-1, S, -1, -1)
                deltas_pred[di] = pred - base_b
        df = F.normalize(deltas_pred.reshape(K, n_states, -1), dim=-1)
        same_z = 0.0
        for di in range(K):
            sim = df[di] @ df[di].T
            same_z += (sim.sum() - torch.diag(sim).sum()).item() / (n_states * (n_states - 1))
        same_z /= K
        diff_z = 0.0
        for si in range(n_states):
            sim = df[:, si] @ df[:, si].T
            diff_z += (sim.sum() - torch.diag(sim).sum()).item() / (K * (K - 1))
        diff_z /= n_states
        sel = same_z / max(abs(diff_z), 1e-6)
        print(f"  same-Z, diff-H delta-cos = {same_z:+.4f}")
        print(f"  diff-Z, same-H delta-cos = {diff_z:+.4f}")
        print(f"  selectivity (same/diff)  = {sel:+.2f}x")
        results["selectivity"] = {"same_z": same_z, "diff_z": diff_z, "ratio": sel}

        # ---- portability (batch swap) ----
        print("\n=== portability (batch swap) ===")
        g3 = torch.Generator(device=device).manual_seed(7)
        perm = torch.randperm(N, generator=g3, device=device)
        perm = torch.where(perm == torch.arange(N, device=device), (perm + 1) % N, perm)
        with torch.no_grad():
            H_pred_swap = tr(H_t, Z[perm]).mean(2)         # [N, S, D]
            H_tgt_own_mean = H_targets.mean(2)             # [N, S, D]
            H_tgt_donor_mean = H_targets[perm].mean(2)     # [N, S, D]

            def retr(q, k, name):
                rows_t = []
                for ss in range(S):
                    qn = F.normalize(q[:, ss], dim=-1); kn = F.normalize(k[:, ss], dim=-1)
                    sim = qn @ kn.T
                    r = sim.argsort(dim=-1, descending=True)
                    gt = torch.arange(N, device=device).unsqueeze(-1)
                    t1 = (r[:, :1] == gt).any(-1).float().mean().item()
                    rows_t.append(t1)
                avg = sum(rows_t) / S
                print(f"  {name:36s} per-step {[round(x,3) for x in rows_t]}  avg={avg:.3f}")
                return rows_t, avg

            own_rows, own_avg = retr(H_pred_swap, H_tgt_own_mean, "T(H_t_i, Z_j) vs OWN H_tp_i")
            don_rows, don_avg = retr(H_pred_swap, H_tgt_donor_mean, "T(H_t_i, Z_j) vs DONOR H_tp_j")
            corr = tr(H_t, Z).mean(2)
            corr_rows, corr_avg = retr(corr, H_tgt_own_mean, "T(H_t, Z) vs own (reference)")

        results["portability"] = {
            "swap_vs_own": {"per_step": own_rows, "avg": own_avg},
            "swap_vs_donor": {"per_step": don_rows, "avg": don_avg},
            "correct": {"per_step": corr_rows, "avg": corr_avg},
        }

        # ---- k-means cluster purity by film ----
        print("\n=== k-means cluster purity by film ===")
        try:
            from sklearn.cluster import KMeans
            from collections import Counter
            Z_np = Z_flat.detach().cpu().numpy()
            y_np = fids.detach().cpu().numpy()
            for KK in [8, 16, 32]:
                km = KMeans(n_clusters=KK, random_state=0, n_init=10).fit(Z_np)
                lab = km.labels_
                purity = 0
                for c in range(KK):
                    members = y_np[lab == c]
                    if len(members) == 0: continue
                    most = Counter(members.tolist()).most_common(1)[0][1]
                    purity += most
                purity = purity / N
                print(f"  K={KK:2d}  purity={purity:.3f}  (chance={1.0/n_films:.3f})")
                results[f"kmeans{KK}_film_purity"] = purity
        except ImportError:
            print("  sklearn not available; install + rerun for cluster purity")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
