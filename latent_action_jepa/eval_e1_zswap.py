"""E1 — z-swap controllability and portability on V-JEPA-2 + LA-runB teacher.

Three tests:

  (a) **Portability.** For each val anchor i, apply Z_j from a donor j and
      check whether `T(H_t_i, Z_j)` retrieves H_targets_j (donor's future)
      better than H_targets_i. Aggregated over many random donors.

  (b) **Controllability.** For a fixed H_t_i, sample K different donor Z's and
      compute pairwise cosine of `(H_pred − base(H_t_i))` across donors. If
      different Z's give similar deltas, z is degenerate / ignored. We compare
      same-z (same donor Z applied to multiple H_t) vs diff-z (different donors
      applied to same H_t) — analogous to v1's selectivity ratio.

  (c) **Z* step diversity.** Re-report mean off-step cos(Z*_i, Z*_j) (already
      done in C1) for completeness.
"""
import argparse
import json
import os
import random
import torch
import torch.nn.functional as F

from data.vjepa2_clips import sample_clip_batch
from models.vit3d import VisionTransformer3D
from models.la_modules import (
    InverseActionMultiStep, MultiStepTransition, MultiStepTransitionPerStep,
)


def per_step_top1(pred, tgt):
    """pred,tgt: [N,S,P,D]. Return per-step top-1 retrieval avg."""
    N, S, P, D = pred.shape
    tops = []
    for s in range(S):
        q = F.normalize(pred[:, s].mean(1), dim=-1)
        k = F.normalize(tgt[:, s].mean(1), dim=-1)
        sim = q @ k.T
        r = sim.argsort(dim=-1, descending=True)
        gt = torch.arange(N, device=q.device).unsqueeze(-1)
        tops.append((r[:, :1] == gt).any(-1).float().mean().item())
    return tops, sum(tops) / S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--vjepa_ckpt", required=True)
    ap.add_argument("--la_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_donors", type=int, default=16, help="K donor Z's per anchor for controllability")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed); random.seed(args.seed)

    # ---- V-JEPA-2 frozen ----
    vjepa = torch.load(args.vjepa_ckpt, map_location=device, weights_only=False)
    va = vjepa["args"]
    enc = VisionTransformer3D(
        num_frames=va["num_frames"], tubelet_size=va["tubelet_size"],
        patch_size=va["patch_size"], img_size=va["img_size"],
        embed_dim=va["embed_dim"], depth=va["depth"], num_heads=va["num_heads"],
        mlp_ratio=4.0,
    ).to(device).eval()
    enc.load_state_dict(vjepa["target"])
    for p in enc.parameters(): p.requires_grad_(False)
    grid_d = va["num_frames"] // va["tubelet_size"]
    grid_s = va["img_size"] // va["patch_size"]
    n_patch = grid_s * grid_s
    state_dim = va["embed_dim"]

    # ---- LA teacher ----
    la = torch.load(args.la_ckpt, map_location=device, weights_only=False)
    la_args = la["args"]
    S = la_args["action_tokens"]; A = la_args["action_dim"]
    use_per_step = la_args.get("per_step_transition", False)
    tr_cls = MultiStepTransitionPerStep if use_per_step else MultiStepTransition
    inv = InverseActionMultiStep(state_dim=state_dim, action_dim=A, n_steps=S, n_patch=n_patch,
                                  depth=la_args["inverse_depth"]).to(device).eval()
    tr = tr_cls(state_dim=state_dim, action_dim=A, n_steps=S, n_patch=n_patch,
                depth=la_args["transition_depth"]).to(device).eval()
    inv.load_state_dict(la["inv"])
    tr.load_state_dict(la["tr"])
    for p in list(inv.parameters()) + list(tr.parameters()):
        p.requires_grad_(False)
    print(f"loaded teacher: tr={tr_cls.__name__}  alpha_mean={tr.alpha.mean().item():.3f}")

    # ---- val anchors ----
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])
    ctx_span = (la_args["num_frames"] - 1) * la_args["frame_stride"]
    fut_span = ctx_span
    fo = la_args["future_offset"]
    anchor_stride = la_args["anchor_stride"]
    val_specs = []
    film_ids_per_spec = []
    for fi in range(n_films):
        n_frames_film = cache["frames"][fi].shape[0]
        a_min = ctx_span
        a_max = n_frames_film - fo - fut_span - 1
        train_end = int(a_max * 0.9)
        val_start = train_end + ctx_span + fut_span + fo
        for a in range(val_start, a_max, anchor_stride):
            val_specs.append((fi, a)); film_ids_per_spec.append(fi)
    N = len(val_specs)
    print(f"val anchors N={N}")

    # ---- gather features ----
    all_H_t, all_H_targets, all_Z = [], [], []
    B_eval = 4
    with torch.no_grad():
        for s in range(0, N, B_eval):
            batch = val_specs[s:s + B_eval]
            ctx_clip, _ = sample_clip_batch(cache, [(fi, a - ctx_span) for (fi, a) in batch],
                                             la_args["num_frames"], la_args["frame_stride"], device)
            fut_clip, _ = sample_clip_batch(cache, [(fi, a + fo) for (fi, a) in batch],
                                             la_args["num_frames"], la_args["frame_stride"], device)
            H_ctx = enc(ctx_clip); H_fut = enc(fut_clip)
            H_t = H_ctx[:, -n_patch:, :]
            H_targets = H_fut[:, : S * n_patch, :].reshape(-1, S, n_patch, state_dim)
            Z = inv(H_t, H_targets)
            all_H_t.append(H_t); all_H_targets.append(H_targets); all_Z.append(Z)
    H_t = torch.cat(all_H_t, 0)              # [N,P,D]
    H_targets = torch.cat(all_H_targets, 0)   # [N,S,P,D]
    Z = torch.cat(all_Z, 0)                   # [N,S,A]
    fids = torch.tensor(film_ids_per_spec, device=device)

    out = {"N": int(N)}

    # ---- (a) portability: swap donors ----
    # For each anchor i, sample donor j != i; compute T(H_t_i, Z_j) and retrieve
    # against (a1) own target H_targets_i, (a2) donor target H_targets_j.
    print("\n=== (a) portability ===")
    # use deterministic permutation seeded
    g = torch.Generator(device=device).manual_seed(7)
    perm = torch.randperm(N, device=device, generator=g)
    # avoid identity matches
    perm = torch.where(perm == torch.arange(N, device=device),
                       (perm + 1) % N, perm)
    with torch.no_grad():
        H_pred_swap = tr(H_t, Z[perm])                                  # [N,S,P,D]
        # vs own target i (per step)
        own_tops, own_avg = per_step_top1(H_pred_swap, H_targets)
        # vs donor target j (need targets[perm])
        donor_tops, donor_avg = per_step_top1(H_pred_swap, H_targets[perm])
        # also: correct Z reference
        H_pred_correct = tr(H_t, Z)
        corr_tops, corr_avg = per_step_top1(H_pred_correct, H_targets)
    print(f"  T(H_t_i, Z_j)  vs own H_targets_i:    avg per-step top1 = {own_avg:.4f}  per-step {[round(x,3) for x in own_tops]}")
    print(f"  T(H_t_i, Z_j)  vs donor H_targets_j:  avg per-step top1 = {donor_avg:.4f}  per-step {[round(x,3) for x in donor_tops]}")
    print(f"  T(H_t_i, Z_i)  vs own H_targets_i:    avg per-step top1 = {corr_avg:.4f}  (reference)")
    out["portability"] = {
        "swap_vs_own_avg_top1": own_avg, "swap_vs_donor_avg_top1": donor_avg,
        "swap_vs_own_per_step": own_tops, "swap_vs_donor_per_step": donor_tops,
        "correct_avg_top1": corr_avg, "correct_per_step": corr_tops,
    }

    # ---- (b) controllability: K donor Z's on a fixed H_t (per-anchor diversity) ----
    # For each anchor i, pick K random donors (not i). Compute delta = T(H_t_i, Z_j) - base(H_t_i).
    # We measure:
    #   same_donor:  same Z applied to different H_t (across-anchor) -- diversity across STATES.
    #   diff_donor:  different Z applied to same H_t (across-donors) -- diversity across ACTIONS.
    # Selectivity ratio = same_donor_cos / diff_donor_cos; a healthy action latent has the same-Z
    # delta-cos HIGHER than different-Z delta-cos (Z determines motion direction).
    print("\n=== (b) controllability ===")
    K = args.n_donors
    n_states = min(64, N)
    n_donors = min(K, N - 1)
    state_idx = torch.randperm(N, device=device, generator=torch.Generator(device=device).manual_seed(11))[:n_states]
    donor_idx = torch.randperm(N, device=device, generator=torch.Generator(device=device).manual_seed(13))[:n_donors]
    # compute base = broadcast H_t per step
    base_states = H_t[state_idx].unsqueeze(1).expand(-1, S, -1, -1)  # [n_states,S,P,D]

    # deltas[d, s_idx] : [S, P, D]
    deltas = torch.zeros(n_donors, n_states, S, n_patch, state_dim, device=device)
    with torch.no_grad():
        for di, d in enumerate(donor_idx):
            z = Z[d:d+1].expand(n_states, -1, -1)                       # [n_states,S,A]
            pred = tr(H_t[state_idx], z)                                 # [n_states,S,P,D]
            base = H_t[state_idx].unsqueeze(1).expand(-1, S, -1, -1)
            deltas[di] = pred - base

    df = F.normalize(deltas.reshape(n_donors, n_states, -1), dim=-1)
    # same-Z, different-H: across states for each donor (rows)
    same_z = 0.0
    for di in range(n_donors):
        m = df[di]
        sim = m @ m.T
        N_ = m.shape[0]
        off = sim - torch.eye(N_, device=device)
        same_z += (off.sum() / (N_ * (N_ - 1))).item()
    same_z /= n_donors

    # diff-Z, same-H: across donors for each state (columns)
    diff_z = 0.0
    for si in range(n_states):
        m = df[:, si]
        sim = m @ m.T
        D_ = m.shape[0]
        off = sim - torch.eye(D_, device=device)
        diff_z += (off.sum() / (D_ * (D_ - 1))).item()
    diff_z /= n_states

    print(f"  same-Z, different-H delta-cos = {same_z:+.4f}")
    print(f"  diff-Z, same-H      delta-cos = {diff_z:+.4f}")
    print(f"  selectivity ratio (same/diff) = {same_z / max(abs(diff_z), 1e-6):+.2f}x")
    out["controllability"] = {
        "n_donors": n_donors, "n_states": n_states,
        "same_z_diff_h_cos": same_z, "diff_z_same_h_cos": diff_z,
        "selectivity": same_z / max(abs(diff_z), 1e-6),
    }

    # ---- (c) Z* step diversity (mean cos across steps) ----
    Zn = F.normalize(Z, dim=-1)
    cos_step = torch.einsum("nsa,nta->nst", Zn, Zn)
    eye = torch.eye(S, device=device).bool()
    off_step = cos_step[:, ~eye].reshape(N, -1).mean(-1).mean().item()
    out["z_step_off_cos_mean"] = off_step
    print(f"\n  Z step-axis off-diag cos = {off_step:+.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
