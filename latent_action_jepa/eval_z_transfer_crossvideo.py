"""Cross-video Z transfer: apply a BBB Z to a Sintel H_t. Does the predicted
delta stay consistent across multiple Sintel H_t's (state-portable across videos)?
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from models.transition_residual import ResidualTransitionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bbb_bundle", required=True)
    ap.add_argument("--sintel_bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_zs", type=int, default=32)
    ap.add_argument("--n_ht", type=int, default=64)
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

    bbb = torch.load(args.bbb_bundle, map_location="cpu", weights_only=False)
    sintel = torch.load(args.sintel_bundle, map_location="cpu", weights_only=False)

    Z_bbb = bbb["Z_star_train"][:2000].to(device)
    H_t_bbb = bbb["H_t_train"][:2000].to(device)
    H_t_sintel = torch.cat([sintel["H_t_train"], sintel["H_t_val"]], 0).to(device)

    z_idx = torch.randperm(Z_bbb.shape[0])[: args.n_zs]
    donor_Zs = Z_bbb[z_idx]
    ht_idx_bbb = torch.randperm(H_t_bbb.shape[0])[: args.n_ht]
    Hts_bbb = H_t_bbb[ht_idx_bbb]
    ht_idx_sin = torch.randperm(H_t_sintel.shape[0])[: args.n_ht]
    Hts_sin = H_t_sintel[ht_idx_sin]

    def compute_deltas(Hts):
        deltas = torch.zeros(args.n_zs, args.n_ht, 16, Hts.shape[-1], device=device)
        with torch.no_grad():
            for zi in range(args.n_zs):
                z = donor_Zs[zi:zi+1].expand(args.n_ht, -1, -1)
                pred = tr(Hts, z)
                base = tr.base(Hts)
                deltas[zi] = pred - base
        return F.normalize(deltas.reshape(args.n_zs, args.n_ht, -1), dim=-1)

    d_bbb = compute_deltas(Hts_bbb)
    d_sin = compute_deltas(Hts_sin)

    def same_z_cos(d):
        s = 0.0; c = 0
        for zi in range(args.n_zs):
            m = d[zi]
            sim = m @ m.T
            off = sim - torch.eye(m.shape[0], device=device)
            s += (off.sum() / (m.shape[0]*(m.shape[0]-1))).item(); c += 1
        return s / c

    print(f"Same-Z avg cos on BBB H_t's     = {same_z_cos(d_bbb):+.4f}")
    print(f"Same-Z avg cos on Sintel H_t's  = {same_z_cos(d_sin):+.4f}")

    # Cross-video: same Z applied to BBB and Sintel — average cos between matched pairs
    # delta_bbb[zi, hi_bbb]  vs  delta_sin[zi, hi_sin]
    cross_cos = 0.0; cnt = 0
    for zi in range(args.n_zs):
        sim = d_bbb[zi] @ d_sin[zi].T  # [n_ht_bbb, n_ht_sin]
        # average all pairs
        cross_cos += sim.mean().item(); cnt += 1
    cross_cos /= cnt
    print(f"Same Z, BBB H_t vs Sintel H_t   = {cross_cos:+.4f}")

    # Random Z control
    perm_z = torch.randperm(args.n_zs, device=device)
    diff_z_cross = 0.0; cnt = 0
    for zi in range(args.n_zs):
        zj = perm_z[zi].item()
        if zj == zi: continue
        sim = d_bbb[zi] @ d_sin[zj].T
        diff_z_cross += sim.mean().item(); cnt += 1
    diff_z_cross /= max(cnt, 1)
    print(f"DIFF Z, BBB H_t vs Sintel H_t   = {diff_z_cross:+.4f}")

    res = {
        "same_z_bbb": same_z_cos(d_bbb),
        "same_z_sintel": same_z_cos(d_sin),
        "same_z_cross_video": cross_cos,
        "diff_z_cross_video": diff_z_cross,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f: json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
