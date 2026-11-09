"""Leave-one-video-out trainer. Loads the multi-video mixed bundle, masks one
film out of TRAIN, trains Stage 1, evaluates on that film's val.

Loops over all films, reporting per-held-out-film diagnostics.
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from models.inverse_encoder import InverseActionEncoder
from models.transition_residual import ResidualTransitionModel
from diag_stage0 import effective_rank


def cosine_distance(pred, target):
    pred = F.normalize(pred, dim=-1); target = F.normalize(target, dim=-1)
    return (2.0 - 2.0 * (pred * target).sum(-1)).mean()


def info_nce(q, k, tau=0.1):
    q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
    logits = q @ k.T / tau
    labels = torch.arange(q.shape[0], device=q.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(k @ q.T / tau, labels))


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


def train_one_loo(bundle, held_film_idx, steps, bs, lr, wd, device,
                  nce_tau, l_nce, l_l2, l_var, l_cov, log_every,
                  action_tokens=8, action_dim=256, future_tokens=16,
                  inv_depth=4, tr_depth=2, seed=0):
    torch.manual_seed(seed)
    state_dim = 384
    H_t = bundle["H_t_train"].to(device); H_tp = bundle["H_tp_train"].to(device)
    vid_tr = bundle["video_id_train"].to(device)
    # Filter train: exclude held-out film
    mask = (vid_tr != held_film_idx)
    H_t = H_t[mask]; H_tp = H_tp[mask]
    N_tr = H_t.shape[0]
    # Val: ONLY held-out film
    H_t_val = bundle["H_t_val"].to(device); H_tp_val = bundle["H_tp_val"].to(device)
    vid_va = bundle["video_id_val"].to(device)
    vmask = (vid_va == held_film_idx)
    H_t_val = H_t_val[vmask]; H_tp_val = H_tp_val[vmask]
    N_va = H_t_val.shape[0]
    print(f"\n=== held-out film {held_film_idx}: train N={N_tr}  val N={N_va} ===")

    inv = InverseActionEncoder(state_dim, action_dim, action_tokens, depth=inv_depth).to(device)
    tr = ResidualTransitionModel(state_dim, action_dim, future_tokens, depth=tr_depth).to(device)
    params = list(inv.parameters()) + list(tr.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    inv.train(); tr.train()
    for step in range(steps):
        idx = torch.randint(0, N_tr, (bs,), device=device)
        h_t = H_t[idx]; h_tp = H_tp[idx]
        z = inv(h_t, h_tp)
        h_pred = tr(h_t, z)
        L_t = cosine_distance(h_pred, h_tp.detach())
        L_n = info_nce(h_pred.mean(1), h_tp.mean(1).detach(), tau=nce_tau)
        L_l2 = (z ** 2).mean()
        std_dim = z.flatten(0, 1).std(0)
        L_v = F.relu(1 - std_dim).mean()
        z_flat = z.flatten(0, 1); z_c = z_flat - z_flat.mean(0, keepdim=True)
        cov = (z_c.T @ z_c) / max(1, z_c.shape[0] - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        L_co = (off ** 2).sum() / cov.shape[0]
        loss = L_t + l_nce*L_n + l_l2*L_l2 + l_var*L_v + l_cov*L_co
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:5d}  L_t={L_t.item():.4f}  L_nce={L_n.item():.4f}  alpha={tr.alpha.item():.4f}")

    # Eval on held-out film val
    inv.eval(); tr.eval()
    results = {}
    with torch.no_grad():
        Z = inv(H_t_val, H_tp_val)
        std_z = Z.flatten(1).std(0).mean().item()
        std_zfull = Z.flatten(1).std(0).mean()
        pred_correct = tr(H_t_val, Z)
        pred_zero = tr(H_t_val, torch.zeros_like(Z))
        pred_rand = tr(H_t_val, torch.randn_like(Z) * std_zfull)
        pred_shuf = tr(H_t_val, torch.roll(Z, 1, 0))
        pred_zonly = tr(torch.zeros_like(H_t_val), Z)
        results["correct_Z"] = retrieval(pred_correct.mean(1), H_tp_val.mean(1), N_va, "correct_Z")[0]
        results["zero_Z"] = retrieval(pred_zero.mean(1), H_tp_val.mean(1), N_va, "zero_Z")[0]
        results["random_Z"] = retrieval(pred_rand.mean(1), H_tp_val.mean(1), N_va, "random_Z")[0]
        results["shuffle_Z"] = retrieval(pred_shuf.mean(1), H_tp_val.mean(1), N_va, "shuffle_Z")[0]
        results["copy_Ht"] = retrieval(H_t_val.mean(1), H_tp_val.mean(1), N_va, "copy_Ht")[0]
        results["Z_only"] = retrieval(pred_zonly.mean(1), H_tp_val.mean(1), N_va, "Z_only")[0]
        # wrong_H
        perm = torch.randperm(N_va, device=device, generator=torch.Generator(device=device).manual_seed(0))
        results["wrong_H"] = retrieval(tr(H_t_val[perm], Z).mean(1), H_tp_val.mean(1), N_va, "wrong_H")[0]
        results["alpha"] = tr.alpha.item()
        results["std_Z"] = std_z
        results["eff_rank_Z"] = effective_rank(Z.flatten(1))
        results["N_val"] = N_va

        # selectivity ratio
        N_zs = min(64, N_va); N_ht = min(100, N_va)
        z_idx = torch.randperm(N_va, device=device)[: N_zs]
        ht_idx = torch.randperm(N_va, device=device)[: N_ht]
        donor_Zs = Z[z_idx]; Hts = H_t_val[ht_idx]
        deltas = torch.zeros(N_zs, N_ht, 16, H_t_val.shape[-1], device=device)
        for zi in range(N_zs):
            zexp = donor_Zs[zi:zi+1].expand(N_ht, -1, -1)
            pred = tr(Hts, zexp)
            base = tr.base(Hts)
            deltas[zi] = pred - base
        df = F.normalize(deltas.reshape(N_zs, N_ht, -1), dim=-1)
        sz = 0.0
        for zi in range(N_zs):
            m = df[zi]; off = m @ m.T - torch.eye(N_ht, device=device)
            sz += (off.sum() / (N_ht*(N_ht-1))).item()
        sz /= N_zs
        dz = 0.0
        for hi in range(N_ht):
            m = df[:, hi]; off = m @ m.T - torch.eye(N_zs, device=device)
            dz += (off.sum() / (N_zs*(N_zs-1))).item()
        dz /= N_ht
        results["same_z_cos"] = sz; results["diff_z_cos"] = dz
        results["selectivity"] = sz / max(dz, 1e-6)
        print(f"  selectivity = {results['selectivity']:.2f}x  (same={sz:+.3f}  diff={dz:+.3f})")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--nce_tau", type=float, default=0.1)
    ap.add_argument("--lambda_nce", type=float, default=1.0)
    ap.add_argument("--lambda_z_l2", type=float, default=0.001)
    ap.add_argument("--lambda_z_var", type=float, default=0.05)
    ap.add_argument("--lambda_z_cov", type=float, default=0.01)
    args = ap.parse_args()
    device = "cuda"
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    film_names = bundle["film_names"]
    print(f"films: {film_names}")
    all_results = {}
    for fi, fname in enumerate(film_names):
        r = train_one_loo(bundle, fi, args.steps, args.bs, args.lr, args.wd, device,
                          args.nce_tau, args.lambda_nce, args.lambda_z_l2,
                          args.lambda_z_var, args.lambda_z_cov, args.log_every)
        all_results[fname] = r
        print(f"\n>>> held-out {fname}: correct_Z={r['correct_Z']:.3f}  Z_only={r['Z_only']:.3f}  selectivity={r['selectivity']:.2f}x")

    print("\n=== LOO summary ===")
    print(f"  {'held-out':<20s}  {'N_val':>6s}  {'correct':>8s}  {'Z_only':>8s}  {'shuffle':>8s}  {'wrong_H':>8s}  {'select':>8s}")
    for fname, r in all_results.items():
        print(f"  {fname[:20]:<20s}  {r['N_val']:>6d}  {r['correct_Z']:>8.3f}  {r['Z_only']:>8.3f}  {r['shuffle_Z']:>8.3f}  {r['wrong_H']:>8.3f}  {r['selectivity']:>8.2f}x")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
