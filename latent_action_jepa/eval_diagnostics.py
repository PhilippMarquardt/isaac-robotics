"""Strict diagnostics on a frozen Stage 1 ckpt.

Distinguishes "Z is used" from "Z is a compressed future" by probing:
  correct_Z   = T(H_t,         Z*)
  shuffle_Z   = T(H_t,         roll(Z*))
  zero_Z      = T(H_t,         0)
  random_Z    = T(H_t,         randn * std(Z*))
  copy_Ht     = mean(H_t) directly
  Z_only      = T(zeros_like(H_t), Z*)
  wrong_H     = T(roll(H_t),   Z*)

Pass desiderata:
  - correct_Z best
  - shuffle_Z and random_Z below chance/copy
  - zero_Z below copy_Ht typically
  - Z_only NOT near oracle (would mean Z carries the whole future)
  - wrong_H should NOT reconstruct original H_tp well (otherwise H_t unused)

Reports across N seeds for stochastic conditions; copy/correct are deterministic.
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from models.transition_residual import ResidualTransitionModel


def retrieval(q, k, N, name=None):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    mr = (r == gt).float().argmax(-1).float().mean().item() + 1
    if name: print(f"  {name:18s} top1={top1:.3f}  top5={top5:.3f}  mr={mr:.2f}")
    return top1, top5, mr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    device = "cuda"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tr = ResidualTransitionModel(cfg["state_dim"], cfg["action_dim"], cfg["future_tokens"],
                                 depth=cfg["transition_depth"]).to(device)
    tr.load_state_dict(ckpt["transition_model"])
    for p in tr.parameters(): p.requires_grad_(False)
    tr.eval()
    print(f"loaded ckpt: alpha={tr.alpha.item():.4f}  action {cfg['action_tokens']}x{cfg['action_dim']}")

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t = bundle["H_t_val"].to(device); H_tp = bundle["H_tp_val"].to(device)
    Z = bundle["Z_star_val"].to(device)
    N = H_t.shape[0]
    print(f"N val = {N}  chance top1 = {1/N:.4f}")

    std_z = Z.flatten(1).std(0).mean().item()

    print("\n=== deterministic probes ===")
    results = {"N": N, "chance": 1.0 / N}
    with torch.no_grad():
        t1, t5, mr = retrieval(tr(H_t, Z).mean(1), H_tp.mean(1), N, "correct_Z")
        results["correct_Z"] = {"top1": t1, "top5": t5, "mr": mr}

        t1, t5, mr = retrieval(tr(H_t, torch.zeros_like(Z)).mean(1), H_tp.mean(1), N, "zero_Z")
        results["zero_Z"] = {"top1": t1, "top5": t5, "mr": mr}

        t1, t5, mr = retrieval(H_t.mean(1), H_tp.mean(1), N, "copy_Ht")
        results["copy_Ht"] = {"top1": t1, "top5": t5, "mr": mr}

        # Z_only: feed zero state, ask transition to use only Z*. If high -> Z encodes the future itself.
        t1, t5, mr = retrieval(tr(torch.zeros_like(H_t), Z).mean(1), H_tp.mean(1), N, "Z_only (H=0)")
        results["Z_only"] = {"top1": t1, "top5": t5, "mr": mr}

    print("\n=== stochastic probes (per seed + summary) ===")
    metrics = {"shuffle_Z": [], "random_Z": [], "wrong_H": []}
    for seed in args.seeds:
        g = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            # shuffle Z (permute along batch)
            perm_z = torch.randperm(N, device=device, generator=g)
            Z_shuf = Z[perm_z]
            t1_sz, _, mr_sz = retrieval(tr(H_t, Z_shuf).mean(1), H_tp.mean(1), N)
            metrics["shuffle_Z"].append((t1_sz, mr_sz))
            # random Z
            Z_rand = torch.randn(N, *Z.shape[1:], device=device, generator=g) * std_z
            t1_rz, _, mr_rz = retrieval(tr(H_t, Z_rand).mean(1), H_tp.mean(1), N)
            metrics["random_Z"].append((t1_rz, mr_rz))
            # wrong_H: feed a shuffled H_t with the CORRECT Z; should NOT recover own H_tp
            perm_h = torch.randperm(N, device=device, generator=g)
            t1_wh, _, mr_wh = retrieval(tr(H_t[perm_h], Z).mean(1), H_tp.mean(1), N)
            metrics["wrong_H"].append((t1_wh, mr_wh))
            print(f"  seed={seed:3d}  shuffle_Z top1={t1_sz:.3f}  random_Z top1={t1_rz:.3f}  wrong_H top1={t1_wh:.3f}")

    for name, vals in metrics.items():
        t1s = torch.tensor([v[0] for v in vals])
        mrs = torch.tensor([v[1] for v in vals])
        m, s = t1s.mean().item(), t1s.std().item()
        m_mr, s_mr = mrs.mean().item(), mrs.std().item()
        print(f"  {name:14s} top1 mean±std = {m:.3f} ± {s:.3f}  mr mean±std = {m_mr:.2f} ± {s_mr:.2f}  over {len(vals)} seeds")
        results[name] = {"top1_mean": m, "top1_std": s, "mr_mean": m_mr, "mr_std": s_mr, "n_seeds": len(vals)}

    # Verdicts
    print("\n=== verdict ===")
    c = results["correct_Z"]["top1"]
    z = results["zero_Z"]["top1"]
    cp = results["copy_Ht"]["top1"]
    zo = results["Z_only"]["top1"]
    sm = results["shuffle_Z"]["top1_mean"]
    rm = results["random_Z"]["top1_mean"]
    wh = results["wrong_H"]["top1_mean"]
    print(f"  correct_Z={c:.3f}  zero_Z={z:.3f}  copy={cp:.3f}  Z_only={zo:.3f}  shuffle={sm:.3f}  random={rm:.3f}  wrong_H={wh:.3f}")
    print(f"  Z_gap (correct - max(zero,random,copy,shuffle)) = {c - max(z, rm, cp, sm):+.3f}")
    print(f"  Z_only ratio (Z_only / correct) = {zo / max(c,1e-6):.2f}  (low = Z is action-like, high = Z is future-like)")
    print(f"  wrong_H ratio (wrong_H / correct) = {wh / max(c,1e-6):.2f}  (low = H_t carries info, high = H_t ignored)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
