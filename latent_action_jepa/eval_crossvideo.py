"""Cross-video evaluation: how does Stage 1C+ teacher + goal-flow (trained on BBB)
behave when given a *Sintel* (held-out video) bundle?

Reports:
- oracle_Z (inverse encoder) on Sintel
- empZ best-of-K using BBB train Z* pool
- empZ best-of-K using SINTEL train Z* pool
- goal-flow K=1 + best-of-K on Sintel
- copy_Ht baseline
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from models.transition_residual import ResidualTransitionModel
from models.flow_prior_goal import GoalConditionedFlowPrior


def retrieval(q, k, N, name=None):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    mr = (r == gt).float().argmax(-1).float().mean().item() + 1
    if name: print(f"  {name:38s} top1={top1:.3f}  top5={top5:.3f}  mr={mr:.2f}")
    return top1, top5, mr


def best_of_k(preds_list, H_tp, N, name):
    target = F.normalize(H_tp.mean(1), dim=-1)
    best_rank = torch.full((N,), N + 1, device=H_tp.device, dtype=torch.long)
    for kp in preds_list:
        q = F.normalize(kp.mean(1), dim=-1)
        sim = q @ target.T
        r = sim.argsort(dim=-1, descending=True)
        gt = torch.arange(N, device=H_tp.device).unsqueeze(-1)
        rank = (r == gt).float().argmax(-1).long() + 1
        best_rank = torch.minimum(best_rank, rank)
    top1 = (best_rank == 1).float().mean().item()
    mr = best_rank.float().mean().item()
    print(f"  bestK={len(preds_list):3d} {name:24s} top1={top1:.3f}  mr={mr:.2f}")
    return top1, mr


@torch.no_grad()
def euler_sample(flow, h_t, h_goal, action_tokens, action_dim, steps=30, seed=42):
    b = h_t.shape[0]
    g = torch.Generator(device=h_t.device).manual_seed(seed)
    z = torch.randn(b, action_tokens, action_dim, device=h_t.device, generator=g)
    for i in range(steps):
        tau = torch.full((b, 1, 1), i / steps, device=h_t.device)
        v = flow(z, tau, h_t, h_goal)
        z = z + v / steps
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bbb_bundle", required=True, help="bundle of BBB Z* used as codebook")
    ap.add_argument("--sintel_bundle", required=True)
    ap.add_argument("--goal_flow", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128])
    ap.add_argument("--sample_steps", type=int, default=30)
    args = ap.parse_args()
    device = "cuda"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tr = ResidualTransitionModel(cfg["state_dim"], cfg["action_dim"], cfg["future_tokens"],
                                 depth=cfg["transition_depth"]).to(device)
    tr.load_state_dict(ckpt["transition_model"])
    for p in tr.parameters(): p.requires_grad_(False); tr.eval()

    fc = torch.load(args.goal_flow, map_location=device, weights_only=False)
    flow = GoalConditionedFlowPrior(cfg["state_dim"], cfg["action_dim"],
                                    depth=fc["args"]["flow_depth"]).to(device)
    flow.load_state_dict(fc["flow"])
    for p in flow.parameters(): p.requires_grad_(False); flow.eval()

    bbb = torch.load(args.bbb_bundle, map_location="cpu", weights_only=False)
    Z_pool_bbb = bbb["Z_star_train"].to(device)

    s = torch.load(args.sintel_bundle, map_location="cpu", weights_only=False)
    H_t = s["H_t_val"].to(device); H_tp = s["H_tp_val"].to(device); Z_star = s["Z_star_val"].to(device)
    Z_pool_sintel = s["Z_star_train"].to(device)
    N = H_t.shape[0]
    print(f"Sintel val N={N}  BBB pool N={Z_pool_bbb.shape[0]}  Sintel pool N={Z_pool_sintel.shape[0]}")

    results = {}
    with torch.no_grad():
        t1, _, mr = retrieval(tr(H_t, Z_star).mean(1), H_tp.mean(1), N, "oracle_Z (Sintel)")
        results["oracle"] = {"top1": t1, "mr": mr}
        t1, _, mr = retrieval(H_t.mean(1), H_tp.mean(1), N, "copy_Ht")
        results["copy"] = {"top1": t1, "mr": mr}
        t1, _, mr = retrieval(tr(H_t, torch.zeros_like(Z_star)).mean(1), H_tp.mean(1), N, "zero_Z")
        results["zero"] = {"top1": t1, "mr": mr}

        # empZ using BBB pool (the codebook trained on BBB)
        K_max = max(args.ks)
        print("\nempZ (BBB pool):")
        bbb_preds = [tr(H_t, Z_pool_bbb[torch.randint(0, Z_pool_bbb.shape[0], (N,), device=device)]) for _ in range(K_max)]
        for k in args.ks:
            t1, mr = best_of_k(bbb_preds[:k], H_tp, N, "BBB_pool")
            results[f"empZ_bbb_K{k}"] = {"top1": t1, "mr": mr}

        # empZ using Sintel pool
        print("\nempZ (Sintel pool):")
        sintel_preds = [tr(H_t, Z_pool_sintel[torch.randint(0, Z_pool_sintel.shape[0], (N,), device=device)]) for _ in range(K_max)]
        for k in args.ks:
            t1, mr = best_of_k(sintel_preds[:k], H_tp, N, "Sintel_pool")
            results[f"empZ_sintel_K{k}"] = {"top1": t1, "mr": mr}

        # goal-flow on Sintel pairs (using own H_tp as goal)
        print("\ngoal-flow on Sintel:")
        K_gf = max(args.ks)
        gf_preds = [tr(H_t, euler_sample(flow, H_t, H_tp, cfg["action_tokens"], cfg["action_dim"], steps=args.sample_steps, seed=k)) for k in range(K_gf)]
        t1_k1, _, mr_k1 = retrieval(gf_preds[0].mean(1), H_tp.mean(1), N, "goal_flow K=1")
        results["goal_flow_K1"] = {"top1": t1_k1, "mr": mr_k1}
        for k in args.ks:
            t1, mr = best_of_k(gf_preds[:k], H_tp, N, "goal_flow")
            results[f"goal_flow_K{k}"] = {"top1": t1, "mr": mr}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
