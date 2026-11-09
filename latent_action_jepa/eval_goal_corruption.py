"""Evaluate goal-conditioned flow with structured goal corruption:
- partial goal (mask k of 16 tokens to zero/mean)
- shuffled goal (other sample's H_tp)
- interpolated goal (lambda * H_tp_i + (1-lambda) * H_tp_j)
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from models.transition_residual import ResidualTransitionModel
from models.flow_prior_goal import GoalConditionedFlowPrior


def retrieval_against_true(pred, H_tp, N):
    qn = F.normalize(pred.mean(1), dim=-1); kn = F.normalize(H_tp.mean(1), dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=pred.device).unsqueeze(-1)
    return (r[:, :1] == gt).any(-1).float().mean().item()


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
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--goal_flow", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample_steps", type=int, default=30)
    args = ap.parse_args()
    device = "cuda"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tr = ResidualTransitionModel(cfg["state_dim"], cfg["action_dim"], cfg["future_tokens"],
                                 depth=cfg["transition_depth"]).to(device)
    tr.load_state_dict(ckpt["transition_model"])
    for p in tr.parameters(): p.requires_grad_(False)
    tr.eval()
    fc = torch.load(args.goal_flow, map_location=device, weights_only=False)
    flow = GoalConditionedFlowPrior(cfg["state_dim"], cfg["action_dim"],
                                    depth=fc["args"]["flow_depth"]).to(device)
    flow.load_state_dict(fc["flow"])
    flow.eval()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t = bundle["H_t_val"].to(device); H_tp = bundle["H_tp_val"].to(device)
    Z_star = bundle["Z_star_val"].to(device)
    N = H_t.shape[0]

    oracle_top1 = retrieval_against_true(tr(H_t, Z_star), H_tp, N)
    copy_top1 = retrieval_against_true(H_t.mean(1).unsqueeze(1).expand(-1, 16, -1), H_tp, N)
    print(f"oracle val top1={oracle_top1:.3f}  copy_Ht val top1={copy_top1:.3f}")
    results = {"oracle_top1": oracle_top1, "copy_top1": copy_top1}

    # 1. Clean goal
    z_clean = euler_sample(flow, H_t, H_tp, cfg["action_tokens"], cfg["action_dim"], steps=args.sample_steps)
    t1 = retrieval_against_true(tr(H_t, z_clean), H_tp, N)
    print(f"clean_goal           top1={t1:.3f}")
    results["clean_goal_top1"] = t1

    # 2. Partial goal: keep first k tokens, zero rest
    for keep in [1, 2, 4, 8, 12, 16]:
        mask = torch.zeros(16, device=device); mask[:keep] = 1.0
        H_partial = H_tp * mask.view(1, 16, 1)
        z = euler_sample(flow, H_t, H_partial, cfg["action_tokens"], cfg["action_dim"], steps=args.sample_steps)
        t1 = retrieval_against_true(tr(H_t, z), H_tp, N)
        print(f"partial_goal({keep}/16) top1={t1:.3f}")
        results[f"partial_{keep}_top1"] = t1

    # 3. Shuffled goal: another sample's H_tp
    perm = torch.randperm(N, device=device)
    z_sh = euler_sample(flow, H_t, H_tp[perm], cfg["action_tokens"], cfg["action_dim"], steps=args.sample_steps)
    # retrieval against TRUE H_tp (not the shuffled one)
    t1_correct = retrieval_against_true(tr(H_t, z_sh), H_tp, N)
    # also vs the SHUFFLED target (so we can see if flow correctly produced the shuffled action)
    pred_sh = tr(H_t, z_sh)
    qn = F.normalize(pred_sh.mean(1), dim=-1); kn = F.normalize(H_tp[perm].mean(1), dim=-1)
    sim = qn @ kn.T
    rr = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=device).unsqueeze(-1)
    t1_shuffled = (rr[:, :1] == gt).any(-1).float().mean().item()
    print(f"shuffled_goal       top1 vs true={t1_correct:.3f}  top1 vs shuffled target={t1_shuffled:.3f}")
    results["shuffled_goal_top1_vs_true"] = t1_correct
    results["shuffled_goal_top1_vs_shuffled"] = t1_shuffled

    # 4. Interpolated goal
    for lam in [0.25, 0.5, 0.75]:
        H_interp = lam * H_tp + (1 - lam) * H_tp[perm]
        z = euler_sample(flow, H_t, H_interp, cfg["action_tokens"], cfg["action_dim"], steps=args.sample_steps)
        t1 = retrieval_against_true(tr(H_t, z), H_tp, N)
        print(f"interp_goal lam={lam} top1 vs true={t1:.3f}")
        results[f"interp_{lam}_top1"] = t1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
