"""Evaluate goal-conditioned flow with noisy / corrupted goals."""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from models.transition_residual import ResidualTransitionModel
from models.flow_prior_goal import GoalConditionedFlowPrior


def retrieval(q, k, N):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    mr = (r == gt).float().argmax(-1).float().mean().item() + 1
    return top1, top5, mr


@torch.no_grad()
def euler_sample(flow, h_t, h_goal, action_tokens, action_dim, steps=30, seed=None):
    b = h_t.shape[0]
    g = torch.Generator(device=h_t.device).manual_seed(seed) if seed is not None else None
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
    ap.add_argument("--noises", type=float, nargs="+", default=[0.0, 0.1, 0.3, 1.0, 3.0])
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

    flow_ckpt = torch.load(args.goal_flow, map_location=device, weights_only=False)
    flow = GoalConditionedFlowPrior(cfg["state_dim"], cfg["action_dim"],
                                    depth=flow_ckpt["args"]["flow_depth"]).to(device)
    flow.load_state_dict(flow_ckpt["flow"])
    flow.eval()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t_val = bundle["H_t_val"].to(device); H_tp_val = bundle["H_tp_val"].to(device)
    Z_star_val = bundle["Z_star_val"].to(device)
    N = H_t_val.shape[0]

    # Reference points
    oracle_pred = tr(H_t_val, Z_star_val)
    oracle_top1, _, _ = retrieval(oracle_pred.mean(1), H_tp_val.mean(1), N)
    copy_top1, _, _ = retrieval(H_t_val.mean(1), H_tp_val.mean(1), N)
    print(f"oracle val top1={oracle_top1:.3f}  copy_Ht val top1={copy_top1:.3f}")

    # Goal std per dim, for noise scaling
    g_std = H_tp_val.flatten(1).std(0).mean().item()

    results = {"oracle_top1": oracle_top1, "copy_top1": copy_top1, "goal_std": g_std, "noises": {}}
    print("\nGoal noise level (relative to H_tp std):")
    for sigma in args.noises:
        # Perturb H_tp
        noise = torch.randn_like(H_tp_val) * (sigma * g_std)
        H_goal_noisy = H_tp_val + noise
        z = euler_sample(flow, H_t_val, H_goal_noisy, cfg["action_tokens"], cfg["action_dim"],
                         steps=args.sample_steps, seed=42)
        pred = tr(H_t_val, z)
        t1, t5, mr = retrieval(pred.mean(1), H_tp_val.mean(1), N)
        # Also: goal_random = random vector with H_tp moments
        H_goal_random = torch.randn_like(H_tp_val) * H_tp_val.std() + H_tp_val.mean()
        z_r = euler_sample(flow, H_t_val, H_goal_random, cfg["action_tokens"], cfg["action_dim"],
                           steps=args.sample_steps, seed=43)
        pred_r = tr(H_t_val, z_r)
        t1_r, t5_r, mr_r = retrieval(pred_r.mean(1), H_tp_val.mean(1), N)
        print(f"  sigma={sigma:.2f}  goal+noise top1={t1:.3f}  random-goal top1={t1_r:.3f}")
        results["noises"][str(sigma)] = {"goal_noisy_top1": t1, "goal_noisy_top5": t5, "goal_noisy_mr": mr,
                                          "goal_random_top1": t1_r, "goal_random_top5": t5_r, "goal_random_mr": mr_r}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
