"""Stage 2B-goal: conditional flow p(Z | H_t, H_goal).

H_goal is set to H_tp at training time. At eval time, the user can provide
H_tp (oracle) or a perturbed/coarse target. We report both.
"""
import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transition_residual import ResidualTransitionModel
from models.flow_prior_goal import GoalConditionedFlowPrior
from diag_stage0 import effective_rank


def retrieval(q, k, N, name=None, print_=True):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    rank = (r == gt).float().argmax(-1).float().mean().item() + 1
    if print_ and name:
        print(f"  retrieval[{name:24s}] top1={top1:.3f}  top5={top5:.3f}  mean_rank={rank:.2f}")
    return top1, top5, rank


def best_of_k_retrieval(preds_list, H_tp, N, name):
    target = F.normalize(H_tp.mean(1), dim=-1)
    best_rank = torch.full((N,), N + 1, device=H_tp.device, dtype=torch.long)
    for kp in preds_list:
        q = F.normalize(kp.mean(1), dim=-1)
        sim = q @ target.T
        r = sim.argsort(dim=-1, descending=True)
        gt = torch.arange(N, device=H_tp.device).unsqueeze(-1)
        rank_i = (r == gt).float().argmax(-1).long() + 1
        best_rank = torch.minimum(best_rank, rank_i)
    top1 = (best_rank == 1).float().mean().item()
    top5 = (best_rank <= 5).float().mean().item()
    mean_rank = best_rank.float().mean().item()
    print(f"  bestK[K={len(preds_list):2d}|{name:18s}] top1={top1:.3f}  top5={top5:.3f}  mean_rank={mean_rank:.2f}")
    return {"top1": top1, "top5": top5, "mean_rank": mean_rank}


@torch.no_grad()
def euler_sample(flow, h_t, h_goal, action_tokens, action_dim, steps=20, seed=None):
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
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--flow_depth", type=int, default=6)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sample_steps", type=int, default=30)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(0)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tr = ResidualTransitionModel(cfg["state_dim"], cfg["action_dim"], cfg["future_tokens"],
                                 depth=cfg["transition_depth"]).to(device)
    tr.load_state_dict(ckpt["transition_model"])
    for p in tr.parameters(): p.requires_grad_(False)
    tr.eval()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t_train = bundle["H_t_train"].to(device); H_tp_train = bundle["H_tp_train"].to(device)
    Z_star_train = bundle["Z_star_train"].to(device)
    H_t_val = bundle["H_t_val"].to(device); H_tp_val = bundle["H_tp_val"].to(device)
    Z_star_val = bundle["Z_star_val"].to(device)
    N_tr = H_t_train.shape[0]; N_va = H_t_val.shape[0]

    flow = GoalConditionedFlowPrior(cfg["state_dim"], cfg["action_dim"],
                                    depth=args.flow_depth).to(device)
    print(f"goal-conditioned flow params: {sum(p.numel() for p in flow.parameters())/1e6:.2f} M")

    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    print("\n=== Training goal-conditioned flow ===")
    flow.train()
    for step in range(args.steps):
        idx = torch.randint(0, N_tr, (args.bs,), device=device)
        h_t = H_t_train[idx]; h_g = H_tp_train[idx]; z_star = Z_star_train[idx]
        eps = torch.randn_like(z_star)
        tau = torch.rand(z_star.shape[0], 1, 1, device=device)
        z_tau = (1 - tau) * eps + tau * z_star
        v_target = z_star - eps
        v_pred = flow(z_tau, tau, h_t, h_g)
        loss = F.mse_loss(v_pred, v_target)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  L_flow={loss.item():.4f}  lr={opt.param_groups[0]['lr']:.2e}")

    # eval val
    flow.eval()
    print(f"\n=== Eval val (N={N_va}) ===")
    results = {}
    with torch.no_grad():
        K_max = max(args.ks)
        samples = [euler_sample(flow, H_t_val, H_tp_val, cfg["action_tokens"], cfg["action_dim"],
                                steps=args.sample_steps, seed=10 + k) for k in range(K_max)]
        z0 = samples[0]
        cos_to_star = F.cosine_similarity(z0.flatten(1), Z_star_val.flatten(1), dim=-1).mean().item()
        print(f"sample0 cos(s0,Z*)={cos_to_star:+.3f}")
        preds = [tr(H_t_val, z) for z in samples]
        # single sample
        retrieval(preds[0].mean(1), H_tp_val.mean(1), N_va, "goal_flow_K=1")
        retrieval(tr(H_t_val, Z_star_val).mean(1), H_tp_val.mean(1), N_va, "oracle_Z")
        retrieval(H_t_val.mean(1), H_tp_val.mean(1), N_va, "copy_Ht")
        for k in args.ks:
            r = best_of_k_retrieval(preds[:k], H_tp_val, N_va, "goal_flow_Z")
            results[f"bestK_{k}"] = r
    torch.save({"flow": flow.state_dict(), "args": vars(args)}, os.path.join(args.out_dir, "flow.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
