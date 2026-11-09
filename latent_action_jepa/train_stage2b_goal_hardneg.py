"""Goal-conditioned flow trained with hard negatives.

At each train step, a fraction p_neg of samples uses a SHUFFLED goal:
  (H_t_i, H_goal=H_tp_perm(i)) with target Z = inv(H_t_i, H_goal).
The rest use the natural (H_tp_i) goal.

This forces the flow to map arbitrary (state, goal) pairs to a Z that achieves
the goal, not just the in-distribution (H_t, H_tp) pairings.
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from models.inverse_encoder import InverseActionEncoder
from models.transition_residual import ResidualTransitionModel
from models.flow_prior_goal import GoalConditionedFlowPrior


def retrieval(q, k, N, name=None, print_=True):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    mr = (r == gt).float().argmax(-1).float().mean().item() + 1
    if print_ and name: print(f"  {name:32s} top1={top1:.3f}  top5={top5:.3f}  mr={mr:.2f}")
    return top1, top5, mr


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
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--flow_depth", type=int, default=6)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sample_steps", type=int, default=30)
    ap.add_argument("--p_neg", type=float, default=0.5, help="fraction of batch with shuffled goal")
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
    inv = InverseActionEncoder(cfg["state_dim"], cfg["action_dim"], cfg["action_tokens"],
                               depth=cfg["inverse_depth"]).to(device)
    inv.load_state_dict(ckpt["inverse_encoder"])
    for p in list(tr.parameters()) + list(inv.parameters()):
        p.requires_grad_(False)
    tr.eval(); inv.eval()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t_train = bundle["H_t_train"].to(device); H_tp_train = bundle["H_tp_train"].to(device)
    H_t_val = bundle["H_t_val"].to(device); H_tp_val = bundle["H_tp_val"].to(device)
    Z_star_val = bundle["Z_star_val"].to(device)
    N_tr = H_t_train.shape[0]; N_va = H_t_val.shape[0]

    flow = GoalConditionedFlowPrior(cfg["state_dim"], cfg["action_dim"],
                                    depth=args.flow_depth).to(device)
    print(f"hardneg goal-flow params: {sum(p.numel() for p in flow.parameters())/1e6:.2f} M")
    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    print(f"\n=== Training (p_neg={args.p_neg}) ===")
    flow.train()
    for step in range(args.steps):
        idx = torch.randint(0, N_tr, (args.bs,), device=device)
        h_t = H_t_train[idx]
        # default: H_goal = own future
        h_goal = H_tp_train[idx].clone()
        # mix: for fraction p_neg use shuffled goal
        mask_shuf = (torch.rand(args.bs, device=device) < args.p_neg)
        if mask_shuf.any():
            shuf_idx = idx[torch.randperm(args.bs, device=device)]
            h_goal[mask_shuf] = H_tp_train[shuf_idx][mask_shuf]
        # compute target Z for whichever goal was given
        with torch.no_grad():
            z_target = inv(h_t, h_goal)
        eps = torch.randn_like(z_target)
        tau = torch.rand(z_target.shape[0], 1, 1, device=device)
        z_tau = (1 - tau) * eps + tau * z_target
        v_target = z_target - eps
        v_pred = flow(z_tau, tau, h_t, h_goal)
        loss = F.mse_loss(v_pred, v_target)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  L_flow={loss.item():.4f}  frac_shuf={mask_shuf.float().mean().item():.2f}  lr={opt.param_groups[0]['lr']:.2e}")

    # ---- Eval ----
    flow.eval()
    print(f"\n=== Eval val (N={N_va}) ===")
    results = {}
    with torch.no_grad():
        # 1) Goal = true H_tp (own goal)
        K_max = max(args.ks)
        samples_own = [euler_sample(flow, H_t_val, H_tp_val, cfg["action_tokens"], cfg["action_dim"],
                                     steps=args.sample_steps, seed=k) for k in range(K_max)]
        preds_own = [tr(H_t_val, z) for z in samples_own]
        retrieval(preds_own[0].mean(1), H_tp_val.mean(1), N_va, "K=1 own_goal")
        for k in args.ks:
            target = F.normalize(H_tp_val.mean(1), dim=-1)
            best_rank = torch.full((N_va,), N_va + 1, device=device, dtype=torch.long)
            for j in range(k):
                q = F.normalize(preds_own[j].mean(1), dim=-1)
                sim = q @ target.T
                r = sim.argsort(dim=-1, descending=True)
                gt = torch.arange(N_va, device=device).unsqueeze(-1)
                rank_i = (r == gt).float().argmax(-1).long() + 1
                best_rank = torch.minimum(best_rank, rank_i)
            t1 = (best_rank == 1).float().mean().item()
            mr = best_rank.float().mean().item()
            print(f"  bestK={k:2d} own_goal      top1={t1:.3f}  mr={mr:.2f}")
            results[f"own_K{k}"] = {"top1": t1, "mr": mr}

        # 2) Shuffled goal: H_goal = H_tp[perm]. Goal-flow should produce Z that achieves the PERMUTED target.
        perm = torch.randperm(N_va, device=device)
        H_goal_shuf = H_tp_val[perm]
        target_shuf = F.normalize(H_goal_shuf.mean(1), dim=-1)
        samples_shuf = [euler_sample(flow, H_t_val, H_goal_shuf, cfg["action_tokens"], cfg["action_dim"],
                                      steps=args.sample_steps, seed=1000 + k) for k in range(8)]
        preds_shuf = [tr(H_t_val, z) for z in samples_shuf]
        # retrieve against shuffled targets
        q = F.normalize(preds_shuf[0].mean(1), dim=-1)
        sim = q @ target_shuf.T
        r = sim.argsort(dim=-1, descending=True)
        gt = torch.arange(N_va, device=device).unsqueeze(-1)
        rank_i = (r == gt).float().argmax(-1).long() + 1
        t1_shuf = (rank_i == 1).float().mean().item()
        # retrieve against TRUE targets (own future) — should be near zero
        q_true = F.normalize(preds_shuf[0].mean(1), dim=-1)
        sim_t = q_true @ F.normalize(H_tp_val.mean(1), dim=-1).T
        r_t = sim_t.argsort(dim=-1, descending=True)
        rank_t = (r_t == gt).float().argmax(-1).long() + 1
        t1_true = (rank_t == 1).float().mean().item()
        print(f"\n  shuffled-goal K=1: top1 vs shuffled_target={t1_shuf:.3f}  top1 vs own_target={t1_true:.3f}")
        results["shuf_K1_vs_shuf"] = t1_shuf
        results["shuf_K1_vs_own"] = t1_true

    torch.save({"flow": flow.state_dict(), "args": vars(args)}, os.path.join(args.out_dir, "flow.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"\nsaved -> {args.out_dir}")


if __name__ == "__main__":
    main()
