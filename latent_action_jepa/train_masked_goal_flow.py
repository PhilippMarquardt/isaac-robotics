"""Exp 2: masked / partial-goal flow.

For each batch, sample keep ∈ {0,1,2,4,8,16} uniformly, then randomly choose
that many of the 16 future tokens to reveal. The rest are replaced by [MASK].
Targets the SAME one-step teacher Z*.

Eval per-keep × K table over the same val set.
"""
import argparse
import json
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transition_residual import ResidualTransitionModel
from models.flow_prior_masked_goal import MaskedGoalFlowPrior


KEEP_VALUES = [0, 1, 2, 4, 8, 16]


def retrieval(q, k, N, name=""):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    mr = (r == gt).float().argmax(-1).float().mean().item() + 1
    if name: print(f"  {name:32s} top1={top1:.3f}  mr={mr:.2f}")
    return top1, mr


def best_of_k_retrieval(preds_list, H_tp, N, name=""):
    target = F.normalize(H_tp.mean(1), dim=-1)
    best_rank = torch.full((N,), N + 1, device=H_tp.device, dtype=torch.long)
    for p in preds_list:
        q = F.normalize(p.mean(1), dim=-1)
        sim = q @ target.T
        r = sim.argsort(dim=-1, descending=True)
        gt = torch.arange(N, device=H_tp.device).unsqueeze(-1)
        rank = (r == gt).float().argmax(-1).long() + 1
        best_rank = torch.minimum(best_rank, rank)
    top1 = (best_rank == 1).float().mean().item()
    mr = best_rank.float().mean().item()
    if name: print(f"  bestK={len(preds_list):3d} {name:18s} top1={top1:.3f}  mr={mr:.2f}")
    return top1, mr


def sample_keep_mask(B, T, keep, device, generator=None):
    """Return a [B, T] boolean visibility mask with exactly `keep` Trues per row,
    chosen uniformly per row."""
    if keep == 0:
        return torch.zeros(B, T, dtype=torch.bool, device=device)
    if keep >= T:
        return torch.ones(B, T, dtype=torch.bool, device=device)
    # Build by random argsort
    scores = torch.rand(B, T, device=device, generator=generator)
    idx = scores.argsort(dim=-1, descending=False)[:, :keep]  # pick `keep` lowest
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return mask


@torch.no_grad()
def euler_sample(flow, H_t, H_goal, vis, action_tokens, action_dim, steps=30, seed=42):
    b = H_t.shape[0]
    g = torch.Generator(device=H_t.device).manual_seed(seed)
    z = torch.randn(b, action_tokens, action_dim, device=H_t.device, generator=g)
    for i in range(steps):
        tau = torch.full((b, 1, 1), i / steps, device=H_t.device)
        v = flow(z, tau, H_t, H_goal, vis)
        z = z + v / steps
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Stage 1 ckpt (tight-Z baseline)")
    ap.add_argument("--bundle", required=True, help="any bundle with H_t/H_tp/Z_star/video_id")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--flow_depth", type=int, default=6)
    ap.add_argument("--flow_heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--sample_steps", type=int, default=30)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--log_every", type=int, default=500)
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
    print(f"loaded teacher: alpha={tr.alpha.item():.4f}  action {cfg['action_tokens']}x{cfg['action_dim']}")

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t = bundle["H_t_train"].to(device); H_tp = bundle["H_tp_train"].to(device); Z = bundle["Z_star_train"].to(device)
    H_t_va = bundle["H_t_val"].to(device); H_tp_va = bundle["H_tp_val"].to(device); Z_va = bundle["Z_star_val"].to(device)
    N_tr = H_t.shape[0]; N_va = H_t_va.shape[0]
    print(f"train {N_tr}  val {N_va}")

    flow = MaskedGoalFlowPrior(state_dim=cfg["state_dim"], action_dim=cfg["action_dim"],
                               future_tokens=cfg["future_tokens"], depth=args.flow_depth,
                               heads=args.flow_heads).to(device)
    print(f"flow params: {sum(p.numel() for p in flow.parameters())/1e6:.2f} M")

    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    flow.train()
    for step in range(args.steps):
        idx = torch.randint(0, N_tr, (args.bs,), device=device)
        h_t = H_t[idx]; h_tp = H_tp[idx]; z_target = Z[idx]
        # uniform sample keep per batch (single keep for whole batch keeps it simple)
        keep = KEEP_VALUES[random.randrange(len(KEEP_VALUES))]
        vis = sample_keep_mask(args.bs, cfg["future_tokens"], keep, device)

        eps = torch.randn_like(z_target)
        tau = torch.rand(z_target.shape[0], 1, 1, device=device)
        z_tau = (1 - tau) * eps + tau * z_target
        v_target = z_target - eps
        v_pred = flow(z_tau, tau, h_t, h_tp, vis)
        loss = F.mse_loss(v_pred, v_target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  L_flow={loss.item():.4f}  keep={keep}  lr={opt.param_groups[0]['lr']:.2e}")

    # ---- Eval ----
    flow.eval()
    print(f"\n=== Eval val (N={N_va}) ===")
    results = {"per_keep": {}}
    with torch.no_grad():
        retrieval(tr(H_t_va, Z_va).mean(1), H_tp_va.mean(1), N_va, "oracle_Z")
        std_z = Z_va.flatten(1).std(0).mean()
        retrieval(tr(H_t_va, torch.zeros_like(Z_va)).mean(1), H_tp_va.mean(1), N_va, "zero_Z")
        retrieval(tr(H_t_va, torch.randn_like(Z_va) * std_z).mean(1), H_tp_va.mean(1), N_va, "random_Z")
        retrieval(H_t_va.mean(1), H_tp_va.mean(1), N_va, "copy_Ht")

        K_max = max(args.ks)
        for keep in KEEP_VALUES:
            vis = sample_keep_mask(N_va, cfg["future_tokens"], keep, device,
                                   generator=torch.Generator(device=device).manual_seed(7 + keep))
            samples = [euler_sample(flow, H_t_va, H_tp_va, vis,
                                    cfg["action_tokens"], cfg["action_dim"],
                                    steps=args.sample_steps, seed=10 * keep + k) for k in range(K_max)]
            preds = [tr(H_t_va, z) for z in samples]
            print(f"\n  keep={keep}:")
            row = {}
            for k in args.ks:
                top1, mr = best_of_k_retrieval(preds[:k], H_tp_va, N_va, f"K={k}")
                row[f"K{k}"] = {"top1": top1, "mr": mr}
            results["per_keep"][keep] = row

    torch.save({"flow": flow.state_dict(), "args": vars(args)}, os.path.join(args.out_dir, "flow.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"\nsaved -> {args.out_dir}")

    # Print summary table
    print("\nkeep |   K=1   K=4   K=8  K=16  K=32  K=64  K=128")
    for keep in KEEP_VALUES:
        row = results["per_keep"][keep]
        cells = "  ".join(f"{row.get(f'K{k}', {'top1': 0})['top1']:.3f}" for k in args.ks)
        print(f"  {keep:2d}   {cells}")


if __name__ == "__main__":
    main()
