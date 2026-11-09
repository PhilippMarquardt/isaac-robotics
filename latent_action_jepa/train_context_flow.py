"""Exp 1: train ContextReducer + ContextFlowPrior on a frozen Stage 1 teacher.

Freezes: ViT, reducers, inverse_encoder, transition.
Trains:  ContextReducer + ContextFlowPrior.
Loss:    flow-matching MSE on velocity v_target = Z* - eps.
"""
import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transition_residual import ResidualTransitionModel
from models.context_encoder import ContextReducer
from models.context_flow import ContextFlowPrior


def retrieval(q, k, N, name=""):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    mr = (r == gt).float().argmax(-1).float().mean().item() + 1
    if name: print(f"  {name:24s} top1={top1:.3f}  top5={top5:.3f}  mr={mr:.2f}")
    return top1, top5, mr


def best_of_k(preds_list, H_tp, N, name=""):
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
    if name: print(f"  bestK={len(preds_list):3d} {name:16s} top1={top1:.3f}  mr={mr:.2f}")
    return top1, mr


@torch.no_grad()
def euler_sample(flow, C_t, action_tokens, action_dim, steps=30, seed=42):
    b = C_t.shape[0]
    g = torch.Generator(device=C_t.device).manual_seed(seed)
    z = torch.randn(b, action_tokens, action_dim, device=C_t.device, generator=g)
    for i in range(steps):
        tau = torch.full((b, 1, 1), i / steps, device=C_t.device)
        v = flow(z, tau, C_t)
        z = z + v / steps
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Stage 1 ckpt (tight-Z baseline)")
    ap.add_argument("--bundle", required=True, help="context bundle from precompute_context_bundle.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--context_tokens", type=int, default=16)
    ap.add_argument("--context_depth", type=int, default=2)
    ap.add_argument("--context_heads", type=int, default=4)
    ap.add_argument("--flow_depth", type=int, default=6)
    ap.add_argument("--flow_heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--sample_steps", type=int, default=30)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16, 32])
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
    print(f"loaded teacher ckpt: alpha={tr.alpha.item():.4f}  action {cfg['action_tokens']}x{cfg['action_dim']}")

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_ctx_tr = bundle["H_ctx_train"].to(device)
    H_t_tr = bundle["H_t_train"].to(device)
    Z_tr = bundle["Z_star_train"].to(device)
    H_ctx_va = bundle["H_ctx_val"].to(device)
    H_t_va = bundle["H_t_val"].to(device)
    H_tp_va = bundle["H_tp_val"].to(device)
    Z_va = bundle["Z_star_val"].to(device)
    vid_va = bundle["video_id_val"].to(device)
    N_tr = H_ctx_tr.shape[0]; N_va = H_ctx_va.shape[0]
    print(f"train {N_tr}  val {N_va}  H_ctx shape {tuple(H_ctx_tr.shape)}")

    ctx = ContextReducer(state_dim=cfg["state_dim"], action_dim=cfg["action_dim"],
                         out_tokens=args.context_tokens, depth=args.context_depth,
                         heads=args.context_heads, n_history_frames=H_ctx_tr.shape[1]).to(device)
    flow = ContextFlowPrior(action_dim=cfg["action_dim"], depth=args.flow_depth, heads=args.flow_heads).to(device)
    print(f"ctx params: {sum(p.numel() for p in ctx.parameters())/1e6:.2f}M  flow params: {sum(p.numel() for p in flow.parameters())/1e6:.2f}M")

    params = list(ctx.parameters()) + list(flow.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    ctx.train(); flow.train()
    for step in range(args.steps):
        idx = torch.randint(0, N_tr, (args.bs,), device=device)
        z_target = Z_tr[idx]
        eps = torch.randn_like(z_target)
        tau = torch.rand(z_target.shape[0], 1, 1, device=device)
        z_tau = (1 - tau) * eps + tau * z_target
        v_target = z_target - eps

        C_t = ctx(H_ctx_tr[idx])
        v_pred = flow(z_tau, tau, C_t)
        loss = F.mse_loss(v_pred, v_target)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  L_flow={loss.item():.4f}  lr={opt.param_groups[0]['lr']:.2e}")

    # ---- Eval ----
    ctx.eval(); flow.eval()
    print(f"\n=== Eval val (N={N_va}) ===")
    results = {}
    with torch.no_grad():
        # oracle baseline
        t1_or, _, mr_or = retrieval(tr(H_t_va, Z_va).mean(1), H_tp_va.mean(1), N_va, "oracle_Z")
        results["oracle"] = {"top1": t1_or, "mr": mr_or}
        # copy / zero / random
        t1, _, mr = retrieval(H_t_va.mean(1), H_tp_va.mean(1), N_va, "copy_Ht"); results["copy"] = {"top1": t1, "mr": mr}
        t1, _, mr = retrieval(tr(H_t_va, torch.zeros_like(Z_va)).mean(1), H_tp_va.mean(1), N_va, "zero_Z"); results["zero"] = {"top1": t1, "mr": mr}
        std_z = Z_va.flatten(1).std(0).mean()
        t1, _, mr = retrieval(tr(H_t_va, torch.randn_like(Z_va) * std_z).mean(1), H_tp_va.mean(1), N_va, "random_Z"); results["random"] = {"top1": t1, "mr": mr}

        # Single random_Z best-of-K baseline for honesty
        rand_preds = [tr(H_t_va, torch.randn_like(Z_va) * std_z) for _ in range(max(args.ks))]
        print("\n  random_Z best-of-K:")
        for k in args.ks:
            top1, mr = best_of_k(rand_preds[:k], H_tp_va, N_va, f"random_Z")
            results[f"random_K{k}"] = {"top1": top1, "mr": mr}

        # Context-flow samples
        K_max = max(args.ks)
        C_va = ctx(H_ctx_va)
        samples = [euler_sample(flow, C_va, cfg["action_tokens"], cfg["action_dim"],
                                steps=args.sample_steps, seed=10 + k) for k in range(K_max)]
        preds = [tr(H_t_va, z) for z in samples]
        print("\n  context-flow best-of-K:")
        for k in args.ks:
            top1, mr = best_of_k(preds[:k], H_tp_va, N_va, f"ctx_flow")
            results[f"ctx_flow_K{k}"] = {"top1": top1, "mr": mr}

        # Z film probe on a single sample (must escape the no_grad wrap)
        s0 = samples[0]
        std_s = s0.flatten(1).std(0).mean().item()
        s0_pool = s0.mean(1).detach()  # [N_va, action_dim]
        results["sample0_std"] = std_s
        results["context_flow_K1_vs_oracle"] = results["ctx_flow_K1"]["top1"] / max(t1_or, 1e-6)

    # outside no_grad: train film probe head
    n_films = int(vid_va.max().item() + 1)
    head = nn.Linear(cfg["action_dim"], n_films).to(device)
    opt2 = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for _ in range(1500):
        i = torch.randint(0, s0_pool.shape[0], (min(256, s0_pool.shape[0]),), device=device)
        l = F.cross_entropy(head(s0_pool[i]), vid_va[i])
        opt2.zero_grad(); l.backward(); opt2.step()
    with torch.no_grad():
        acc = (head(s0_pool).argmax(-1) == vid_va).float().mean().item()
    results["sample0_film_probe_val_acc"] = acc
    print(f"\n  sample0 std={std_s:.4f}  film_probe_val_acc={acc:.3f}  chance={1/n_films:.4f}")

    torch.save({"ctx": ctx.state_dict(), "flow": flow.state_dict(), "args": vars(args)},
               os.path.join(args.out_dir, "ctx_flow.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
