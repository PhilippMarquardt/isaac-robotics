"""Stage 2B: conditional flow-matching prior p(Z | H_t).

Loads frozen Stage 1C ckpt + oracle bundle. Trains FlowPrior on
v_target = Z_star - eps. Eval single-sample + best-of-K retrieval.

Best-of-K is ORACLE selection (pick the sample whose rank against true H_tp is
best). That measures whether samples *contain* the right future. We separately
report self-scored selection where applicable.
"""
import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.inverse_encoder import InverseActionEncoder
from models.transition_residual import ResidualTransitionModel
from models.flow_prior import FlowPrior
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
        print(f"  retrieval[{name:22s}] top1={top1:.3f}  top5={top5:.3f}  mean_rank={rank:.2f}")
    return top1, top5, rank


def best_of_k_retrieval(preds_list, H_tp, N, name):
    """Oracle selection: per anchor, take min rank across K samples."""
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
    print(f"  bestK[K={len(preds_list):2d}|{name:14s}] top1={top1:.3f}  top5={top5:.3f}  mean_rank={mean_rank:.2f}")
    return {"top1": top1, "top5": top5, "mean_rank": mean_rank}


@torch.no_grad()
def euler_sample(flow, h_t, action_tokens, action_dim, steps=20, seed=None):
    b = h_t.shape[0]
    g = torch.Generator(device=h_t.device).manual_seed(seed) if seed is not None else None
    z = torch.randn(b, action_tokens, action_dim, device=h_t.device, generator=g)
    for i in range(steps):
        tau = torch.full((b, 1, 1), i / steps, device=h_t.device)
        v = flow(z, tau, h_t)
        z = z + v / steps
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/latent_action_jepa/runs/stage1c/ckpt.pt")
    ap.add_argument("--bundle", default="/workspace/latent_action_jepa/runs/stage1c/oracle_bundle.pt")
    ap.add_argument("--out_dir", default="/workspace/latent_action_jepa/runs/stage2b_flow_prior/run0")
    ap.add_argument("--flow_depth", type=int, default=6)
    ap.add_argument("--flow_heads", type=int, default=8)
    ap.add_argument("--flow_dropout", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--sample_steps", type=int, default=30)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(0)

    # ---- Load Stage 1C frozen ----
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    inv = InverseActionEncoder(cfg["state_dim"], cfg["action_dim"], cfg["action_tokens"],
                               depth=cfg["inverse_depth"]).to(device)
    inv.load_state_dict(ckpt["inverse_encoder"])
    tr = ResidualTransitionModel(cfg["state_dim"], cfg["action_dim"], cfg["future_tokens"],
                                 depth=cfg["transition_depth"]).to(device)
    tr.load_state_dict(ckpt["transition_model"])
    for p in list(inv.parameters()) + list(tr.parameters()):
        p.requires_grad_(False)
    inv.eval(); tr.eval()
    print(f"loaded Stage 1C ckpt: alpha={tr.alpha.item():.4f}")

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t_train = bundle["H_t_train"].to(device); H_tp_train = bundle["H_tp_train"].to(device)
    Z_star_train = bundle["Z_star_train"].to(device)
    H_t_val = bundle["H_t_val"].to(device); H_tp_val = bundle["H_tp_val"].to(device)
    Z_star_val = bundle["Z_star_val"].to(device)
    N_tr = H_t_train.shape[0]; N_va = H_t_val.shape[0]
    print(f"train N={N_tr}  val N={N_va}  action {cfg['action_tokens']}x{cfg['action_dim']}")

    flow = FlowPrior(cfg["state_dim"], cfg["action_dim"],
                     depth=args.flow_depth, heads=args.flow_heads, drop=args.flow_dropout).to(device)
    print(f"flow params: {sum(p.numel() for p in flow.parameters())/1e6:.2f} M")

    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # ---- Training ----
    print("\n=== Training flow ===")
    flow.train()
    log = []
    for step in range(args.steps):
        idx = torch.randint(0, N_tr, (args.bs,), device=device)
        h_t = H_t_train[idx]; z_star = Z_star_train[idx]
        eps = torch.randn_like(z_star)
        tau = torch.rand(z_star.shape[0], 1, 1, device=device)
        z_tau = (1 - tau) * eps + tau * z_star
        v_target = z_star - eps
        v_pred = flow(z_tau, tau, h_t)
        loss = F.mse_loss(v_pred, v_target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  L_flow={loss.item():.4f}  lr={opt.param_groups[0]['lr']:.2e}")
            log.append({"step": step, "L_flow": loss.item()})

    # ---- Eval ----
    def eval_split(H_t, H_tp, Z_star, tag):
        N = H_t.shape[0]
        flow.eval()
        print(f"\n=== [{tag}]  N={N} ===")
        results = {"split": tag, "N": N}

        with torch.no_grad():
            # Sample K_max once
            K_max = max(args.ks)
            all_samples = []
            for k in range(K_max):
                z = euler_sample(flow, H_t, cfg["action_tokens"], cfg["action_dim"], steps=args.sample_steps, seed=1000 + k)
                all_samples.append(z)

            # Stats on first sample
            z0 = all_samples[0]
            std_zh = z0.flatten(1).std(0).mean().item()
            std_zs = Z_star.flatten(1).std(0).mean().item()
            er_zh = effective_rank(z0.flatten(1))
            er_zs = effective_rank(Z_star.flatten(1))
            cos_to_star = F.cosine_similarity(z0.flatten(1), Z_star.flatten(1), dim=-1).mean().item()
            print(f"  sample0  std={std_zh:.4f} (Z*={std_zs:.4f})  eff_rank={er_zh:.1f} (Z*={er_zs:.1f})  cos(s0,Z*)={cos_to_star:+.3f}")
            results.update({
                "sample0_std": std_zh, "Z_star_std": std_zs,
                "sample0_eff_rank": er_zh, "Z_star_eff_rank": er_zs,
                "sample0_cos_to_Zstar": cos_to_star,
            })

            # Predictions
            preds_per_k = [tr(H_t, z) for z in all_samples]

            # Single-sample retrieval
            print("\n  -- single-sample retrieval --")
            t1, t5, mr = retrieval(preds_per_k[0].mean(1), H_tp.mean(1), N, "flow_Z(K=1)")
            results["flow_Z_K1"] = {"top1": t1, "top5": t5, "mean_rank": mr}

            # Baselines
            t1, t5, mr = retrieval(tr(H_t, Z_star).mean(1), H_tp.mean(1), N, "oracle_Z")
            results["oracle_Z"] = {"top1": t1, "top5": t5, "mean_rank": mr}
            t1, t5, mr = retrieval(tr(H_t, torch.zeros_like(Z_star)).mean(1), H_tp.mean(1), N, "zero_Z")
            results["zero_Z"] = {"top1": t1, "top5": t5, "mean_rank": mr}
            t1, t5, mr = retrieval(tr(H_t, torch.randn_like(Z_star) * std_zs).mean(1), H_tp.mean(1), N, "random_Z")
            results["random_Z"] = {"top1": t1, "top5": t5, "mean_rank": mr}
            t1, t5, mr = retrieval(H_t.mean(1), H_tp.mean(1), N, "copy_Ht")
            results["copy_Ht"] = {"top1": t1, "top5": t5, "mean_rank": mr}

            # Best-of-K oracle selection
            print("\n  -- best-of-K (oracle selection by true H_tp) --")
            for k in args.ks:
                r = best_of_k_retrieval(preds_per_k[:k], H_tp, N, "flow_Z")
                results[f"bestK_{k}"] = r

            # Best-of-K baseline: K random_Z samples
            print("\n  -- best-of-K baseline: random_Z samples --")
            rand_preds = [tr(H_t, torch.randn_like(Z_star) * std_zs) for _ in range(max(args.ks))]
            for k in args.ks:
                r = best_of_k_retrieval(rand_preds[:k], H_tp, N, "random_Z")
                results[f"bestK_random_{k}"] = r

        return results

    pre_train = None
    print("\n=== Eval ===")
    metrics_train = eval_split(H_t_train, H_tp_train, Z_star_train, "train post")
    metrics_val = eval_split(H_t_val, H_tp_val, Z_star_val, "val post")

    # ---- Save ----
    torch.save({"flow": flow.state_dict(), "args": vars(args)}, os.path.join(args.out_dir, "flow.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"train": metrics_train, "val": metrics_val, "args": vars(args), "train_log": log}, f, indent=2)
    print(f"\nsaved -> {args.out_dir}")
    print("\nSTAGE 2B DONE")


if __name__ == "__main__":
    main()
