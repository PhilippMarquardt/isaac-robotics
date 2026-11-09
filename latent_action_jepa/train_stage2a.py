"""Stage 2A: deterministic prior Z_hat = Prior(H_t).

Loads frozen Stage 1C checkpoint (inverse_encoder + ResidualTransitionModel) and
the oracle bundle (pre-encoded H_t/H_tp + Z_star) from runs/stage1c/.

Trains only the Prior. Evaluates retrieval for:
  oracle_Z   = T(H_t, Z_star)
  prior_Z    = T(H_t, Z_hat)
  zero_Z     = T(H_t, 0)
  random_Z   = T(H_t, randn * z_scale)
  shuffle_prior_Z = T(H_t, roll(Z_hat,1))
  copy_Ht baseline
"""
import argparse
import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.inverse_encoder import InverseActionEncoder
from models.transition_residual import ResidualTransitionModel
from models.prior import PooledMLPPrior, CrossAttnPrior
from diag_stage0 import effective_rank


def cosine_distance(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return (2.0 - 2.0 * (pred * target).sum(-1)).mean()


def retrieval(q, k, N, name=None, print_=True):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    rank = (r == gt).float().argmax(-1).float().mean().item() + 1
    if print_ and name:
        print(f"  retrieval[{name:18s}] top1={top1:.3f}  top5={top5:.3f}  mean_rank={rank:.2f}")
    return top1, top5, rank


def vicreg_var(z):
    return F.relu(1 - z.flatten(0, 1).std(0)).mean()


def vicreg_cov(z):
    z_flat = z.flatten(0, 1)
    z_c = z_flat - z_flat.mean(0, keepdim=True)
    cov = (z_c.T @ z_c) / max(1, z_c.shape[0] - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return (off ** 2).sum() / cov.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/latent_action_jepa/runs/stage1c/ckpt.pt")
    ap.add_argument("--bundle", default="/workspace/latent_action_jepa/runs/stage1c/oracle_bundle.pt")
    ap.add_argument("--out_dir", default="/workspace/latent_action_jepa/runs/stage2a_deterministic_prior/run0")
    ap.add_argument("--prior_arch", choices=["mlp", "xattn"], default="mlp")
    ap.add_argument("--prior_hidden", type=int, default=1024)
    ap.add_argument("--prior_depth", type=int, default=3)
    ap.add_argument("--prior_dropout", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--lambda_token_cos", type=float, default=1.0)
    ap.add_argument("--lambda_nce", type=float, default=1.0)
    ap.add_argument("--nce_tau", type=float, default=0.1)
    ap.add_argument("--lambda_z_distill", type=float, default=0.5)
    ap.add_argument("--lambda_z_var", type=float, default=0.05)
    ap.add_argument("--lambda_z_cov", type=float, default=0.01)
    ap.add_argument("--z_distill_loss", choices=["smoothl1", "mse", "cos"], default="smoothl1")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(0)

    # ---- Load frozen Stage 1C ----
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
    print(f"loaded Stage 1C ckpt: action_tokens={cfg['action_tokens']}, action_dim={cfg['action_dim']}, alpha={tr.alpha.item():.4f}")

    # ---- Load oracle bundle ----
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t_train = bundle["H_t_train"].to(device); H_tp_train = bundle["H_tp_train"].to(device)
    Z_star_train = bundle["Z_star_train"].to(device)
    H_t_val = bundle["H_t_val"].to(device); H_tp_val = bundle["H_tp_val"].to(device)
    Z_star_val = bundle["Z_star_val"].to(device)
    N_tr = H_t_train.shape[0]; N_va = H_t_val.shape[0]
    print(f"train H_t {tuple(H_t_train.shape)} H_tp {tuple(H_tp_train.shape)} Z* {tuple(Z_star_train.shape)}")
    print(f"val   H_t {tuple(H_t_val.shape)} H_tp {tuple(H_tp_val.shape)} Z* {tuple(Z_star_val.shape)}")

    # ---- Build prior ----
    if args.prior_arch == "mlp":
        prior = PooledMLPPrior(cfg["state_dim"], cfg["action_tokens"], cfg["action_dim"],
                               hidden=args.prior_hidden, depth=args.prior_depth,
                               drop=args.prior_dropout).to(device)
    else:
        prior = CrossAttnPrior(cfg["state_dim"], cfg["action_tokens"], cfg["action_dim"],
                               depth=args.prior_depth, heads=8, drop=args.prior_dropout).to(device)
    print(f"prior arch={args.prior_arch}  params={sum(p.numel() for p in prior.parameters())/1e6:.2f} M")

    opt = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # ---- Eval function ----
    def eval_split(H_t, H_tp, Z_star, tag, file=None):
        N = H_t.shape[0]
        prior.eval()
        with torch.no_grad():
            Z_hat = prior(H_t)

            # diagnostic stats
            std_zh = Z_hat.flatten(1).std(0).mean().item()
            std_zs = Z_star.flatten(1).std(0).mean().item()
            er_zh = effective_rank(Z_hat.flatten(1))
            er_zs = effective_rank(Z_star.flatten(1))
            cos_to_star = F.cosine_similarity(Z_hat.flatten(1), Z_star.flatten(1), dim=-1).mean().item()
            mse_to_star = F.mse_loss(Z_hat, Z_star).item()

            # match random-Z scale to mean ||Z_star||
            z_scale = Z_star.flatten(1).norm(dim=-1).mean().item() / (cfg["action_tokens"] * cfg["action_dim"]) ** 0.5
            rand = torch.randn_like(Z_star) * z_scale

            preds = {
                "oracle_Z":   tr(H_t, Z_star),
                "prior_Z":    tr(H_t, Z_hat),
                "zero_Z":     tr(H_t, torch.zeros_like(Z_star)),
                "random_Z":   tr(H_t, rand),
                "shuffle_prior_Z": tr(H_t, torch.roll(Z_hat, 1, 0)),
            }

            print(f"\n[{tag}]  N={N}  std(Z_hat)={std_zh:.4f}  std(Z*)={std_zs:.4f}  eff_rank(Z_hat)={er_zh:.1f}  eff_rank(Z*)={er_zs:.1f}  cos(Z_hat,Z*)={cos_to_star:+.3f}  mse(Z_hat,Z*)={mse_to_star:.4f}")
            results = {}
            for name, pred in preds.items():
                t1, t5, mr = retrieval(pred.mean(1), H_tp.mean(1), N, name)
                results[name] = {"top1": t1, "top5": t5, "mean_rank": mr}
            # copy baseline
            t1, t5, mr = retrieval(H_t.mean(1), H_tp.mean(1), N, "copy_Ht")
            results["copy_Ht"] = {"top1": t1, "top5": t5, "mean_rank": mr}

            wrong_t1 = max(results["zero_Z"]["top1"], results["random_Z"]["top1"], results["copy_Ht"]["top1"])
            wrong_mr = min(results["zero_Z"]["mean_rank"], results["random_Z"]["mean_rank"], results["copy_Ht"]["mean_rank"])
            gap_t1 = results["prior_Z"]["top1"] - wrong_t1
            gap_mr = wrong_mr - results["prior_Z"]["mean_rank"]
            gap_to_oracle = results["oracle_Z"]["top1"] - results["prior_Z"]["top1"]
            print(f"  >>> Z_gap_top1 (prior - max(zero,random,copy)) = {gap_t1:+.3f}")
            print(f"  >>> Z_gap_rank (min(zero,random,copy) - prior) = {gap_mr:+.2f}")
            print(f"  >>> gap_to_oracle (oracle - prior)              = {gap_to_oracle:+.3f}")
            return {
                "split": tag, "N": N,
                "std_Z_hat": std_zh, "std_Z_star": std_zs,
                "eff_rank_Z_hat": er_zh, "eff_rank_Z_star": er_zs,
                "cos_Z_hat_to_star": cos_to_star, "mse_Z_hat_to_star": mse_to_star,
                "retrieval": results,
                "Z_gap_top1_vs_baselines": gap_t1,
                "Z_gap_rank_vs_baselines": gap_mr,
                "gap_to_oracle_top1": gap_to_oracle,
            }
        prior.train()

    print("\n=== Pre-train eval ===")
    pre_train = eval_split(H_t_train, H_tp_train, Z_star_train, "train pre")
    pre_val = eval_split(H_t_val, H_tp_val, Z_star_val, "val pre")

    # ---- Training ----
    print("\n=== Training prior ===")
    prior.train()
    log_lines = []
    for step in range(args.steps):
        idx = torch.randint(0, N_tr, (args.bs,), device=device)
        h_t = H_t_train[idx]; h_tp = H_tp_train[idx]; z_star = Z_star_train[idx]
        z_hat = prior(h_t)
        h_pred = tr(h_t, z_hat)

        # transition retrieval losses
        L_token = cosine_distance(h_pred, h_tp)
        q = F.normalize(h_pred.mean(1), dim=-1)
        k = F.normalize(h_tp.mean(1), dim=-1)
        logits = q @ k.T / args.nce_tau
        labels = torch.arange(q.shape[0], device=device)
        L_nce = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(k @ q.T / args.nce_tau, labels))

        # distillation against oracle Z*
        if args.z_distill_loss == "smoothl1":
            L_dist = F.smooth_l1_loss(z_hat, z_star)
        elif args.z_distill_loss == "mse":
            L_dist = F.mse_loss(z_hat, z_star)
        else:
            L_dist = (1 - F.cosine_similarity(z_hat, z_star, dim=-1)).mean()

        # VICReg on Z_hat
        L_var = vicreg_var(z_hat)
        L_cov = vicreg_cov(z_hat)

        loss = (args.lambda_token_cos * L_token
                + args.lambda_nce * L_nce
                + args.lambda_z_distill * L_dist
                + args.lambda_z_var * L_var
                + args.lambda_z_cov * L_cov)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 200 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  L_tok={L_token.item():.4f}  L_nce={L_nce.item():.4f}  L_dist={L_dist.item():.4f}  L_var={L_var.item():.4f}  L_cov={L_cov.item():.4f}  lr={opt.param_groups[0]['lr']:.2e}")
            log_lines.append({"step": step, "L_tok": L_token.item(), "L_nce": L_nce.item(),
                              "L_dist": L_dist.item(), "L_var": L_var.item(), "L_cov": L_cov.item(),
                              "lr": opt.param_groups[0]['lr']})

    print("\n=== Post-train eval ===")
    post_train = eval_split(H_t_train, H_tp_train, Z_star_train, "train post")
    post_val = eval_split(H_t_val, H_tp_val, Z_star_val, "val post")

    # ---- Save ----
    torch.save({
        "prior": prior.state_dict(),
        "args": vars(args),
        "prior_arch": args.prior_arch,
    }, os.path.join(args.out_dir, "prior.pt"))
    metrics = {
        "pre_train": pre_train, "pre_val": pre_val,
        "post_train": post_train, "post_val": post_val,
        "train_log": log_lines, "args": vars(args),
        "stage1c_alpha": tr.alpha.item(),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nsaved prior + metrics to {args.out_dir}")
    print("\nSTAGE 2A DONE")


if __name__ == "__main__":
    main()
