"""End-to-end V-JEPA-style pretraining with flow-matching predictor.

Strict V-JEPA-2 protocol (no shortcuts):
  - Online encoder forward on VISIBLE TOKENS ONLY (context block).
  - EMA target encoder forward on the full clip, no grad.
  - Predictor is a flow-matching denoiser over target latents.
  - Loss = MSE(v_pred, v_target) where v_target = h_target.detach() - epsilon.
  - EMA update of target encoder each step.

Future-block masking: context = first M tubelets, target = last N tubelets.
"Predict next N target latents from past context" — the design target.

No latent action Z. No inverse. No transition. Flow IS the predictor.
"""
import argparse
import copy
import json
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.vjepa2_clips import sample_clip_batch
from models.vit3d import VisionTransformer3D
from models.flow_predictor import FlowVJEPAPredictor


def effective_rank(x):
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x.float())
    p = (s ** 2) / (s ** 2).sum().clamp_min(1e-12)
    H = -(p * (p.clamp_min(1e-12).log())).sum()
    return torch.exp(H).item()


def per_step_retrieval(pred_pool, tgt_pool, n_pred_per_step, name=""):
    """pred_pool, tgt_pool: [N, N_pred, D]. Group N_pred by step (=n_pred_per_step
    per step). Mean-pool patches per step then retrieve."""
    N, N_pred, D = pred_pool.shape
    S = N_pred // n_pred_per_step
    rows = []
    for s in range(S):
        a = s * n_pred_per_step; b = (s + 1) * n_pred_per_step
        q = F.normalize(pred_pool[:, a:b].mean(1), dim=-1)
        k = F.normalize(tgt_pool[:, a:b].mean(1), dim=-1)
        sim = q @ k.T
        r = sim.argsort(dim=-1, descending=True)
        gt = torch.arange(N, device=q.device).unsqueeze(-1)
        rows.append({
            "step": s,
            "top1": (r[:, :1] == gt).any(-1).float().mean().item(),
            "mr": (r == gt).float().argmax(-1).float().mean().item() + 1,
        })
    avg = sum(r["top1"] for r in rows) / S
    print(f"  [{name:30s}]  per-step top1: " +
          " ".join(f"{r['top1']:.3f}" for r in rows) + f"   avg={avg:.3f}")
    return {"per_step": rows, "avg_top1": avg}


@torch.no_grad()
def ema_update(online, target, m):
    for po, pt in zip(online.parameters(), target.parameters()):
        pt.data.mul_(m).add_(po.data, alpha=1 - m)
    for bo, bt in zip(online.buffers(), target.buffers()):
        bt.data.copy_(bo.data)


def warmup_cosine(step, warmup, total, lr_start, lr_peak, lr_end=0.0):
    if step < warmup:
        return lr_start + (lr_peak - lr_start) * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return lr_end + 0.5 * (lr_peak - lr_end) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out_dir", required=True)
    # clip
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--frame_stride", type=int, default=2)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--clip_anchor_stride", type=int, default=4)
    # encoder
    ap.add_argument("--tubelet_size", type=int, default=2)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--embed_dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=6)
    # predictor
    ap.add_argument("--pred_embed_dim", type=int, default=384)
    ap.add_argument("--pred_depth", type=int, default=8)
    ap.add_argument("--pred_num_heads", type=int, default=6)
    # split context vs target
    ap.add_argument("--n_context_tubelets", type=int, default=4,
                     help="first N tubelets are context (visible to online encoder)")
    # training
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr_start", type=float, default=1.0e-4)
    ap.add_argument("--lr_peak", type=float, default=5.25e-4)
    ap.add_argument("--lr_end", type=float, default=1.0e-6)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--ema_start", type=float, default=0.99925)
    ap.add_argument("--ema_end", type=float, default=0.99925)
    # flow
    ap.add_argument("--sample_steps", type=int, default=30)
    ap.add_argument("--n_val_eval", type=int, default=800)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed); random.seed(args.seed)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])

    grid_d = args.num_frames // args.tubelet_size
    grid_s = args.img_size // args.patch_size
    n_patch = grid_s * grid_s
    n_tokens = grid_d * n_patch
    n_ctx_tubs = args.n_context_tubelets
    n_pred_tubs = grid_d - n_ctx_tubs
    assert n_pred_tubs >= 1
    n_vis = n_ctx_tubs * n_patch
    n_pred = n_pred_tubs * n_patch
    print(f"grid: T_tube={grid_d}  P={n_patch}  n_tokens={n_tokens}  "
          f"ctx_tubs={n_ctx_tubs}  pred_tubs={n_pred_tubs}  "
          f"N_vis={n_vis}  N_pred={n_pred}")

    # static vis/pred indices (deterministic future-block mask)
    flat_grid = torch.arange(n_tokens, device=device)
    vis_idx_base = flat_grid[:n_vis]                                    # [N_vis]
    pred_idx_base = flat_grid[n_vis:]                                   # [N_pred]

    # train/val anchors
    span = (args.num_frames - 1) * args.frame_stride + 1
    train_specs, val_specs = [], []
    for fi in range(n_films):
        n_frames = cache["frames"][fi].shape[0]
        last_anchor = n_frames - span - 1
        train_end = int(last_anchor * 0.9)
        val_start = train_end + args.num_frames * args.frame_stride
        for a in range(0, train_end, args.clip_anchor_stride):
            train_specs.append((fi, a))
        for a in range(val_start, last_anchor, args.clip_anchor_stride):
            val_specs.append((fi, a))
    N_tr, N_va = len(train_specs), len(val_specs)
    print(f"train clips={N_tr}  val clips={N_va}")

    # encoders
    online = VisionTransformer3D(
        num_frames=args.num_frames, tubelet_size=args.tubelet_size, patch_size=args.patch_size,
        img_size=args.img_size, embed_dim=args.embed_dim, depth=args.depth,
        num_heads=args.num_heads, mlp_ratio=4.0,
    ).to(device).train()
    target = copy.deepcopy(online).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    predictor = FlowVJEPAPredictor(
        embed_dim=args.embed_dim, pred_embed_dim=args.pred_embed_dim,
        depth=args.pred_depth, heads=args.pred_num_heads,
        grid_depth=grid_d, grid_size=grid_s, mlp_ratio=4.0,
    ).to(device).train()
    print(f"params  enc={sum(p.numel() for p in online.parameters())/1e6:.1f}M  "
          f"pred={sum(p.numel() for p in predictor.parameters())/1e6:.1f}M")

    params = list(online.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr_peak, weight_decay=args.wd, betas=(0.9, 0.999))

    def set_lr(lr):
        for g in opt.param_groups:
            g["lr"] = lr

    # ---- Training ----
    print(f"\n=== Train  steps={args.steps}  bs={args.bs} ===")
    for step in range(args.steps):
        idx = [random.randrange(N_tr) for _ in range(args.bs)]
        specs = [train_specs[i] for i in idx]
        clip, _ = sample_clip_batch(cache, specs, args.num_frames, args.frame_stride, device)
        B = clip.shape[0]
        vis_idx = vis_idx_base.unsqueeze(0).expand(B, -1)
        pred_idx = pred_idx_base.unsqueeze(0).expand(B, -1)
        # ONLINE encoder forward on VISIBLE TOKENS ONLY
        H_vis = online(clip, keep_indices=vis_idx)                       # [B, N_vis, D]
        # EMA TARGET encoder forward on full clip, no grad
        with torch.no_grad():
            H_full = target(clip)                                         # [B, n_tokens, D]
            D = H_full.shape[-1]
            H_target = H_full.gather(1, pred_idx.unsqueeze(-1).expand(-1, -1, D))   # [B, N_pred, D]
        # flow matching
        eps = torch.randn_like(H_target)
        tau = torch.rand(B, device=device)
        tau_b = tau.view(B, 1, 1)
        z_tau = (1 - tau_b) * eps + tau_b * H_target.detach()
        v_target = H_target.detach() - eps
        v_pred = predictor(H_vis, vis_idx, pred_idx, z_tau, tau)
        L = F.mse_loss(v_pred, v_target)

        lr = warmup_cosine(step, args.warmup, args.steps, args.lr_start, args.lr_peak, args.lr_end)
        set_lr(lr)
        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        m = args.ema_start + (args.ema_end - args.ema_start) * step / max(1, args.steps - 1)
        ema_update(online, target, m)

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                std_t = H_target.std().item()
                std_v = v_pred.std().item()
            print(f"  step {step:5d}  L_flow={L.item():.4f}  "
                  f"std(tgt)={std_t:.3f}  std(v_pred)={std_v:.3f}  "
                  f"lr={lr:.2e}  ema={m:.5f}")

    # save ckpt BEFORE eval
    torch.save({
        "online": online.state_dict(),
        "target": target.state_dict(),
        "predictor": predictor.state_dict(),
        "args": vars(args),
    }, os.path.join(args.out_dir, "ckpt.pt"))
    print(f"\nsaved ckpt -> {args.out_dir}/ckpt.pt")

    # ---- Eval ----
    online.eval(); target.eval(); predictor.eval()
    N_eval = min(N_va, args.n_val_eval)
    print(f"\n=== Eval val (N={N_eval} of {N_va}) ===")
    all_H_vis, all_H_target = [], []
    B_eval = 2
    with torch.no_grad():
        for s in range(0, N_eval, B_eval):
            specs = val_specs[s:s + B_eval]
            clip, _ = sample_clip_batch(cache, specs, args.num_frames, args.frame_stride, device)
            B_v = clip.shape[0]
            vis_idx_v = vis_idx_base.unsqueeze(0).expand(B_v, -1)
            pred_idx_v = pred_idx_base.unsqueeze(0).expand(B_v, -1)
            H_vis = online(clip, keep_indices=vis_idx_v).cpu()
            H_full = target(clip)
            D = H_full.shape[-1]
            H_target = H_full.gather(1, pred_idx_v.unsqueeze(-1).expand(-1, -1, D)).cpu()
            all_H_vis.append(H_vis); all_H_target.append(H_target)
            del H_full, clip
        torch.cuda.empty_cache()
        H_vis = torch.cat(all_H_vis, 0)
        H_target = torch.cat(all_H_target, 0)
        N = H_vis.shape[0]
        D = H_target.shape[-1]
        vis_idx = vis_idx_base.unsqueeze(0).expand(N, -1)
        pred_idx = pred_idx_base.unsqueeze(0).expand(N, -1)

        # 1) deterministic copy baseline: visible context mean as predicted target
        H_vis_mean = H_vis.reshape(N, n_ctx_tubs, n_patch, D).mean(1)        # avg over context tubelets -> [N, P, D]
        # tile per pred tubelet
        copy_pred = H_vis_mean.unsqueeze(1).expand(-1, n_pred_tubs, -1, -1).reshape(N, n_pred, D)
        copy_pred = copy_pred.to(device)

        results = {}
        # copy baseline
        results["copy_context"] = per_step_retrieval(copy_pred, H_target.to(device),
                                                     n_patch, "copy_context")

        # flow K=1 single-seed sample (chunked)
        def euler_chunk(seed):
            preds = []
            chunk = 8
            for s_ in range(0, N, chunk):
                e = min(s_ + chunk, N)
                Hv = H_vis[s_:e].to(device)
                vi = vis_idx[s_:e]
                pi = pred_idx[s_:e]
                b = e - s_
                g = torch.Generator(device=device).manual_seed(seed + s_)
                z = torch.randn(b, n_pred, D, device=device, generator=g)
                for i in range(args.sample_steps):
                    tau_v = torch.full((b,), i / args.sample_steps, device=device)
                    v = predictor(Hv, vi, pi, z, tau_v)
                    z = z + v / args.sample_steps
                preds.append(z.cpu())
                del Hv
            return torch.cat(preds, 0)

        Z_pred = euler_chunk(seed=11).to(device)
        results["flow_K1_s11"] = per_step_retrieval(Z_pred, H_target.to(device),
                                                     n_patch, "flow_K1_s11")

        # best-of-K
        Ks = [1, 4, 8, 16]
        H_tgt_dev = H_target.to(device)
        ranks_per_pos = torch.full((N, n_pred), N + 1, device=device, dtype=torch.long)
        # using top1-only per group; will compute later
        # we'll track best-rank per STEP group (each step group = n_patch positions)
        ranks_per_step = {s: torch.full((N,), N + 1, dtype=torch.long, device=device)
                           for s in range(n_pred_tubs)}
        for ki in range(max(Ks)):
            Z = euler_chunk(seed=100 + ki).to(device)
            for s_ in range(n_pred_tubs):
                a = s_ * n_patch; b = (s_ + 1) * n_patch
                q = F.normalize(Z[:, a:b].mean(1), dim=-1)
                k = F.normalize(H_tgt_dev[:, a:b].mean(1), dim=-1)
                sim = q @ k.T
                r = sim.argsort(dim=-1, descending=True)
                gt = torch.arange(N, device=device).unsqueeze(-1)
                rank = (r == gt).float().argmax(-1).long() + 1
                ranks_per_step[s_] = torch.minimum(ranks_per_step[s_], rank)
            del Z
            if (ki + 1) in Ks:
                rows = []
                for s_ in range(n_pred_tubs):
                    t1 = (ranks_per_step[s_] == 1).float().mean().item()
                    rows.append({"step": s_, "top1": t1,
                                  "mr": ranks_per_step[s_].float().mean().item()})
                avg = sum(r["top1"] for r in rows) / n_pred_tubs
                print(f"  [bestK={ki+1:3d} flow                 ]  per-step top1: " +
                      " ".join(f"{r['top1']:.3f}" for r in rows) + f"   avg={avg:.3f}")
                results[f"flow_bestK{ki+1}"] = {"per_step": rows, "avg_top1": avg}

        # collapse stats
        H_tgt_pool = H_tgt_dev.reshape(N, n_pred_tubs, n_patch, D).mean(2)  # [N, S, D]
        std_t = H_tgt_pool.std().item()
        er_t = effective_rank(H_tgt_pool.reshape(N * n_pred_tubs, -1))
        results["std_target_pooled"] = std_t
        results["eff_rank_target_pooled"] = er_t
        print(f"  std(tgt pooled)={std_t:.4f}  eff_rank={er_t:.1f}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"post": results, "args": vars(args)}, f, indent=2)
    print(f"saved metrics -> {args.out_dir}")


if __name__ == "__main__":
    main()
