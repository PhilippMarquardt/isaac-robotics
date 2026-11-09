"""End-to-end JEPA-style pretraining with latent-action + flow-matching prior.

Single online ViT (trainable) + EMA target ViT (no grad). Trains:

  L_jepa  : V-JEPA-2 masked-target prediction loss (encoder + predictor backprop).
  L_la    : multi-step latent action — inverse(H_t, H_targets_TARGET) -> Z*;
            transition(H_t, Z*) -> H_pred; cosine vs H_targets_TARGET.
  L_flow  : flow matching of Z conditioned on context tokens (online).

Tubelet structure: 16-frame clip @ stride 2 -> 8 tubelets * 196 patches.
  * Context tubelets: 0..3 (first half).
  * Future tubelets:  4..7 (second half).
  * H_t  = last context tubelet (3).
  * H_targets = future tubelets [B,4,196,D] (from EMA target encoder).
  * H_context for flow = all 4 context tubelets flat (online).
  * Z shape = [B,4,action_dim].

Nothing is frozen.
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
from models.predictor_vjepa2 import VJEPAPredictor
from models.la_modules import (
    InverseActionMultiStep, MultiStepTransition, ContextOnlyMultiStepFlow,
    ContextDiTFlow,
)
from masks.multiblock3d import batched_mask, DEFAULT_SPECS


def effective_rank(x):
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x.float())
    p = (s ** 2) / (s ** 2).sum().clamp_min(1e-12)
    H = -(p * (p.clamp_min(1e-12).log())).sum()
    return torch.exp(H).item()


def retrieval_per_step(pred, tgt, name=""):
    N, S, P, D = pred.shape
    rows = []
    for s in range(S):
        q = F.normalize(pred[:, s].mean(1), dim=-1)
        k = F.normalize(tgt[:, s].mean(1), dim=-1)
        sim = q @ k.T
        r = sim.argsort(dim=-1, descending=True)
        gt = torch.arange(N, device=q.device).unsqueeze(-1)
        rows.append({
            "step": s,
            "top1": (r[:, :1] == gt).any(-1).float().mean().item(),
            "mr": (r == gt).float().argmax(-1).float().mean().item() + 1,
        })
    avg = sum(r["top1"] for r in rows) / S
    print(f"  [{name:30s}]  per-step top1: " + " ".join(f"{r['top1']:.3f}" for r in rows) + f"   avg={avg:.3f}")
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
    # JEPA model
    ap.add_argument("--tubelet_size", type=int, default=2)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--embed_dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=6)
    ap.add_argument("--pred_embed_dim", type=int, default=384)
    ap.add_argument("--pred_depth", type=int, default=6)
    ap.add_argument("--pred_num_heads", type=int, default=6)
    # LA model
    ap.add_argument("--action_tokens", type=int, default=4)
    ap.add_argument("--action_dim", type=int, default=128)
    ap.add_argument("--n_context_tubelets", type=int, default=4)
    ap.add_argument("--inverse_depth", type=int, default=4)
    ap.add_argument("--transition_depth", type=int, default=4)
    ap.add_argument("--flow_depth", type=int, default=6)
    ap.add_argument("--flow_type", choices=["xattn", "dit"], default="xattn")
    ap.add_argument("--cross_target_aug", type=float, default=0.0,
                     help="probability of cross-target inverse augmentation (V1 run-B trick). 0 = off.")
    # training
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr_start", type=float, default=1.0e-4)
    ap.add_argument("--lr_peak", type=float, default=5.25e-4)
    ap.add_argument("--lr_end", type=float, default=1.0e-6)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--ema_start", type=float, default=0.99925)
    ap.add_argument("--ema_end", type=float, default=0.99925)
    ap.add_argument("--loss_exp", type=float, default=1.0)
    ap.add_argument("--lambda_jepa", type=float, default=1.0)
    ap.add_argument("--lambda_la", type=float, default=0.5)
    ap.add_argument("--lambda_flow", type=float, default=0.5)
    ap.add_argument("--lambda_z_l2", type=float, default=0.001)
    ap.add_argument("--lambda_z_var", type=float, default=0.05)
    ap.add_argument("--lambda_z_cov", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed); random.seed(args.seed)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])

    grid_d = args.num_frames // args.tubelet_size                   # 8
    grid_s = args.img_size // args.patch_size                        # 14
    n_patch = grid_s * grid_s                                        # 196
    n_tokens = grid_d * n_patch                                      # 1568
    assert args.n_context_tubelets * 2 == grid_d, "default expects 4 ctx + 4 future tubelets"
    S = grid_d - args.n_context_tubelets                              # 4 future tubelets = 4 action steps
    n_ctx_tokens = args.n_context_tubelets * n_patch
    print(f"grid: T_tube={grid_d}  P={n_patch}  n_tokens={n_tokens}  "
          f"ctx_tubs={args.n_context_tubelets}  fut_tubs={S}  action_tokens=S={S}")

    # build train/val anchors (clips)
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

    # online + EMA target encoders
    online = VisionTransformer3D(
        num_frames=args.num_frames, tubelet_size=args.tubelet_size, patch_size=args.patch_size,
        img_size=args.img_size, embed_dim=args.embed_dim, depth=args.depth,
        num_heads=args.num_heads, mlp_ratio=4.0,
    ).to(device).train()
    target = copy.deepcopy(online).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    predictor = VJEPAPredictor(
        embed_dim=args.embed_dim, pred_embed_dim=args.pred_embed_dim,
        pred_depth=args.pred_depth, pred_num_heads=args.pred_num_heads,
        grid_depth=grid_d, grid_size=grid_s, mlp_ratio=4.0, num_mask_tokens=2,
    ).to(device).train()
    inv = InverseActionMultiStep(state_dim=args.embed_dim, action_dim=args.action_dim,
                                  n_steps=S, n_patch=n_patch,
                                  depth=args.inverse_depth).to(device).train()
    tr = MultiStepTransition(state_dim=args.embed_dim, action_dim=args.action_dim,
                              n_steps=S, n_patch=n_patch,
                              depth=args.transition_depth).to(device).train()
    if args.flow_type == "dit":
        flow = ContextDiTFlow(state_dim=args.embed_dim, action_dim=args.action_dim,
                              n_steps=S, depth=args.flow_depth).to(device).train()
        print(f"flow: ContextDiTFlow (depth={args.flow_depth})")
    else:
        flow = ContextOnlyMultiStepFlow(state_dim=args.embed_dim, action_dim=args.action_dim,
                                         n_steps=S, depth=args.flow_depth).to(device).train()
        print(f"flow: ContextOnlyMultiStepFlow (depth={args.flow_depth})")
    print(f"params  enc={sum(p.numel() for p in online.parameters())/1e6:.1f}M  "
          f"pred={sum(p.numel() for p in predictor.parameters())/1e6:.1f}M  "
          f"inv={sum(p.numel() for p in inv.parameters())/1e6:.1f}M  "
          f"tr={sum(p.numel() for p in tr.parameters())/1e6:.1f}M  "
          f"flow={sum(p.numel() for p in flow.parameters())/1e6:.1f}M")

    params = (list(online.parameters()) + list(predictor.parameters()) + list(inv.parameters())
              + list(tr.parameters()) + list(flow.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr_peak, weight_decay=args.wd, betas=(0.9, 0.999))

    def set_lr(lr):
        for g in opt.param_groups:
            g["lr"] = lr

    # ---- Training ----
    for step in range(args.steps):
        idx = [random.randrange(N_tr) for _ in range(args.bs)]
        specs = [train_specs[i] for i in idx]
        clip, _ = sample_clip_batch(cache, specs, args.num_frames, args.frame_stride, device)
        B = clip.shape[0]

        # online encoder on FULL clip (gradients flow)
        H_online = online(clip)                                         # [B, n_tokens, D]
        # target encoder on FULL clip (no grad)
        with torch.no_grad():
            H_target = target(clip)                                     # [B, n_tokens, D]

        # ---- JEPA branch: masked prediction using online VISIBLE + target MASKED ----
        spec_id = step % 2
        spec_list = [DEFAULT_SPECS[spec_id]]
        vis_idx, pred_idx = batched_mask(B, grid_d, grid_s, grid_s, spec_list, device=device)
        D = H_online.shape[-1]
        H_online_vis = H_online.gather(1, vis_idx.unsqueeze(-1).expand(-1, -1, D))
        H_target_at_pred = H_target.gather(1, pred_idx.unsqueeze(-1).expand(-1, -1, D))
        pred_at_pred = predictor(H_online_vis, vis_idx, pred_idx, mask_token_id=spec_id)
        L_jepa = torch.mean(torch.abs(pred_at_pred - H_target_at_pred) ** args.loss_exp) / args.loss_exp

        # ---- LA branch: H_t from online, H_targets_future from target ----
        # Last context tubelet of online features: tokens [n_ctx_tokens-n_patch : n_ctx_tokens]
        H_t_online = H_online[:, n_ctx_tokens - n_patch : n_ctx_tokens, :]   # [B, P, D]
        H_targets_future_target = H_target[:, n_ctx_tokens:, :].reshape(B, S, n_patch, D)
        H_context_online = H_online[:, :n_ctx_tokens, :]                   # [B, n_ctx_tokens, D]

        # ---- own-pair LA ----
        Z = inv(H_t_online, H_targets_future_target.detach())
        H_pred = tr(H_t_online, Z)
        p_la = F.normalize(H_pred, dim=-1)
        t_la = F.normalize(H_targets_future_target.detach(), dim=-1)
        L_la_own = (2.0 - 2.0 * (p_la * t_la).sum(-1)).mean()
        L_la_cross = torch.zeros((), device=device)
        if args.cross_target_aug > 0 and torch.rand(1).item() < args.cross_target_aug:
            perm = torch.randperm(B, device=device)
            while bool((perm == torch.arange(B, device=device)).all().item()):
                perm = torch.randperm(B, device=device)
            H_targets_donor = H_targets_future_target[perm].detach()
            Z_donor = inv(H_t_online, H_targets_donor)
            H_pred_donor = tr(H_t_online, Z_donor)
            p_c = F.normalize(H_pred_donor, dim=-1)
            t_c = F.normalize(H_targets_donor, dim=-1)
            L_la_cross = (2.0 - 2.0 * (p_c * t_c).sum(-1)).mean()
        L_la_teacher = L_la_own + 0.5 * L_la_cross

        # ---- Flow branch: context -> Z prior ----
        eps = torch.randn_like(Z)
        tau = torch.rand(Z.shape[0], 1, 1, device=device)
        z_tau = (1 - tau) * eps + tau * Z.detach()
        v_target = Z.detach() - eps
        v_pred = flow(z_tau, tau, H_context_online)
        L_flow = F.mse_loss(v_pred, v_target)

        # Z VICReg regs
        z_flat = Z.flatten(0, 1)
        L_l2 = (Z ** 2).mean()
        std = z_flat.std(0)
        L_var = F.relu(1 - std).mean()
        z_c = z_flat - z_flat.mean(0, keepdim=True)
        cov = (z_c.T @ z_c) / max(1, z_c.shape[0] - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        L_cov = (off ** 2).sum() / cov.shape[0]

        loss = (args.lambda_jepa * L_jepa
                + args.lambda_la * L_la_teacher
                + args.lambda_flow * L_flow
                + args.lambda_z_l2 * L_l2
                + args.lambda_z_var * L_var
                + args.lambda_z_cov * L_cov)

        lr = warmup_cosine(step, args.warmup, args.steps, args.lr_start, args.lr_peak, args.lr_end)
        set_lr(lr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        m = args.ema_start + (args.ema_end - args.ema_start) * step / max(1, args.steps - 1)
        ema_update(online, target, m)

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                std_t = H_targets_future_target.std().item()
                std_z = Z.std().item()
            print(f"  step {step:5d}  L_jepa={L_jepa.item():.4f}  L_la={L_la_teacher.item():.4f}  "
                  f"L_flow={L_flow.item():.4f}  L_var={L_var.item():.4f}  "
                  f"alpha={tr.alpha.mean().item():.3f}  std_z={std_z:.3f}  std_tgt={std_t:.3f}  "
                  f"lr={lr:.2e}  ema={m:.5f}")

    # ---- save ckpt BEFORE eval so we don't lose weights if eval OOMs ----
    torch.save({
        "online": online.state_dict(),
        "target": target.state_dict(),
        "predictor": predictor.state_dict(),
        "inv": inv.state_dict(),
        "tr": tr.state_dict(),
        "flow": flow.state_dict(),
        "args": vars(args),
    }, os.path.join(args.out_dir, "ckpt.pt"))
    print(f"\nsaved ckpt -> {args.out_dir}/ckpt.pt (training done; running eval)")

    # ---- Eval (chunked, capped) ----
    online.eval(); target.eval(); predictor.eval(); inv.eval(); tr.eval(); flow.eval()
    N_va_eval = min(N_va, 800)
    print(f"\n=== Eval val (N={N_va_eval} of {N_va}) ===")
    all_H_t, all_H_targets, all_H_context, all_Z = [], [], [], []
    B_eval = 2
    val_specs_eval = val_specs[:N_va_eval]
    with torch.no_grad():
        for s in range(0, N_va_eval, B_eval):
            specs = val_specs_eval[s:s + B_eval]
            clip, _ = sample_clip_batch(cache, specs, args.num_frames, args.frame_stride, device)
            H_online = online(clip)
            H_target = target(clip)
            B_v = H_online.shape[0]
            D = H_online.shape[-1]
            H_t = H_online[:, n_ctx_tokens - n_patch : n_ctx_tokens, :].cpu()
            H_targets = H_target[:, n_ctx_tokens:, :].reshape(B_v, S, n_patch, D).cpu()
            H_context = H_online[:, :n_ctx_tokens, :].cpu()
            Z = inv(H_t.to(device), H_targets.to(device)).cpu()
            all_H_t.append(H_t); all_H_targets.append(H_targets)
            all_H_context.append(H_context); all_Z.append(Z)
            del H_online, H_target, clip
        torch.cuda.empty_cache()
        H_t = torch.cat(all_H_t, 0).to(device)
        H_targets = torch.cat(all_H_targets, 0).to(device)
        H_context = torch.cat(all_H_context, 0).to(device)
        Z = torch.cat(all_Z, 0).to(device)
        N = H_t.shape[0]

        results = {}
        results["oracle"] = retrieval_per_step(tr(H_t, Z), H_targets, "oracle z*")
        results["zero_Z"] = retrieval_per_step(tr(H_t, torch.zeros_like(Z)), H_targets, "zero_z")
        std_z = Z.flatten(2).std(0).mean()
        results["random_Z"] = retrieval_per_step(tr(H_t, torch.randn_like(Z) * std_z), H_targets, "random_z")
        perm = torch.randperm(N, device=device)
        results["shuffle_Z_batch"] = retrieval_per_step(tr(H_t, Z[perm]), H_targets, "shuffle_z(batch)")
        time_perm = torch.randperm(Z.shape[1], device=device)
        results["time_shuffle_Z"] = retrieval_per_step(tr(H_t, Z[:, time_perm]), H_targets, "time_shuffle_z")
        results["Z_only"] = retrieval_per_step(tr(torch.zeros_like(H_t), Z), H_targets, "Z_only(H=0)")
        results["wrong_H"] = retrieval_per_step(tr(H_t[perm], Z), H_targets, "wrong_H")
        base_only = H_t.unsqueeze(1).expand(-1, S, -1, -1)
        results["copy_Ht"] = retrieval_per_step(base_only, H_targets, "copy_Ht(broadcast)")

        # flow eval (context -> Z prior)
        def euler_sample(steps_=30, seed=11):
            b = H_context.shape[0]
            g = torch.Generator(device=device).manual_seed(seed)
            z = torch.randn(b, S, args.action_dim, device=device, generator=g)
            for i in range(steps_):
                tau = torch.full((b, 1, 1), i / steps_, device=device)
                v = flow(z, tau, H_context)
                z = z + v / steps_
            return z

        Z_flow = euler_sample()
        results["context_flow"] = retrieval_per_step(tr(H_t, Z_flow), H_targets, "context_flow")

        # JEPA pred->target masked-position cosine (sanity)
        # use ONE mask sample over val
        vis_idx, pred_idx = batched_mask(N, grid_d, grid_s, grid_s, [DEFAULT_SPECS[0]], device=device)
        # need to re-run online on each clip; we don't have raw clips here, so skip in eval

        # collapse stats on H_target features
        std_tgt = H_targets.std(0).mean().item()
        er_tgt = effective_rank(H_targets.mean(2).reshape(N * S, -1))
        results["std_tgt"] = std_tgt
        results["eff_rank_tgt"] = er_tgt
        print(f"  std(tgt)={std_tgt:.4f}  eff_rank(tgt features)={er_tgt:.1f}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"post": results, "args": vars(args)}, f, indent=2)
    print(f"saved metrics -> {args.out_dir}")


if __name__ == "__main__":
    main()
