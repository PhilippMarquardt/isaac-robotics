"""Flow-Action JEPA v2 — residual-input inverse, one shared encoder, two consumers.

Per the spec:
  - ONE shared online encoder produces H_context (visible context tubelets).
  - FLOW prior gets full H_context to generate compact action tokens Z.
  - TRANSITION gets only H_t (last context tubelet) plus Z_i for step i.
  - INVERSE (training-only posterior) sees ONLY the motion residual
        delta_i = stopgrad(H_target_i) - stopgrad(base_i(H_t))
    so Z* encodes motion, not appearance / film identity.

Stack:
  encoder (online)  : VisionTransformer3D, visible context tokens only.
  encoder (EMA tgt) : VisionTransformer3D, full clip, no grad.
  StepBase          : per-step MLP from H_t -> base_i.
  InverseFromDelta  : delta_1..S -> Z* [B, S, A].
  MultiStepTransitionPerStep : (H_t, Z) -> H_pred [B, S, P, D] with step routing.
  ContextDiTFlow    : (z_tau, tau, H_context.detach()) -> velocity.

Losses:
  L_pred = cosine(H_pred, H_target.detach())
  L_flow = MSE(v_pred, Z*.detach() - epsilon)
  L_z    = VICReg variance + cov + small L2 on Z*

H_context is detached for flow in v1 (safer); allow grad in v2.
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
from models.la_modules import (
    StepBase, InverseFromDelta, InverseFromDeltaPerStep,
    MultiStepTransitionPerStep, ContextDiTFlow,
)


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
        rows.append({"step": s,
                     "top1": (r[:, :1] == gt).any(-1).float().mean().item(),
                     "mr": (r == gt).float().argmax(-1).float().mean().item() + 1})
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


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = float(lam)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lam * grad_out, None


def grad_reverse(x, lam):
    return GradReverse.apply(x, lam)


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
    # action stack
    ap.add_argument("--action_steps", type=int, default=4)
    ap.add_argument("--action_dim", type=int, default=128)
    ap.add_argument("--inverse_depth", type=int, default=4)
    ap.add_argument("--per_step_inverse", action="store_true", default=False,
                     help="if set, use InverseFromDeltaPerStep (Z_i depends only on delta_i)")
    ap.add_argument("--base_depth", type=int, default=2)
    ap.add_argument("--transition_depth", type=int, default=4)
    ap.add_argument("--alpha_min", type=float, default=0.02,
                     help="floor on transition alpha gate; raise to force Z usage")
    ap.add_argument("--alpha_init", type=float, default=0.05)
    ap.add_argument("--flow_depth", type=int, default=8)
    ap.add_argument("--flow_heads", type=int, default=4)
    ap.add_argument("--n_context_tubelets", type=int, default=4)
    # training
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr_start", type=float, default=1.0e-4)
    ap.add_argument("--lr_peak", type=float, default=5.25e-4)
    ap.add_argument("--lr_end", type=float, default=1.0e-6)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--ema_start", type=float, default=0.99925)
    ap.add_argument("--ema_end", type=float, default=0.99925)
    # losses
    ap.add_argument("--lambda_pred", type=float, default=1.0)
    ap.add_argument("--lambda_flow", type=float, default=0.5)
    ap.add_argument("--lambda_z_l2", type=float, default=0.001)
    ap.add_argument("--lambda_z_var", type=float, default=0.05)
    ap.add_argument("--lambda_z_cov", type=float, default=0.01)
    ap.add_argument("--lambda_enc_var", type=float, default=0.0,
                     help="VICReg variance reg on ONLINE encoder features (anti-collapse)")
    ap.add_argument("--lambda_enc_cov", type=float, default=0.0,
                     help="VICReg covariance reg on ONLINE encoder features")
    ap.add_argument("--film_adv_lambda", type=float, default=0.0,
                     help="gradient reversal lambda on Z -> film head (anti-domain leakage)")
    # runtime
    ap.add_argument("--ctx_grad_to_flow", action="store_true", default=False,
                     help="if set, flow gradient flows into encoder via H_context (v2 mode); "
                          "default is to detach H_context for flow (safer)")
    ap.add_argument("--sample_steps", type=int, default=30)
    ap.add_argument("--n_val_eval", type=int, default=800)
    ap.add_argument("--log_every", type=int, default=200)
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
    n_fut_tubs = grid_d - n_ctx_tubs
    assert n_fut_tubs == args.action_steps, (
        f"action_steps must equal grid_d - n_context_tubelets ({n_fut_tubs}); got {args.action_steps}")
    n_ctx_tokens = n_ctx_tubs * n_patch
    S = args.action_steps
    A = args.action_dim
    print(f"grid_d={grid_d} P={n_patch} ctx_tubs={n_ctx_tubs} fut_tubs={n_fut_tubs} S={S} A={A}")
    print(f"flow ctx-grad: {args.ctx_grad_to_flow}")

    flat_grid = torch.arange(n_tokens, device=device)
    vis_idx_base = flat_grid[:n_ctx_tokens]
    fut_idx_base = flat_grid[n_ctx_tokens:]

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

    online = VisionTransformer3D(
        num_frames=args.num_frames, tubelet_size=args.tubelet_size, patch_size=args.patch_size,
        img_size=args.img_size, embed_dim=args.embed_dim, depth=args.depth,
        num_heads=args.num_heads, mlp_ratio=4.0,
    ).to(device).train()
    target_enc = copy.deepcopy(online).eval()
    for p in target_enc.parameters():
        p.requires_grad_(False)
    base = StepBase(state_dim=args.embed_dim, n_steps=S, depth=args.base_depth).to(device).train()
    inv_cls = InverseFromDeltaPerStep if args.per_step_inverse else InverseFromDelta
    inv = inv_cls(state_dim=args.embed_dim, action_dim=A, n_steps=S, n_patch=n_patch,
                  depth=args.inverse_depth).to(device).train()
    print(f"inverse: {inv_cls.__name__}")
    tr = MultiStepTransitionPerStep(state_dim=args.embed_dim, action_dim=A, n_steps=S,
                                     n_patch=n_patch, depth=args.transition_depth,
                                     alpha_min=args.alpha_min,
                                     alpha_init=args.alpha_init).to(device).train()
    print(f"transition alpha_min={args.alpha_min} alpha_init={args.alpha_init}")
    flow = ContextDiTFlow(state_dim=args.embed_dim, action_dim=A, n_steps=S,
                           depth=args.flow_depth, heads=args.flow_heads).to(device).train()
    print(f"params  enc={sum(p.numel() for p in online.parameters())/1e6:.1f}M  "
          f"base={sum(p.numel() for p in base.parameters())/1e6:.1f}M  "
          f"inv={sum(p.numel() for p in inv.parameters())/1e6:.1f}M  "
          f"tr={sum(p.numel() for p in tr.parameters())/1e6:.1f}M  "
          f"flow={sum(p.numel() for p in flow.parameters())/1e6:.1f}M")

    film_head = None
    if args.film_adv_lambda > 0:
        film_head = nn.Linear(S * A, n_films).to(device).train()
        print(f"film-adv head enabled  n_films={n_films}  lambda={args.film_adv_lambda}")
    params = (list(online.parameters()) + list(base.parameters()) + list(inv.parameters())
              + list(tr.parameters()) + list(flow.parameters()))
    if film_head is not None:
        params += list(film_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr_peak, weight_decay=args.wd, betas=(0.9, 0.999))

    def set_lr(lr):
        for g in opt.param_groups:
            g["lr"] = lr

    print(f"\n=== Train  steps={args.steps}  bs={args.bs} ===")
    for step in range(args.steps):
        idx = [random.randrange(N_tr) for _ in range(args.bs)]
        specs = [train_specs[i] for i in idx]
        film_ids_b = torch.tensor([fi for fi, _ in specs], dtype=torch.long, device=device)
        clip, _ = sample_clip_batch(cache, specs, args.num_frames, args.frame_stride, device)
        B = clip.shape[0]
        vis_idx = vis_idx_base.unsqueeze(0).expand(B, -1)
        fut_idx = fut_idx_base.unsqueeze(0).expand(B, -1)

        # ONE encoder forward: online on visible context only
        H_ctx_online = online(clip, keep_indices=vis_idx)                  # [B, N_vis, D]
        H_t_online = H_ctx_online[:, -n_patch:, :]                          # [B, P, D]

        # EMA target encoder on FULL clip, no grad
        with torch.no_grad():
            H_full_target = target_enc(clip)
            D = H_full_target.shape[-1]
            H_targets = H_full_target.gather(
                1, fut_idx.unsqueeze(-1).expand(-1, -1, D)
            ).reshape(B, S, n_patch, D)                                    # [B, S, P, D]

        # 1) StepBase: predicted future under no-action
        base_pred = base(H_t_online)                                       # [B, S, P, D]
        # 2) Motion residual at the inverse input
        deltas = (H_targets.detach() - base_pred.detach())                 # detach both: inverse uses residual as data
        Z_star = inv(deltas)                                               # [B, S, A]
        # 3) Per-step transition: predicts H_pred from H_t + Z (per-step routed)
        H_pred = tr(H_t_online, Z_star)                                    # [B, S, P, D]

        # JEPA latent prediction loss (cosine, mean over batch/step/patch)
        p_la = F.normalize(H_pred, dim=-1)
        t_la = F.normalize(H_targets.detach(), dim=-1)
        L_pred = (2.0 - 2.0 * (p_la * t_la).sum(-1)).mean()

        # Also push base to be a decent no-action approximation: cosine(base_pred, H_target)
        # This trains the base; otherwise StepBase floats arbitrarily.
        p_b = F.normalize(base_pred, dim=-1)
        L_base = (2.0 - 2.0 * (p_b * t_la).sum(-1)).mean()

        # Flow matching on Z*
        H_ctx_for_flow = H_ctx_online if args.ctx_grad_to_flow else H_ctx_online.detach()
        eps = torch.randn_like(Z_star)
        tau = torch.rand(B, 1, 1, device=device)
        z_tau = (1 - tau) * eps + tau * Z_star.detach()
        v_target = Z_star.detach() - eps
        v_pred = flow(z_tau, tau, H_ctx_for_flow)
        L_flow = F.mse_loss(v_pred, v_target)

        # Z regularizers
        z_flat = Z_star.flatten(0, 1)
        L_l2 = (Z_star ** 2).mean()
        std = z_flat.std(0)
        L_var = F.relu(1 - std).mean()
        z_c = z_flat - z_flat.mean(0, keepdim=True)
        cov = (z_c.T @ z_c) / max(1, z_c.shape[0] - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        L_cov = (off ** 2).sum() / cov.shape[0]

        # online encoder VICReg (anti-collapse on encoder features)
        L_enc_var = torch.zeros((), device=device)
        L_enc_cov = torch.zeros((), device=device)
        if args.lambda_enc_var > 0 or args.lambda_enc_cov > 0:
            feats = H_ctx_online.flatten(0, 1)                         # [B*N_vis, D]
            std_d = feats.std(0)
            L_enc_var = F.relu(1.0 - std_d).mean()
            feats_c = feats - feats.mean(0, keepdim=True)
            covm = (feats_c.T @ feats_c) / max(1, feats_c.shape[0] - 1)
            offm = covm - torch.diag(torch.diagonal(covm))
            L_enc_cov = (offm ** 2).sum() / covm.shape[0]

        # film-adv on Z (GRL reverses gradient toward Z)
        L_film = torch.zeros((), device=device)
        if film_head is not None:
            Z_rev = grad_reverse(Z_star.flatten(1), args.film_adv_lambda)
            film_logits = film_head(Z_rev)
            L_film = F.cross_entropy(film_logits, film_ids_b)

        loss = (args.lambda_pred * L_pred
                + 0.5 * L_base
                + args.lambda_flow * L_flow
                + args.lambda_z_l2 * L_l2
                + args.lambda_z_var * L_var
                + args.lambda_z_cov * L_cov
                + args.lambda_enc_var * L_enc_var
                + args.lambda_enc_cov * L_enc_cov
                + L_film)

        lr = warmup_cosine(step, args.warmup, args.steps, args.lr_start, args.lr_peak, args.lr_end)
        set_lr(lr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        m = args.ema_start + (args.ema_end - args.ema_start) * step / max(1, args.steps - 1)
        ema_update(online, target_enc, m)

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                std_z = Z_star.std().item()
                std_t = H_targets.std().item()
                alpha = tr.alpha.mean().item()
            print(f"  step {step:5d}  L_pred={L_pred.item():.4f}  L_base={L_base.item():.4f}  "
                  f"L_flow={L_flow.item():.4f}  L_var={L_var.item():.4f}  "
                  f"L_enc_var={L_enc_var.item():.4f}  alpha={alpha:.3f}  "
                  f"std_z={std_z:.3f}  std_tgt={std_t:.3f}  lr={lr:.2e}  ema={m:.5f}")

    torch.save({
        "online": online.state_dict(),
        "target": target_enc.state_dict(),
        "base": base.state_dict(),
        "inv": inv.state_dict(),
        "tr": tr.state_dict(),
        "flow": flow.state_dict(),
        "args": vars(args),
    }, os.path.join(args.out_dir, "ckpt.pt"))
    print(f"\nsaved ckpt -> {args.out_dir}/ckpt.pt")

    # ---- Eval ----
    online.eval(); target_enc.eval(); base.eval(); inv.eval(); tr.eval(); flow.eval()
    N_eval = min(N_va, args.n_val_eval)
    print(f"\n=== Eval val (N={N_eval} of {N_va}) ===")
    all_H_t, all_H_targets, all_H_ctx, all_Z = [], [], [], []
    B_eval = 2
    with torch.no_grad():
        for s in range(0, N_eval, B_eval):
            specs = val_specs[s:s + B_eval]
            clip, _ = sample_clip_batch(cache, specs, args.num_frames, args.frame_stride, device)
            B_v = clip.shape[0]
            vis_idx_v = vis_idx_base.unsqueeze(0).expand(B_v, -1)
            fut_idx_v = fut_idx_base.unsqueeze(0).expand(B_v, -1)
            H_ctx = online(clip, keep_indices=vis_idx_v)
            H_t = H_ctx[:, -n_patch:, :]
            H_full = target_enc(clip)
            D = H_full.shape[-1]
            H_targets = H_full.gather(1, fut_idx_v.unsqueeze(-1).expand(-1, -1, D)).reshape(B_v, S, n_patch, D)
            base_pred = base(H_t)
            deltas = H_targets - base_pred
            Z = inv(deltas)
            all_H_t.append(H_t.cpu()); all_H_targets.append(H_targets.cpu())
            all_H_ctx.append(H_ctx.cpu()); all_Z.append(Z.cpu())
            del H_full, clip
        torch.cuda.empty_cache()
        H_t = torch.cat(all_H_t, 0).to(device)
        H_targets = torch.cat(all_H_targets, 0).to(device)
        H_ctx = torch.cat(all_H_ctx, 0).to(device)
        Z = torch.cat(all_Z, 0).to(device)
        N = H_t.shape[0]

        results = {}
        results["oracle"] = retrieval_per_step(tr(H_t, Z), H_targets, "oracle z*")
        results["base_only"] = retrieval_per_step(base(H_t), H_targets, "base_only")
        results["zero_Z"] = retrieval_per_step(tr(H_t, torch.zeros_like(Z)), H_targets, "zero_z")
        std_z = Z.flatten(0, 1).std(0)
        results["random_Z"] = retrieval_per_step(tr(H_t, torch.randn_like(Z) * std_z), H_targets, "random_z")
        perm = torch.randperm(N, device=device)
        results["batch_shuffle_Z"] = retrieval_per_step(tr(H_t, Z[perm]), H_targets, "batch_shuffle_z")
        tperm = torch.randperm(S, device=device)
        results["time_shuffle_Z"] = retrieval_per_step(tr(H_t, Z[:, tperm]), H_targets, "time_shuffle_z")
        results["Z_only"] = retrieval_per_step(tr(torch.zeros_like(H_t), Z), H_targets, "Z_only(H=0)")
        results["wrong_H"] = retrieval_per_step(tr(H_t[perm], Z), H_targets, "wrong_H")
        base_only_pred = H_t.unsqueeze(1).expand(-1, S, -1, -1)
        results["copy_Ht"] = retrieval_per_step(base_only_pred, H_targets, "copy_Ht")

        # flow path
        def euler_chunk(seed):
            preds = []
            chunk = 8
            for s_ in range(0, N, chunk):
                e = min(s_ + chunk, N)
                Hc = H_ctx[s_:e]
                b = e - s_
                g = torch.Generator(device=device).manual_seed(seed + s_)
                z = torch.randn(b, S, A, device=device, generator=g)
                for i in range(args.sample_steps):
                    tau_v = torch.full((b, 1, 1), i / args.sample_steps, device=device)
                    v = flow(z, tau_v, Hc)
                    z = z + v / args.sample_steps
                preds.append(z)
            return torch.cat(preds, 0)

        Z_flow1 = euler_chunk(seed=11)
        results["flow_K1_s11"] = retrieval_per_step(tr(H_t, Z_flow1), H_targets, "flow_K1_s11")

        Ks = [1, 4, 8, 16, 32]
        ranks_per_step = {ss: torch.full((N,), N + 1, device=device, dtype=torch.long)
                           for ss in range(S)}
        for ki in range(max(Ks)):
            Zk = euler_chunk(seed=100 + ki)
            H_pred = tr(H_t, Zk)
            for ss in range(S):
                q = F.normalize(H_pred[:, ss].mean(1), dim=-1)
                k = F.normalize(H_targets[:, ss].mean(1), dim=-1)
                sim = q @ k.T
                r = sim.argsort(dim=-1, descending=True)
                gt = torch.arange(N, device=device).unsqueeze(-1)
                rank = (r == gt).float().argmax(-1).long() + 1
                ranks_per_step[ss] = torch.minimum(ranks_per_step[ss], rank)
            del Zk, H_pred
            if (ki + 1) in Ks:
                rows = []
                for ss in range(S):
                    t1 = (ranks_per_step[ss] == 1).float().mean().item()
                    rows.append({"step": ss, "top1": t1,
                                  "mr": ranks_per_step[ss].float().mean().item()})
                avg = sum(r["top1"] for r in rows) / S
                print(f"  [bestK={ki+1:3d} flow                 ]  per-step top1: " +
                      " ".join(f"{r['top1']:.3f}" for r in rows) + f"   avg={avg:.3f}")
                results[f"flow_bestK{ki+1}"] = {"per_step": rows, "avg_top1": avg}

        tgt_pool = H_targets.mean(2)
        results["std_target_pooled"] = tgt_pool.std().item()
        results["eff_rank_target_pooled"] = effective_rank(tgt_pool.reshape(N * S, -1))
        print(f"  std(tgt pooled)={results['std_target_pooled']:.4f}  "
              f"eff_rank={results['eff_rank_target_pooled']:.1f}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"post": results, "args": vars(args)}, f, indent=2)
    print(f"saved metrics -> {args.out_dir}")


if __name__ == "__main__":
    main()
