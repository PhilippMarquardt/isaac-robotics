"""Goal-conditioned multi-step latent-action JEPA on frozen V-JEPA-2 features.

Per anchor t in a film:
  context clip = 16 frames at stride 2, last frame = t           -> H_context [1568,D], H_t = last 196 tokens
  future  clip = 16 frames at stride 2, first frame = t+4        -> H_future  [1568,D]
                                                                  reshape to H_targets [8, 196, D]
                                                                  H_goal = H_targets[:, -1]

Train (with frozen V-JEPA-2 encoder):
  Z* = Inverse(H_t, H_targets)
  H_pred = MultiStepTransition(H_t, Z*)
  L_teacher = cosine_loss_per_step(H_pred, H_targets.detach())
  ε ~ N(0,I); τ ~ U(0,1); z_τ = (1-τ)ε + τ Z*.detach()
  v_pred = Flow(z_τ, τ, H_context.detach(), H_goal.detach())
  L_flow = MSE(v_pred, Z*.detach() - ε)
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
    InverseActionMultiStep, MultiStepTransition, MultiStepTransitionPerStep,
    MultiStepTransitionNoBase, GoalConditionedMultiStepFlow, VectorQuantizer,
)


def cosine_per_step(H_pred, H_tgt):
    """H_pred, H_tgt: [B,S,P,D]. Return mean (2 - 2 cos) over batch/step/patch."""
    p = F.normalize(H_pred, dim=-1)
    t = F.normalize(H_tgt, dim=-1)
    return (2.0 - 2.0 * (p * t).sum(-1)).mean()


def retrieval_per_step(pred, tgt, name=""):
    """pred, tgt: [N,S,P,D]. For each step s, retrieve via mean-pool over patches."""
    N, S, P, D = pred.shape
    results = []
    for s in range(S):
        q = F.normalize(pred[:, s].mean(1), dim=-1)
        k = F.normalize(tgt[:, s].mean(1), dim=-1)
        sim = q @ k.T
        r = sim.argsort(dim=-1, descending=True)
        gt = torch.arange(N, device=q.device).unsqueeze(-1)
        top1 = (r[:, :1] == gt).any(-1).float().mean().item()
        mr = (r == gt).float().argmax(-1).float().mean().item() + 1
        results.append({"step": s, "top1": top1, "mr": mr})
    avg_top1 = sum(r["top1"] for r in results) / S
    avg_mr = sum(r["mr"] for r in results) / S
    print(f"  [{name:20s}]  per-step top1: " + " ".join(f"{r['top1']:.3f}" for r in results)
          + f"   avg_top1={avg_top1:.3f}  avg_mr={avg_mr:.1f}")
    return {"per_step": results, "avg_top1": avg_top1, "avg_mr": avg_mr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="vjepa2_clips cache (frames per film)")
    ap.add_argument("--vjepa_ckpt", required=True, help="V-JEPA-2 ckpt (uses 'target' encoder)")
    ap.add_argument("--out_dir", required=True)
    # data
    ap.add_argument("--num_frames", type=int, default=16, help="frames per clip (ctx and future)")
    ap.add_argument("--frame_stride", type=int, default=2)
    ap.add_argument("--future_offset", type=int, default=4, help="future clip starts at t + future_offset")
    ap.add_argument("--anchor_stride", type=int, default=8)
    # model
    ap.add_argument("--action_tokens", type=int, default=8)
    ap.add_argument("--action_dim", type=int, default=128)
    ap.add_argument("--inverse_depth", type=int, default=4)
    ap.add_argument("--transition_depth", type=int, default=4)
    ap.add_argument("--flow_depth", type=int, default=6)
    ap.add_argument("--per_step_transition", action="store_true",
                     help="use MultiStepTransitionPerStep: output step i sees only Z[i]")
    ap.add_argument("--no_base_transition", action="store_true",
                     help="use MultiStepTransitionNoBase: no skip from H_t")
    ap.add_argument("--delta_loss", action="store_true",
                     help="loss in delta space: cosine(H_pred - H_t, H_target - H_t)")
    ap.add_argument("--vq_codes", type=int, default=0,
                     help="if > 0, apply VectorQuantizer with this codebook size between inverse and transition")
    ap.add_argument("--vq_commit", type=float, default=0.25)
    ap.add_argument("--lambda_commit", type=float, default=1.0)
    ap.add_argument("--goal_hardneg", type=float, default=0.0,
                     help="probability p in [0,1] of replacing (H_goal, Z_target) with a shuffled donor's. 0 = off.")
    # training
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--lambda_teacher", type=float, default=1.0)
    ap.add_argument("--lambda_flow", type=float, default=1.0)
    ap.add_argument("--lambda_z_l2", type=float, default=0.001)
    ap.add_argument("--lambda_z_var", type=float, default=0.05)
    ap.add_argument("--lambda_z_cov", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed); random.seed(args.seed)

    # ---- frozen V-JEPA-2 encoder ----
    vjepa = torch.load(args.vjepa_ckpt, map_location=device, weights_only=False)
    va = vjepa["args"]
    enc = VisionTransformer3D(
        num_frames=va["num_frames"], tubelet_size=va["tubelet_size"],
        patch_size=va["patch_size"], img_size=va["img_size"],
        embed_dim=va["embed_dim"], depth=va["depth"], num_heads=va["num_heads"],
        mlp_ratio=4.0,
    ).to(device).eval()
    enc.load_state_dict(vjepa["target"])  # use the EMA target weights as the frozen encoder
    for p in enc.parameters(): p.requires_grad_(False)
    grid_d = va["num_frames"] // va["tubelet_size"]                 # 8
    grid_s = va["img_size"] // va["patch_size"]                       # 14
    n_patch = grid_s * grid_s                                         # 196
    n_tokens = grid_d * n_patch                                       # 1568
    state_dim = va["embed_dim"]                                       # 384
    print(f"frozen V-JEPA-2 encoder: tubelet={va['tubelet_size']}  grid_d={grid_d}  P={n_patch}  D={state_dim}")

    # ---- dataset (clips) ----
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])
    # context span = (num_frames-1)*frame_stride
    ctx_span = (args.num_frames - 1) * args.frame_stride
    # future first frame = t + future_offset; future span = (num_frames-1)*frame_stride
    fut_span = (args.num_frames - 1) * args.frame_stride
    # need cache frames at [t - ctx_span, t + future_offset + fut_span]
    train_specs, val_specs = [], []
    for fi in range(n_films):
        n_frames = cache["frames"][fi].shape[0]
        a_min = ctx_span
        a_max = n_frames - args.future_offset - fut_span - 1
        last_anchor = a_max
        train_end = int(last_anchor * 0.9)
        val_start = train_end + ctx_span + fut_span + args.future_offset
        for a in range(a_min, train_end, args.anchor_stride):
            train_specs.append((fi, a))
        for a in range(val_start, last_anchor, args.anchor_stride):
            val_specs.append((fi, a))
    N_tr = len(train_specs); N_va = len(val_specs)
    print(f"train anchors={N_tr}  val anchors={N_va}")

    # ---- modules ----
    inv = InverseActionMultiStep(state_dim=state_dim, action_dim=args.action_dim,
                                  n_steps=args.action_tokens, n_patch=n_patch,
                                  depth=args.inverse_depth).to(device)
    if args.no_base_transition:
        tr_cls = MultiStepTransitionNoBase
    elif args.per_step_transition:
        tr_cls = MultiStepTransitionPerStep
    else:
        tr_cls = MultiStepTransition
    tr = tr_cls(state_dim=state_dim, action_dim=args.action_dim,
                n_steps=args.action_tokens, n_patch=n_patch,
                depth=args.transition_depth).to(device)
    print(f"transition: {tr_cls.__name__}")
    flow = GoalConditionedMultiStepFlow(state_dim=state_dim, action_dim=args.action_dim,
                                         n_steps=args.action_tokens,
                                         depth=args.flow_depth).to(device)
    vq = None
    if args.vq_codes > 0:
        vq = VectorQuantizer(num_codes=args.vq_codes, code_dim=args.action_dim,
                              commit=args.vq_commit).to(device)
        print(f"VQ enabled: K={args.vq_codes}  commit={args.vq_commit}")
    params = list(inv.parameters()) + list(tr.parameters()) + list(flow.parameters())
    if vq is not None:
        params += list(vq.parameters())
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f} M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    def encode_clips(specs):
        """Returns H_context [B, N_ctx, D], H_future_flat [B, N_ctx, D]."""
        # context clip: frames at offsets -ctx_span..0 stride frame_stride relative to anchor a.
        ctx_clip, _ = sample_clip_batch(cache, [(fi, a - ctx_span) for (fi, a) in specs],
                                         args.num_frames, args.frame_stride, device)
        # future clip: frames starting at a + future_offset
        fut_clip, _ = sample_clip_batch(cache, [(fi, a + args.future_offset) for (fi, a) in specs],
                                         args.num_frames, args.frame_stride, device)
        with torch.no_grad():
            H_ctx = enc(ctx_clip)                                       # [B, n_tokens, D]
            H_fut = enc(fut_clip)
        return H_ctx, H_fut

    def split_features(H_ctx, H_fut):
        """Return H_t [B,P,D], H_targets [B,S,P,D], H_context [B,N,D], H_goal [B,P,D]."""
        H_t = H_ctx[:, -n_patch:, :]                                     # last tubelet
        H_targets = H_fut[:, : args.action_tokens * n_patch, :].reshape(
            -1, args.action_tokens, n_patch, state_dim)
        H_goal = H_targets[:, -1]                                         # final tubelet
        return H_t, H_targets, H_ctx, H_goal

    # ---- Eval helpers (full retrieval at end + intermediate) ----
    @torch.no_grad()
    def euler_sample(z_shape, H_context, H_goal, steps=30, seed=0):
        b = H_context.shape[0]
        g = torch.Generator(device=device).manual_seed(seed)
        z = torch.randn(b, *z_shape, device=device, generator=g)
        for i in range(steps):
            tau = torch.full((b, 1, 1), i / steps, device=device)
            v = flow(z, tau, H_context, H_goal)
            z = z + v / steps
        return z

    def eval_all(label="val"):
        inv.eval(); tr.eval(); flow.eval()
        with torch.no_grad():
            all_H_t, all_targets, all_ctx, all_goal = [], [], [], []
            all_Z = []
            B_eval = 4
            specs = val_specs
            for s in range(0, len(specs), B_eval):
                batch = specs[s:s + B_eval]
                H_ctx, H_fut = encode_clips(batch)
                H_t, H_targets, H_context, H_goal = split_features(H_ctx, H_fut)
                Z = inv(H_t, H_targets)
                all_H_t.append(H_t); all_targets.append(H_targets); all_ctx.append(H_context); all_goal.append(H_goal); all_Z.append(Z)
            H_t = torch.cat(all_H_t, 0); H_targets = torch.cat(all_targets, 0)
            H_context = torch.cat(all_ctx, 0); H_goal = torch.cat(all_goal, 0); Z = torch.cat(all_Z, 0)
            N = H_t.shape[0]
            print(f"\n[{label}]  N={N}  Z shape {tuple(Z.shape)}")

            # apply VQ to Z if enabled
            if vq is not None:
                Z_continuous = Z
                Z, vq_idx, _ = vq(Z)
            # oracle
            H_pred_correct = tr(H_t, Z)
            oracle = retrieval_per_step(H_pred_correct, H_targets, "oracle Z*")
            # zero_Z
            H_pred_zero = tr(H_t, torch.zeros_like(Z))
            zerores = retrieval_per_step(H_pred_zero, H_targets, "zero_Z")
            # random_Z
            std_z = Z.flatten(2).std(0).mean()
            H_pred_rand = tr(H_t, torch.randn_like(Z) * std_z)
            randres = retrieval_per_step(H_pred_rand, H_targets, "random_Z")
            # shuffle (batch)
            perm = torch.randperm(N, device=device)
            H_pred_shuf = tr(H_t, Z[perm])
            shufres = retrieval_per_step(H_pred_shuf, H_targets, "shuffle_Z(batch)")
            # time-shuffle within sequence
            time_perm = torch.randperm(Z.shape[1], device=device)
            H_pred_tshuf = tr(H_t, Z[:, time_perm])
            tshufres = retrieval_per_step(H_pred_tshuf, H_targets, "time_shuffle_Z")
            # Z_only (zero H_t)
            H_pred_zo = tr(torch.zeros_like(H_t), Z)
            zores = retrieval_per_step(H_pred_zo, H_targets, "Z_only(H=0)")
            # wrong_H
            H_pred_wh = tr(H_t[perm], Z)
            whres = retrieval_per_step(H_pred_wh, H_targets, "wrong_H")
            # copy_Ht (broadcast H_t across steps and predict ≈ H_t)
            base_only = H_t.unsqueeze(1).expand(-1, args.action_tokens, -1, -1)
            copy_res = retrieval_per_step(base_only, H_targets, "copy_Ht(broadcast)")

            # ---- goal-conditioned flow eval ----
            # own goal
            Z_flow_own = euler_sample(Z.shape[1:], H_context, H_goal, steps=30, seed=11)
            flow_own = retrieval_per_step(tr(H_t, Z_flow_own), H_targets, "goal_flow(own)")
            # shuffled goal
            H_goal_shuf = H_goal[perm]
            Z_flow_shuf = euler_sample(Z.shape[1:], H_context, H_goal_shuf, steps=30, seed=12)
            # vs own H_targets
            flow_shuf_vs_own = retrieval_per_step(tr(H_t, Z_flow_shuf), H_targets, "goal_flow(shuf) vs own")
            # vs shuffled H_targets (the donor's future)
            flow_shuf_vs_shuf = retrieval_per_step(tr(H_t, Z_flow_shuf), H_targets[perm], "goal_flow(shuf) vs shuffled")

        results = {
            "N": N, "oracle": oracle, "zero_Z": zerores, "random_Z": randres,
            "shuffle_Z_batch": shufres, "time_shuffle_Z": tshufres,
            "Z_only": zores, "wrong_H": whres, "copy_Ht": copy_res,
            "flow_own": flow_own, "flow_shuf_vs_own": flow_shuf_vs_own,
            "flow_shuf_vs_shuf": flow_shuf_vs_shuf,
        }
        inv.train(); tr.train(); flow.train()
        return results

    # ---- Train ----
    print(f"\n=== Train  steps={args.steps}  bs={args.bs} ===")
    pre_results = eval_all("val pre")
    inv.train(); tr.train(); flow.train()
    for step in range(args.steps):
        idx = [random.randrange(N_tr) for _ in range(args.bs)]
        specs = [train_specs[i] for i in idx]
        H_ctx, H_fut = encode_clips(specs)
        H_t, H_targets, H_context, H_goal = split_features(H_ctx, H_fut)

        Z = inv(H_t, H_targets)
        commit_loss = torch.zeros((), device=device)
        if vq is not None:
            Z, vq_idx, commit_loss = vq(Z)
        H_pred = tr(H_t, Z)
        if args.delta_loss:
            delta_pred = H_pred - H_t.unsqueeze(1)
            delta_target = (H_targets - H_t.unsqueeze(1)).detach()
            p = F.normalize(delta_pred, dim=-1)
            t = F.normalize(delta_target, dim=-1)
            L_teacher_cos = (2.0 - 2.0 * (p * t).sum(-1)).mean()
            L_teacher_mse = F.mse_loss(delta_pred, delta_target)
            L_teacher = L_teacher_cos + 0.5 * L_teacher_mse
        else:
            L_teacher = cosine_per_step(H_pred, H_targets.detach())

        # ---- Flow: optionally swap goal+target trajectory with a donor (hard-neg) ----
        B = Z.shape[0]
        H_goal_flow = H_goal
        Z_target_flow = Z.detach()
        if args.goal_hardneg > 0:
            swap_mask = (torch.rand(B, device=device) < args.goal_hardneg)
            if swap_mask.any():
                perm = torch.randperm(B, device=device)
                while bool((perm == torch.arange(B, device=device)).all().item()):
                    perm = torch.randperm(B, device=device)
                # donor's goal + trajectory; recompute Z via inverse(H_t_i, H_targets_j)
                with torch.no_grad():
                    Z_donor = inv(H_t, H_targets[perm])
                H_goal_flow = torch.where(swap_mask.view(B, 1, 1), H_goal[perm], H_goal)
                Z_target_flow = torch.where(swap_mask.view(B, 1, 1), Z_donor, Z.detach())

        eps = torch.randn_like(Z_target_flow)
        tau = torch.rand(Z_target_flow.shape[0], 1, 1, device=device)
        z_tau = (1 - tau) * eps + tau * Z_target_flow
        v_target = Z_target_flow - eps
        v_pred = flow(z_tau, tau, H_context.detach(), H_goal_flow.detach())
        L_flow = F.mse_loss(v_pred, v_target)

        L_l2 = (Z ** 2).mean()
        std = Z.flatten(0, 1).std(0)
        L_var = F.relu(1 - std).mean()
        z_flat = Z.flatten(0, 1); z_c = z_flat - z_flat.mean(0, keepdim=True)
        cov = (z_c.T @ z_c) / max(1, z_c.shape[0] - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        L_cov = (off ** 2).sum() / cov.shape[0]

        loss = (args.lambda_teacher * L_teacher
                + args.lambda_flow * L_flow
                + args.lambda_z_l2 * L_l2
                + args.lambda_z_var * L_var
                + args.lambda_z_cov * L_cov
                + args.lambda_commit * commit_loss)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            extra = ""
            if vq is not None:
                with torch.no_grad():
                    used = vq.ema_count.gt(1e-3).sum().item()
                extra = f"  L_commit={commit_loss.item():.4f}  codes_used={used}/{vq.K}"
            print(f"  step {step:5d}  L_t={L_teacher.item():.4f}  L_flow={L_flow.item():.4f}  "
                  f"L_var={L_var.item():.4f}  L_cov={L_cov.item():.4f}  "
                  f"alpha_mean={tr.alpha.mean().item():.3f}{extra}")

    print(f"\n=== Train done ===")
    post_results = eval_all("val post")

    save_d = {
        "inv": inv.state_dict(), "tr": tr.state_dict(), "flow": flow.state_dict(),
        "args": vars(args), "va_args": va,
    }
    if vq is not None:
        save_d["vq"] = vq.state_dict()
    torch.save(save_d, os.path.join(args.out_dir, "ckpt.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"pre": pre_results, "post": post_results, "args": vars(args)}, f, indent=2)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
