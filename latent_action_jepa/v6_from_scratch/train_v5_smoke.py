"""V5 smoke trainer: stripped v4 — goal channel disabled, action_dim=16,
running on the Charades dataloader.

Reuses every module from the v4 codebase (encoder, StepBase, InverseFromDeltaPerStep,
MultiStepTransitionPerStep, GoalConditionedDiTFlow) — the DiT just receives goal_dropout=1.0
on every batch so the "goal" channel always carries the learned null token.

Smoke target: 100 steps, bs=4 on H100. Should produce non-degenerate cosine-pred and a
non-collapsed Z (std(Z) > 0.01 after 100 steps).

Add KL bottleneck next iteration. This run is just "does the pipeline ingest Charades
and produce sensible losses?".
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the existing v4 code package importable.
THIS = os.path.dirname(os.path.abspath(__file__))
LAJ = os.path.normpath(os.path.join(THIS, ".."))
sys.path.insert(0, LAJ)
sys.path.insert(0, THIS)

from charades_clips import sample_clip_batch  # noqa: E402
from models.vit3d import VisionTransformer3D  # noqa: E402
from models.la_modules import (  # noqa: E402
    StepBase, InverseFromDeltaPerStep,
    MultiStepTransitionPerStep, GoalConditionedDiTFlow,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/root/jepa_v5/charades_index.json")
    ap.add_argument("--out_dir", default="/root/jepa_v5/runs/v5_smoke")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--frame_stride", type=int, default=2)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--tubelet_size", type=int, default=2)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--embed_dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=6)
    ap.add_argument("--action_steps", type=int, default=4)
    ap.add_argument("--action_dim", type=int, default=16)
    ap.add_argument("--inverse_depth", type=int, default=4)
    ap.add_argument("--base_depth", type=int, default=2)
    ap.add_argument("--transition_depth", type=int, default=4)
    ap.add_argument("--flow_depth", type=int, default=8)
    ap.add_argument("--flow_heads", type=int, default=4)
    ap.add_argument("--n_context_tubelets", type=int, default=4)
    ap.add_argument("--alpha_min", type=float, default=0.02)
    ap.add_argument("--alpha_init", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--lambda_pred", type=float, default=1.0)
    ap.add_argument("--lambda_flow", type=float, default=0.5)
    ap.add_argument("--lambda_z_l2", type=float, default=0.001)
    ap.add_argument("--lambda_z_var", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.99925)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed); random.seed(args.seed)

    with open(args.index) as f:
        index = json.load(f)
    # Held-out val split.
    rng = random.Random(args.seed)
    rng.shuffle(index)
    n_val = max(1, int(len(index) * args.val_frac))
    val_index = index[:n_val]
    train_index = index[n_val:]
    print(f"[v5] index has {len(index)} videos  (train={len(train_index)}, val={len(val_index)})")

    grid_d = args.num_frames // args.tubelet_size
    grid_s = args.img_size // args.patch_size
    n_patch = grid_s * grid_s
    n_tokens = grid_d * n_patch
    n_ctx_tubs = args.n_context_tubelets
    n_fut_tubs = grid_d - n_ctx_tubs
    assert n_fut_tubs == args.action_steps, f"future tubs {n_fut_tubs} != action_steps {args.action_steps}"
    n_ctx_tokens = n_ctx_tubs * n_patch
    n_fut_tokens = n_fut_tubs * n_patch
    S = args.action_steps
    A = args.action_dim
    print(f"[v5] grid_d={grid_d} P={n_patch} S={S} A={A}")

    flat_grid = torch.arange(n_tokens, device=device)
    vis_idx_base = flat_grid[:n_ctx_tokens]
    fut_idx_base = flat_grid[n_ctx_tokens:]
    goal_start = n_ctx_tokens + (n_fut_tubs - 1) * n_patch
    goal_idx_base = flat_grid[goal_start:goal_start + n_patch]

    online = VisionTransformer3D(
        num_frames=args.num_frames, tubelet_size=args.tubelet_size, patch_size=args.patch_size,
        img_size=args.img_size, embed_dim=args.embed_dim, depth=args.depth,
        num_heads=args.num_heads, mlp_ratio=4.0,
    ).to(device).train()
    target_enc = copy.deepcopy(online).eval()
    for p in target_enc.parameters():
        p.requires_grad_(False)

    base = StepBase(state_dim=args.embed_dim, n_steps=S, depth=args.base_depth).to(device).train()
    inv = InverseFromDeltaPerStep(state_dim=args.embed_dim, action_dim=A, n_steps=S, n_patch=n_patch,
                                   depth=args.inverse_depth).to(device).train()
    tr = MultiStepTransitionPerStep(state_dim=args.embed_dim, action_dim=A, n_steps=S,
                                     n_patch=n_patch, depth=args.transition_depth,
                                     alpha_min=args.alpha_min, alpha_init=args.alpha_init).to(device).train()
    flow = GoalConditionedDiTFlow(state_dim=args.embed_dim, action_dim=A, n_steps=S,
                                   depth=args.flow_depth, heads=args.flow_heads).to(device).train()

    n_params = lambda m: sum(p.numel() for p in m.parameters())
    print(f"[v5] params  enc={n_params(online)/1e6:.1f}M  inv={n_params(inv)/1e6:.1f}M  "
          f"tr={n_params(tr)/1e6:.1f}M  flow={n_params(flow)/1e6:.1f}M")

    params = (list(online.parameters()) + list(base.parameters())
              + list(inv.parameters()) + list(tr.parameters())
              + list(flow.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))

    # === eval helpers ===
    def retrieval_top1(pred, tgt):
        """pred, tgt: [B, S, P, D]. Per-step batch-retrieval top-1 (mean across S)."""
        B, Ss, P, D = pred.shape
        scores = []
        for s in range(Ss):
            q = F.normalize(pred[:, s].mean(dim=1), dim=-1)
            k = F.normalize(tgt[:, s].mean(dim=1), dim=-1)
            sim = q @ k.T
            gt = torch.arange(B, device=sim.device)
            top1 = (sim.argmax(dim=-1) == gt).float().mean().item()
            scores.append(top1)
        return sum(scores) / max(1, len(scores)), scores

    @torch.no_grad()
    def run_eval():
        online.eval(); target_enc.eval(); base.eval(); inv.eval(); tr.eval(); flow.eval()
        try:
            clips = sample_clip_batch(val_index, bs=args.eval_bs, num_frames=args.num_frames,
                                       frame_stride=args.frame_stride, img_size=args.img_size,
                                       augment=False)
        except Exception as e:
            print(f"[eval] sample failed: {e}")
            online.train(); base.train(); inv.train(); tr.train(); flow.train()
            return None
        clips = clips.permute(0, 4, 1, 2, 3).contiguous().to(device)
        B = clips.shape[0]
        H_full = target_enc(clips, keep_indices=None)
        H_full = H_full.view(B, grid_d, n_patch, -1)
        H_target = H_full[:, n_ctx_tubs:]
        H_t = H_full[:, :n_ctx_tubs][:, -1]
        delta = H_target - base(H_t)
        Z = inv(delta)
        H_pred = tr(H_t, Z)

        oracle_avg, oracle_steps = retrieval_top1(H_pred, H_target)
        # copy_Ht baseline: predict H_t for every step.
        H_t_rep = H_t.unsqueeze(1).expand(-1, S, -1, -1)
        copy_avg, _ = retrieval_top1(H_t_rep, H_target)
        # time_shuffle baseline: shuffle Z across batch dim, predict.
        perm = torch.randperm(B, device=device)
        Z_shuf = Z[perm]
        H_ts = tr(H_t, Z_shuf)
        ts_avg, _ = retrieval_top1(H_ts, H_target)
        # Z_only baseline: zero out H_t (mean), keep Z.
        zero_state = H_t.mean(dim=1, keepdim=True).expand_as(H_t)
        H_zo = tr(zero_state, Z)
        zo_avg, _ = retrieval_top1(H_zo, H_target)

        # Effective rank of Z.
        Z_flat = Z.flatten(0, 1).float()
        Z_c = Z_flat - Z_flat.mean(dim=0, keepdim=True)
        try:
            s = torch.linalg.svdvals(Z_c)
            p = (s ** 2) / (s ** 2).sum().clamp_min(1e-12)
            eff_rank = float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item())
        except Exception:
            eff_rank = float("nan")

        online.train(); base.train(); inv.train(); tr.train(); flow.train()
        return {
            "oracle": oracle_avg,
            "oracle_per_step": oracle_steps,
            "copy_Ht": copy_avg,
            "gap": oracle_avg - copy_avg,
            "time_shuffle": ts_avg,
            "Z_only": zo_avg,
            "z_eff_rank": eff_rank,
            "z_std": Z.flatten(0, 1).std().item(),
        }

    eval_history = []
    print(f"\n[v5] === train {args.steps} steps  bs={args.bs} ===")
    t_train_start = time.time()
    losses_pred, losses_flow, z_stds = [], [], []
    for step in range(args.steps):
        t_step = time.time()
        # 1. Sample a batch of clips from train split.
        clips = sample_clip_batch(
            train_index, bs=args.bs, num_frames=args.num_frames,
            frame_stride=args.frame_stride, img_size=args.img_size, augment=True,
        )  # [B, T, H, W, 3] in [-1, 1]
        clips = clips.permute(0, 4, 1, 2, 3).contiguous().to(device)  # [B, C, T, H, W]

        # 2. Encode with online (visible context only) and target (full clip).
        with torch.no_grad():
            H_full = target_enc(clips, keep_indices=None)  # [B, n_tokens, D]
        vis_idx = vis_idx_base.expand(args.bs, -1)
        H_ctx_pred = online(clips, keep_indices=vis_idx)  # [B, n_ctx_tokens, D]
        # rearrange to [B, T, P, D]
        D = H_full.shape[-1]
        H_full = H_full.view(args.bs, grid_d, n_patch, D)
        H_ctx_full = H_full[:, :n_ctx_tubs]                      # [B, n_ctx_tubs, P, D]
        H_target = H_full[:, n_ctx_tubs:]                         # [B, S, P, D]
        H_goal = H_full[:, -1]                                    # [B, P, D]  (last tubelet)
        H_t = H_ctx_full[:, -1]                                   # [B, P, D]  (last context tubelet)
        H_ctx_flat = H_ctx_pred                                   # [B, n_ctx_tokens, D]  online prediction

        # 3. Base + delta + inverse Z.
        base_steps = base(H_t)                                    # [B, S, P, D]
        delta = H_target - base_steps
        Z = inv(delta)                                            # [B, S, A]  (per-step inverse)

        # 4. Transition reconstruction.
        H_pred = tr(H_t, Z)                                       # [B, S, P, D]
        L_pred = 1.0 - F.cosine_similarity(
            H_pred.reshape(-1, D), H_target.detach().reshape(-1, D), dim=-1
        ).mean()

        # 5. Flow matching on Z. goal_dropout=1.0 -> always null goal token.
        eps = torch.randn_like(Z)
        tau = torch.rand(args.bs, device=device).clamp(1e-3, 1.0 - 1e-3)
        z_t = (1 - tau).view(args.bs, 1, 1) * eps + tau.view(args.bs, 1, 1) * Z.detach()
        v_target = Z.detach() - eps
        goal_mask = torch.zeros(args.bs, device=device)            # zero = "null", learned token
        v_pred = flow(z_t, tau, H_ctx_flat.detach(), H_goal.detach(), goal_mask=goal_mask)
        L_flow = F.mse_loss(v_pred, v_target)

        # 6. Regularisers on Z (variance + L2).
        z_var = Z.flatten(0, 1).var(dim=0).mean()
        L_z_var = F.relu(1.0 - z_var.sqrt())
        L_z_l2 = Z.pow(2).mean()

        loss = (args.lambda_pred * L_pred + args.lambda_flow * L_flow
                + args.lambda_z_var * L_z_var + args.lambda_z_l2 * L_z_l2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        # EMA update.
        with torch.no_grad():
            m = args.ema
            for po, pt in zip(online.parameters(), target_enc.parameters()):
                pt.data.mul_(m).add_(po.data, alpha=1 - m)
            for bo, bt in zip(online.buffers(), target_enc.buffers()):
                bt.data.copy_(bo.data)

        z_std = Z.flatten(0, 1).std().item()
        losses_pred.append(L_pred.item())
        losses_flow.append(L_flow.item())
        z_stds.append(z_std)
        if (step + 1) % args.log_every == 0 or step == 0:
            dt = time.time() - t_step
            print(f"  step {step+1:4d}/{args.steps}  "
                  f"L_pred={L_pred.item():.3f}  L_flow={L_flow.item():.3f}  "
                  f"z_std={z_std:.3f}  z_var={z_var.item():.4f}  "
                  f"step_t={dt:.2f}s", flush=True)

        # Periodic eval — the headline metric is `oracle` (per-step retrieval
        # of the real H_target given a Z computed from the real future).
        if (step + 1) % args.eval_every == 0:
            m = run_eval()
            if m is not None:
                eval_history.append({"step": step + 1, **m})
                with open(os.path.join(args.out_dir, "eval_history.json"), "w") as f:
                    json.dump(eval_history, f)
                gap = m["gap"]
                rule_ts = m["time_shuffle"] < m["oracle"]
                rule_zo = m["Z_only"] < m["oracle"]
                print(f"  [EVAL step {step+1}]  oracle={m['oracle']:.3f}  "
                      f"copy_Ht={m['copy_Ht']:.3f}  gap={gap:+.3f}  "
                      f"time_shuf={m['time_shuffle']:.3f}  Z_only={m['Z_only']:.3f}  "
                      f"eff_rank={m['z_eff_rank']:.2f}  "
                      f"hard_rules={'OK' if rule_ts and rule_zo else 'BROKEN'}",
                      flush=True)

        # Intermediate checkpoint.
        if (step + 1) % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_step{step+1}.pt")
            torch.save({
                "step": step + 1, "online": online.state_dict(),
                "target": target_enc.state_dict(), "base": base.state_dict(),
                "inv": inv.state_dict(), "tr": tr.state_dict(), "flow": flow.state_dict(),
            }, ckpt_path)
            print(f"  [CKPT]  saved {ckpt_path}", flush=True)

    total_t = time.time() - t_train_start
    print(f"\n[v5] DONE  total={total_t:.1f}s  ({total_t/args.steps:.2f}s/step)")
    print(f"[v5] final L_pred={losses_pred[-1]:.3f}  L_flow={losses_flow[-1]:.3f}  z_std={z_stds[-1]:.3f}")
    print(f"[v5] mean last-10 L_pred={sum(losses_pred[-10:])/10:.3f}")

    # Save tiny ckpt + stats.
    torch.save({
        "online": online.state_dict(),
        "target": target_enc.state_dict(),
        "base": base.state_dict(),
        "inv": inv.state_dict(),
        "tr": tr.state_dict(),
        "flow": flow.state_dict(),
    }, os.path.join(args.out_dir, "smoke_ckpt.pt"))
    with open(os.path.join(args.out_dir, "smoke_stats.json"), "w") as f:
        json.dump({"L_pred": losses_pred, "L_flow": losses_flow, "z_std": z_stds}, f)
    print(f"[v5] saved to {args.out_dir}")


if __name__ == "__main__":
    main()
