"""Stage-B: flow-only training on a frozen E2E teacher.

Load run2_big ckpt. Freeze online encoder, target encoder, inverse, transition.
Re-initialize OR keep flow weights. Train flow alone with full attention to
matching the inverse-derived Z* from past context, no JEPA / LA loss
competing.

This addresses the weakness in run2: context_flow=0.064 vs oracle=0.248. The
flow lost ground during E2E because cross-target aug shifted the Z target
distribution mid-training. Freezing the rest stops that and lets the flow
catch up.
"""
import argparse
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
    InverseActionMultiStep, MultiStepTransition, ContextDiTFlow,
    ContextOnlyMultiStepFlow,
)


def retrieval_per_step(pred, tgt, name=""):
    N, S, P, D = pred.shape
    rows = []
    for s in range(S):
        q = F.normalize(pred[:, s].mean(1), dim=-1)
        k = F.normalize(tgt[:, s].mean(1), dim=-1)
        sim = q @ k.T
        r = sim.argsort(dim=-1, descending=True)
        gt = torch.arange(N, device=q.device).unsqueeze(-1)
        rows.append({"step": s, "top1": (r[:, :1] == gt).any(-1).float().mean().item(),
                     "mr": (r == gt).float().argmax(-1).float().mean().item() + 1})
    avg = sum(r["top1"] for r in rows) / S
    print(f"  [{name:30s}]  per-step top1: " + " ".join(f"{r['top1']:.3f}" for r in rows) + f"   avg={avg:.3f}")
    return {"per_step": rows, "avg_top1": avg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5.25e-4)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--reinit_flow", action="store_true",
                     help="reset flow weights and retrain from scratch")
    ap.add_argument("--n_val_eval", type=int, default=800)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--sample_steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed); random.seed(args.seed)

    ck = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    grid_d = a["num_frames"] // a["tubelet_size"]
    grid_s = a["img_size"] // a["patch_size"]
    n_patch = grid_s * grid_s
    n_tokens = grid_d * n_patch
    n_ctx_tubs = a["n_context_tubelets"]
    n_ctx_tokens = n_ctx_tubs * n_patch
    S = grid_d - n_ctx_tubs
    print(f"teacher: grid_d={grid_d} P={n_patch} ctx_tubs={n_ctx_tubs} S={S}")

    online = VisionTransformer3D(
        num_frames=a["num_frames"], tubelet_size=a["tubelet_size"], patch_size=a["patch_size"],
        img_size=a["img_size"], embed_dim=a["embed_dim"], depth=a["depth"],
        num_heads=a["num_heads"], mlp_ratio=4.0,
    ).to(device).eval()
    target = VisionTransformer3D(
        num_frames=a["num_frames"], tubelet_size=a["tubelet_size"], patch_size=a["patch_size"],
        img_size=a["img_size"], embed_dim=a["embed_dim"], depth=a["depth"],
        num_heads=a["num_heads"], mlp_ratio=4.0,
    ).to(device).eval()
    inv = InverseActionMultiStep(state_dim=a["embed_dim"], action_dim=a["action_dim"],
                                  n_steps=S, n_patch=n_patch,
                                  depth=a["inverse_depth"]).to(device).eval()
    tr = MultiStepTransition(state_dim=a["embed_dim"], action_dim=a["action_dim"],
                              n_steps=S, n_patch=n_patch,
                              depth=a["transition_depth"]).to(device).eval()
    if a.get("flow_type", "xattn") == "dit":
        flow = ContextDiTFlow(state_dim=a["embed_dim"], action_dim=a["action_dim"],
                              n_steps=S, depth=a["flow_depth"]).to(device).train()
    else:
        flow = ContextOnlyMultiStepFlow(state_dim=a["embed_dim"], action_dim=a["action_dim"],
                                         n_steps=S, depth=a["flow_depth"]).to(device).train()
    online.load_state_dict(ck["online"])
    target.load_state_dict(ck["target"])
    inv.load_state_dict(ck["inv"])
    tr.load_state_dict(ck["tr"])
    if not args.reinit_flow:
        flow.load_state_dict(ck["flow"])
    for m in (online, target, inv, tr):
        for p in m.parameters():
            p.requires_grad_(False)
    print(f"frozen: online + target + inv + tr  |  trainable: flow ({sum(p.numel() for p in flow.parameters())/1e6:.2f} M)")
    print(f"reinit_flow={args.reinit_flow}")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])
    span = (a["num_frames"] - 1) * a["frame_stride"] + 1
    train_specs, val_specs = [], []
    for fi in range(n_films):
        n_frames = cache["frames"][fi].shape[0]
        last_anchor = n_frames - span - 1
        train_end = int(last_anchor * 0.9)
        val_start = train_end + a["num_frames"] * a["frame_stride"]
        for av in range(0, train_end, a["clip_anchor_stride"]):
            train_specs.append((fi, av))
        for av in range(val_start, last_anchor, a["clip_anchor_stride"]):
            val_specs.append((fi, av))
    N_tr, N_va = len(train_specs), len(val_specs)
    print(f"train clips={N_tr}  val clips={N_va}")

    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.wd)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * p))

    print("\n=== Train flow only ===")
    for step in range(args.steps):
        idx = [random.randrange(N_tr) for _ in range(args.bs)]
        specs = [train_specs[i] for i in idx]
        clip, _ = sample_clip_batch(cache, specs, a["num_frames"], a["frame_stride"], device)
        with torch.no_grad():
            H_online = online(clip)
            H_target = target(clip)
            D = H_online.shape[-1]
            B = H_online.shape[0]
            H_t = H_online[:, n_ctx_tokens - n_patch : n_ctx_tokens, :]
            H_targets = H_target[:, n_ctx_tokens:, :].reshape(B, S, n_patch, D)
            H_context = H_online[:, :n_ctx_tokens, :]
            Z = inv(H_t, H_targets)
        eps = torch.randn_like(Z)
        tau = torch.rand(Z.shape[0], 1, 1, device=device)
        z_tau = (1 - tau) * eps + tau * Z
        v_target = Z - eps
        v_pred = flow(z_tau, tau, H_context)
        L = F.mse_loss(v_pred, v_target)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  L_flow={L.item():.4f}  lr={opt.param_groups[0]['lr']:.2e}")

    # save flow
    torch.save({"flow": flow.state_dict(), "args": vars(args), "teacher_ckpt": args.teacher_ckpt},
               os.path.join(args.out_dir, "flow.pt"))

    # ---- Eval ----
    flow.eval()
    N_eval = min(N_va, args.n_val_eval)
    print(f"\n=== Eval val (N={N_eval} of {N_va}) ===")
    all_H_t, all_H_targets, all_H_context, all_Z = [], [], [], []
    B_eval = 2
    with torch.no_grad():
        for s in range(0, N_eval, B_eval):
            specs = val_specs[s:s + B_eval]
            clip, _ = sample_clip_batch(cache, specs, a["num_frames"], a["frame_stride"], device)
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

        # context flow sample at multiple seeds
        for seed in [11, 22, 33]:
            b = H_context.shape[0]
            g = torch.Generator(device=device).manual_seed(seed)
            z = torch.randn(b, S, a["action_dim"], device=device, generator=g)
            for i in range(args.sample_steps):
                tau_v = torch.full((b, 1, 1), i / args.sample_steps, device=device)
                v = flow(z, tau_v, H_context)
                z = z + v / args.sample_steps
            results[f"context_flow_seed{seed}"] = retrieval_per_step(
                tr(H_t, z), H_targets, f"context_flow s{seed}")

        # best-of-K flow
        Ks = [1, 4, 16]
        ranks_per_step = {s: torch.full((N,), N + 1, device=device, dtype=torch.long) for s in range(S)}
        best_tops = {s: [] for s in range(S)}
        for ki in range(max(Ks)):
            b = N
            g = torch.Generator(device=device).manual_seed(100 + ki)
            z = torch.randn(b, S, a["action_dim"], device=device, generator=g)
            for i in range(args.sample_steps):
                tau_v = torch.full((b, 1, 1), i / args.sample_steps, device=device)
                v = flow(z, tau_v, H_context)
                z = z + v / args.sample_steps
            H_pred = tr(H_t, z)
            for s in range(S):
                q = F.normalize(H_pred[:, s].mean(1), dim=-1)
                k = F.normalize(H_targets[:, s].mean(1), dim=-1)
                sim = q @ k.T
                r = sim.argsort(dim=-1, descending=True)
                gt = torch.arange(N, device=device).unsqueeze(-1)
                rank = (r == gt).float().argmax(-1).long() + 1
                ranks_per_step[s] = torch.minimum(ranks_per_step[s], rank)
            if (ki + 1) in Ks:
                rows = []
                for s in range(S):
                    t1 = (ranks_per_step[s] == 1).float().mean().item()
                    rows.append({"step": s, "top1": t1, "mr": ranks_per_step[s].float().mean().item()})
                avg = sum(r["top1"] for r in rows) / S
                print(f"  [bestK={ki+1:3d} context_flow         ]  per-step top1: " +
                      " ".join(f"{r['top1']:.3f}" for r in rows) + f"   avg={avg:.3f}")
                results[f"context_flow_bestK{ki+1}"] = {"per_step": rows, "avg_top1": avg}

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"post": results, "args": vars(args)}, f, indent=2)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
