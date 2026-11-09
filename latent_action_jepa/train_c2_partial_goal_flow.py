"""C2: partial-goal flow only. Freeze inverse + transition from a LA-JEPA ckpt
(typically run B), train a new MaskedGoalMultiStepFlow with matched samples
only. Sweep keep counts at eval.
"""
import argparse
import json
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.vjepa2_clips import sample_clip_batch
from models.vit3d import VisionTransformer3D
from models.la_modules import (
    InverseActionMultiStep, MultiStepTransition, MultiStepTransitionPerStep,
    MaskedGoalMultiStepFlow,
)


KEEP_CHOICES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 196]


def sample_keep_mask(B, P, keep, device, generator=None):
    if keep == 0:
        return torch.zeros(B, P, dtype=torch.bool, device=device)
    if keep >= P:
        return torch.ones(B, P, dtype=torch.bool, device=device)
    scores = torch.rand(B, P, device=device, generator=generator)
    idx = scores.argsort(dim=-1, descending=False)[:, :keep]
    mask = torch.zeros(B, P, dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return mask


def retrieval_per_step(pred, tgt, name=""):
    N, S, P, D = pred.shape
    rows = []
    for s in range(S):
        q = F.normalize(pred[:, s].mean(1), dim=-1)
        k = F.normalize(tgt[:, s].mean(1), dim=-1)
        sim = q @ k.T
        r = sim.argsort(dim=-1, descending=True)
        gt = torch.arange(N, device=q.device).unsqueeze(-1)
        t1 = (r[:, :1] == gt).any(-1).float().mean().item()
        mr = (r == gt).float().argmax(-1).float().mean().item() + 1
        rows.append({"step": s, "top1": t1, "mr": mr})
    avg_top1 = sum(r["top1"] for r in rows) / S
    avg_mr = sum(r["mr"] for r in rows) / S
    if name:
        print(f"  [{name:30s}]  per-step top1: " + " ".join(f"{r['top1']:.3f}" for r in rows)
              + f"   avg={avg_top1:.3f}  mr={avg_mr:.1f}")
    return {"per_step": rows, "avg_top1": avg_top1, "avg_mr": avg_mr}


@torch.no_grad()
def euler_sample(flow, z_shape, H_context, H_goal, visible, steps=30, seed=0, device="cuda"):
    b = H_context.shape[0]
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(b, *z_shape, device=device, generator=g)
    for i in range(steps):
        tau = torch.full((b, 1, 1), i / steps, device=device)
        v = flow(z, tau, H_context, H_goal, visible)
        z = z + v / steps
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--vjepa_ckpt", required=True)
    ap.add_argument("--la_ckpt", required=True, help="LA-JEPA ckpt with inv+transition to freeze (use run B)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--flow_depth", type=int, default=6)
    ap.add_argument("--flow_heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--sample_steps", type=int, default=30)
    ap.add_argument("--ks_eval", type=int, nargs="+", default=[1, 32])
    ap.add_argument("--log_every", type=int, default=200)
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
    enc.load_state_dict(vjepa["target"])
    for p in enc.parameters(): p.requires_grad_(False)
    grid_d = va["num_frames"] // va["tubelet_size"]
    grid_s = va["img_size"] // va["patch_size"]
    n_patch = grid_s * grid_s
    state_dim = va["embed_dim"]

    # ---- frozen LA teacher ----
    la = torch.load(args.la_ckpt, map_location=device, weights_only=False)
    la_args = la["args"]
    S = la_args["action_tokens"]
    A = la_args["action_dim"]
    use_per_step_tr = la_args.get("per_step_transition", False)
    tr_cls = MultiStepTransitionPerStep if use_per_step_tr else MultiStepTransition
    inv = InverseActionMultiStep(state_dim=state_dim, action_dim=A, n_steps=S, n_patch=n_patch,
                                  depth=la_args["inverse_depth"]).to(device).eval()
    tr = tr_cls(state_dim=state_dim, action_dim=A, n_steps=S, n_patch=n_patch,
                depth=la_args["transition_depth"]).to(device).eval()
    inv.load_state_dict(la["inv"])
    tr.load_state_dict(la["tr"])
    for p in list(inv.parameters()) + list(tr.parameters()):
        p.requires_grad_(False)
    print(f"loaded LA teacher from {args.la_ckpt} (transition: {tr_cls.__name__})")

    # ---- data ----
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])
    ctx_span = (la_args["num_frames"] - 1) * la_args["frame_stride"]
    fut_span = ctx_span
    fo = la_args["future_offset"]
    anchor_stride = la_args["anchor_stride"]
    train_specs, val_specs = [], []
    for fi in range(n_films):
        n_frames_film = cache["frames"][fi].shape[0]
        a_min = ctx_span
        a_max = n_frames_film - fo - fut_span - 1
        train_end = int(a_max * 0.9)
        val_start = train_end + ctx_span + fut_span + fo
        for a in range(a_min, train_end, anchor_stride):
            train_specs.append((fi, a))
        for a in range(val_start, a_max, anchor_stride):
            val_specs.append((fi, a))
    N_tr, N_va = len(train_specs), len(val_specs)
    print(f"train anchors={N_tr}  val anchors={N_va}")

    def encode_pair(specs):
        ctx_clip, _ = sample_clip_batch(cache, [(fi, a - ctx_span) for (fi, a) in specs],
                                         la_args["num_frames"], la_args["frame_stride"], device)
        fut_clip, _ = sample_clip_batch(cache, [(fi, a + fo) for (fi, a) in specs],
                                         la_args["num_frames"], la_args["frame_stride"], device)
        with torch.no_grad():
            H_ctx = enc(ctx_clip)
            H_fut = enc(fut_clip)
        return H_ctx, H_fut

    def split(H_ctx, H_fut):
        H_t = H_ctx[:, -n_patch:, :]
        H_targets = H_fut[:, : S * n_patch, :].reshape(-1, S, n_patch, state_dim)
        H_goal = H_targets[:, -1]
        return H_t, H_targets, H_ctx, H_goal

    # ---- flow ----
    flow = MaskedGoalMultiStepFlow(state_dim=state_dim, action_dim=A, n_steps=S, n_patch=n_patch,
                                    depth=args.flow_depth, heads=args.flow_heads).to(device)
    print(f"flow params: {sum(p.numel() for p in flow.parameters())/1e6:.2f} M")
    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    flow.train()
    for step in range(args.steps):
        idx = [random.randrange(N_tr) for _ in range(args.bs)]
        specs = [train_specs[i] for i in idx]
        H_ctx, H_fut = encode_pair(specs)
        H_t, H_targets, H_context, H_goal = split(H_ctx, H_fut)
        with torch.no_grad():
            Z = inv(H_t, H_targets)

        keep = KEEP_CHOICES[random.randrange(len(KEEP_CHOICES))]
        visible = sample_keep_mask(args.bs, n_patch, keep, device)

        eps = torch.randn_like(Z)
        tau = torch.rand(Z.shape[0], 1, 1, device=device)
        z_tau = (1 - tau) * eps + tau * Z
        v_target = Z - eps
        v_pred = flow(z_tau, tau, H_context, H_goal, visible)
        loss = F.mse_loss(v_pred, v_target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step(); sched.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  L_flow={loss.item():.4f}  keep={keep}  lr={opt.param_groups[0]['lr']:.2e}")

    # ---- Eval ----
    flow.eval()
    # collect val features once
    H_ts, H_targets_l, H_ctx_l, H_goal_l, Zs = [], [], [], [], []
    with torch.no_grad():
        for s in range(0, N_va, args.bs):
            batch = val_specs[s:s + args.bs]
            H_ctx, H_fut = encode_pair(batch)
            H_t, H_targets, H_context, H_goal = split(H_ctx, H_fut)
            Z = inv(H_t, H_targets)
            H_ts.append(H_t); H_targets_l.append(H_targets); H_ctx_l.append(H_context)
            H_goal_l.append(H_goal); Zs.append(Z)
    H_t = torch.cat(H_ts, 0); H_targets = torch.cat(H_targets_l, 0)
    H_context = torch.cat(H_ctx_l, 0); H_goal = torch.cat(H_goal_l, 0); Z_oracle = torch.cat(Zs, 0)
    N = H_t.shape[0]
    print(f"\n=== Eval val (N={N}) ===")

    results = {}
    # oracle reference
    with torch.no_grad():
        H_pred_oracle = tr(H_t, Z_oracle)
        results["oracle"] = retrieval_per_step(H_pred_oracle, H_targets, "oracle Z*")

    # keep-count sweep
    for keep in KEEP_CHOICES:
        visible = sample_keep_mask(N, n_patch, keep, device,
                                    generator=torch.Generator(device=device).manual_seed(7 + keep))
        for k_sample in args.ks_eval:
            preds = []
            with torch.no_grad():
                # best-of-K via taking min-rank across K euler samples
                K_max = max(args.ks_eval)
                if k_sample == 1:
                    z = euler_sample(flow, Z_oracle.shape[1:], H_context, H_goal, visible,
                                     steps=args.sample_steps, seed=10 * keep, device=device)
                    H_pred = tr(H_t, z)
                    r = retrieval_per_step(H_pred, H_targets, f"keep={keep:3d} K=1")
                    results[f"keep{keep}_K1"] = r
                elif k_sample == 32:
                    # best-of-K oracle selection per step per anchor
                    samples = [euler_sample(flow, Z_oracle.shape[1:], H_context, H_goal, visible,
                                             steps=args.sample_steps, seed=10 * keep + ki, device=device)
                               for ki in range(k_sample)]
                    # per-step best rank across K samples
                    per_step_best = {s: [] for s in range(Z_oracle.shape[1])}
                    target_means = [F.normalize(H_targets[:, s].mean(1), dim=-1) for s in range(Z_oracle.shape[1])]
                    best_ranks = [torch.full((N,), N + 1, device=device, dtype=torch.long)
                                  for _ in range(Z_oracle.shape[1])]
                    for z in samples:
                        H_pred = tr(H_t, z)
                        for s in range(Z_oracle.shape[1]):
                            q = F.normalize(H_pred[:, s].mean(1), dim=-1)
                            sim = q @ target_means[s].T
                            r = sim.argsort(dim=-1, descending=True)
                            gt = torch.arange(N, device=device).unsqueeze(-1)
                            rank = (r == gt).float().argmax(-1).long() + 1
                            best_ranks[s] = torch.minimum(best_ranks[s], rank)
                    rows = []
                    for s in range(Z_oracle.shape[1]):
                        t1 = (best_ranks[s] == 1).float().mean().item()
                        mr = best_ranks[s].float().mean().item()
                        rows.append({"step": s, "top1": t1, "mr": mr})
                    avg_top1 = sum(r["top1"] for r in rows) / Z_oracle.shape[1]
                    avg_mr = sum(r["mr"] for r in rows) / Z_oracle.shape[1]
                    print(f"  [keep={keep:3d} K=32 bestK             ]  per-step top1: "
                          + " ".join(f"{r['top1']:.3f}" for r in rows) + f"   avg={avg_top1:.3f}")
                    results[f"keep{keep}_K32"] = {"per_step": rows, "avg_top1": avg_top1, "avg_mr": avg_mr}

    # print summary table
    print("\n=== keep-count sweep summary ===")
    print(f"  oracle avg top1 = {results['oracle']['avg_top1']:.3f}")
    print("  keep |   K=1   |   K=32  ")
    for keep in KEEP_CHOICES:
        k1 = results.get(f"keep{keep}_K1", {}).get("avg_top1", 0)
        k32 = results.get(f"keep{keep}_K32", {}).get("avg_top1", 0)
        print(f"  {keep:4d} | {k1:.4f} | {k32:.4f}")

    torch.save({"flow": flow.state_dict(), "args": vars(args)}, os.path.join(args.out_dir, "flow.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
