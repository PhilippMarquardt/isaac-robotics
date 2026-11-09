"""Phase 5 — synthetic robot-adapter test on flow-action JEPA ckpt.

Setup:
  For each anchor we sample a 4-dim "action" [tx, ty, scale, theta], apply it
  as an identical affine transform to every frame in the FUTURE half of the
  clip, then encode the MODIFIED clip via the EMA target encoder. The inverse
  on the motion residual produces Z*. We have (params, Z*) pairs.

  We then train a small MLP `params -> Z` adapter and evaluate via
  `T(H_t, adapter(params))` retrieval against the modified-future targets.

This answers: can a small supervised adapter map low-dim action labels onto
the learned compact Z space, such that the frozen transition reproduces the
labeled future representation?
"""
import argparse
import json
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from data.vjepa2_clips import sample_clip_batch, IMAGENET_MEAN, IMAGENET_STD
from models.vit3d import VisionTransformer3D
from models.la_modules import (
    StepBase, InverseFromDelta, InverseFromDeltaPerStep,
    MultiStepTransitionPerStep,
)


PARAM_RANGES = {
    "tx": (-30, 30), "ty": (-30, 30),
    "scale": (0.8, 1.2), "theta": (-15, 15),
}


def sample_action(rng):
    return [
        rng.uniform(*PARAM_RANGES["tx"]),
        rng.uniform(*PARAM_RANGES["ty"]),
        rng.uniform(*PARAM_RANGES["scale"]),
        rng.uniform(*PARAM_RANGES["theta"]),
    ]


def apply_affine_to_future(clip, params, n_future_frames):
    """clip: [B, 3, T, H, W] ImageNet-normalized float. params: list of B param lists.
    Applies the same affine per-clip to the LAST n_future_frames frames.
    Returns the modified clip (clone)."""
    B, C, T, H, W = clip.shape
    out = clip.clone()
    fut_start = T - n_future_frames
    for i in range(B):
        tx, ty, sc, th = params[i]
        for t in range(fut_start, T):
            out[i, :, t] = TF.affine(out[i, :, t], angle=th, translate=(tx, ty),
                                       scale=sc, shear=(0, 0))
    return out


def per_step_retrieval(pred, tgt, name=""):
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
                     "top5": (r[:, :5] == gt).any(-1).float().mean().item()})
    avg_t1 = sum(r["top1"] for r in rows) / S
    avg_t5 = sum(r["top5"] for r in rows) / S
    print(f"  [{name:36s}]  per-step top1: " +
          " ".join(f"{r['top1']:.3f}" for r in rows) +
          f"   avg_t1={avg_t1:.3f}  avg_t5={avg_t5:.3f}")
    return {"per_step": rows, "avg_top1": avg_t1, "avg_top5": avg_t5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_pairs", type=int, default=4000,
                     help="number of (anchor, action_params) pairs to build")
    ap.add_argument("--adapter_steps", type=int, default=3000)
    ap.add_argument("--adapter_lr", type=float, default=3e-4)
    ap.add_argument("--adapter_hidden", type=int, default=512)
    ap.add_argument("--adapter_depth", type=int, default=3)
    ap.add_argument("--adapter_bs", type=int, default=128)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed); random.seed(args.seed)
    rng = random.Random(args.seed)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    grid_d = a["num_frames"] // a["tubelet_size"]
    grid_s = a["img_size"] // a["patch_size"]
    n_patch = grid_s * grid_s
    n_tokens = grid_d * n_patch
    n_ctx_tubs = a["n_context_tubelets"]
    n_fut_tubs = grid_d - n_ctx_tubs
    n_ctx_tokens = n_ctx_tubs * n_patch
    S = a["action_steps"]; A = a["action_dim"]
    n_future_frames = n_fut_tubs * a["tubelet_size"]

    online = VisionTransformer3D(
        num_frames=a["num_frames"], tubelet_size=a["tubelet_size"], patch_size=a["patch_size"],
        img_size=a["img_size"], embed_dim=a["embed_dim"], depth=a["depth"],
        num_heads=a["num_heads"], mlp_ratio=4.0,
    ).to(device).eval()
    target_enc = VisionTransformer3D(
        num_frames=a["num_frames"], tubelet_size=a["tubelet_size"], patch_size=a["patch_size"],
        img_size=a["img_size"], embed_dim=a["embed_dim"], depth=a["depth"],
        num_heads=a["num_heads"], mlp_ratio=4.0,
    ).to(device).eval()
    base = StepBase(state_dim=a["embed_dim"], n_steps=S, depth=a.get("base_depth", 2)).to(device).eval()
    inv_cls = InverseFromDeltaPerStep if a.get("per_step_inverse", False) else InverseFromDelta
    inv = inv_cls(state_dim=a["embed_dim"], action_dim=A, n_steps=S, n_patch=n_patch,
                  depth=a["inverse_depth"]).to(device).eval()
    tr = MultiStepTransitionPerStep(state_dim=a["embed_dim"], action_dim=A, n_steps=S,
                                     n_patch=n_patch, depth=a["transition_depth"],
                                     alpha_min=a.get("alpha_min", 0.02),
                                     alpha_init=a.get("alpha_init", 0.05)).to(device).eval()
    online.load_state_dict(ck["online"])
    target_enc.load_state_dict(ck["target"])
    base.load_state_dict(ck["base"])
    inv.load_state_dict(ck["inv"])
    tr.load_state_dict(ck["tr"])
    for m in (online, target_enc, base, inv, tr):
        for p in m.parameters(): p.requires_grad_(False)
    print(f"loaded teacher: alpha={tr.alpha.mean().item():.3f}  S={S} A={A} n_future_frames={n_future_frames}")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])
    span = (a["num_frames"] - 1) * a["frame_stride"] + 1
    # pool of anchors across films
    pool = []
    for fi in range(n_films):
        n_frames = cache["frames"][fi].shape[0]
        last_anchor = n_frames - span - 1
        if last_anchor <= 0: continue
        for av in range(0, last_anchor, max(1, a["clip_anchor_stride"])):
            pool.append((fi, av))
    rng.shuffle(pool)
    pool = pool[:args.n_pairs]
    print(f"pool size {len(pool)}")

    flat_grid = torch.arange(n_tokens, device=device)
    vis_idx_base = flat_grid[:n_ctx_tokens]
    fut_idx_base = flat_grid[n_ctx_tokens:]

    # ---- build (params, Z*, H_t, H_targets) dataset ----
    print("\n=== building (params, Z*) dataset ===")
    all_params, all_Z, all_Ht, all_Htargets = [], [], [], []
    Bsz = 4
    with torch.no_grad():
        for s in range(0, len(pool), Bsz):
            batch = pool[s:s + Bsz]
            clip, _ = sample_clip_batch(cache, batch, a["num_frames"], a["frame_stride"], device)
            B = clip.shape[0]
            params = [sample_action(rng) for _ in range(B)]
            clip_mod = apply_affine_to_future(clip, params, n_future_frames)
            vis_idx = vis_idx_base.unsqueeze(0).expand(B, -1)
            fut_idx = fut_idx_base.unsqueeze(0).expand(B, -1)
            # online sees only context — which is identical in clip and clip_mod
            H_ctx = online(clip_mod, keep_indices=vis_idx)
            H_t = H_ctx[:, -n_patch:, :]
            # EMA target sees full modified clip
            H_full = target_enc(clip_mod)
            D = H_full.shape[-1]
            H_targets = H_full.gather(1, fut_idx.unsqueeze(-1).expand(-1, -1, D)).reshape(B, S, n_patch, D)
            base_p = base(H_t)
            delta = H_targets - base_p
            Z = inv(delta)
            all_params.extend(params)
            all_Z.append(Z.cpu())
            all_Ht.append(H_t.cpu())
            all_Htargets.append(H_targets.cpu())
            if (s // Bsz) % 20 == 0:
                print(f"  built {s+B}/{len(pool)}")
    params_t = torch.tensor(all_params, dtype=torch.float32)
    Z_t = torch.cat(all_Z, 0)
    Ht_t = torch.cat(all_Ht, 0)
    Htgt_t = torch.cat(all_Htargets, 0)
    N = params_t.shape[0]
    print(f"built {N} pairs  Z shape {tuple(Z_t.shape)}  Ht shape {tuple(Ht_t.shape)}  Htgt shape {tuple(Htgt_t.shape)}")

    # normalize params
    p_mean = params_t.mean(0); p_std = params_t.std(0).clamp_min(1e-6)
    params_norm = (params_t - p_mean) / p_std

    # split
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    n_tr = int(N * args.train_frac)
    tr_idx, va_idx = perm[:n_tr], perm[n_tr:]
    print(f"train {len(tr_idx)}  val {len(va_idx)}")

    # adapter
    Ka, Da = Z_t.shape[1], Z_t.shape[2]
    out_dim = Ka * Da
    layers = []
    in_dim = 4
    for _ in range(args.adapter_depth - 1):
        layers += [nn.Linear(in_dim, args.adapter_hidden), nn.GELU()]
        in_dim = args.adapter_hidden
    layers += [nn.Linear(in_dim, out_dim)]
    adapter = nn.Sequential(*layers).to(device)
    print(f"adapter params: {sum(p.numel() for p in adapter.parameters())/1e3:.1f} K  out_dim={out_dim}")
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.adapter_lr, weight_decay=1e-3)

    params_norm = params_norm.to(device)
    Z_flat = Z_t.reshape(N, -1).to(device)

    for step in range(args.adapter_steps):
        idx = tr_idx[torch.randint(0, len(tr_idx), (args.adapter_bs,))]
        x = params_norm[idx]; y = Z_flat[idx]
        pred = adapter(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == args.adapter_steps - 1:
            with torch.no_grad():
                cos = F.cosine_similarity(pred, y, dim=-1).mean().item()
            print(f"  adapter step {step:5d}  mse={loss.item():.4f}  cos={cos:+.4f}")

    # ---- eval ----
    print(f"\n=== Eval adapter on held-out val (N={len(va_idx)}) ===")
    adapter.eval()
    with torch.no_grad():
        x_va = params_norm[va_idx]
        z_pred = adapter(x_va).reshape(-1, Ka, Da)
        z_true = Z_t[va_idx].to(device)
        cos_full = F.cosine_similarity(z_pred.flatten(1), z_true.flatten(1), dim=-1).mean().item()
        print(f"  cos(z_pred, z*) = {cos_full:+.4f}")

        Ht_v = Ht_t[va_idx].to(device)
        Htgt_v = Htgt_t[va_idx].to(device)

        results = {"cos_zpred_zstar": cos_full}
        results["oracle"] = per_step_retrieval(tr(Ht_v, z_true), Htgt_v, "T(H_t, z*) oracle")
        results["adapter"] = per_step_retrieval(tr(Ht_v, z_pred), Htgt_v, "T(H_t, adapter(params))")
        results["zero_z"] = per_step_retrieval(tr(Ht_v, torch.zeros_like(z_true)), Htgt_v, "T(H_t, 0) zero")
        std_z = z_true.flatten(0, 1).std(0)
        results["random_z"] = per_step_retrieval(tr(Ht_v, torch.randn_like(z_true) * std_z),
                                                  Htgt_v, "T(H_t, random)")
        # data-scaling curve (need enable_grad since outer with is no_grad)
        print("\n--- adapter data-scaling curve (re-train at different sizes) ---")
        ks = [50, 200, 800, 3000]
        scaling = []
        with torch.enable_grad():
            for k in ks:
                if k > len(tr_idx): break
                tr_k = tr_idx[:k]
                ad2 = nn.Sequential(
                    *([nn.Linear(4, args.adapter_hidden), nn.GELU()] * 1),
                    nn.Linear(args.adapter_hidden, out_dim),
                ).to(device)
                opt2 = torch.optim.AdamW(ad2.parameters(), lr=args.adapter_lr, weight_decay=1e-3)
                for step in range(args.adapter_steps):
                    idx = tr_k[torch.randint(0, k, (min(args.adapter_bs, k),))]
                    pred = ad2(params_norm[idx])
                    loss = F.mse_loss(pred, Z_flat[idx])
                    opt2.zero_grad(); loss.backward(); opt2.step()
                with torch.no_grad():
                    z_pk = ad2(x_va).reshape(-1, Ka, Da)
                    cos_k = F.cosine_similarity(z_pk.flatten(1), z_true.flatten(1), dim=-1).mean().item()
                    ret_k = per_step_retrieval(tr(Ht_v, z_pk), Htgt_v, f"adapter@{k}")
                scaling.append({"k": k, "cos": cos_k, "ret": ret_k})
        results["scaling"] = scaling

    with open(os.path.join(args.out_dir, "phase5.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.save({"adapter": adapter.state_dict(), "p_mean": p_mean, "p_std": p_std},
               os.path.join(args.out_dir, "adapter.pt"))
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
