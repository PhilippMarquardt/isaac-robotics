"""Proper V-JEPA-2 eval: predict only masked positions and report cosine to
EMA target at those positions, plus per-patch retrieval and encoder retrieval.
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
from models.predictor_vjepa2 import VJEPAPredictor
from masks.multiblock3d import batched_mask, DEFAULT_SPECS


def effective_rank(x):
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x.float())
    p = (s ** 2) / (s ** 2).sum().clamp_min(1e-12)
    H = -(p * (p.clamp_min(1e-12).log())).sum()
    return torch.exp(H).item()


def retrieval(q, k, N, name=""):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    t1 = (r[:, :1] == gt).any(-1).float().mean().item()
    t5 = (r[:, :5] == gt).any(-1).float().mean().item()
    t10 = (r[:, :10] == gt).any(-1).float().mean().item()
    ranks = (r == gt).float().argmax(-1).float() + 1
    mr = ranks.mean().item(); medr = ranks.median().item()
    print(f"  {name:32s} top1={t1:.4f}  top5={t5:.4f}  top10={t10:.4f}  mr={mr:.1f}  medr={medr:.1f}")
    return {"top1": t1, "top5": t5, "top10": t10, "mr": mr, "medr": medr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mask_spec", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed); random.seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ckpt["args"]
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])

    # rebuild train/val anchors same as in trainer
    span = (a["num_frames"] - 1) * a["frame_stride"] + 1
    val_specs = []
    for fi in range(n_films):
        n_frames_film = cache["frames"][fi].shape[0]
        last_anchor = n_frames_film - span - 1
        train_end = int(last_anchor * 0.9)
        val_start = train_end + a["num_frames"] * a["frame_stride"]
        for av in range(val_start, last_anchor, a["clip_anchor_stride"]):
            val_specs.append((fi, av))
    N_va = len(val_specs)
    print(f"val clips N={N_va}")

    grid_d = a["num_frames"] // a["tubelet_size"]
    grid_s = a["img_size"] // a["patch_size"]
    n_tokens = grid_d * grid_s * grid_s

    encoder = VisionTransformer3D(
        num_frames=a["num_frames"], tubelet_size=a["tubelet_size"], patch_size=a["patch_size"],
        img_size=a["img_size"], embed_dim=a["embed_dim"], depth=a["depth"],
        num_heads=a["num_heads"], mlp_ratio=4.0,
    ).to(device).eval()
    target = VisionTransformer3D(
        num_frames=a["num_frames"], tubelet_size=a["tubelet_size"], patch_size=a["patch_size"],
        img_size=a["img_size"], embed_dim=a["embed_dim"], depth=a["depth"],
        num_heads=a["num_heads"], mlp_ratio=4.0,
    ).to(device).eval()
    predictor = VJEPAPredictor(
        embed_dim=a["embed_dim"], pred_embed_dim=a["pred_embed_dim"],
        pred_depth=a["pred_depth"], pred_num_heads=a["pred_num_heads"],
        grid_depth=grid_d, grid_size=grid_s, mlp_ratio=4.0, num_mask_tokens=2,
    ).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    target.load_state_dict(ckpt["target"])
    predictor.load_state_dict(ckpt["predictor"])
    for m in (encoder, target, predictor):
        for p in m.parameters(): p.requires_grad_(False)

    spec_list = [DEFAULT_SPECS[args.mask_spec]]

    # Per-clip metrics: cosine at masked positions (pred vs target), per-clip mean.
    cos_pred = []
    cos_ctx = []        # cosine between (online full clip @ masked positions) and target at same positions
    per_clip_pred_mean = []
    per_clip_tgt_mean = []
    per_clip_ctx_mean = []
    with torch.no_grad():
        for s in range(0, N_va, args.bs):
            specs = val_specs[s:s + args.bs]
            clip, fids = sample_clip_batch(cache, specs, a["num_frames"], a["frame_stride"], device)
            B = clip.shape[0]
            vis_idx, pred_idx = batched_mask(B, grid_d, grid_s, grid_s, spec_list, device=device)
            ctx_tokens = encoder(clip, keep_indices=vis_idx)        # [B, n_vis, D]
            tgt_all = target(clip)                                   # [B, N, D]
            D = tgt_all.shape[-1]
            tgt_at_pred = tgt_all.gather(1, pred_idx.unsqueeze(-1).expand(-1, -1, D))
            pred_at = predictor(ctx_tokens, vis_idx, pred_idx, mask_token_id=args.mask_spec)
            # cosine per-token, average over tokens, per-clip mean
            cos_t = F.cosine_similarity(pred_at, tgt_at_pred, dim=-1).mean(dim=-1)   # [B]
            cos_pred.extend(cos_t.cpu().tolist())
            # online-full as baseline
            ctx_all = encoder(clip)
            ctx_at_pred = ctx_all.gather(1, pred_idx.unsqueeze(-1).expand(-1, -1, D))
            cos_c = F.cosine_similarity(ctx_at_pred, tgt_at_pred, dim=-1).mean(dim=-1)
            cos_ctx.extend(cos_c.cpu().tolist())
            per_clip_pred_mean.append(pred_at.mean(1))
            per_clip_tgt_mean.append(tgt_all.mean(1))
            per_clip_ctx_mean.append(ctx_all.mean(1))

    cos_pred = torch.tensor(cos_pred)
    cos_ctx = torch.tensor(cos_ctx)
    pred_mean = torch.cat(per_clip_pred_mean, 0)
    tgt_mean = torch.cat(per_clip_tgt_mean, 0)
    ctx_mean = torch.cat(per_clip_ctx_mean, 0)

    print(f"\n=== mean cosine at masked positions (over {N_va} clips) ===")
    print(f"  predictor vs target  cos mean = {cos_pred.mean().item():+.4f}  median = {cos_pred.median().item():+.4f}")
    print(f"  online-ctx vs target cos mean = {cos_ctx.mean().item():+.4f}  median = {cos_ctx.median().item():+.4f}")

    print(f"\n=== retrieval (mean-pool per clip, may collapse) ===")
    N = pred_mean.shape[0]
    res = {}
    res["pred->tgt"] = retrieval(pred_mean, tgt_mean, N, "pred -> target")
    res["ctx->tgt"] = retrieval(ctx_mean, tgt_mean, N, "ctx -> target")
    res["tgt->tgt"] = retrieval(tgt_mean, tgt_mean, N, "tgt identity")

    print(f"\n=== collapse ===")
    print(f"  std(pred)={pred_mean.std(0).mean().item():.4f}  std(tgt)={tgt_mean.std(0).mean().item():.4f}  std(ctx)={ctx_mean.std(0).mean().item():.4f}")
    print(f"  eff_rank pred={effective_rank(pred_mean):.1f}  tgt={effective_rank(tgt_mean):.1f}  ctx={effective_rank(ctx_mean):.1f}")

    out = {
        "cos_pred_mean": cos_pred.mean().item(),
        "cos_pred_median": cos_pred.median().item(),
        "cos_ctx_mean": cos_ctx.mean().item(),
        "cos_ctx_median": cos_ctx.median().item(),
        "retrieval": res,
        "std_pred": pred_mean.std(0).mean().item(),
        "std_tgt": tgt_mean.std(0).mean().item(),
        "std_ctx": ctx_mean.std(0).mean().item(),
        "eff_rank_pred": effective_rank(pred_mean),
        "eff_rank_tgt": effective_rank(tgt_mean),
        "eff_rank_ctx": effective_rank(ctx_mean),
        "N_val": N, "mask_spec": args.mask_spec,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
