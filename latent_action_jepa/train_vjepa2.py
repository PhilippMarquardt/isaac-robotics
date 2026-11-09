"""V-JEPA-2 video pretraining (action-free). Small-scale (ViT-S) reproduction
of facebookresearch/vjepa2 app/vjepa/train.py with same: 3D tubelet patch
embed, 3D sin-cos pos embed, multi-block 3D mask collator, L1 loss between
predictor and EMA target encoder at masked positions, Adam + warmup, EMA
momentum schedule.

Differences from the official:
  * ViT-S (384/12/6) instead of ViT-L (1024/24/16) — VRAM scale.
  * Predictor depth 12, embed 384, heads 12 — same as official (matches at 384).
  * Single mask spec per step alternates between the 2 specs (round-robin).
  * No distributed wrapping (single-GPU).
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
from masks.multiblock3d import batched_mask, DEFAULT_SPECS


def effective_rank(x):
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x.float())
    p = (s ** 2) / (s ** 2).sum().clamp_min(1e-12)
    H = -(p * (p.clamp_min(1e-12).log())).sum()
    return torch.exp(H).item()


def warmup_cosine(step, warmup, total, lr_start, lr_peak, lr_end=0.0):
    if step < warmup:
        return lr_start + (lr_peak - lr_start) * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_end + 0.5 * (lr_peak - lr_end) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def ema_update(online, target, m):
    for po, pt in zip(online.parameters(), target.parameters()):
        pt.data.mul_(m).add_(po.data, alpha=1 - m)
    for bo, bt in zip(online.buffers(), target.buffers()):
        bt.data.copy_(bo.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out_dir", required=True)
    # clip
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--frame_stride", type=int, default=2)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--clip_anchor_stride", type=int, default=4)
    # model
    ap.add_argument("--tubelet_size", type=int, default=2)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--embed_dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=6)
    ap.add_argument("--pred_embed_dim", type=int, default=384)
    ap.add_argument("--pred_depth", type=int, default=12)
    ap.add_argument("--pred_num_heads", type=int, default=12)
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
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed); random.seed(args.seed)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    film_names = cache["film_names"]
    n_films = len(film_names)
    print(f"loaded cache: {n_films} films")

    # Build anchor list: per film, anchors at clip_anchor_stride; clip needs frames
    # at positions [a, a + frame_stride, ..., a + (num_frames-1)*frame_stride].
    span = (args.num_frames - 1) * args.frame_stride + 1
    train_specs = []
    val_specs = []
    for fi in range(n_films):
        n_frames_film = cache["frames"][fi].shape[0]
        last_anchor = n_frames_film - span - 1
        train_end = int(last_anchor * 0.9)
        val_start = train_end + args.num_frames * args.frame_stride  # gap
        for a in range(0, train_end, args.clip_anchor_stride):
            train_specs.append((fi, a))
        for a in range(val_start, last_anchor, args.clip_anchor_stride):
            val_specs.append((fi, a))
    N_tr = len(train_specs); N_va = len(val_specs)
    print(f"train clips={N_tr}  val clips={N_va}  span_frames={span}")

    grid_d = args.num_frames // args.tubelet_size
    grid_s = args.img_size // args.patch_size
    n_tokens = grid_d * grid_s * grid_s
    print(f"grid: T_tube={grid_d}  H={grid_s}  W={grid_s}  total_tokens={n_tokens}")

    encoder = VisionTransformer3D(
        num_frames=args.num_frames, tubelet_size=args.tubelet_size, patch_size=args.patch_size,
        img_size=args.img_size, embed_dim=args.embed_dim, depth=args.depth,
        num_heads=args.num_heads, mlp_ratio=4.0,
    ).to(device).train()
    target = copy.deepcopy(encoder).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    predictor = VJEPAPredictor(
        embed_dim=args.embed_dim, pred_embed_dim=args.pred_embed_dim,
        pred_depth=args.pred_depth, pred_num_heads=args.pred_num_heads,
        grid_depth=grid_d, grid_size=grid_s, mlp_ratio=4.0, num_mask_tokens=2,
    ).to(device).train()
    print(f"params  enc={sum(p.numel() for p in encoder.parameters())/1e6:.1f}M  "
          f"pred={sum(p.numel() for p in predictor.parameters())/1e6:.1f}M")

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr_peak, weight_decay=args.wd, betas=(0.9, 0.999))

    def set_lr(lr):
        for g in opt.param_groups:
            g["lr"] = lr

    for step in range(args.steps):
        idx = [random.randrange(N_tr) for _ in range(args.bs)]
        specs = [train_specs[i] for i in idx]
        clip, fids = sample_clip_batch(cache, specs, args.num_frames, args.frame_stride, device)

        # alternate between mask spec A and B
        spec_id = step % 2
        spec_list = [DEFAULT_SPECS[spec_id]]
        vis_idx, pred_idx = batched_mask(args.bs, grid_d, grid_s, grid_s, spec_list, device=device)

        # encoder sees only visible positions
        ctx_tokens = encoder(clip, keep_indices=vis_idx)            # [B, n_vis, D]
        with torch.no_grad():
            tgt_all = target(clip)                                   # [B, n_tokens, D]
            B, _, D = tgt_all.shape
            tgt_at_pred = tgt_all.gather(
                1, pred_idx.unsqueeze(-1).expand(-1, -1, D))

        pred_at_pred = predictor(ctx_tokens, vis_idx, pred_idx, mask_token_id=spec_id)

        # L1 ** loss_exp
        loss = torch.mean(torch.abs(pred_at_pred - tgt_at_pred.detach()) ** args.loss_exp) / args.loss_exp

        lr = warmup_cosine(step, args.warmup, args.steps, args.lr_start, args.lr_peak, args.lr_end)
        set_lr(lr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), 1.0)
        opt.step()

        m = args.ema_start + (args.ema_end - args.ema_start) * step / max(1, args.steps - 1)
        ema_update(encoder, target, m)

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                std_p = pred_at_pred.std().item()
                std_t = tgt_at_pred.std().item()
            print(f"  step {step:5d}  L={loss.item():.4f}  std(pred)={std_p:.3f}  std(tgt)={std_t:.3f}  "
                  f"mask=spec{spec_id}  n_vis={vis_idx.shape[1]}  n_pred={pred_idx.shape[1]}  "
                  f"lr={lr:.2e}  ema={m:.5f}")

    # ---- Eval: retrieval of predicted target vs true target via mean-pool ----
    encoder.eval(); target.eval(); predictor.eval()
    print(f"\n=== Eval val (N={N_va}) ===")
    B_eval = 4
    pred_means, tgt_means, ctx_means, fids_all = [], [], [], []
    with torch.no_grad():
        for s in range(0, N_va, B_eval):
            specs = val_specs[s:s + B_eval]
            clip, fids = sample_clip_batch(cache, specs, args.num_frames, args.frame_stride, device)
            B = clip.shape[0]
            # encode full clip with target encoder
            tgt_all = target(clip)                                   # [B, N, D]
            # encoder also encodes full clip (no masking)
            ctx_all = encoder(clip)                                  # [B, N, D]
            # predictor: feed all encoder tokens as context, predict ALL tokens
            full_idx = torch.arange(n_tokens, device=device).unsqueeze(0).expand(B, -1)
            pred_all = predictor(ctx_all, full_idx, full_idx, mask_token_id=0)
            tgt_means.append(tgt_all.mean(1))
            ctx_means.append(ctx_all.mean(1))
            pred_means.append(pred_all.mean(1))
            fids_all.append(fids)
    tgt_mean = torch.cat(tgt_means, 0); ctx_mean = torch.cat(ctx_means, 0)
    pred_mean = torch.cat(pred_means, 0); fids = torch.cat(fids_all, 0)

    def retrieval(q, k, N, name):
        qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
        sim = qn @ kn.T
        r = sim.argsort(dim=-1, descending=True)
        gt = torch.arange(N, device=q.device).unsqueeze(-1)
        t1 = (r[:, :1] == gt).any(-1).float().mean().item()
        t5 = (r[:, :5] == gt).any(-1).float().mean().item()
        t10 = (r[:, :10] == gt).any(-1).float().mean().item()
        ranks = (r == gt).float().argmax(-1).float() + 1
        mr = ranks.mean().item(); medr = ranks.median().item()
        print(f"  {name:24s}  top1={t1:.4f}  top5={t5:.4f}  top10={t10:.4f}  mr={mr:.1f}  medr={medr:.1f}")
        return {"top1": t1, "top5": t5, "top10": t10, "mr": mr, "medr": medr}

    res = {}
    res["pred->tgt"] = retrieval(pred_mean, tgt_mean, len(fids), "pred -> target")
    res["ctx->tgt"] = retrieval(ctx_mean, tgt_mean, len(fids), "ctx (online) -> target")
    res["tgt->tgt"] = retrieval(tgt_mean, tgt_mean, len(fids), "target identity")
    res["std_pred"] = pred_mean.std(0).mean().item()
    res["std_tgt"] = tgt_mean.std(0).mean().item()
    res["eff_rank_pred"] = effective_rank(pred_mean)
    res["eff_rank_tgt"] = effective_rank(tgt_mean)
    print(f"  std(pred)={res['std_pred']:.4f}  std(tgt)={res['std_tgt']:.4f}  "
          f"eff_rank(pred)={res['eff_rank_pred']:.1f}  eff_rank(tgt)={res['eff_rank_tgt']:.1f}")

    torch.save({"encoder": encoder.state_dict(), "target": target.state_dict(),
                 "predictor": predictor.state_dict(), "args": vars(args)},
               os.path.join(args.out_dir, "ckpt.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"post": res, "args": vars(args)}, f, indent=2)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
