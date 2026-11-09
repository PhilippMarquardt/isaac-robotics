"""V-JEPA baseline trainer.

Online ViT-S (trainable). EMA target ViT-S. VJEPAPredictor on context patch tokens.
Loss: smooth-L1 between predicted target tokens and stopgrad(EMA(target)) at masked positions.

Diagnostics:
  * online/target patch-feature variance & effective rank (collapse check)
  * predicted-target retrieval (mean-pool) vs copy baseline (mean-pool of last context frame)
  * per-film retrieval
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

from data.vjepa_data import load_clip_batch
from models.encoder import ViTEncoder
from models.predictor_vjepa import VJEPAPredictor


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
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    top10 = (r[:, :10] == gt).any(-1).float().mean().item()
    ranks = (r == gt).float().argmax(-1).float() + 1
    mr = ranks.mean().item()
    medr = ranks.median().item()
    if name: print(f"  {name:30s} top1={top1:.3f}  top5={top5:.3f}  top10={top10:.3f}  mr={mr:.1f}  medr={medr:.1f}")
    return {"top1": top1, "top5": top5, "top10": top10, "mr": mr, "medr": medr}


def pairwise_offdiag_cos(x):
    """Mean off-diagonal cosine over rows."""
    xn = F.normalize(x, dim=-1)
    sim = xn @ xn.T
    N = sim.shape[0]
    off_sum = (sim.sum() - torch.diagonal(sim).sum()).item()
    return off_sum / (N * (N - 1))


@torch.no_grad()
def ema_update(online, target, m):
    for po, pt in zip(online.parameters(), target.parameters()):
        pt.data.mul_(m).add_(po.data, alpha=1 - m)
    for bo, bt in zip(online.buffers(), target.buffers()):
        bt.data.copy_(bo.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="vjepa_data cache .pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--encoder_name", default="vit_small_patch16_224")
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--pred_depth", type=int, default=6)
    ap.add_argument("--pred_heads", type=int, default=6)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr_enc", type=float, default=1.5e-5)
    ap.add_argument("--lr_pred", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ema_start", type=float, default=0.99)
    ap.add_argument("--ema_end", type=float, default=0.999)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--loss", choices=["smoothl1", "cos"], default="smoothl1")
    ap.add_argument("--ctx_offsets", type=int, nargs="+", default=[-16, -8, 0],
                     help="time offsets used as context (e.g. -16 -8 0 for 3 frames)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(0)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    film_names = cache["film_names"]
    n_films = len(film_names)
    ctx_offsets = args.ctx_offsets
    n_ctx = len(ctx_offsets)
    cache_offsets = cache["ctx_offsets"]
    for o in ctx_offsets:
        assert o in cache_offsets, f"ctx_offset {o} not in cache (cache has {cache_offsets})"
    print(f"loaded cache: {n_films} films, ctx_offsets={ctx_offsets} (cache supports {cache_offsets})")

    train_specs = []
    val_specs = []
    for fi in range(n_films):
        for a in cache["train_anchors"][fi]:
            train_specs.append((fi, a))
        for a in cache["val_anchors"][fi]:
            val_specs.append((fi, a))
    N_tr, N_va = len(train_specs), len(val_specs)
    print(f"train {N_tr}  val {N_va}")

    # online + EMA target encoders
    online = ViTEncoder(args.encoder_name, pretrained=True, layer_idx=None).to(device).train()
    target = copy.deepcopy(online).eval()
    for p in target.parameters():
        p.requires_grad_(False)

    grid_h = grid_w = 14
    n_patch = grid_h * grid_w
    dim = online.embed_dim
    pred = VJEPAPredictor(dim=dim, depth=args.pred_depth, heads=args.pred_heads,
                          n_ctx_frames=n_ctx, grid_h=grid_h, grid_w=grid_w).to(device)
    print(f"params  online={sum(p.numel() for p in online.parameters())/1e6:.1f}M  pred={sum(p.numel() for p in pred.parameters())/1e6:.1f}M")

    opt = torch.optim.AdamW(
        [{"params": online.parameters(), "lr": args.lr_enc},
         {"params": pred.parameters(), "lr": args.lr_pred}],
        weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    def eval_val():
        online.eval(); target.eval(); pred.eval()
        with torch.no_grad():
            B_eval = 32
            ctx_means, tgt_means_target, pred_means_at_target = [], [], []
            ctx_flats, tgt_flats, pred_flats = [], [], []
            fids_all = []
            for s in range(0, N_va, B_eval):
                specs = val_specs[s:s+B_eval]
                ctx, tgt, fids = load_clip_batch(cache, specs, device, ctx_offsets=ctx_offsets)
                Bc = ctx.shape[0]
                ctx_flat = ctx.reshape(Bc * n_ctx, 3, 224, 224)
                ctx_feats = online(ctx_flat).reshape(Bc, n_ctx, n_patch, dim)
                mask_idx = torch.arange(n_patch, device=device).unsqueeze(0).expand(Bc, -1)
                pred_tgt = pred(ctx_feats, mask_idx)
                target_feats = target(tgt)
                ctx_means.append(ctx_feats[:, -1].mean(1))
                tgt_means_target.append(target_feats.mean(1))
                pred_means_at_target.append(pred_tgt.mean(1))
                ctx_flats.append(ctx_feats[:, -1].flatten(1))      # [Bc, n_patch*D]
                tgt_flats.append(target_feats.flatten(1))
                pred_flats.append(pred_tgt.flatten(1))
                fids_all.append(fids)
            ctx_mean = torch.cat(ctx_means, 0); tgt_mean = torch.cat(tgt_means_target, 0)
            pred_mean = torch.cat(pred_means_at_target, 0)
            ctx_flat = torch.cat(ctx_flats, 0); tgt_flat = torch.cat(tgt_flats, 0)
            pred_flat = torch.cat(pred_flats, 0)
            print(f"\n[eval]  N={N_va}  ctx_offsets={ctx_offsets}")
            res = {}
            res["pred_mean->tgt"] = retrieval(pred_mean, tgt_mean, N_va, "pred mean -> tgt mean")
            res["copy_mean->tgt"] = retrieval(ctx_mean, tgt_mean, N_va, "copy mean -> tgt mean")
            res["pred_flat->tgt"] = retrieval(pred_flat, tgt_flat, N_va, "pred flat -> tgt flat")
            res["copy_flat->tgt"] = retrieval(ctx_flat, tgt_flat, N_va, "copy flat -> tgt flat")
            res["tgt->tgt"] = retrieval(tgt_mean, tgt_mean, N_va, "tgt identity (mean)")
            res["tgt_flat->tgt_flat"] = retrieval(tgt_flat, tgt_flat, N_va, "tgt identity (flat)")
            # collapse
            std_pred = pred_mean.std(0).mean().item()
            std_tgt = tgt_mean.std(0).mean().item()
            er_pred = effective_rank(pred_mean)
            er_tgt = effective_rank(tgt_mean)
            er_pred_flat = effective_rank(pred_flat)
            er_tgt_flat = effective_rank(tgt_flat)
            off_pred = pairwise_offdiag_cos(pred_mean)
            off_tgt = pairwise_offdiag_cos(tgt_mean)
            print(f"  std(pred)={std_pred:.4f}  std(tgt)={std_tgt:.4f}")
            print(f"  eff_rank(pred,mean)={er_pred:.1f}  eff_rank(tgt,mean)={er_tgt:.1f}  eff_rank(pred,flat)={er_pred_flat:.1f}  eff_rank(tgt,flat)={er_tgt_flat:.1f}")
            print(f"  pairwise off-diag cos: pred={off_pred:+.3f}  tgt={off_tgt:+.3f}")
            res.update({"std_pred": std_pred, "std_tgt": std_tgt,
                          "eff_rank_pred_mean": er_pred, "eff_rank_tgt_mean": er_tgt,
                          "eff_rank_pred_flat": er_pred_flat, "eff_rank_tgt_flat": er_tgt_flat,
                          "pairwise_offdiag_cos_pred": off_pred, "pairwise_offdiag_cos_tgt": off_tgt})
        online.train(); pred.train()
        return res

    # initial baseline (no training): copy baseline already meaningful
    print("\n=== Eval pre-train ===")
    pre_eval = eval_val()

    online.train(); pred.train()
    for step in range(args.steps):
        idx = [random.randrange(N_tr) for _ in range(args.bs)]
        specs = [train_specs[i] for i in idx]
        ctx, tgt, fids = load_clip_batch(cache, specs, device, ctx_offsets=ctx_offsets)
        B = ctx.shape[0]

        # encode context with ONLINE
        ctx_flat = ctx.reshape(B * n_ctx, 3, 224, 224)
        ctx_feats = online(ctx_flat).reshape(B, n_ctx, n_patch, dim)
        # target encoded with EMA TARGET
        with torch.no_grad():
            target_feats = target(tgt)                                    # [B, n_patch, dim]

        # sample mask indices per row
        n_mask = max(1, int(n_patch * args.mask_ratio))
        mask_idx = torch.argsort(torch.rand(B, n_patch, device=device), dim=-1)[:, :n_mask]
        target_at_mask = target_feats.gather(1, mask_idx.unsqueeze(-1).expand(-1, -1, dim))

        # predict
        pred_at_mask = pred(ctx_feats, mask_idx)

        if args.loss == "smoothl1":
            loss = F.smooth_l1_loss(pred_at_mask, target_at_mask.detach())
        else:
            loss = (1 - F.cosine_similarity(pred_at_mask, target_at_mask.detach(), dim=-1)).mean()

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(online.parameters()) + list(pred.parameters()), 1.0)
        opt.step(); sched.step()

        # EMA momentum: 0.99 -> 0.999 linear
        prog = step / max(1, args.steps - 1)
        m = args.ema_start + (args.ema_end - args.ema_start) * prog
        ema_update(online, target, m)

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                std_p = pred_at_mask.std().item()
                std_t = target_at_mask.std().item()
            print(f"  step {step:5d}  L={loss.item():.4f}  std(pred)={std_p:.3f}  std(tgt)={std_t:.3f}  ema={m:.4f}  lr_enc={opt.param_groups[0]['lr']:.2e}")

        if args.eval_every and step > 0 and step % args.eval_every == 0:
            _ = eval_val()
            online.train(); pred.train()

    print("\n=== Eval post-train ===")
    post_eval = eval_val()

    # save
    torch.save({
        "online": online.state_dict(),
        "target": target.state_dict(),
        "pred": pred.state_dict(),
        "args": vars(args),
        "config": {"dim": dim, "grid_h": grid_h, "grid_w": grid_w, "n_ctx_frames": n_ctx},
    }, os.path.join(args.out_dir, "ckpt.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"pre": pre_eval, "post": post_eval, "args": vars(args)}, f, indent=2,
                  default=lambda o: list(o) if isinstance(o, tuple) else float(o))
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
