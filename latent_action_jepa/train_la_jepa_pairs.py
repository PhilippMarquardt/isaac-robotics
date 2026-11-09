"""F6: per-frame frozen ViT-S substrate + single-step + FSQ.

Substrate switch. V-JEPA-2 features are temporally smooth (the encoder is
trained to predict ahead, so motion gets compressed). For latent-action
discovery we want a substrate where consecutive frames are visibly distinct.
Plain per-frame ImageNet ViT-S is exactly that.

For each anchor t in a film, sample (frame_t, frame_{t+delta}). Encode each
frame to [196, 384] patch tokens. Run:

    z_cont = inverse(H_t, H_next)        in R^{action_dim}
    z_q    = FSQ(z_cont)                 K = prod(levels)
    H_pred = transition(H_t, z_q)

Loss: cosine(H_pred, H_next.detach()).

Eval the standard battery (oracle, zero_z, random_z, shuffle_z, time_shuffle_z
across pairs from the same clip, Z_only, wrong_H, copy_H).
"""
import argparse
import json
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.vjepa2_clips import IMAGENET_MEAN, IMAGENET_STD
from models.encoder import ViTEncoder
from models.la_modules import SingleStepInverse, SingleStepTransition, FSQBottleneck


def cosine_loss(p, t):
    pn = F.normalize(p, dim=-1); tn = F.normalize(t, dim=-1)
    return (2.0 - 2.0 * (pn * tn).sum(-1)).mean()


def retrieval(q, k, N, name=""):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    t1 = (r[:, :1] == gt).any(-1).float().mean().item()
    t5 = (r[:, :5] == gt).any(-1).float().mean().item()
    mr = (r == gt).float().argmax(-1).float().mean().item() + 1
    print(f"  {name:30s} top1={t1:.3f}  top5={t5:.3f}  mr={mr:.1f}")
    return {"top1": t1, "top5": t5, "mr": mr}


def per_pair_retrieval(pred_pool, tgt_pool, name=""):
    """[N, NPair, P, D] each. Per-pair retrieval at mean-pool over patches."""
    N, NP, P, D = pred_pool.shape
    rows = []
    for p in range(NP):
        q = F.normalize(pred_pool[:, p].mean(1), dim=-1)
        k = F.normalize(tgt_pool[:, p].mean(1), dim=-1)
        sim = q @ k.T
        r = sim.argsort(dim=-1, descending=True)
        gt = torch.arange(N, device=q.device).unsqueeze(-1)
        rows.append({
            "pair": p,
            "top1": (r[:, :1] == gt).any(-1).float().mean().item(),
            "mr": (r == gt).float().argmax(-1).float().mean().item() + 1,
        })
    avg = sum(r["top1"] for r in rows) / NP
    mr = sum(r["mr"] for r in rows) / NP
    print(f"  [{name:30s}]  per-pair top1: " + " ".join(f"{r['top1']:.3f}" for r in rows)
          + f"   avg={avg:.3f}  mr={mr:.1f}")
    return {"per_pair": rows, "avg_top1": avg, "avg_mr": mr}


# We sample NPair consecutive pairs per anchor: (a, a+delta), (a+delta, a+2*delta), ...
def sample_pairs(cache, anchors, num_pairs, delta, device):
    """anchors: list of (film_id, a). Returns frames tensor [B, NPair+1, 3, H, W]."""
    out = []
    for fi, a in anchors:
        frames = cache["frames"][fi]
        # cache stores frames as either dict {"data": ...} or tensor (depending on version);
        # vjepa2_clips uses direct tensor at frames[fi]
        if isinstance(frames, dict):
            data = frames["data"]
        else:
            data = frames
        idxs = [a + i * delta for i in range(num_pairs + 1)]
        clip = data[idxs]
        out.append(clip)
    clip = torch.stack(out, 0).to(device).float() / 255.0           # [B, NPair+1, 3, H, W]
    clip = (clip - IMAGENET_MEAN.squeeze(0).to(device).view(1, 1, 3, 1, 1)) / IMAGENET_STD.squeeze(0).to(device).view(1, 1, 3, 1, 1)
    return clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--encoder_name", default="vit_small_patch16_224")
    ap.add_argument("--num_pairs", type=int, default=7, help="consecutive pairs per anchor")
    ap.add_argument("--delta", type=int, default=8, help="frame distance per pair")
    ap.add_argument("--anchor_stride", type=int, default=8)
    ap.add_argument("--action_dim", type=int, default=128)
    ap.add_argument("--fsq_levels", type=int, nargs="+", default=[5, 5, 5, 5, 5])
    ap.add_argument("--inverse_depth", type=int, default=3)
    ap.add_argument("--transition_depth", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed); random.seed(args.seed)

    # frozen per-frame ViT-S
    enc = ViTEncoder(args.encoder_name, pretrained=True, layer_idx=None).to(device).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    state_dim = enc.embed_dim
    n_patch = 196
    NPair = args.num_pairs
    print(f"encoder: {args.encoder_name}  D={state_dim}  pairs/anchor={NPair}  delta={args.delta}")

    # modules
    inv = SingleStepInverse(state_dim=state_dim, action_dim=args.action_dim,
                             depth=args.inverse_depth).to(device)
    tr = SingleStepTransition(state_dim=state_dim, action_dim=args.action_dim,
                               n_patch=n_patch, depth=args.transition_depth).to(device)
    fsq = FSQBottleneck(action_dim=args.action_dim, levels=tuple(args.fsq_levels)).to(device)
    print(f"FSQ K={fsq.codebook_size}")
    params = list(inv.parameters()) + list(tr.parameters()) + list(fsq.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # data
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])
    span = NPair * args.delta
    train_specs, val_specs = [], []
    for fi in range(n_films):
        first = cache["frames"][fi] if isinstance(cache["frames"][fi], torch.Tensor) else cache["frames"][fi]["data"]
        n_frames = first.shape[0]
        last_anchor = n_frames - span - 1
        train_end = int(last_anchor * 0.9)
        val_start = train_end + span + 10
        for a in range(0, train_end, args.anchor_stride):
            train_specs.append((fi, a))
        for a in range(val_start, last_anchor, args.anchor_stride):
            val_specs.append((fi, a))
    N_tr, N_va = len(train_specs), len(val_specs)
    print(f"train anchors={N_tr}  val anchors={N_va}")

    def encode_pairs(specs):
        clip = sample_pairs(cache, specs, NPair, args.delta, device)             # [B, NP+1, 3, H, W]
        B = clip.shape[0]
        flat = clip.reshape(B * (NPair + 1), 3, 224, 224)
        with torch.no_grad():
            feats = enc(flat)                                                     # [B*(NP+1), P, D]
        feats = feats.reshape(B, NPair + 1, n_patch, state_dim)
        return feats

    # training
    for step in range(args.steps):
        idx = [random.randrange(N_tr) for _ in range(args.bs)]
        specs = [train_specs[i] for i in idx]
        H = encode_pairs(specs)                                                   # [B, NP+1, P, D]
        Ht = H[:, :NPair].reshape(-1, n_patch, state_dim)
        Hn = H[:, 1:].reshape(-1, n_patch, state_dim)
        z_cont = inv(Ht, Hn)
        z_q, codes = fsq(z_cont)
        H_pred = tr(Ht, z_q)
        L = cosine_loss(H_pred, Hn.detach())
        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                uniq = codes.unique().numel()
            print(f"  step {step:5d}  L={L.item():.4f}  codes_used={uniq}/{fsq.codebook_size}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    # eval
    inv.eval(); tr.eval(); fsq.eval()
    print(f"\n=== Eval val (N={N_va}, NPair={NPair}, delta={args.delta}) ===")
    all_Ht, all_Hn, all_Z, all_codes = [], [], [], []
    B_eval = 8
    with torch.no_grad():
        for s in range(0, N_va, B_eval):
            specs = val_specs[s:s + B_eval]
            H = encode_pairs(specs)
            B = H.shape[0]
            Ht = H[:, :NPair]                                                     # [B,NP,P,D]
            Hn = H[:, 1:]
            z_cont = inv(Ht.reshape(-1, n_patch, state_dim),
                          Hn.reshape(-1, n_patch, state_dim))
            z_q, codes = fsq(z_cont)
            z_q = z_q.reshape(B, NPair, args.action_dim)
            codes = codes.reshape(B, NPair)
            all_Ht.append(Ht); all_Hn.append(Hn); all_Z.append(z_q); all_codes.append(codes)
        Ht = torch.cat(all_Ht, 0); Hn = torch.cat(all_Hn, 0)
        Z = torch.cat(all_Z, 0); codes = torch.cat(all_codes, 0)
        N = Ht.shape[0]

        def runtest(Z_use, name):
            preds = tr(Ht.reshape(-1, n_patch, state_dim),
                       Z_use.reshape(-1, args.action_dim)).reshape(N, NPair, n_patch, state_dim)
            return per_pair_retrieval(preds, Hn, name)

        results = {}
        results["oracle"] = runtest(Z, "oracle z*")
        results["zero_Z"] = runtest(torch.zeros_like(Z), "zero_z")
        std_z = Z.flatten(0, 1).std(0)
        results["random_Z"] = runtest(torch.randn_like(Z) * std_z, "random_z")
        perm = torch.randperm(N, device=device)
        results["shuffle_Z_batch"] = runtest(Z[perm], "shuffle_z(batch)")
        time_perm = torch.randperm(NPair, device=device)
        results["time_shuffle_Z"] = runtest(Z[:, time_perm], "time_shuffle_z")
        H0 = torch.zeros_like(Ht).reshape(-1, n_patch, state_dim)
        preds = tr(H0, Z.reshape(-1, args.action_dim)).reshape(N, NPair, n_patch, state_dim)
        results["Z_only_H0"] = per_pair_retrieval(preds, Hn, "Z_only(H=0)")
        Ht_wrong = Ht[perm].reshape(-1, n_patch, state_dim)
        preds = tr(Ht_wrong, Z.reshape(-1, args.action_dim)).reshape(N, NPair, n_patch, state_dim)
        results["wrong_H"] = per_pair_retrieval(preds, Hn, "wrong_H")
        results["copy_H"] = per_pair_retrieval(Ht, Hn, "copy_H")

        # code stats
        K = fsq.codebook_size
        hist = torch.bincount(codes.reshape(-1), minlength=K).float()
        p = hist / hist.sum().clamp_min(1)
        ent = -(p * (p + 1e-12).log()).sum().item()
        uniq = (hist > 0).sum().item()
        max_ent = math.log(K)
        top10 = hist.topk(10).values.tolist()
        print(f"\ncode usage: unique={uniq}/{K}  entropy={ent:.2f}  max={max_ent:.2f}  "
              f"utilization={ent/max_ent:.3f}  top10_counts={top10}")
        results["code_unique"] = uniq
        results["code_entropy"] = ent
        results["code_max_entropy"] = max_ent

    torch.save({
        "inv": inv.state_dict(), "tr": tr.state_dict(), "fsq": fsq.state_dict(),
        "args": vars(args),
    }, os.path.join(args.out_dir, "ckpt.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"results": results, "N": int(N), "NPair": int(NPair), "args": vars(args)},
                  f, indent=2)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
