"""F5: single-step Genie-style world model with FSQ bottleneck on V-JEPA-2 features.

Architecture
------------
Encoder (frozen V-JEPA-2): a 16-frame clip -> 8 tubelet slots × 196 patches.
For each clip, form 7 consecutive (H_step_i, H_step_{i+1}) pairs.

For each pair:
    z_cont = inverse(H_step_i, H_step_{i+1})            in R^{action_dim}
    z_q    = FSQ_bottleneck(z_cont)                     discrete code
    H_pred = transition(H_step_i, z_q)                  -> R^{P, D}

Loss = cosine(H_pred, H_step_{i+1}.detach()) averaged over the 7 pairs.

Diagnostics (val):
    oracle z*               correct per-pair z         <-- expect high
    zero_z, random_z         z replaced                 <-- expect low
    shuffle_z (batch)        z from random other anchor <-- expect low
    time_shuffle_z (in clip) z permuted across 7 pairs  <-- expect low (since
                                                           each z is bound
                                                           to its specific
                                                           pair)
    Z_only (H=0)
    wrong_H (donor)
    copy_H (broadcast H_t -- already done at step level)
    code usage histogram (entropy)
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
from models.la_modules import SingleStepInverse, SingleStepTransition, FSQBottleneck


def cosine_loss(p, t):
    pn = F.normalize(p, dim=-1)
    tn = F.normalize(t, dim=-1)
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
    """pred_pool, tgt_pool: [N, NPair, P, D]. Per-pair retrieval averaged."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--vjepa_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--frame_stride", type=int, default=2)
    ap.add_argument("--anchor_stride", type=int, default=4)
    ap.add_argument("--action_dim", type=int, default=128)
    ap.add_argument("--fsq_levels", type=int, nargs="+", default=[5, 5, 5, 5, 5])
    ap.add_argument("--inverse_depth", type=int, default=3)
    ap.add_argument("--transition_depth", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
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
    grid_d = va["num_frames"] // va["tubelet_size"]                  # 8
    grid_s = va["img_size"] // va["patch_size"]                       # 14
    n_patch = grid_s * grid_s                                         # 196
    n_tokens = grid_d * n_patch
    state_dim = va["embed_dim"]
    NPair = grid_d - 1                                                # 7
    print(f"frozen V-JEPA-2: grid_d={grid_d}  P={n_patch}  D={state_dim}  pairs/clip={NPair}")

    # ---- modules ----
    inv = SingleStepInverse(state_dim=state_dim, action_dim=args.action_dim,
                             depth=args.inverse_depth).to(device)
    tr = SingleStepTransition(state_dim=state_dim, action_dim=args.action_dim,
                               n_patch=n_patch, depth=args.transition_depth).to(device)
    fsq = FSQBottleneck(action_dim=args.action_dim, levels=tuple(args.fsq_levels)).to(device)
    print(f"FSQ codebook size = {fsq.codebook_size}  (dim={fsq.fsq.D})")
    params = list(inv.parameters()) + list(tr.parameters()) + list(fsq.parameters())
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f} M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # ---- data ----
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])
    span = (args.num_frames - 1) * args.frame_stride + 1
    train_specs, val_specs = [], []
    for fi in range(n_films):
        n_frames_film = cache["frames"][fi].shape[0]
        last_anchor = n_frames_film - span - 1
        train_end = int(last_anchor * 0.9)
        val_start = train_end + args.num_frames * args.frame_stride
        for a in range(0, train_end, args.anchor_stride):
            train_specs.append((fi, a))
        for a in range(val_start, last_anchor, args.anchor_stride):
            val_specs.append((fi, a))
    N_tr, N_va = len(train_specs), len(val_specs)
    print(f"train clips={N_tr}  val clips={N_va}")

    def encode_clip(specs):
        clip, _ = sample_clip_batch(cache, specs, args.num_frames, args.frame_stride, device)
        with torch.no_grad():
            H_all = enc(clip)                                          # [B, n_tokens, D]
        # reshape to [B, grid_d, P, D]
        return H_all.reshape(-1, grid_d, n_patch, state_dim)

    # ---- training ----
    for step in range(args.steps):
        idx = [random.randrange(N_tr) for _ in range(args.bs)]
        specs = [train_specs[i] for i in idx]
        H_tubes = encode_clip(specs)                                   # [B, 8, P, D]
        # build (H_t, H_next) pairs, flatten across B and pairs
        Ht = H_tubes[:, :NPair].reshape(-1, n_patch, state_dim)        # [B*NPair, P, D]
        Hn = H_tubes[:, 1:].reshape(-1, n_patch, state_dim)
        z_cont = inv(Ht, Hn)                                            # [B*NPair, action_dim]
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

    # ---- eval ----
    inv.eval(); tr.eval(); fsq.eval()
    print(f"\n=== Eval val (N={N_va}, NPair={NPair}) ===")
    all_Ht, all_Hn, all_Z = [], [], []
    all_codes = []
    B_eval = 4
    with torch.no_grad():
        for s in range(0, N_va, B_eval):
            specs = val_specs[s:s + B_eval]
            H_tubes = encode_clip(specs)                                # [B,8,P,D]
            B = H_tubes.shape[0]
            Ht = H_tubes[:, :NPair]                                     # [B,NPair,P,D]
            Hn = H_tubes[:, 1:]
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
        # H=0
        H0 = torch.zeros_like(Ht).reshape(-1, n_patch, state_dim)
        z_flat = Z.reshape(-1, args.action_dim)
        preds = tr(H0, z_flat).reshape(N, NPair, n_patch, state_dim)
        results["Z_only_H0"] = per_pair_retrieval(preds, Hn, "Z_only(H=0)")
        # wrong_H
        Ht_wrong = Ht[perm].reshape(-1, n_patch, state_dim)
        preds = tr(Ht_wrong, z_flat).reshape(N, NPair, n_patch, state_dim)
        results["wrong_H"] = per_pair_retrieval(preds, Hn, "wrong_H")
        # copy_H baseline (no transition, just predict H_t == H_t+1)
        results["copy_H"] = per_pair_retrieval(Ht, Hn, "copy_H")

        # ---- code usage histogram ----
        K = fsq.codebook_size
        hist = torch.bincount(codes.reshape(-1), minlength=K).float()
        p = hist / hist.sum().clamp_min(1)
        ent = -(p * (p + 1e-12).log()).sum().item()
        uniq = (hist > 0).sum().item()
        max_ent = math.log(K)
        print(f"\ncode usage: unique={uniq}/{K}  entropy={ent:.2f}  max={max_ent:.2f}  utilization={ent/max_ent:.3f}")
        results["code_unique"] = uniq
        results["code_entropy"] = ent
        results["code_max_entropy"] = max_ent

    torch.save({
        "inv": inv.state_dict(), "tr": tr.state_dict(), "fsq": fsq.state_dict(),
        "args": vars(args), "va_args": va,
    }, os.path.join(args.out_dir, "ckpt.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"results": results, "N": int(N), "NPair": int(NPair), "args": vars(args)},
                  f, indent=2)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
