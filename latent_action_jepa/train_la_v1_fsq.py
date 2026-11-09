"""F7: V1 stage1 architecture (frozen ViT-S + FixedReducer + InverseActionEncoder
+ ResidualTransition) plus FSQ bottleneck on Z. Multi-film, single-step pair
(frame_t, frame_{t+delta}). delta=16 default to match V1.

z [B, action_tokens, action_dim] -> per-token FSQ -> z_q -> transition.

Loss: per-token cosine on H_pred vs reduced H_tp + InfoNCE on mean pool + Z VICReg.

Diagnostics: oracle/zero/random/shuffle/wrong_H/copy_Ht + per-token shuffle
+ Z code-usage stats + selectivity.
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
from models.fixed_reducer import FixedSpatialReducer
from models.inverse_encoder import InverseActionEncoder
from models.transition_residual import ResidualTransitionModel
from models.la_modules import FSQBottleneck


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = float(lam)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lam * grad_out, None


def grad_reverse(x, lam):
    return GradReverse.apply(x, lam)


def cosine_distance(pred, target):
    p = F.normalize(pred, dim=-1)
    t = F.normalize(target, dim=-1)
    return (2.0 - 2.0 * (p * t).sum(-1)).mean()


def retrieval(q, k, N, name=""):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    t1 = (r[:, :1] == gt).any(-1).float().mean().item()
    t5 = (r[:, :5] == gt).any(-1).float().mean().item()
    mr = (r == gt).float().argmax(-1).float().mean().item() + 1
    print(f"  {name:24s} top1={t1:.3f}  top5={t5:.3f}  mr={mr:.1f}")
    return {"top1": t1, "top5": t5, "mr": mr}


def sample_pair_batch(cache, anchors, delta, device):
    """anchors: list of (film_id, a). Returns (x_t, x_tp) tensors [B,3,H,W]."""
    xt, xtp = [], []
    for fi, a in anchors:
        frames = cache["frames"][fi]
        data = frames["data"] if isinstance(frames, dict) else frames
        xt.append(data[a])
        xtp.append(data[a + delta])
    xt = torch.stack(xt, 0).to(device).float() / 255.0
    xtp = torch.stack(xtp, 0).to(device).float() / 255.0
    m = IMAGENET_MEAN.squeeze(0).to(device).view(1, 3, 1, 1)
    s = IMAGENET_STD.squeeze(0).to(device).view(1, 3, 1, 1)
    xt = (xt - m) / s
    xtp = (xtp - m) / s
    return xt, xtp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--encoder_name", default="vit_small_patch16_224")
    ap.add_argument("--delta", type=int, default=16)
    ap.add_argument("--anchor_stride", type=int, default=4)
    # model
    ap.add_argument("--action_tokens", type=int, default=4)
    ap.add_argument("--action_dim", type=int, default=128)
    ap.add_argument("--state_tokens", type=int, default=64)
    ap.add_argument("--future_tokens", type=int, default=16)
    ap.add_argument("--inverse_depth", type=int, default=4)
    ap.add_argument("--transition_depth", type=int, default=6)
    ap.add_argument("--use_fsq", action="store_true", default=False)
    ap.add_argument("--fsq_levels", type=int, nargs="+", default=[5, 5, 5, 5, 5])
    ap.add_argument("--film_adv_lambda", type=float, default=0.0,
                     help="GRL coefficient on Z -> film head. 0 = disabled.")
    ap.add_argument("--delta_input", action="store_true",
                     help="feed inverse(H_tp - red_future(H_t)) instead of (H_t, H_tp)")
    # train
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--lambda_nce", type=float, default=1.0)
    ap.add_argument("--nce_tau", type=float, default=0.1)
    ap.add_argument("--lambda_z_l2", type=float, default=0.001)
    ap.add_argument("--lambda_z_var", type=float, default=0.05)
    ap.add_argument("--lambda_z_cov", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed); random.seed(args.seed)

    enc = ViTEncoder(args.encoder_name, pretrained=True, layer_idx=None).to(device).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    state_dim = enc.embed_dim
    red_state = FixedSpatialReducer(args.state_tokens).to(device)
    red_future = FixedSpatialReducer(args.future_tokens).to(device)

    inv = InverseActionEncoder(state_dim, args.action_dim, args.action_tokens,
                                depth=args.inverse_depth).to(device)
    tr = ResidualTransitionModel(state_dim, args.action_dim, args.future_tokens,
                                  depth=args.transition_depth).to(device)
    fsq = None
    if args.use_fsq:
        fsq = FSQBottleneck(action_dim=args.action_dim, levels=tuple(args.fsq_levels)).to(device)
        print(f"FSQ enabled, K={fsq.codebook_size}, levels={args.fsq_levels}")

    params = list(inv.parameters()) + list(tr.parameters())
    if fsq is not None:
        params += list(fsq.parameters())
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f} M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])
    film_head = None
    if args.film_adv_lambda > 0:
        film_head = nn.Linear(args.action_tokens * args.action_dim, n_films).to(device)
        params += list(film_head.parameters())
        print(f"film-adv head enabled, n_films={n_films}, lambda={args.film_adv_lambda}")
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    train_specs, val_specs = [], []
    film_ids_val = []
    for fi in range(n_films):
        data = cache["frames"][fi]["data"] if isinstance(cache["frames"][fi], dict) else cache["frames"][fi]
        n_frames = data.shape[0]
        last_anchor = n_frames - args.delta - 1
        train_end = int(last_anchor * 0.9)
        val_start = train_end + args.delta + 16
        for a in range(0, train_end, args.anchor_stride):
            train_specs.append((fi, a))
        for a in range(val_start, last_anchor, args.anchor_stride):
            val_specs.append((fi, a)); film_ids_val.append(fi)
    N_tr, N_va = len(train_specs), len(val_specs)
    print(f"train pairs={N_tr}  val pairs={N_va}  delta={args.delta}")

    def encode_pair(specs):
        xt, xtp = sample_pair_batch(cache, specs, args.delta, device)
        with torch.no_grad():
            Ht_raw = enc(xt)                # [B, 196, D]
            Htp_raw = enc(xtp)
            Ht = red_state(Ht_raw)            # [B, 64, D]
            Htp = red_future(Htp_raw)         # [B, 16, D]
        return Ht, Htp

    print("\n=== Train ===")
    for step in range(args.steps):
        idx = [random.randrange(N_tr) for _ in range(args.bs)]
        specs = [train_specs[i] for i in idx]
        film_ids = torch.tensor([fi for fi, _ in specs], dtype=torch.long, device=device)
        Ht, Htp = encode_pair(specs)
        if args.delta_input:
            Ht_red = red_future(Ht)
            diff = Htp - Ht_red
            zero_h = torch.zeros_like(Ht_red)
            Z = inv(zero_h, diff)
        else:
            Z = inv(Ht, Htp)
        if fsq is not None:
            B, Ka, A = Z.shape
            Z_q_flat, codes = fsq(Z.reshape(-1, A))
            Z_q = Z_q_flat.reshape(B, Ka, A)
        else:
            Z_q = Z; codes = None
        H_pred = tr(Ht, Z_q)
        L_t = cosine_distance(H_pred, Htp.detach())
        q = F.normalize(H_pred.mean(1), dim=-1)
        k = F.normalize(Htp.mean(1).detach(), dim=-1)
        logits = q @ k.T / args.nce_tau
        labels = torch.arange(q.shape[0], device=device)
        L_nce = 0.5 * (F.cross_entropy(logits, labels) +
                       F.cross_entropy(k @ q.T / args.nce_tau, labels))
        z_flat = Z.flatten(0, 1)
        L_l2 = (Z ** 2).mean()
        std = z_flat.std(0)
        L_var = F.relu(1 - std).mean()
        z_c = z_flat - z_flat.mean(0, keepdim=True)
        cov = (z_c.T @ z_c) / max(1, z_c.shape[0] - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        L_cov = (off ** 2).sum() / cov.shape[0]
        L_film = torch.zeros((), device=device)
        if film_head is not None:
            Z_rev = grad_reverse(Z.flatten(1), args.film_adv_lambda)
            film_logits = film_head(Z_rev)
            L_film = F.cross_entropy(film_logits, film_ids)
        loss = L_t + args.lambda_nce * L_nce + args.lambda_z_l2 * L_l2 \
               + args.lambda_z_var * L_var + args.lambda_z_cov * L_cov + L_film
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            extra = ""
            if codes is not None:
                with torch.no_grad():
                    uniq = codes.unique().numel()
                extra = f"  codes_used={uniq}/{fsq.codebook_size}"
            if film_head is not None:
                with torch.no_grad():
                    film_acc = (film_logits.argmax(-1) == film_ids).float().mean().item()
                extra += f"  L_film={L_film.item():.3f}  film_acc={film_acc:.3f}"
            print(f"  step {step:5d}  L_t={L_t.item():.4f}  L_nce={L_nce.item():.4f}  "
                  f"L_var={L_var.item():.4f}  alpha={tr.alpha.item():+.3f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}{extra}")

    # ---- Eval ----
    inv.eval(); tr.eval()
    if fsq is not None: fsq.eval()
    print(f"\n=== Eval val (N={N_va}) ===")
    all_Ht, all_Htp, all_Z, all_codes = [], [], [], []
    with torch.no_grad():
        for s in range(0, N_va, args.bs):
            specs = val_specs[s:s + args.bs]
            Ht, Htp = encode_pair(specs)
            if args.delta_input:
                Ht_red = red_future(Ht)
                Z = inv(torch.zeros_like(Ht_red), Htp - Ht_red)
            else:
                Z = inv(Ht, Htp)
            if fsq is not None:
                B, Ka, A = Z.shape
                Z_q_flat, codes = fsq(Z.reshape(-1, A))
                Z = Z_q_flat.reshape(B, Ka, A)
                all_codes.append(codes.reshape(B, Ka))
            all_Ht.append(Ht); all_Htp.append(Htp); all_Z.append(Z)
        Ht = torch.cat(all_Ht, 0); Htp = torch.cat(all_Htp, 0); Z = torch.cat(all_Z, 0)
        N = Ht.shape[0]

        def per_tok_retrieval(pred, name):
            p = F.normalize(pred, dim=-1).flatten(1)
            t = F.normalize(Htp, dim=-1).flatten(1)
            sim = p @ t.T
            r = sim.argsort(dim=-1, descending=True)
            gt = torch.arange(N, device=device).unsqueeze(-1)
            t1 = (r[:, :1] == gt).any(-1).float().mean().item()
            t5 = (r[:, :5] == gt).any(-1).float().mean().item()
            mr = (r == gt).float().argmax(-1).float().mean().item() + 1
            print(f"  pertok[{name:20s}] top1={t1:.3f}  top5={t5:.3f}  mr={mr:.1f}")
            return {"top1": t1, "top5": t5, "mr": mr}

        def mp_retrieval(pred, name):
            return retrieval(pred.mean(1), Htp.mean(1), N, "mp " + name)

        results = {}
        results["oracle_mp"] = mp_retrieval(tr(Ht, Z), "oracle")
        results["zero_mp"] = mp_retrieval(tr(Ht, torch.zeros_like(Z)), "zero")
        std_z = Z.flatten(0, 1).std(0)
        results["random_mp"] = mp_retrieval(tr(Ht, torch.randn_like(Z) * std_z), "random")
        perm = torch.randperm(N, device=device)
        results["shuffle_mp"] = mp_retrieval(tr(Ht, Z[perm]), "shuffle(batch)")
        if Z.shape[1] > 1:
            tperm = torch.randperm(Z.shape[1], device=device)
            results["token_shuffle_mp"] = mp_retrieval(tr(Ht, Z[:, tperm]), "token_shuffle")
        results["Z_only_mp"] = mp_retrieval(tr(torch.zeros_like(Ht), Z), "Z_only")
        results["wrong_H_mp"] = mp_retrieval(tr(Ht[perm], Z), "wrong_H")
        # copy_Ht broadcast / reduce
        Ht_red16 = red_future(Ht.reshape(-1, args.state_tokens, state_dim))
        results["copy_Ht_mp"] = mp_retrieval(Ht_red16, "copy_Ht")
        # ---- selectivity (same-Z vs diff-Z) ----
        K = min(24, N)
        n_states = min(48, N)
        donor_idx = torch.randperm(N, device=device,
                                    generator=torch.Generator(device=device).manual_seed(13))[:K]
        state_idx = torch.randperm(N, device=device,
                                    generator=torch.Generator(device=device).manual_seed(11))[:n_states]
        deltas = torch.zeros(K, n_states, args.future_tokens, state_dim, device=device)
        for di, d in enumerate(donor_idx):
            z = Z[d:d+1].expand(n_states, -1, -1)
            pred = tr(Ht[state_idx], z)
            base = red_future(Ht[state_idx])
            deltas[di] = pred - base
        df = F.normalize(deltas.reshape(K, n_states, -1), dim=-1)
        same_z = 0.0
        for di in range(K):
            sim = df[di] @ df[di].T
            same_z += (sim.sum() - torch.diag(sim).sum()).item() / (n_states * (n_states - 1))
        same_z /= K
        diff_z = 0.0
        for si in range(n_states):
            sim = df[:, si] @ df[:, si].T
            diff_z += (sim.sum() - torch.diag(sim).sum()).item() / (K * (K - 1))
        diff_z /= n_states
        sel = same_z / max(abs(diff_z), 1e-6)
        print(f"  selectivity: same-Z={same_z:+.4f}  diff-Z={diff_z:+.4f}  ratio={sel:+.2f}x")
        results["selectivity"] = {"same_z": same_z, "diff_z": diff_z, "ratio": sel}

        # code usage
        if all_codes:
            codes = torch.cat(all_codes, 0)
            K = fsq.codebook_size
            hist = torch.bincount(codes.reshape(-1), minlength=K).float()
            p = hist / hist.sum().clamp_min(1)
            ent = -(p * (p + 1e-12).log()).sum().item()
            uniq = (hist > 0).sum().item()
            max_ent = math.log(K)
            top10 = hist.topk(10).values.tolist()
            print(f"\ncode usage: unique={uniq}/{K}  entropy={ent:.2f}  max={max_ent:.2f}  util={ent/max_ent:.3f}  top10={top10}")
            results["code_unique"] = uniq
            results["code_entropy"] = ent

    save = {"inv": inv.state_dict(), "tr": tr.state_dict(), "args": vars(args)}
    if fsq is not None: save["fsq"] = fsq.state_dict()
    torch.save(save, os.path.join(args.out_dir, "ckpt.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"results": results, "N": int(N), "args": vars(args)}, f, indent=2)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
