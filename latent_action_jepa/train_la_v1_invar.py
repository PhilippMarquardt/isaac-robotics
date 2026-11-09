"""F10: V1 stage1 architecture + augmentation-invariance loss on Z.

Per anchor, sample TWO views by applying the SAME random spatial transform
(crop + horizontal flip + color jitter) to BOTH frame_t and frame_{t+delta}.
The pair's motion content is preserved, but the appearance changes.

Loss:
    L = L_t + L_nce + lambda_invar * (1 - cos(Z_view1, Z_view2))

If lambda_invar > 0, Z is pushed toward view-invariance, which should
suppress appearance/film leakage in Z and keep motion.
"""
import argparse
import json
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from data.vjepa2_clips import IMAGENET_MEAN, IMAGENET_STD
from models.encoder import ViTEncoder
from models.fixed_reducer import FixedSpatialReducer
from models.inverse_encoder import InverseActionEncoder
from models.transition_residual import ResidualTransitionModel


def cosine_distance(pred, target):
    p = F.normalize(pred, dim=-1)
    t = F.normalize(target, dim=-1)
    return (2.0 - 2.0 * (p * t).sum(-1)).mean()


def make_view(xt_u8, xtp_u8, size=224, crop_min=0.5, crop_max=1.0, jitter=True):
    """Same spatial+flip transform applied to xt and xtp identically.

    xt_u8, xtp_u8: uint8 [B, 3, H, W] tensors.
    Returns float [B,3,size,size] each (ImageNet-normalized).
    """
    B, C, H, W = xt_u8.shape
    out_t = torch.empty(B, C, size, size, device=xt_u8.device, dtype=torch.float32)
    out_tp = torch.empty_like(out_t)
    for i in range(B):
        s = random.uniform(crop_min, crop_max)
        ch = int(round(H * s)); cw = int(round(W * s))
        top = random.randint(0, H - ch); left = random.randint(0, W - cw)
        a = xt_u8[i, :, top:top + ch, left:left + cw].float()
        b = xtp_u8[i, :, top:top + ch, left:left + cw].float()
        a = TF.resize(a, [size, size], antialias=True)
        b = TF.resize(b, [size, size], antialias=True)
        if random.random() < 0.5:
            a = torch.flip(a, dims=[-1]); b = torch.flip(b, dims=[-1])
        if jitter and random.random() < 0.8:
            bright = 0.8 + 0.4 * random.random()
            a = a * bright; b = b * bright
        out_t[i] = a; out_tp[i] = b
    out_t = out_t / 255.0; out_tp = out_tp / 255.0
    m = IMAGENET_MEAN.squeeze(0).to(xt_u8.device).view(1, 3, 1, 1)
    s = IMAGENET_STD.squeeze(0).to(xt_u8.device).view(1, 3, 1, 1)
    return (out_t - m) / s, (out_tp - m) / s


def sample_pair_uint8(cache, anchors, delta, device):
    xt, xtp = [], []
    for fi, a in anchors:
        frames = cache["frames"][fi]
        data = frames["data"] if isinstance(frames, dict) else frames
        xt.append(data[a]); xtp.append(data[a + delta])
    return torch.stack(xt, 0).to(device), torch.stack(xtp, 0).to(device)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--encoder_name", default="vit_small_patch16_224")
    ap.add_argument("--delta", type=int, default=16)
    ap.add_argument("--anchor_stride", type=int, default=4)
    ap.add_argument("--action_tokens", type=int, default=4)
    ap.add_argument("--action_dim", type=int, default=128)
    ap.add_argument("--state_tokens", type=int, default=64)
    ap.add_argument("--future_tokens", type=int, default=16)
    ap.add_argument("--inverse_depth", type=int, default=4)
    ap.add_argument("--transition_depth", type=int, default=6)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--lambda_nce", type=float, default=1.0)
    ap.add_argument("--nce_tau", type=float, default=0.1)
    ap.add_argument("--lambda_z_l2", type=float, default=0.001)
    ap.add_argument("--lambda_z_var", type=float, default=0.05)
    ap.add_argument("--lambda_z_cov", type=float, default=0.01)
    ap.add_argument("--lambda_invar", type=float, default=0.5)
    ap.add_argument("--crop_min", type=float, default=0.6)
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

    params = list(inv.parameters()) + list(tr.parameters())
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f} M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])
    train_specs, val_specs = [], []
    for fi in range(n_films):
        data = cache["frames"][fi]["data"] if isinstance(cache["frames"][fi], dict) else cache["frames"][fi]
        n_frames = data.shape[0]
        last_anchor = n_frames - args.delta - 1
        train_end = int(last_anchor * 0.9)
        val_start = train_end + args.delta + 16
        for a in range(0, train_end, args.anchor_stride):
            train_specs.append((fi, a))
        for a in range(val_start, last_anchor, args.anchor_stride):
            val_specs.append((fi, a))
    N_tr, N_va = len(train_specs), len(val_specs)
    print(f"train pairs={N_tr}  val pairs={N_va}  lambda_invar={args.lambda_invar}")

    for step in range(args.steps):
        idx = [random.randrange(N_tr) for _ in range(args.bs)]
        specs = [train_specs[i] for i in idx]
        xt_u8, xtp_u8 = sample_pair_uint8(cache, specs, args.delta, device)
        # two augmented views
        xt1, xtp1 = make_view(xt_u8, xtp_u8, crop_min=args.crop_min)
        xt2, xtp2 = make_view(xt_u8, xtp_u8, crop_min=args.crop_min)
        with torch.no_grad():
            Ht1 = red_state(enc(xt1)); Htp1 = red_future(enc(xtp1))
            Ht2 = red_state(enc(xt2)); Htp2 = red_future(enc(xtp2))
        Z1 = inv(Ht1, Htp1)
        Z2 = inv(Ht2, Htp2)
        H_pred1 = tr(Ht1, Z1)
        H_pred2 = tr(Ht2, Z2)
        L_t = 0.5 * (cosine_distance(H_pred1, Htp1.detach()) +
                     cosine_distance(H_pred2, Htp2.detach()))
        # NCE on view 1 (matches retrieval at eval)
        q = F.normalize(H_pred1.mean(1), dim=-1)
        k = F.normalize(Htp1.mean(1).detach(), dim=-1)
        logits = q @ k.T / args.nce_tau
        labels = torch.arange(q.shape[0], device=device)
        L_nce = 0.5 * (F.cross_entropy(logits, labels) +
                       F.cross_entropy(k @ q.T / args.nce_tau, labels))
        # invariance
        L_invar = (1.0 - F.cosine_similarity(Z1.flatten(1), Z2.flatten(1), dim=-1)).mean()
        # VICReg
        Z = torch.cat([Z1, Z2], dim=0)
        z_flat = Z.flatten(0, 1)
        L_l2 = (Z ** 2).mean()
        std = z_flat.std(0)
        L_var = F.relu(1 - std).mean()
        z_c = z_flat - z_flat.mean(0, keepdim=True)
        cov = (z_c.T @ z_c) / max(1, z_c.shape[0] - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        L_cov = (off ** 2).sum() / cov.shape[0]
        loss = L_t + args.lambda_nce * L_nce + args.lambda_z_l2 * L_l2 \
               + args.lambda_z_var * L_var + args.lambda_z_cov * L_cov \
               + args.lambda_invar * L_invar
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  L_t={L_t.item():.4f}  L_nce={L_nce.item():.4f}  "
                  f"L_invar={L_invar.item():.4f}  alpha={tr.alpha.item():+.3f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    # ---- Eval ----
    inv.eval(); tr.eval()
    print(f"\n=== Eval val (N={N_va}, no aug) ===")
    all_Ht, all_Htp, all_Z = [], [], []
    with torch.no_grad():
        for s in range(0, N_va, args.bs):
            specs = val_specs[s:s + args.bs]
            xt_u8, xtp_u8 = sample_pair_uint8(cache, specs, args.delta, device)
            m = IMAGENET_MEAN.squeeze(0).to(device).view(1, 3, 1, 1)
            stdimg = IMAGENET_STD.squeeze(0).to(device).view(1, 3, 1, 1)
            xt = (xt_u8.float() / 255.0 - m) / stdimg
            xtp = (xtp_u8.float() / 255.0 - m) / stdimg
            Ht = red_state(enc(xt)); Htp = red_future(enc(xtp))
            Z = inv(Ht, Htp)
            all_Ht.append(Ht); all_Htp.append(Htp); all_Z.append(Z)
        Ht = torch.cat(all_Ht, 0); Htp = torch.cat(all_Htp, 0); Z = torch.cat(all_Z, 0)
        N = Ht.shape[0]

        def mp_r(pred, name):
            return retrieval(pred.mean(1), Htp.mean(1), N, name)

        results = {}
        results["oracle"] = mp_r(tr(Ht, Z), "oracle")
        results["zero"] = mp_r(tr(Ht, torch.zeros_like(Z)), "zero")
        std_z = Z.flatten(0, 1).std(0)
        results["random"] = mp_r(tr(Ht, torch.randn_like(Z) * std_z), "random")
        perm = torch.randperm(N, device=device)
        results["shuffle"] = mp_r(tr(Ht, Z[perm]), "shuffle(batch)")
        results["Z_only"] = mp_r(tr(torch.zeros_like(Ht), Z), "Z_only")
        results["wrong_H"] = mp_r(tr(Ht[perm], Z), "wrong_H")
        Ht_r16 = red_future(Ht)
        results["copy_Ht"] = mp_r(Ht_r16, "copy_Ht")

        # selectivity
        K = min(24, N); n_states = min(48, N)
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

    torch.save({"inv": inv.state_dict(), "tr": tr.state_dict(), "args": vars(args)},
               os.path.join(args.out_dir, "ckpt.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"results": results, "N": int(N), "args": vars(args)}, f, indent=2)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
