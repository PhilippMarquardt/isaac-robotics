"""Phase B: synthetic robot-adapter test on F12.

Pipeline (simulates what robot post-train would look like):
  1. Take real frames from cache.
  2. Generate a synthetic "action" = 4-dim transform params (tx, ty, scale, theta).
  3. Apply transform to frame_t to get frame_t'.
  4. Encode both -> H_t, H_t'. Compute z* via F12 inverse on delta.
  5. Save (action_params, z*) tuples.
  6. Train an MLP adapter: action_params (4-dim) -> z (4*128 dim).
  7. Eval: predict z from params, run transition, check retrieval of H_t'.

If the adapter can recover z accurately AND drive the transition to retrieve
the true H_t', the robot post-train path is viable: replace synthetic action
params with real robot actions and the same recipe works.
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

from data.vjepa2_clips import IMAGENET_MEAN, IMAGENET_STD
from models.encoder import ViTEncoder
from models.fixed_reducer import FixedSpatialReducer
from models.inverse_encoder import InverseActionEncoder
from models.transition_residual import ResidualTransitionModel


PARAM_RANGES = {
    "tx": (-30, 30), "ty": (-30, 30),
    "scale": (0.8, 1.2), "theta": (-15, 15),
}


def sample_action():
    tx = random.uniform(*PARAM_RANGES["tx"])
    ty = random.uniform(*PARAM_RANGES["ty"])
    sc = random.uniform(*PARAM_RANGES["scale"])
    th = random.uniform(*PARAM_RANGES["theta"])
    return [tx, ty, sc, th]


def apply_action(img_u8, params):
    """img_u8: [3,H,W] uint8. params: [tx,ty,scale,theta]. Returns [3,H,W] uint8."""
    tx, ty, sc, th = params
    img = img_u8.float()
    out = TF.affine(img, angle=th, translate=(tx, ty), scale=sc, shear=(0, 0))
    return out.clamp(0, 255).to(torch.uint8)


def normalize(img_u8, device):
    """[B,3,H,W] uint8 -> [B,3,H,W] ImageNet-normalized float on device."""
    x = img_u8.to(device).float() / 255.0
    m = IMAGENET_MEAN.squeeze(0).to(device).view(1, 3, 1, 1)
    s = IMAGENET_STD.squeeze(0).to(device).view(1, 3, 1, 1)
    return (x - m) / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--la_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_pairs_per_film", type=int, default=200)
    ap.add_argument("--anchor_stride", type=int, default=8)
    ap.add_argument("--adapter_steps", type=int, default=2000)
    ap.add_argument("--adapter_hidden", type=int, default=256)
    ap.add_argument("--adapter_depth", type=int, default=3)
    ap.add_argument("--adapter_lr", type=float, default=3e-4)
    ap.add_argument("--adapter_bs", type=int, default=256)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed); random.seed(args.seed)

    ck = torch.load(args.la_ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    enc = ViTEncoder(a["encoder_name"], pretrained=True, layer_idx=None).to(device).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    state_dim = enc.embed_dim
    red_state = FixedSpatialReducer(a["state_tokens"]).to(device)
    red_future = FixedSpatialReducer(a["future_tokens"]).to(device)
    inv = InverseActionEncoder(state_dim, a["action_dim"], a["action_tokens"],
                                depth=a["inverse_depth"]).to(device).eval()
    tr = ResidualTransitionModel(state_dim, a["action_dim"], a["future_tokens"],
                                  depth=a["transition_depth"]).to(device).eval()
    inv.load_state_dict(ck["inv"]); tr.load_state_dict(ck["tr"])
    for p in list(inv.parameters()) + list(tr.parameters()): p.requires_grad_(False)
    delta_input = a.get("delta_input", False)
    print(f"loaded teacher: delta_input={delta_input} alpha={tr.alpha.item():.3f}")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])

    print(f"\n=== build (action_params, z*) dataset ===")
    all_params, all_Ht, all_Htp, all_Zstar = [], [], [], []
    g = torch.Generator().manual_seed(args.seed)

    for fi in range(n_films):
        frames = cache["frames"][fi]
        data = frames["data"] if isinstance(frames, dict) else frames
        n_frames = data.shape[0]
        max_a = n_frames - 50
        anchors = list(range(0, max_a, args.anchor_stride))
        random.shuffle(anchors)
        anchors = anchors[:args.n_pairs_per_film]
        # batch-compute features
        Bs = 16
        for s in range(0, len(anchors), Bs):
            batch_a = anchors[s:s + Bs]
            xt_u8 = torch.stack([data[a] for a in batch_a], dim=0)         # [B,3,H,W]
            params = [sample_action() for _ in range(len(batch_a))]
            xtp_u8 = torch.stack([apply_action(xt_u8[i], params[i]) for i in range(len(batch_a))], dim=0)
            xt = normalize(xt_u8, device)
            xtp = normalize(xtp_u8, device)
            with torch.no_grad():
                Ht = red_state(enc(xt))                                     # [B,64,D]
                Htp = red_future(enc(xtp))                                  # [B,16,D]
                if delta_input:
                    Ht_r = red_future(Ht)
                    Z = inv(torch.zeros_like(Ht_r), Htp - Ht_r)
                else:
                    Z = inv(Ht, Htp)
            all_params.extend(params)
            all_Ht.append(Ht.cpu()); all_Htp.append(Htp.cpu())
            all_Zstar.append(Z.cpu())
        print(f"  film {fi} '{cache['film_names'][fi]}': {len(anchors)} pairs")

    params_t = torch.tensor(all_params, dtype=torch.float32)                 # [N,4]
    Ht_all = torch.cat(all_Ht, 0)
    Htp_all = torch.cat(all_Htp, 0)
    Zstar_all = torch.cat(all_Zstar, 0)
    N_total = Ht_all.shape[0]
    print(f"total pairs: {N_total}, params dim={params_t.shape[-1]}, Z shape per pair {tuple(Zstar_all[0].shape)}")

    # normalize action params to [-1,1] roughly
    p_mean = params_t.mean(0); p_std = params_t.std(0).clamp_min(1e-6)
    params_norm = (params_t - p_mean) / p_std

    # split
    perm = torch.randperm(N_total, generator=g)
    n_tr = int(N_total * args.train_frac)
    tr_idx, va_idx = perm[:n_tr], perm[n_tr:]
    print(f"train {len(tr_idx)}  val {len(va_idx)}")

    # adapter MLP
    Ka, Da = Zstar_all.shape[1], Zstar_all.shape[2]
    out_dim = Ka * Da
    layers = []
    in_dim = 4
    for _ in range(args.adapter_depth - 1):
        layers += [nn.Linear(in_dim, args.adapter_hidden), nn.GELU()]
        in_dim = args.adapter_hidden
    layers += [nn.Linear(in_dim, out_dim)]
    adapter = nn.Sequential(*layers).to(device)
    print(f"adapter params: {sum(p.numel() for p in adapter.parameters())/1e3:.1f} K")
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.adapter_lr, weight_decay=1e-3)

    params_norm = params_norm.to(device)
    Zstar_flat = Zstar_all.reshape(N_total, -1).to(device)

    for step in range(args.adapter_steps):
        idx = tr_idx[torch.randint(0, len(tr_idx), (args.adapter_bs,))]
        x = params_norm[idx]; y = Zstar_flat[idx]
        p = adapter(x)
        loss = F.mse_loss(p, y)
        # also align direction
        cos = F.cosine_similarity(p, y, dim=-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == args.adapter_steps - 1:
            print(f"  step {step:5d}  mse={loss.item():.4f}  cos={cos.item():+.4f}")

    # ---- eval ----
    print(f"\n=== Eval adapter on held-out val ===")
    adapter.eval()
    with torch.no_grad():
        x_va = params_norm[va_idx]
        z_pred = adapter(x_va).reshape(-1, Ka, Da)
        z_true = Zstar_all[va_idx].to(device)
        cos_full = F.cosine_similarity(z_pred.flatten(1), z_true.flatten(1), dim=-1).mean().item()
        # drive transition with predicted z, retrieve true Htp
        Ht_va = Ht_all[va_idx].to(device)
        Htp_va = Htp_all[va_idx].to(device)
        H_pred = tr(Ht_va, z_pred)
        H_pred_oracle = tr(Ht_va, z_true)
        # retrieval
        def retr(q, k, name):
            qn = F.normalize(q.mean(1), dim=-1); kn = F.normalize(k.mean(1), dim=-1)
            sim = qn @ kn.T
            r = sim.argsort(dim=-1, descending=True)
            gt = torch.arange(qn.shape[0], device=device).unsqueeze(-1)
            t1 = (r[:, :1] == gt).any(-1).float().mean().item()
            t5 = (r[:, :5] == gt).any(-1).float().mean().item()
            mr = (r == gt).float().argmax(-1).float().mean().item() + 1
            print(f"  {name:36s} top1={t1:.4f}  top5={t5:.4f}  mr={mr:.1f}")
            return {"top1": t1, "top5": t5, "mr": mr}

        print(f"  cos(z_pred, z*) = {cos_full:+.4f}")
        results = {"cos_zpred_zstar": cos_full}
        results["oracle_drive"] = retr(H_pred_oracle, Htp_va, "T(H_t, z*) (oracle) -> H_tp")
        results["adapter_drive"] = retr(H_pred, Htp_va,         "T(H_t, adapter(params)) -> H_tp")

    out = {"adapter_args": vars(args), "results": results, "n_val": int(len(va_idx))}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    torch.save({"adapter": adapter.state_dict(), "p_mean": p_mean, "p_std": p_std,
                "args": vars(args)}, os.path.join(args.out_dir, "adapter.pt"))
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
