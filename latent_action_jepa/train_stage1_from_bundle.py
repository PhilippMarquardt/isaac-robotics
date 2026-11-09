"""Stage 1 trainer that loads a pre-encoded bundle (multi-video friendly).

Trains inverse_encoder + ResidualTransitionModel with the same losses as
train_stage1.py (token cosine + symmetric InfoNCE + VICReg var/cov on Z + tiny L2).
No video decoding here — bundle already has H_t / H_tp encoded.
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from models.inverse_encoder import InverseActionEncoder
from models.transition_residual import ResidualTransitionModel
from diag_stage0 import effective_rank


def cosine_distance(pred, target):
    pred = F.normalize(pred, dim=-1); target = F.normalize(target, dim=-1)
    return (2.0 - 2.0 * (pred * target).sum(-1)).mean()


def info_nce(q, k, tau=0.1, symmetric=True):
    q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
    logits = q @ k.T / tau
    labels = torch.arange(q.shape[0], device=q.device)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(k @ q.T / tau, labels))
    return loss


def retrieval(q, k, N, name):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    mr = (r == gt).float().argmax(-1).float().mean().item() + 1
    print(f"  retrieval[{name:18s}] top1={top1:.3f}  top5={top5:.3f}  mr={mr:.2f}")
    return top1, top5, mr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--save_ckpt", required=True)
    ap.add_argument("--save_z", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action_tokens", type=int, default=8)
    ap.add_argument("--action_dim", type=int, default=256)
    ap.add_argument("--future_tokens", type=int, default=16)
    ap.add_argument("--inverse_depth", type=int, default=4)
    ap.add_argument("--transition_depth", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--nce_tau", type=float, default=0.1)
    ap.add_argument("--lambda_nce", type=float, default=1.0)
    ap.add_argument("--lambda_z_l2", type=float, default=0.001)
    ap.add_argument("--lambda_z_var", type=float, default=0.05)
    ap.add_argument("--lambda_z_cov", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    state_dim = 384

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t = bundle["H_t_train"].to(device); H_tp = bundle["H_tp_train"].to(device)
    H_t_val = bundle["H_t_val"].to(device); H_tp_val = bundle["H_tp_val"].to(device)
    N_tr = H_t.shape[0]; N_va = H_t_val.shape[0]
    film_names = bundle.get("film_names", ["video"])
    vid_tr = bundle.get("video_id_train", torch.zeros(N_tr, dtype=torch.long)).to(device)
    vid_va = bundle.get("video_id_val", torch.zeros(N_va, dtype=torch.long)).to(device)
    print(f"loaded bundle: train {N_tr}  val {N_va}  films {film_names}")

    inv = InverseActionEncoder(state_dim, args.action_dim, args.action_tokens,
                               depth=args.inverse_depth).to(device)
    tr = ResidualTransitionModel(state_dim, args.action_dim, args.future_tokens,
                                 depth=args.transition_depth).to(device)
    print(f"trainable params: {sum(p.numel() for p in list(inv.parameters()) + list(tr.parameters()))/1e6:.2f} M")

    params = list(inv.parameters()) + list(tr.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    def eval_split(tag, Ht, Htp, vid):
        N = Ht.shape[0]
        inv.eval(); tr.eval()
        with torch.no_grad():
            Z = inv(Ht, Htp)
            pred_correct = tr(Ht, Z)
            pred_zero = tr(Ht, torch.zeros_like(Z))
            std_z = Z.flatten(1).std(0).mean()
            pred_rand = tr(Ht, torch.randn_like(Z) * std_z)
            pred_shuf = tr(Ht, torch.roll(Z, 1, 0))
            print(f"\n[{tag}]  N={N}  alpha={tr.alpha.item():.4f}  std(Z)={std_z.item():.4f}  eff_rank(Z)={effective_rank(Z.flatten(1)):.1f}")
            retrieval(pred_correct.mean(1), Htp.mean(1), N, "correct_Z")
            retrieval(pred_zero.mean(1), Htp.mean(1), N, "zero_Z")
            retrieval(pred_rand.mean(1), Htp.mean(1), N, "random_Z")
            retrieval(pred_shuf.mean(1), Htp.mean(1), N, "shuffle_Z")
            retrieval(Ht.mean(1), Htp.mean(1), N, "copy_Ht")
            # per-film
            for fi, fname in enumerate(film_names):
                mask = (vid == fi)
                if mask.sum() < 5: continue
                Nf = mask.sum().item()
                t1 = retrieval(pred_correct[mask].mean(1), Htp[mask].mean(1), Nf, f"  film {fname[:14]}")
        inv.train(); tr.train()

    eval_split("val pre", H_t_val, H_tp_val, vid_va)

    inv.train(); tr.train()
    for step in range(args.steps):
        idx = torch.randint(0, N_tr, (args.bs,), device=device)
        h_t = H_t[idx]; h_tp = H_tp[idx]
        z = inv(h_t, h_tp)
        h_pred = tr(h_t, z)
        L_t = cosine_distance(h_pred, h_tp.detach())
        L_nce = info_nce(h_pred.mean(1), h_tp.mean(1).detach(), tau=args.nce_tau, symmetric=True)
        L_l2 = (z ** 2).mean()
        std = z.flatten(0, 1).std(0)
        L_var = F.relu(1 - std).mean()
        z_flat = z.flatten(0, 1); z_c = z_flat - z_flat.mean(0, keepdim=True)
        cov = (z_c.T @ z_c) / max(1, z_c.shape[0] - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        L_cov = (off ** 2).sum() / cov.shape[0]
        loss = (L_t + args.lambda_nce * L_nce + args.lambda_z_l2 * L_l2
                + args.lambda_z_var * L_var + args.lambda_z_cov * L_cov)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  L_t={L_t.item():.4f}  L_nce={L_nce.item():.4f}  L_var={L_var.item():.4f}  L_cov={L_cov.item():.4f}  alpha={tr.alpha.item():+.4f}  lr={opt.param_groups[0]['lr']:.2e}")

    eval_split("train post", H_t, H_tp, vid_tr)
    eval_split("val post", H_t_val, H_tp_val, vid_va)

    os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
    torch.save({
        "inverse_encoder": inv.state_dict(),
        "transition_model": tr.state_dict(),
        "config": {
            "state_dim": state_dim, "action_dim": args.action_dim,
            "action_tokens": args.action_tokens, "future_tokens": args.future_tokens,
            "inverse_depth": args.inverse_depth, "transition_depth": args.transition_depth,
            "encoder_name": "vit_small_patch16_224", "encoder_layer_idx": None,
            "state_tokens": 64, "k": bundle.get("k", 16),
        },
        "args": vars(args),
        "film_names": film_names,
    }, args.save_ckpt)
    print(f"saved -> {args.save_ckpt}")

    if args.save_z:
        os.makedirs(os.path.dirname(args.save_z) or ".", exist_ok=True)
        inv.eval()
        with torch.no_grad():
            Z_tr = inv(H_t, H_tp).cpu()
            Z_va = inv(H_t_val, H_tp_val).cpu()
        torch.save({
            "H_t_train": H_t.cpu(), "H_tp_train": H_tp.cpu(), "Z_star_train": Z_tr,
            "video_id_train": vid_tr.cpu(),
            "H_t_val": H_t_val.cpu(), "H_tp_val": H_tp_val.cpu(), "Z_star_val": Z_va,
            "video_id_val": vid_va.cpu(),
            "film_names": film_names,
        }, args.save_z)
        print(f"saved Z-bundle -> {args.save_z}")


if __name__ == "__main__":
    main()
