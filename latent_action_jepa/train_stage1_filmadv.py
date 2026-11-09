"""Stage 1 trainer with film-adversarial Z regularization.

Adds a film classifier on mean-pooled Z via a Gradient Reversal Layer (GRL).
The classifier itself learns to predict film_id from pooled Z (normal gradient).
The reversed gradient flows back into the inverse_encoder, pushing it to make
Z hard to classify by film.

Important: the transition model receives the un-reversed Z. Only the inverse
encoder is pressured to drop film identity from Z. The transition can still
use H_t for appearance/style context.
"""
import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.inverse_encoder import InverseActionEncoder
from models.transition_residual import ResidualTransitionModel
from diag_stage0 import effective_rank


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


def grad_reverse(x, lam):
    return GradReverse.apply(x, lam)


class FilmClassifier(nn.Module):
    def __init__(self, action_dim, n_films, depth=1, hidden=128, drop=0.0):
        super().__init__()
        if depth == 1:
            self.net = nn.Linear(action_dim, n_films)
        else:
            layers = [nn.Linear(action_dim, hidden), nn.GELU(), nn.Dropout(drop)]
            for _ in range(depth - 2):
                layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop)]
            layers += [nn.Linear(hidden, n_films)]
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def cosine_distance(pred, target):
    pred = F.normalize(pred, dim=-1); target = F.normalize(target, dim=-1)
    return (2.0 - 2.0 * (pred * target).sum(-1)).mean()


def info_nce(q, k, tau=0.1):
    q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
    logits = q @ k.T / tau
    labels = torch.arange(q.shape[0], device=q.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(k @ q.T / tau, labels))


def retrieval(q, k, N, name):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    mr = (r == gt).float().argmax(-1).float().mean().item() + 1
    print(f"  retrieval[{name:14s}] top1={top1:.3f}  top5={top5:.3f}  mr={mr:.2f}")
    return top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--save_ckpt", required=True)
    ap.add_argument("--save_z", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action_tokens", type=int, default=4)
    ap.add_argument("--action_dim", type=int, default=128)
    ap.add_argument("--future_tokens", type=int, default=16)
    ap.add_argument("--inverse_depth", type=int, default=4)
    ap.add_argument("--transition_depth", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_film", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--nce_tau", type=float, default=0.1)
    ap.add_argument("--lambda_nce", type=float, default=1.0)
    ap.add_argument("--lambda_z_l2", type=float, default=0.001)
    ap.add_argument("--lambda_z_var", type=float, default=0.05)
    ap.add_argument("--lambda_z_cov", type=float, default=0.01)
    ap.add_argument("--lambda_film_adv", type=float, default=0.1)
    ap.add_argument("--film_classifier_depth", type=int, default=2)
    ap.add_argument("--film_classifier_hidden", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=500)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    state_dim = 384

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t = bundle["H_t_train"].to(device); H_tp = bundle["H_tp_train"].to(device)
    vid_tr = bundle["video_id_train"].to(device)
    H_t_val = bundle["H_t_val"].to(device); H_tp_val = bundle["H_tp_val"].to(device)
    vid_va = bundle["video_id_val"].to(device)
    N_tr = H_t.shape[0]; N_va = H_t_val.shape[0]
    n_films = int(vid_tr.max().item() + 1)
    film_names = bundle.get("film_names", [f"film{i}" for i in range(n_films)])
    print(f"train N={N_tr}  val N={N_va}  n_films={n_films}  λ_film_adv={args.lambda_film_adv}")

    inv = InverseActionEncoder(state_dim, args.action_dim, args.action_tokens,
                               depth=args.inverse_depth).to(device)
    tr = ResidualTransitionModel(state_dim, args.action_dim, args.future_tokens,
                                 depth=args.transition_depth).to(device)
    film_cls = FilmClassifier(args.action_dim, n_films, depth=args.film_classifier_depth,
                              hidden=args.film_classifier_hidden).to(device)
    print(f"params  inv+tr={sum(p.numel() for p in list(inv.parameters())+list(tr.parameters()))/1e6:.2f}M  filmcls={sum(p.numel() for p in film_cls.parameters())/1e6:.2f}M")

    params = list(inv.parameters()) + list(tr.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    opt_film = torch.optim.AdamW(film_cls.parameters(), lr=args.lr_film, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    sched_film = torch.optim.lr_scheduler.CosineAnnealingLR(opt_film, T_max=args.steps)

    inv.train(); tr.train(); film_cls.train()
    for step in range(args.steps):
        idx = torch.randint(0, N_tr, (args.bs,), device=device)
        h_t = H_t[idx]; h_tp = H_tp[idx]; vid = vid_tr[idx]
        z = inv(h_t, h_tp)
        h_pred = tr(h_t, z)

        # main losses
        L_t = cosine_distance(h_pred, h_tp.detach())
        L_n = info_nce(h_pred.mean(1), h_tp.mean(1).detach(), tau=args.nce_tau)
        L_l2 = (z ** 2).mean()
        std_dim = z.flatten(0, 1).std(0)
        L_v = F.relu(1 - std_dim).mean()
        z_flat = z.flatten(0, 1); z_c = z_flat - z_flat.mean(0, keepdim=True)
        cov = (z_c.T @ z_c) / max(1, z_c.shape[0] - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        L_co = (off ** 2).sum() / cov.shape[0]

        # film-adversarial: GRL on pooled Z
        z_pool = z.mean(1)
        z_grl = grad_reverse(z_pool, args.lambda_film_adv)
        film_logits = film_cls(z_grl)
        L_film = F.cross_entropy(film_logits, vid)

        loss = (L_t + args.lambda_nce * L_n + args.lambda_z_l2 * L_l2
                + args.lambda_z_var * L_v + args.lambda_z_cov * L_co
                + L_film)  # GRL already incorporates λ on the reversed-grad side

        opt.zero_grad(); opt_film.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        torch.nn.utils.clip_grad_norm_(film_cls.parameters(), 1.0)
        opt.step(); opt_film.step()
        sched.step(); sched_film.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                film_acc = (film_logits.argmax(-1) == vid).float().mean().item()
            print(f"  step {step:5d}  L_t={L_t.item():.4f}  L_nce={L_n.item():.4f}  L_film={L_film.item():.4f}  film_acc={film_acc:.3f}  alpha={tr.alpha.item():.3f}")

    # ---- Eval ----
    inv.eval(); tr.eval(); film_cls.eval()
    print("\n=== val eval ===")
    with torch.no_grad():
        Z_va = inv(H_t_val, H_tp_val)
        std_z = Z_va.flatten(1).std(0).mean().item()
        pred_c = tr(H_t_val, Z_va)
        retrieval(pred_c.mean(1), H_tp_val.mean(1), N_va, "correct_Z")
        retrieval(tr(H_t_val, torch.zeros_like(Z_va)).mean(1), H_tp_val.mean(1), N_va, "zero_Z")
        retrieval(tr(H_t_val, torch.randn_like(Z_va) * std_z).mean(1), H_tp_val.mean(1), N_va, "random_Z")
        retrieval(tr(H_t_val, torch.roll(Z_va, 1, 0)).mean(1), H_tp_val.mean(1), N_va, "shuffle_Z")
        retrieval(H_t_val.mean(1), H_tp_val.mean(1), N_va, "copy_Ht")
        retrieval(tr(torch.zeros_like(H_t_val), Z_va).mean(1), H_tp_val.mean(1), N_va, "Z_only")
        perm = torch.randperm(N_va, device=device, generator=torch.Generator(device=device).manual_seed(0))
        retrieval(tr(H_t_val[perm], Z_va).mean(1), H_tp_val.mean(1), N_va, "wrong_H")
        # film probe by the trained head
        film_va = film_cls(Z_va.mean(1))
        film_acc_va = (film_va.argmax(-1) == vid_va).float().mean().item()
        print(f"  in-loop film classifier val acc on Z: {film_acc_va:.3f}  (chance={1/n_films:.3f})")

    os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
    torch.save({
        "inverse_encoder": inv.state_dict(),
        "transition_model": tr.state_dict(),
        "film_classifier": film_cls.state_dict(),
        "config": {
            "state_dim": state_dim, "action_dim": args.action_dim,
            "action_tokens": args.action_tokens, "future_tokens": args.future_tokens,
            "inverse_depth": args.inverse_depth, "transition_depth": args.transition_depth,
            "encoder_name": "vit_small_patch16_224", "encoder_layer_idx": None,
            "state_tokens": 64, "k": bundle.get("k", 16),
            "n_films": n_films,
        },
        "args": vars(args),
        "film_names": film_names,
    }, args.save_ckpt)

    if args.save_z:
        os.makedirs(os.path.dirname(args.save_z) or ".", exist_ok=True)
        inv.eval()
        with torch.no_grad():
            Z_tr = inv(H_t, H_tp).cpu()
            Z_va2 = inv(H_t_val, H_tp_val).cpu()
        torch.save({
            "H_t_train": H_t.cpu(), "H_tp_train": H_tp.cpu(), "Z_star_train": Z_tr,
            "video_id_train": vid_tr.cpu(),
            "H_t_val": H_t_val.cpu(), "H_tp_val": H_tp_val.cpu(), "Z_star_val": Z_va2,
            "video_id_val": vid_va.cpu(),
            "film_names": film_names,
        }, args.save_z)


if __name__ == "__main__":
    main()
