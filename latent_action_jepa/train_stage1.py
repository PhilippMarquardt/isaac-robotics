"""Stage 1: frozen encoder + fixed reducers; train inverse + transition only.

Compare on held-in (overfit) and held-out splits:
  correct_Z:  T(H_t, Z*)
  copy_Ht:    H_t mean (no transition)
  zero_Z:     T(H_t, 0)
  random_Z:   T(H_t, random)
  shuffle_Z:  T(H_t, roll(Z*))
  Z_only:     mean(Z*)                (with linear probe head)
  H_only:     mean(H_t)               (== copy_Ht)
Log std(Z), effective_rank(Z).
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import ViTEncoder
from models.fixed_reducer import FixedSpatialReducer
from models.inverse_encoder import InverseActionEncoder
from models.transition import TransitionModel
from models.transition_residual import ResidualTransitionModel
from data.video_pairs import make_or_load_contig
from diag_stage0 import effective_rank


def cosine_distance(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return (2.0 - 2.0 * (pred * target).sum(-1)).mean()


def retrieval(q, k, N, name=None, print_=True):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    rank = (r == gt).float().argmax(-1).float().mean().item() + 1
    if print_ and name:
        print(f"  retrieval[{name:18s}] top1={top1:.3f}  top5={top5:.3f}  mean_rank={rank:.2f}")
    return top1, top5, rank


def retrieval_pertok(pred, tgt, N, name=None, print_=True):
    """Per-token paired cosine retrieval. Matches the training loss objective.
    pred,tgt: [N,K,D]. Each token is L2-normalized then flattened; cosine sim then.
    """
    p = F.normalize(pred, dim=-1).flatten(1)
    t = F.normalize(tgt, dim=-1).flatten(1)
    # Inner product after per-token norm gives K * mean_t cos(p_t, t_t); ranking is order-preserving.
    sim = p @ t.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=p.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    rank = (r == gt).float().argmax(-1).float().mean().item() + 1
    if print_ and name:
        print(f"  pertok[{name:21s}] top1={top1:.3f}  top5={top5:.3f}  mean_rank={rank:.2f}")
    return top1, top5, rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/workspace/latent_action_jepa/bbb.mp4")
    ap.add_argument("--cache_train", default="/workspace/latent_action_jepa/bbb_contig_k16.pt")
    ap.add_argument("--cache_val", default="/workspace/latent_action_jepa/bbb_val_k16.pt")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--train_start", type=int, default=200)
    ap.add_argument("--stride_train", type=int, default=50)
    ap.add_argument("--val_start", type=int, default=-1, help="-1 = auto (train_end + block_gap)")
    ap.add_argument("--stride_val", type=int, default=25)
    ap.add_argument("--block_gap", type=int, default=80, help="frames gap between train and val blocks (>=k+64 recommended)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--action_tokens", type=int, default=8)
    ap.add_argument("--action_dim", type=int, default=256)
    ap.add_argument("--inverse_depth", type=int, default=4)
    ap.add_argument("--transition_depth", type=int, default=6)
    ap.add_argument("--residual", action="store_true", help="use ResidualTransitionModel with alpha-gated delta")
    ap.add_argument("--lambda_z_l2", type=float, default=0.001)
    ap.add_argument("--lambda_z_var", type=float, default=0.05)
    ap.add_argument("--lambda_z_cov", type=float, default=0.01)
    ap.add_argument("--lambda_nce", type=float, default=1.0)
    ap.add_argument("--nce_tau", type=float, default=0.1)
    ap.add_argument("--nce_symmetric", action="store_true", default=True)
    ap.add_argument("--z_dropout", type=float, default=0.0)
    ap.add_argument("--save_ckpt", type=str, default="", help="path to save final inv+tr+cfg state_dict")
    ap.add_argument("--save_z", type=str, default="", help="path to save oracle Z_star for train and val")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    state_dim = 384

    enc = ViTEncoder("vit_small_patch16_224", pretrained=True, layer_idx=None).to(device).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    red_state = FixedSpatialReducer(64).to(device)
    red_future = FixedSpatialReducer(16).to(device)

    inv = InverseActionEncoder(state_dim, args.action_dim, args.action_tokens,
                               depth=args.inverse_depth).to(device)
    if args.residual:
        tr = ResidualTransitionModel(state_dim, args.action_dim, 16,
                                     depth=args.transition_depth).to(device)
        print(f"transition: residual (alpha-gated), depth={args.transition_depth}")
    else:
        tr = TransitionModel(state_dim, args.action_dim, 16,
                             depth=args.transition_depth).to(device)
        print(f"transition: dense, depth={args.transition_depth}")

    # Train data — contiguous block
    from data.video_pairs import decode_contiguous_pairs
    import os
    if os.path.exists(args.cache_train):
        d = torch.load(args.cache_train, map_location="cpu")
        x_t, x_tp = d["x_t"], d["x_tp"]
        train_end = int(d.get("train_end", args.train_start + args.n_train * args.stride_train + args.k))
    else:
        x_t, x_tp, ts, _ = decode_contiguous_pairs(args.video, n_pairs=args.n_train, k=args.k,
                                                  stride=args.stride_train, start_skip=args.train_start)
        train_end = int(ts[-1].item() + args.k)
        torch.save({"x_t": x_t, "x_tp": x_tp, "train_end": train_end}, args.cache_train)

    val_start = args.val_start if args.val_start >= 0 else train_end + args.block_gap
    print(f"train block: [{args.train_start}, {train_end}]  stride={args.stride_train}")
    print(f"val   block: [{val_start}, ?]  stride={args.stride_val}  gap={val_start - train_end}")
    if os.path.exists(args.cache_val):
        d = torch.load(args.cache_val, map_location="cpu")
        x_v_t, x_v_tp = d["x_t"], d["x_tp"]
    else:
        x_v_t, x_v_tp, _, _ = decode_contiguous_pairs(args.video, n_pairs=args.n_val,
                                                     k=args.k, stride=args.stride_val,
                                                     start_skip=val_start)
        torch.save({"x_t": x_v_t, "x_tp": x_v_tp}, args.cache_val)
    print(f"train pairs={x_t.shape[0]}  val pairs={x_v_t.shape[0]}  k={args.k}")

    x_t = x_t.to(device); x_tp = x_tp.to(device)
    x_v_t = x_v_t.to(device); x_v_tp = x_v_tp.to(device)

    # Pre-compute frozen features for speed
    def encode_pair(xt, xtp):
        with torch.no_grad():
            chunks = max(1, xt.shape[0] // 64)
            Hts, Htps = [], []
            for c in range(chunks + 1):
                s = c * 64; e = min((c + 1) * 64, xt.shape[0])
                if s >= e: continue
                Hts.append(red_state(enc(xt[s:e])))
                Htps.append(red_future(enc(xtp[s:e])))
            return torch.cat(Hts, 0), torch.cat(Htps, 0)

    H_t_train, H_tp_train = encode_pair(x_t, x_tp)
    H_t_val, H_tp_val = encode_pair(x_v_t, x_v_tp)
    print(f"H_t_train {tuple(H_t_train.shape)}  H_tp_train {tuple(H_tp_train.shape)}")
    print(f"H_t_val   {tuple(H_t_val.shape)}    H_tp_val   {tuple(H_tp_val.shape)}")

    params = list(inv.parameters()) + list(tr.parameters())
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f} M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    N_tr = H_t_train.shape[0]

    def eval_split(H_t, H_tp, tag):
        N = H_t.shape[0]
        inv.eval(); tr.eval()
        with torch.no_grad():
            Z_star = inv(H_t, H_tp)
            H_pred_correct = tr(H_t, Z_star)
            H_pred_zero = tr(H_t, torch.zeros_like(Z_star))
            H_pred_rand = tr(H_t, torch.randn_like(Z_star))
            H_pred_shuf = tr(H_t, torch.roll(Z_star, 1, 0))

            print(f"\n[{tag}]  N={N}")
            std_Z = Z_star.flatten(1).std(0).mean().item()
            er_Z = effective_rank(Z_star.flatten(1))
            print(f"  std(Z)={std_Z:.4f}  eff_rank(Z)={er_Z:.1f}  ||Z||={Z_star.pow(2).mean().sqrt().item():.4f}")

            print("  -- mean-pool cosine --")
            stats_mp = {}
            for name, pred in [
                ("correct_Z",  H_pred_correct),
                ("zero_Z",     H_pred_zero),
                ("random_Z",   H_pred_rand),
                ("shuffle_Z",  H_pred_shuf),
            ]:
                t1, t5, mr = retrieval(pred.mean(1), H_tp.mean(1), N, name)
                stats_mp[name] = (t1, mr)
            ct1, cmr, _ = (*retrieval(H_t.mean(1), H_tp.mean(1), N, "copy_Ht"),)
            copy_t1, copy_mr = ct1, cmr
            corr_t1, corr_mr = stats_mp["correct_Z"]
            wrong_t1 = max(stats_mp["zero_Z"][0], stats_mp["random_Z"][0], stats_mp["shuffle_Z"][0])
            wrong_mr = min(stats_mp["zero_Z"][1], stats_mp["random_Z"][1], stats_mp["shuffle_Z"][1])
            print(f"  >>> Z_gap_top1 = correct - best_wrong = {corr_t1 - wrong_t1:+.3f}   (vs copy {corr_t1 - copy_t1:+.3f})")
            print(f"  >>> Z_gap_rank = best_wrong - correct (positive=good) = {wrong_mr - corr_mr:+.2f}  (vs copy {copy_mr - corr_mr:+.2f})")

            print("  -- per-token paired cosine (matches train loss) --")
            stats_pt = {}
            for name, pred in [
                ("correct_Z",  H_pred_correct),
                ("zero_Z",     H_pred_zero),
                ("random_Z",   H_pred_rand),
                ("shuffle_Z",  H_pred_shuf),
            ]:
                t1, t5, mr = retrieval_pertok(pred, H_tp, N, name)
                stats_pt[name] = (t1, mr)
            corr_t1, corr_mr = stats_pt["correct_Z"]
            wrong_t1 = max(stats_pt["zero_Z"][0], stats_pt["random_Z"][0], stats_pt["shuffle_Z"][0])
            wrong_mr = min(stats_pt["zero_Z"][1], stats_pt["random_Z"][1], stats_pt["shuffle_Z"][1])
            print(f"  >>> [pertok] Z_gap_top1 = {corr_t1 - wrong_t1:+.3f}    Z_gap_rank = {wrong_mr - corr_mr:+.2f}")
        inv.train(); tr.train()

    print("\n=== Pre-train baseline eval (random init inv+tr) ===")
    eval_split(H_t_train, H_tp_train, "train pre")
    eval_split(H_t_val, H_tp_val, "val pre")

    print("\n=== Training ===")
    inv.train(); tr.train()
    for step in range(args.steps):
        idx = torch.randint(0, N_tr, (args.bs,), device=device)
        h_t = H_t_train[idx]; h_tp = H_tp_train[idx]
        z = inv(h_t, h_tp)
        if args.z_dropout > 0:
            mask = (torch.rand_like(z[..., :1]) > args.z_dropout).float()
            z_in = z * mask
        else:
            z_in = z
        h_pred = tr(h_t, z_in)

        # Per-token cosine (existing)
        L_t = cosine_distance(h_pred, h_tp.detach())

        # InfoNCE on mean-pooled, L2-normalized features (matches retrieval metric)
        q = F.normalize(h_pred.mean(1), dim=-1)
        k = F.normalize(h_tp.mean(1).detach(), dim=-1)
        logits_qk = q @ k.T / args.nce_tau
        labels = torch.arange(q.shape[0], device=q.device)
        L_nce = F.cross_entropy(logits_qk, labels)
        if args.nce_symmetric:
            L_nce = 0.5 * (L_nce + F.cross_entropy(k @ q.T / args.nce_tau, labels))

        # VICReg-style Z regs: l2, per-dim variance, covariance decorrelation
        z_flat = z.flatten(0, 1)  # [B*Ka, Da]
        L_l2 = (z ** 2).mean()
        std = z_flat.std(0)
        L_var = F.relu(1 - std).mean()
        z_c = z_flat - z_flat.mean(0, keepdim=True)
        cov = (z_c.T @ z_c) / max(1, z_c.shape[0] - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        L_cov = (off ** 2).sum() / cov.shape[0]

        loss = (L_t
                + args.lambda_nce * L_nce
                + args.lambda_z_l2 * L_l2
                + args.lambda_z_var * L_var
                + args.lambda_z_cov * L_cov)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 200 == 0 or step == args.steps - 1:
            extra = ""
            if args.residual:
                extra = f"  alpha={tr.alpha.item():+.4f}"
            print(f"  step {step:4d}  L_t={L_t.item():.4f}  L_nce={L_nce.item():.4f}  L_var={L_var.item():.4f}  L_cov={L_cov.item():.4f}  lr={opt.param_groups[0]['lr']:.2e}{extra}")

    print("\n=== Post-train eval ===")
    eval_split(H_t_train, H_tp_train, "train post")
    eval_split(H_t_val, H_tp_val, "val post")

    if args.save_ckpt:
        os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
        torch.save({
            "inverse_encoder": inv.state_dict(),
            "transition_model": tr.state_dict(),
            "config": {
                "state_dim": state_dim,
                "action_dim": args.action_dim,
                "action_tokens": args.action_tokens,
                "inverse_depth": args.inverse_depth,
                "transition_depth": args.transition_depth,
                "future_tokens": 16,
                "residual": args.residual,
                "encoder_name": "vit_small_patch16_224",
                "encoder_layer_idx": None,
                "state_tokens": 64,
                "k": args.k,
            },
            "args": vars(args),
        }, args.save_ckpt)
        print(f"saved checkpoint -> {args.save_ckpt}")

    if args.save_z:
        os.makedirs(os.path.dirname(args.save_z) or ".", exist_ok=True)
        inv.eval()
        with torch.no_grad():
            Z_train = inv(H_t_train, H_tp_train).detach().cpu()
            Z_val = inv(H_t_val, H_tp_val).detach().cpu()
        torch.save({
            "H_t_train": H_t_train.cpu(),
            "H_tp_train": H_tp_train.cpu(),
            "Z_star_train": Z_train,
            "H_t_val": H_t_val.cpu(),
            "H_tp_val": H_tp_val.cpu(),
            "Z_star_val": Z_val,
        }, args.save_z)
        print(f"saved feature/Z bundle -> {args.save_z}")

    print("\nSTAGE 1 DONE")


if __name__ == "__main__":
    main()
