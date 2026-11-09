"""Stage 2C: reranker with HARD negatives — sample wrong Z*'s from train pool
and learn to score correct Z* higher than wrong ones.
"""
import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transition_residual import ResidualTransitionModel


class Reranker(nn.Module):
    def __init__(self, state_dim, hidden=512, drop=0.1):
        super().__init__()
        self.proj_t = nn.Sequential(nn.Linear(state_dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, hidden))
        self.proj_p = nn.Sequential(nn.Linear(state_dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, hidden))

    def forward(self, h_t, h_pred):
        t = self.proj_t(h_t.mean(1)); p = self.proj_p(h_pred.mean(1))
        return F.normalize(t, dim=-1), F.normalize(p, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=64, help="anchor batch size")
    ap.add_argument("--K_negs", type=int, default=32, help="hard negatives per anchor at train time")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--K_pool", type=int, default=64)
    ap.add_argument("--sub_ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(0)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tr = ResidualTransitionModel(cfg["state_dim"], cfg["action_dim"], cfg["future_tokens"],
                                 depth=cfg["transition_depth"]).to(device)
    tr.load_state_dict(ckpt["transition_model"])
    for p in tr.parameters(): p.requires_grad_(False)
    tr.eval()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t_train = bundle["H_t_train"].to(device); H_tp_train = bundle["H_tp_train"].to(device)
    Z_star_train = bundle["Z_star_train"].to(device)
    H_t_val = bundle["H_t_val"].to(device); H_tp_val = bundle["H_tp_val"].to(device)
    Z_star_val = bundle["Z_star_val"].to(device)
    N_tr = H_t_train.shape[0]; N_va = H_t_val.shape[0]
    print(f"train N={N_tr}  val N={N_va}")

    reranker = Reranker(cfg["state_dim"], hidden=args.hidden, drop=args.dropout).to(device)
    opt = torch.optim.AdamW(reranker.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    print("\n=== Training reranker with hard negatives ===")
    reranker.train()
    for step in range(args.steps):
        anchor_idx = torch.randint(0, N_tr, (args.bs,), device=device)
        h_t = H_t_train[anchor_idx]
        # correct preds
        with torch.no_grad():
            pred_pos = tr(h_t, Z_star_train[anchor_idx])  # [bs, 16, 384]
            # hard negatives: random other train indices
            neg_idx = torch.randint(0, N_tr, (args.bs, args.K_negs), device=device)
            neg_z = Z_star_train[neg_idx.reshape(-1)].view(args.bs, args.K_negs, *Z_star_train.shape[1:])
            # apply transition for each negative
            h_t_rep = h_t.unsqueeze(1).expand(-1, args.K_negs, -1, -1).reshape(-1, *h_t.shape[1:])
            neg_z_flat = neg_z.reshape(-1, *Z_star_train.shape[1:])
            pred_neg = tr(h_t_rep, neg_z_flat).view(args.bs, args.K_negs, 16, h_t.shape[-1])
        # score pos and negs
        t_emb, p_emb_pos = reranker(h_t, pred_pos)  # [bs, hidden] each
        # build per-anchor negative embeddings
        t_emb_rep = t_emb.unsqueeze(1).expand(-1, args.K_negs, -1).reshape(-1, t_emb.shape[-1])
        h_t_rep_emb = h_t_rep
        _, p_emb_neg = reranker(h_t_rep_emb, pred_neg.reshape(-1, 16, h_t.shape[-1]))
        p_emb_neg = p_emb_neg.view(args.bs, args.K_negs, -1)
        # cosine sim
        pos_sim = (t_emb * p_emb_pos).sum(-1) / args.tau  # [bs]
        neg_sim = (t_emb.unsqueeze(1) * p_emb_neg).sum(-1) / args.tau  # [bs, K_negs]
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        loss = F.cross_entropy(logits, torch.zeros(args.bs, dtype=torch.long, device=device))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0 or step == args.steps - 1:
            with torch.no_grad():
                acc = (logits.argmax(-1) == 0).float().mean().item()
            print(f"  step {step:4d}  L={loss.item():.4f}  train_acc={acc:.3f}  lr={opt.param_groups[0]['lr']:.2e}")

    # Eval on val with empZ K_pool
    reranker.eval()
    torch.manual_seed(123)
    val_zs = [Z_star_train[torch.randint(0, N_tr, (N_va,), device=device)] for _ in range(args.K_pool)]
    target = F.normalize(H_tp_val.mean(1), dim=-1)
    print(f"\n=== Eval val (K_pool={args.K_pool}) ===")
    with torch.no_grad():
        preds = [tr(H_t_val, z) for z in val_zs]
        # score each
        scores = torch.zeros(N_va, args.K_pool, device=device)
        for k, p in enumerate(preds):
            t_emb, p_emb = reranker(H_t_val, p)
            scores[:, k] = (t_emb * p_emb).sum(-1)
        sorted_idx = scores.argsort(dim=-1, descending=True)
        results = {}
        for sub_k in args.sub_ks:
            best_rank = torch.full((N_va,), N_va + 1, device=device, dtype=torch.long)
            for j in range(sub_k):
                ank = sorted_idx[:, j]
                pred_j = torch.stack([preds[k_idx.item()][i] for i, k_idx in enumerate(ank)], dim=0)
                q = F.normalize(pred_j.mean(1), dim=-1)
                sim = q @ target.T
                r = sim.argsort(dim=-1, descending=True)
                gt = torch.arange(N_va, device=device).unsqueeze(-1)
                rank_i = (r == gt).float().argmax(-1).long() + 1
                best_rank = torch.minimum(best_rank, rank_i)
            top1 = (best_rank == 1).float().mean().item()
            mr = best_rank.float().mean().item()
            results[sub_k] = {"top1": top1, "mean_rank": mr}
            print(f"  rerank_hardneg top{sub_k:2d}: top1={top1:.3f}  mr={mr:.2f}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
