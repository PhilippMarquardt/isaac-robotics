"""Stage 2C: learn a reranker over empZ samples.

Pipeline at inference:
  1. Sample K empZ candidates from train Z* pool (uniform).
  2. Apply frozen transition: H_pred_k = T(H_t, Z_k) for each k.
  3. Score(H_t, H_pred) -> s_k. Take argmax_k or top-N'.

Trained with InfoNCE:
  For batch of train (H_t_i, H_tp_i, Z*_i):
    H_pred_correct_i = T(H_t_i, Z*_i)
    Negatives = other batch elements' H_pred (or T(H_t_i, Z*_j) shuffled).
    score(H_t_i, H_pred_correct_i) > score(H_t_i, H_pred_negative_j).
"""
import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transition_residual import ResidualTransitionModel
from diag_stage0 import effective_rank


class Reranker(nn.Module):
    """Score(H_t, H_pred) -> R. Mean-pool both, MLP."""

    def __init__(self, state_dim, hidden=512, drop=0.1):
        super().__init__()
        self.proj_t = nn.Sequential(nn.Linear(state_dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, hidden))
        self.proj_p = nn.Sequential(nn.Linear(state_dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, hidden))

    def forward(self, h_t, h_pred):
        t = self.proj_t(h_t.mean(1))
        p = self.proj_p(h_pred.mean(1))
        return F.normalize(t, dim=-1), F.normalize(p, dim=-1)


def best_of_k_with_scorer(reranker, tr, H_t, candidate_zs_list, H_tp, N, name, sub_k_list, device):
    """Score K candidates, take top-N from rerank; eval best-of-N retrieval."""
    K = len(candidate_zs_list)
    # Score every candidate
    scores = torch.zeros(N, K, device=device)
    preds = []
    with torch.no_grad():
        for k, z in enumerate(candidate_zs_list):
            pred = tr(H_t, z)
            preds.append(pred)
            t_emb, p_emb = reranker(H_t, pred)
            s = (t_emb * p_emb).sum(-1)  # cosine
            scores[:, k] = s
    # Sort candidates by score desc
    sorted_idx = scores.argsort(dim=-1, descending=True)
    results = {}
    for sub_k in sub_k_list:
        # for each anchor i, take its top sub_k candidates
        # Build preds_subset_list (over k_idx in sorted_idx[:, :sub_k])
        # Best-of-sub_k retrieval
        target = F.normalize(H_tp.mean(1), dim=-1)
        best_rank = torch.full((N,), N + 1, device=device, dtype=torch.long)
        for j in range(sub_k):
            anchor_ks = sorted_idx[:, j]  # [N]
            # gather pred for each anchor at its j-th best candidate
            pred_j = torch.stack([preds[k_idx.item()][i] for i, k_idx in enumerate(anchor_ks)], dim=0)
            q = F.normalize(pred_j.mean(1), dim=-1)
            sim = q @ target.T
            r = sim.argsort(dim=-1, descending=True)
            gt = torch.arange(N, device=device).unsqueeze(-1)
            rank_i = (r == gt).float().argmax(-1).long() + 1
            best_rank = torch.minimum(best_rank, rank_i)
        top1 = (best_rank == 1).float().mean().item()
        top5 = (best_rank <= 5).float().mean().item()
        mr = best_rank.float().mean().item()
        results[sub_k] = {"top1": top1, "top5": top5, "mean_rank": mr}
        print(f"  rerank[K={K:3d}|top{sub_k:2d}|{name:12s}] top1={top1:.3f}  top5={top5:.3f}  mean_rank={mr:.2f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/latent_action_jepa/runs/stage1c/ckpt.pt")
    ap.add_argument("--bundle", default="/workspace/latent_action_jepa/runs/stage1c/oracle_bundle_12k.pt")
    ap.add_argument("--out_dir", default="/workspace/latent_action_jepa/runs/stage2_ablations/reranker_12k")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=128)
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
    print(f"reranker params: {sum(p.numel() for p in reranker.parameters())/1e6:.2f} M")

    opt = torch.optim.AdamW(reranker.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    print("\n=== Training reranker ===")
    reranker.train()
    for step in range(args.steps):
        idx = torch.randint(0, N_tr, (args.bs,), device=device)
        h_t = H_t_train[idx]; z = Z_star_train[idx]
        with torch.no_grad():
            h_pred = tr(h_t, z)  # correct preds
        t_emb, p_emb = reranker(h_t, h_pred)
        logits = t_emb @ p_emb.T / args.tau
        labels = torch.arange(args.bs, device=device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0 or step == args.steps - 1:
            with torch.no_grad():
                top1 = (logits.argmax(-1) == labels).float().mean().item()
            print(f"  step {step:4d}  L={loss.item():.4f}  in_batch_top1={top1:.3f}  lr={opt.param_groups[0]['lr']:.2e}")

    # ---- Eval reranker on val ----
    reranker.eval()
    results = {}
    K = args.K_pool
    # Build K empZ candidates per anchor for val
    torch.manual_seed(123)
    val_zs = [Z_star_train[torch.randint(0, N_tr, (N_va,), device=device)] for _ in range(K)]
    print(f"\n=== Reranker eval (K_pool={K} empZ candidates per val anchor) ===")
    out = best_of_k_with_scorer(reranker, tr, H_t_val, val_zs, H_tp_val, N_va, "val", args.sub_ks, device)
    results["val"] = out

    # Compare: plain best-of-K' for K' in sub_ks using same candidates
    print(f"\n--- baseline best-of-K (no rerank) over same candidates ---")
    target = F.normalize(H_tp_val.mean(1), dim=-1)
    plain_preds = [tr(H_t_val, z) for z in val_zs]
    plain = {}
    for sub_k in args.sub_ks:
        best_rank = torch.full((N_va,), N_va + 1, device=device, dtype=torch.long)
        for j in range(sub_k):
            q = F.normalize(plain_preds[j].mean(1), dim=-1)
            sim = q @ target.T
            r = sim.argsort(dim=-1, descending=True)
            gt = torch.arange(N_va, device=device).unsqueeze(-1)
            rank_i = (r == gt).float().argmax(-1).long() + 1
            best_rank = torch.minimum(best_rank, rank_i)
        top1 = (best_rank == 1).float().mean().item()
        mr = best_rank.float().mean().item()
        print(f"  plain[K={K:3d}|top{sub_k:2d}|val         ] top1={top1:.3f}  mean_rank={mr:.2f}")
        plain[sub_k] = {"top1": top1, "mean_rank": mr}
    results["val_plain"] = plain

    torch.save({"reranker": reranker.state_dict(), "args": vars(args)}, os.path.join(args.out_dir, "reranker.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"\nsaved -> {args.out_dir}")


if __name__ == "__main__":
    main()
