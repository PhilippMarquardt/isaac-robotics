"""Baseline priors for Stage 2: kNN (over train Z* indexed by train H_t),
empirical sampling from train Z*, and per-anchor best-of-K on these.

If a free baseline matches/beats flow on val, the flow isn't adding signal.
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from models.inverse_encoder import InverseActionEncoder
from models.transition_residual import ResidualTransitionModel
from diag_stage0 import effective_rank


def retrieval(q, k, N, name=None, print_=True):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    rank = (r == gt).float().argmax(-1).float().mean().item() + 1
    if print_ and name:
        print(f"  retrieval[{name:24s}] top1={top1:.3f}  top5={top5:.3f}  mean_rank={rank:.2f}")
    return top1, top5, rank


def best_of_k_retrieval(preds_list, H_tp, N, name):
    target = F.normalize(H_tp.mean(1), dim=-1)
    best_rank = torch.full((N,), N + 1, device=H_tp.device, dtype=torch.long)
    for kp in preds_list:
        q = F.normalize(kp.mean(1), dim=-1)
        sim = q @ target.T
        r = sim.argsort(dim=-1, descending=True)
        gt = torch.arange(N, device=H_tp.device).unsqueeze(-1)
        rank_i = (r == gt).float().argmax(-1).long() + 1
        best_rank = torch.minimum(best_rank, rank_i)
    top1 = (best_rank == 1).float().mean().item()
    top5 = (best_rank <= 5).float().mean().item()
    mean_rank = best_rank.float().mean().item()
    print(f"  bestK[K={len(preds_list):2d}|{name:18s}] top1={top1:.3f}  top5={top5:.3f}  mean_rank={mean_rank:.2f}")
    return {"top1": top1, "top5": top5, "mean_rank": mean_rank}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/latent_action_jepa/runs/stage1c/ckpt.pt")
    ap.add_argument("--bundle", default="/workspace/latent_action_jepa/runs/stage1c/oracle_bundle.pt")
    ap.add_argument("--out_dir", default="/workspace/latent_action_jepa/runs/stage2_ablations/baselines_2k")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--skip_train", action="store_true", help="only eval val (cheap)")
    ap.add_argument("--chunk", type=int, default=400, help="chunk size for transition forward")
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
    print(f"loaded transition: alpha={tr.alpha.item():.4f}")

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t_train = bundle["H_t_train"].to(device); H_tp_train = bundle["H_tp_train"].to(device)
    Z_star_train = bundle["Z_star_train"].to(device)
    H_t_val = bundle["H_t_val"].to(device); H_tp_val = bundle["H_tp_val"].to(device)
    Z_star_val = bundle["Z_star_val"].to(device)
    N_tr = H_t_train.shape[0]; N_va = H_t_val.shape[0]
    print(f"train N={N_tr}  val N={N_va}")

    # ---- Compute train H_t embeddings for kNN ----
    train_keys = F.normalize(H_t_train.mean(1), dim=-1)  # [N_tr, D]
    K_max = max(args.ks)

    results = {}

    splits = [("val", H_t_val, H_tp_val, Z_star_val, N_va)]
    if not args.skip_train:
        splits.append(("train", H_t_train, H_tp_train, Z_star_train, N_tr))
    for tag, H_t, H_tp, Z_star, N in splits:
        print(f"\n=== [{tag}]  N={N} ===")
        sec = {}

        def tr_chunked(H, Z, chunk=args.chunk):
            outs = []
            for s in range(0, H.shape[0], chunk):
                e = min(s + chunk, H.shape[0])
                outs.append(tr(H[s:e], Z[s:e]))
            return torch.cat(outs, 0)

        with torch.no_grad():
            # oracle
            t1, t5, mr = retrieval(tr_chunked(H_t, Z_star).mean(1), H_tp.mean(1), N, "oracle_Z")
            sec["oracle_Z"] = {"top1": t1, "top5": t5, "mean_rank": mr}

            # zero / random / copy
            t1, t5, mr = retrieval(tr_chunked(H_t, torch.zeros_like(Z_star)).mean(1), H_tp.mean(1), N, "zero_Z")
            sec["zero_Z"] = {"top1": t1, "top5": t5, "mean_rank": mr}
            std_z = Z_star.flatten(1).std(0).mean().item()
            t1, t5, mr = retrieval(tr_chunked(H_t, torch.randn_like(Z_star) * std_z).mean(1), H_tp.mean(1), N, "random_Z(stdZ*)")
            sec["random_Z"] = {"top1": t1, "top5": t5, "mean_rank": mr}
            t1, t5, mr = retrieval(H_t.mean(1), H_tp.mean(1), N, "copy_Ht")
            sec["copy_Ht"] = {"top1": t1, "top5": t5, "mean_rank": mr}

            queries = F.normalize(H_t.mean(1), dim=-1)
            sim = queries @ train_keys.T
            if tag == "train":
                sim.fill_diagonal_(-1e9)
            topk_idx = sim.topk(K_max, dim=-1).indices
            z_knn1 = Z_star_train[topk_idx[:, 0]]
            t1, t5, mr = retrieval(tr_chunked(H_t, z_knn1).mean(1), H_tp.mean(1), N, "kNN1_Z")
            sec["kNN1_Z"] = {"top1": t1, "top5": t5, "mean_rank": mr}

            print("  -- best-of-K, kNN neighbors --")
            knn_preds_per_k = []
            for k in range(K_max):
                z_k = Z_star_train[topk_idx[:, k]]
                knn_preds_per_k.append(tr_chunked(H_t, z_k))
            for k in args.ks:
                r = best_of_k_retrieval(knn_preds_per_k[:k], H_tp, N, f"kNN_top{k}")
                sec[f"bestK_kNN_{k}"] = r

            print("  -- best-of-K, empirical Z* (uniform train sample) --")
            emp_preds = []
            for ki in range(K_max):
                idx = torch.randint(0, N_tr, (N,), device=device)
                emp_preds.append(tr_chunked(H_t, Z_star_train[idx]))
            for k in args.ks:
                r = best_of_k_retrieval(emp_preds[:k], H_tp, N, f"empZ_{k}")
                sec[f"bestK_empZ_{k}"] = r

            z_mu = Z_star_train.flatten(1).mean(0)
            z_std = Z_star_train.flatten(1).std(0)
            print("  -- best-of-K, factored Gaussian (train moments) --")
            fg_preds = []
            for ki in range(K_max):
                noise = torch.randn(N, z_mu.numel(), device=device) * z_std + z_mu
                fg_preds.append(tr_chunked(H_t, noise.view(N, cfg["action_tokens"], cfg["action_dim"])))
            for k in args.ks:
                r = best_of_k_retrieval(fg_preds[:k], H_tp, N, f"fgZ_{k}")
                sec[f"bestK_fgZ_{k}"] = r

        results[tag] = sec

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"\nsaved -> {args.out_dir}")


if __name__ == "__main__":
    main()
