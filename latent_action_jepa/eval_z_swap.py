"""Z-swap analysis: apply Z*_j to H_t_i. Does prediction land on H_tp_j (sample j's future)
applied to sample i's state, or somewhere else?
"""
import argparse
import os
import json
import torch
import torch.nn.functional as F

from models.transition_residual import ResidualTransitionModel


def retrieval_at_top(pred_pooled, target_pooled, gt_idx):
    """pred_pooled [N,D], target_pooled [N,D], gt_idx [N]: returns top1, top5."""
    pn = F.normalize(pred_pooled, dim=-1); tn = F.normalize(target_pooled, dim=-1)
    sim = pn @ tn.T
    r = sim.argsort(dim=-1, descending=True)
    top1 = (r[:, :1] == gt_idx.unsqueeze(-1)).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt_idx.unsqueeze(-1)).any(-1).float().mean().item()
    return top1, top5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tr = ResidualTransitionModel(cfg["state_dim"], cfg["action_dim"], cfg["future_tokens"],
                                 depth=cfg["transition_depth"]).to(device)
    tr.load_state_dict(ckpt["transition_model"])
    for p in tr.parameters(): p.requires_grad_(False)
    tr.eval()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t = bundle["H_t_val"].to(device); H_tp = bundle["H_tp_val"].to(device)
    Z_star = bundle["Z_star_val"].to(device)
    N = H_t.shape[0]

    with torch.no_grad():
        # Identity reference
        H_pred_correct = tr(H_t, Z_star)
        gt = torch.arange(N, device=device)
        t1, t5 = retrieval_at_top(H_pred_correct.mean(1), H_tp.mean(1), gt)
        print(f"correct (T(H_t_i, Z*_i) -> H_tp_i):           top1={t1:.3f}  top5={t5:.3f}")

        # Z-swap: Z*_j applied to H_t_i. Check both retrievals.
        # j = roll by 1.
        perm = torch.roll(torch.arange(N, device=device), 1)
        H_pred_swap = tr(H_t, Z_star[perm])
        # vs sample i's own H_tp
        t1_i, t5_i = retrieval_at_top(H_pred_swap.mean(1), H_tp.mean(1), gt)
        print(f"swap_Z vs H_tp_i (own target):                 top1={t1_i:.3f}  top5={t5_i:.3f}")
        # vs sample j's H_tp (the "Z donor"'s future)
        # If T(H_t_i, Z_j) ≈ H_tp_j, then comparing pred_i to H_tp[perm][i] = H_tp_j should be high.
        t1_j, t5_j = retrieval_at_top(H_pred_swap.mean(1), H_tp[perm].mean(1), gt)
        print(f"swap_Z vs H_tp_j (donor's target):             top1={t1_j:.3f}  top5={t5_j:.3f}")

        # Pure transformation: T(H_t_j, Z_j) -> H_tp_j  (sample j applying its own Z, anchor j)
        H_pred_donor_own = tr(H_t[perm], Z_star[perm])
        t1_o, t5_o = retrieval_at_top(H_pred_donor_own.mean(1), H_tp[perm].mean(1), gt)
        print(f"donor's own (T(H_t_j, Z*_j) -> H_tp_j):        top1={t1_o:.3f}  top5={t5_o:.3f}")

        # Random permutation, multiple seeds
        sw_i_list = []
        sw_j_list = []
        for seed in range(5):
            perm = torch.randperm(N, device=device, generator=torch.Generator(device=device).manual_seed(seed))
            H_pred_sw = tr(H_t, Z_star[perm])
            t1i, _ = retrieval_at_top(H_pred_sw.mean(1), H_tp.mean(1), gt)
            t1j, _ = retrieval_at_top(H_pred_sw.mean(1), H_tp[perm].mean(1), gt)
            sw_i_list.append(t1i); sw_j_list.append(t1j)
        print(f"swap (5 random perm avg)  vs own H_tp_i: {sum(sw_i_list)/5:.3f}")
        print(f"swap (5 random perm avg)  vs donor H_tp_j: {sum(sw_j_list)/5:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"correct_t1": t1, "swap_vs_own_t1": t1_i, "swap_vs_donor_t1": t1_j,
                   "donor_own_t1": t1_o,
                   "swap_random_avg_vs_own": sum(sw_i_list)/5,
                   "swap_random_avg_vs_donor": sum(sw_j_list)/5}, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
