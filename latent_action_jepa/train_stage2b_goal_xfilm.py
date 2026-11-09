"""Goal-flow with CROSS-FILM hard negatives.

p_neg fraction of the batch gets a shuffled goal that is FORCED to come from a
different film than the anchor. Target Z = inv(H_t, H_goal). This pushes the
flow to handle any (state, goal) combination, not just within-film.
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from models.inverse_encoder import InverseActionEncoder
from models.transition_residual import ResidualTransitionModel
from models.flow_prior_goal import GoalConditionedFlowPrior


def retrieval(q, k, N, name=""):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    mr = (r == gt).float().argmax(-1).float().mean().item() + 1
    if name: print(f"  {name:28s} top1={top1:.3f}  top5={top5:.3f}  mr={mr:.2f}")
    return top1, top5, mr


@torch.no_grad()
def euler_sample(flow, h_t, h_goal, action_tokens, action_dim, steps=30, seed=42):
    b = h_t.shape[0]
    g = torch.Generator(device=h_t.device).manual_seed(seed)
    z = torch.randn(b, action_tokens, action_dim, device=h_t.device, generator=g)
    for i in range(steps):
        tau = torch.full((b, 1, 1), i / steps, device=h_t.device)
        v = flow(z, tau, h_t, h_goal)
        z = z + v / steps
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--flow_depth", type=int, default=6)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--sample_steps", type=int, default=30)
    ap.add_argument("--p_neg", type=float, default=0.5)
    ap.add_argument("--xfilm_frac", type=float, default=0.8, help="frac of negs that come from a DIFFERENT film")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(0)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tr = ResidualTransitionModel(cfg["state_dim"], cfg["action_dim"], cfg["future_tokens"],
                                 depth=cfg["transition_depth"]).to(device)
    tr.load_state_dict(ckpt["transition_model"])
    inv = InverseActionEncoder(cfg["state_dim"], cfg["action_dim"], cfg["action_tokens"],
                               depth=cfg["inverse_depth"]).to(device)
    inv.load_state_dict(ckpt["inverse_encoder"])
    for p in list(tr.parameters()) + list(inv.parameters()):
        p.requires_grad_(False)
    tr.eval(); inv.eval()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t = bundle["H_t_train"].to(device); H_tp = bundle["H_tp_train"].to(device)
    vid_tr = bundle["video_id_train"].to(device)
    H_t_val = bundle["H_t_val"].to(device); H_tp_val = bundle["H_tp_val"].to(device)
    Z_star_val = bundle["Z_star_val"].to(device)
    vid_va = bundle["video_id_val"].to(device)
    N_tr = H_t.shape[0]; N_va = H_t_val.shape[0]
    n_films = int(vid_tr.max().item() + 1)
    print(f"train {N_tr}  val {N_va}  n_films={n_films}")

    # Build film index buckets for cross-film sampling
    film_indices = [torch.nonzero(vid_tr == f, as_tuple=False).flatten() for f in range(n_films)]

    flow = GoalConditionedFlowPrior(cfg["state_dim"], cfg["action_dim"], depth=args.flow_depth).to(device)
    print(f"flow params: {sum(p.numel() for p in flow.parameters())/1e6:.2f} M  p_neg={args.p_neg}  xfilm_frac={args.xfilm_frac}")
    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    flow.train()
    for step in range(args.steps):
        idx = torch.randint(0, N_tr, (args.bs,), device=device)
        h_t = H_t[idx]
        film_anchor = vid_tr[idx]
        h_goal = H_tp[idx].clone()

        # Choose negatives
        is_neg = (torch.rand(args.bs, device=device) < args.p_neg)
        if is_neg.any():
            # of the negatives, xfilm_frac should be cross-film
            is_xfilm = is_neg & (torch.rand(args.bs, device=device) < args.xfilm_frac)
            # cross-film: pick a random film != anchor film, then a random idx from that film
            for i in is_xfilm.nonzero(as_tuple=False).flatten().tolist():
                # pick different film
                other_film = torch.randint(0, n_films, (1,), device=device).item()
                while other_film == film_anchor[i].item():
                    other_film = torch.randint(0, n_films, (1,), device=device).item()
                pool = film_indices[other_film]
                j = pool[torch.randint(0, pool.shape[0], (1,), device=device)].item()
                h_goal[i] = H_tp[j]
            # remaining negs: within-batch random shuffle (may be within-film)
            within = is_neg & ~is_xfilm
            if within.any():
                perm = torch.randperm(args.bs, device=device)
                wi = within.nonzero(as_tuple=False).flatten()
                h_goal[wi] = H_tp[idx][perm][wi]

        with torch.no_grad():
            z_target = inv(h_t, h_goal)
        eps = torch.randn_like(z_target)
        tau = torch.rand(z_target.shape[0], 1, 1, device=device)
        z_tau = (1 - tau) * eps + tau * z_target
        v_target = z_target - eps
        v_pred = flow(z_tau, tau, h_t, h_goal)
        loss = F.mse_loss(v_pred, v_target)

        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  L_flow={loss.item():.4f}  lr={opt.param_groups[0]['lr']:.2e}")

    flow.eval()
    print(f"\n=== Eval val (N={N_va}) ===")
    results = {}
    with torch.no_grad():
        K_max = max(args.ks)
        samples = [euler_sample(flow, H_t_val, H_tp_val, cfg["action_tokens"], cfg["action_dim"],
                                steps=args.sample_steps, seed=10 + k) for k in range(K_max)]
        preds = [tr(H_t_val, z) for z in samples]
        retrieval(tr(H_t_val, Z_star_val).mean(1), H_tp_val.mean(1), N_va, "oracle_Z")
        retrieval(preds[0].mean(1), H_tp_val.mean(1), N_va, "goal_flow K=1")
        for k in args.ks:
            target = F.normalize(H_tp_val.mean(1), dim=-1)
            best_rank = torch.full((N_va,), N_va + 1, device=device, dtype=torch.long)
            for j in range(k):
                q = F.normalize(preds[j].mean(1), dim=-1)
                sim = q @ target.T
                r = sim.argsort(dim=-1, descending=True)
                gt = torch.arange(N_va, device=device).unsqueeze(-1)
                rank_i = (r == gt).float().argmax(-1).long() + 1
                best_rank = torch.minimum(best_rank, rank_i)
            top1 = (best_rank == 1).float().mean().item()
            mr = best_rank.float().mean().item()
            print(f"  bestK={k:2d} goal_flow top1={top1:.3f}  mr={mr:.2f}")
            results[f"K{k}"] = {"top1": top1, "mr": mr}

        # cross-film goal test: anchor from film A, goal from film B
        # pick anchor/film pairs and randomly assign goals from different films
        perm = torch.randperm(N_va, device=device)
        # find a goal from different film
        new_goal = H_tp_val.clone()
        for i in range(N_va):
            other = perm[i].item()
            tries = 0
            while vid_va[other].item() == vid_va[i].item() and tries < 50:
                other = torch.randint(0, N_va, (1,)).item()
                tries += 1
            new_goal[i] = H_tp_val[other]
        z_xf = euler_sample(flow, H_t_val, new_goal, cfg["action_tokens"], cfg["action_dim"], steps=args.sample_steps, seed=999)
        pred_xf = tr(H_t_val, z_xf)
        # retrieval against TRUE H_tp (the anchor's own)
        t1_own, _, mr_own = retrieval(pred_xf.mean(1), H_tp_val.mean(1), N_va, "xfilm goal vs own_target")
        # retrieval against the cross-film goal target
        t1_g, _, mr_g = retrieval(pred_xf.mean(1), new_goal.mean(1), N_va, "xfilm goal vs given_target")
        results["xfilm_vs_own"] = t1_own
        results["xfilm_vs_given"] = t1_g

    torch.save({"flow": flow.state_dict(), "args": vars(args)}, os.path.join(args.out_dir, "flow.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
