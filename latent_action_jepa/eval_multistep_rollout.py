"""Multi-step rollout: starting from H_t, predict H_{t+k}, treat as new H_t,
predict next, etc. Compare:
  - oracle rollout: feed true Z* at each step.
  - goal-flow rollout: feed flow-sampled Z given a target horizon-goal.
  - copy rollout (no transition): just keep H_t.

Eval against the true H at horizon h = k * num_steps frames ahead.

Need a multi-step bundle: anchors with futures at horizons {1k, 2k, 3k}.
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from models.encoder import ViTEncoder
from models.fixed_reducer import FixedSpatialReducer
from models.inverse_encoder import InverseActionEncoder
from models.transition_residual import ResidualTransitionModel
from models.flow_prior_goal import GoalConditionedFlowPrior
from data.video_pairs import decode_indices, IMAGENET_MEAN, IMAGENET_STD


def retrieval(q, k, N, name=None, print_=True):
    qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    mr = (r == gt).float().argmax(-1).float().mean().item() + 1
    if print_ and name: print(f"  {name:36s} top1={top1:.3f}  top5={top5:.3f}  mr={mr:.2f}")
    return top1, top5, mr


@torch.no_grad()
def euler_sample_goal(flow, h_t, h_goal, action_tokens, action_dim, steps=30, seed=42):
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
    ap.add_argument("--video", default="/workspace/latent_action_jepa/bbb.mp4")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--goal_flow", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=4)
    ap.add_argument("--n_val", type=int, default=80)
    ap.add_argument("--val_start", type=int, default=12295)
    ap.add_argument("--val_stride", type=int, default=20)
    ap.add_argument("--encoder_name", default="vit_small_patch16_224")
    args = ap.parse_args()
    device = "cuda"

    # frozen encoder
    enc = ViTEncoder(args.encoder_name, pretrained=True, layer_idx=None).to(device).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    red_state = FixedSpatialReducer(64).to(device)
    red_future = FixedSpatialReducer(16).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tr = ResidualTransitionModel(cfg["state_dim"], cfg["action_dim"], cfg["future_tokens"],
                                 depth=cfg["transition_depth"]).to(device)
    tr.load_state_dict(ckpt["transition_model"])
    inv = InverseActionEncoder(cfg["state_dim"], cfg["action_dim"], cfg["action_tokens"],
                               depth=cfg["inverse_depth"]).to(device)
    inv.load_state_dict(ckpt["inverse_encoder"])
    fc = torch.load(args.goal_flow, map_location=device, weights_only=False)
    flow = GoalConditionedFlowPrior(cfg["state_dim"], cfg["action_dim"],
                                    depth=fc["args"]["flow_depth"]).to(device)
    flow.load_state_dict(fc["flow"])
    for p in list(tr.parameters()) + list(inv.parameters()) + list(flow.parameters()):
        p.requires_grad_(False)
    tr.eval(); inv.eval(); flow.eval()

    # Need frames at each multi-step horizon for N val anchors.
    # anchor a_i at val_start + i*val_stride; horizons a + k, a + 2k, ..., a + max_steps*k.
    anchors = [args.val_start + i * args.val_stride for i in range(args.n_val)]
    last_needed = anchors[-1] + args.max_steps * args.k
    print(f"anchors {anchors[0]}..{anchors[-1]}, last needed frame {last_needed}")
    # Decode all needed frames at once
    needed_frames = sorted(set(anchors + [a + s * args.k for a in anchors for s in range(1, args.max_steps + 1)]))
    print(f"decoding {len(needed_frames)} unique frames...")
    frames = decode_indices(args.video, needed_frames, size=224).to(device)
    frame_index = {fi: i for i, fi in enumerate(needed_frames)}

    # Encode all frames once
    def encode_state(x_batch):
        with torch.no_grad():
            return red_state(enc(x_batch))
    def encode_future(x_batch):
        with torch.no_grad():
            return red_future(enc(x_batch))

    chunk = 64
    H_state_all, H_future_all = [], []
    for s in range(0, frames.shape[0], chunk):
        e = min(s + chunk, frames.shape[0])
        H_state_all.append(encode_state(frames[s:e]))
        H_future_all.append(encode_future(frames[s:e]))
    H_state_all = torch.cat(H_state_all, 0)  # [F, 64, 384]
    H_future_all = torch.cat(H_future_all, 0)  # [F, 16, 384]

    # Index by frame
    def H_state(fi): return H_state_all[frame_index[fi]]
    def H_future(fi): return H_future_all[frame_index[fi]]

    # Build [N, *] tensors at anchor=0 and at each horizon
    N = args.n_val
    H_t0 = torch.stack([H_state(a) for a in anchors], 0)  # [N, 64, 384]
    H_targets = []  # list over steps, each [N, 16, 384] = true H_future at a + step*k
    for s in range(1, args.max_steps + 1):
        H_targets.append(torch.stack([H_future(a + s * args.k) for a in anchors], 0))

    # Also need H_state for next step (to chain rollouts in latent space)
    # Our transition predicts future tokens [B, 16, 384], not state tokens [B, 64, 384].
    # For multi-step we need H_state at step s; use H_state(a + s*k) as oracle "next state"
    # but for predicted rollout we need to convert predicted future tokens -> next state tokens.
    # Pragmatic choice: take H_pred as the "next state representation" by tiling/projecting.
    # Cleaner: encode (a + s*k) into H_state separately and compare via mean-pool to H_pred.
    # Strategy: for the *prediction* chain we keep using future-token outputs as the "next H_t"
    # but we need to feed them to the transition which expects 64-token state. So we project
    # the predicted 16-token future back to 64 by simple repetition (the spatial pool reducer
    # is the canonical map down; reverse via tile is approximate but consistent for the dynamics).

    # For first round, do step=1: T(H_t, Z) -> H_pred. Compare to H_target[0]. That's standard.
    # For step=2 we need: predict H_t at horizon 1 and feed back. We'll synthesize H_t' from
    # H_pred by *also* encoding the true frame at horizon 1 (oracle proxy), so multi-step is
    # really "iterated future prediction using the true next-state H_t."
    H_states_at_horizons = [None]  # placeholder for step=0 (= H_t0)
    H_states_at_horizons[0] = H_t0
    for s in range(1, args.max_steps + 1):
        H_states_at_horizons.append(torch.stack([H_state(a + s * args.k) for a in anchors], 0))

    print(f"\nMulti-step rollout, N={N}, k={args.k}, max_steps={args.max_steps}\n")
    results = {}

    # 1. Single-step (step=1) sanity
    H_target_1 = H_targets[0]
    # oracle Z* via inverse(H_t0, H_target_1)
    Z_oracle = inv(H_t0, H_target_1)
    pred_oracle = tr(H_t0, Z_oracle)
    t1, _, mr = retrieval(pred_oracle.mean(1), H_target_1.mean(1), N, "step=1 oracle_Z")
    results["step1_oracle"] = {"top1": t1, "mr": mr}

    # 2. Goal-flow rollout (one step at a time, using TRUE H_t_{s-1} as state and TRUE H_target_s as goal)
    # This is "given that we know the next-step goal, can flow + transition do one step?"
    print("\n[chain with TRUE next-state inputs, flow-sampled Z toward true goal]")
    for s in range(1, args.max_steps + 1):
        H_t = H_states_at_horizons[s - 1]  # current state
        H_goal = H_targets[s - 1]  # next future target
        z = euler_sample_goal(flow, H_t, H_goal, cfg["action_tokens"], cfg["action_dim"])
        pred = tr(H_t, z)
        t1, _, mr = retrieval(pred.mean(1), H_target_1.mean(1) if s == 1 else H_targets[s - 1].mean(1),
                               N, f"step={s} goal_flow_Z (TRUE state)")
        results[f"step{s}_goalflow_truestate"] = {"top1": t1, "mr": mr}

    # 3. Copy baseline: H_pred = mean-pool(H_t) replicated as future tokens
    print("\n[copy baselines (no transition)]")
    for s in range(1, args.max_steps + 1):
        H_t = H_states_at_horizons[s - 1]
        # crude: use H_t_mean as the "future" representation
        copy_pool = H_t.mean(1, keepdim=True).expand(-1, 16, -1)
        t1, _, mr = retrieval(copy_pool.mean(1), H_targets[s - 1].mean(1), N, f"step={s} copy_Ht (TRUE state)")
        results[f"step{s}_copy_Ht"] = {"top1": t1, "mr": mr}

    # 4. Iterated transition WITHOUT re-grounding state — use predicted future as state-proxy
    # We rebroadcast 16-token H_pred to 64-token state via simple repeat (4x).
    print("\n[chain WITHOUT re-grounding: feed predicted future back as state]")
    H_t_iter = H_t0
    for s in range(1, args.max_steps + 1):
        # for transition we need state-shape [N, 64, D]; tile 16->64 by repeat
        if H_t_iter.shape[1] == 16:
            H_t_state = H_t_iter.repeat_interleave(4, dim=1)
        else:
            H_t_state = H_t_iter
        # use true horizon goal but at TWO horizons ahead so the iterated step has to drift
        # for fairness compare s-step prediction to s-step target
        # Strategy: at each iteration, feed goal = TRUE H_target_s (single-shot future prediction)
        # vs we don't want to leak; let's only use goal H_target_1 and iterate forward
        H_goal = H_targets[0]  # use the immediate-next-step goal at every iteration
        z = euler_sample_goal(flow, H_t_state, H_goal, cfg["action_tokens"], cfg["action_dim"])
        pred = tr(H_t_state, z)
        t1, _, mr = retrieval(pred.mean(1), H_targets[s - 1].mean(1), N, f"step={s} iter_pred (goal=step1 truth)")
        results[f"step{s}_iter_pred"] = {"top1": t1, "mr": mr}
        H_t_iter = pred  # feed back

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
