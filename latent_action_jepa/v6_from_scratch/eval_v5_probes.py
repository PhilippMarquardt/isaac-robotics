"""Proper evaluation for a v5 checkpoint.

Three honest metrics that don't suffer from the encoder-fingerprint shortcut:

1. **Z → video_id linear probe**: pick V videos, sample C clips each → V*C clips.
   Train a 1-vs-rest linear classifier on Z to predict video_id. Test accuracy
   vs chance (1/V).
   - High accuracy (>>1/V) = Z encodes video identity = bad transferability.
   - At chance = Z is video-invariant = good.

2. **Within-video next-tubelet retrieval**: for each video, sample two clips
   at different time offsets. Run the transition with the right Z (from
   the actual future) vs the *wrong* Z (from a different time in the same
   video). Retrieval among all candidates from the SAME video. Trivial
   shortcut "encoder fingerprints the video" no longer wins, because all
   candidates come from the same video.

3. **Effective rank of Z** — how many independent dims Z actually uses.

Use:
    python eval_v5_probes.py \\
        --ckpt /root/jepa_v5/runs/v5_c/ckpt_step3000.pt \\
        --index /root/jepa_v5/charades_index.json \\
        --n_videos 64 --clips_per_video 4
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
LAJ = os.path.normpath(os.path.join(THIS, ".."))
sys.path.insert(0, LAJ)
sys.path.insert(0, THIS)

from charades_clips import _decode_clip
from models.vit3d import VisionTransformer3D
from models.la_modules import (
    StepBase, InverseFromDeltaPerStep,
    MultiStepTransitionPerStep, GoalConditionedDiTFlow,
)


def load_ckpt(ckpt_path, device, *, num_frames, tubelet_size, patch_size, img_size,
              embed_dim, depth, num_heads, action_steps, action_dim,
              inverse_depth, base_depth, transition_depth, flow_depth, flow_heads,
              alpha_min, alpha_init, n_patch):
    online = VisionTransformer3D(
        num_frames=num_frames, tubelet_size=tubelet_size, patch_size=patch_size,
        img_size=img_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        mlp_ratio=4.0,
    ).to(device).eval()
    target_enc = VisionTransformer3D(
        num_frames=num_frames, tubelet_size=tubelet_size, patch_size=patch_size,
        img_size=img_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        mlp_ratio=4.0,
    ).to(device).eval()
    base = StepBase(state_dim=embed_dim, n_steps=action_steps, depth=base_depth).to(device).eval()
    inv = InverseFromDeltaPerStep(state_dim=embed_dim, action_dim=action_dim,
                                   n_steps=action_steps, n_patch=n_patch,
                                   depth=inverse_depth).to(device).eval()
    tr = MultiStepTransitionPerStep(state_dim=embed_dim, action_dim=action_dim,
                                     n_steps=action_steps, n_patch=n_patch,
                                     depth=transition_depth, alpha_min=alpha_min,
                                     alpha_init=alpha_init).to(device).eval()
    flow = GoalConditionedDiTFlow(state_dim=embed_dim, action_dim=action_dim,
                                   n_steps=action_steps, depth=flow_depth,
                                   heads=flow_heads).to(device).eval()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    online.load_state_dict(ck["online"])
    target_enc.load_state_dict(ck["target"])
    base.load_state_dict(ck["base"])
    inv.load_state_dict(ck["inv"])
    tr.load_state_dict(ck["tr"])
    flow.load_state_dict(ck["flow"])
    return online, target_enc, base, inv, tr, flow


@torch.no_grad()
def encode_clips(target_enc, clips, grid_d, n_patch):
    """clips: [B, T, H, W, 3] float in [-1, 1]. Returns H_full [B, grid_d, P, D]."""
    x = clips.permute(0, 4, 1, 2, 3).contiguous().to(next(target_enc.parameters()).device)
    H_full = target_enc(x, keep_indices=None)
    B, D = H_full.shape[0], H_full.shape[-1]
    return H_full.view(B, grid_d, n_patch, D)


@torch.no_grad()
def get_Z_for_clips(target_enc, base, inv, clips, grid_d, n_ctx_tubs, n_patch):
    H_full = encode_clips(target_enc, clips, grid_d, n_patch)
    H_target = H_full[:, n_ctx_tubs:]
    H_t = H_full[:, :n_ctx_tubs][:, -1]
    delta = H_target - base(H_t)
    Z = inv(delta)
    return Z, H_t, H_target


def video_id_probe(Z_all, labels_all, n_train_per_class, n_test_per_class):
    """Simple closed-form ridge probe: Z → one-hot label."""
    V = labels_all.max().item() + 1
    Z_all = Z_all.float()
    perm_per_class = {}
    for v in range(V):
        idx = (labels_all == v).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx))]
        perm_per_class[v] = idx
    train_idx, test_idx = [], []
    for v in range(V):
        idx = perm_per_class[v]
        train_idx.extend(idx[:n_train_per_class].tolist())
        test_idx.extend(idx[n_train_per_class:n_train_per_class + n_test_per_class].tolist())
    if not train_idx or not test_idx:
        return None
    Xtr = Z_all[train_idx]; ytr = labels_all[train_idx]
    Xte = Z_all[test_idx];  yte = labels_all[test_idx]
    # Ridge: W = (X^T X + λI)^-1 X^T Y
    Y = F.one_hot(ytr, V).float()
    lam = 1e-3
    XtX = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], device=Xtr.device)
    W = torch.linalg.solve(XtX, Xtr.T @ Y)  # [D, V]
    pred = (Xte @ W).argmax(dim=-1)
    acc = (pred == yte).float().mean().item()
    return acc, 1.0 / V


def within_video_retrieval(target_enc, base, inv, tr, video_clips, grid_d,
                           n_ctx_tubs, n_patch, n_clips_per_video):
    """For each video, take C clips at different times. For each clip i, ask
    "is transition(H_t_i, Z_real_i) closer to H_target_i than to H_target_j for
    j in same-video other times?" Reports per-video top-1 (chance = 1/C)."""
    device = next(target_enc.parameters()).device
    accs = []
    for v_idx in range(0, video_clips.shape[0], n_clips_per_video):
        clips = video_clips[v_idx:v_idx + n_clips_per_video].to(device)
        if clips.shape[0] < 2:
            continue
        Z, H_t, H_target = get_Z_for_clips(target_enc, base, inv, clips,
                                            grid_d, n_ctx_tubs, n_patch)
        H_pred = tr(H_t, Z)  # [C, S, P, D]
        # Compare H_pred[i] vs H_target[j] across all j in same video.
        # use mean-of-patch per step, average sim across steps.
        C, S, P, D = H_pred.shape
        q = F.normalize(H_pred.mean(dim=2), dim=-1)   # [C, S, D]
        k = F.normalize(H_target.mean(dim=2), dim=-1)  # [C, S, D]
        per_step = []
        for s in range(S):
            sim = q[:, s] @ k[:, s].T   # [C, C]
            gt = torch.arange(C, device=device)
            top1 = (sim.argmax(dim=-1) == gt).float().mean().item()
            per_step.append(top1)
        accs.append(sum(per_step) / max(1, len(per_step)))
    if not accs:
        return None
    return sum(accs) / len(accs), 1.0 / n_clips_per_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", default="/root/jepa_v5/charades_index.json")
    ap.add_argument("--n_videos", type=int, default=64,
                    help="number of distinct videos to evaluate")
    ap.add_argument("--clips_per_video", type=int, default=4,
                    help="clips per video at different time offsets")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--frame_stride", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--tubelet_size", type=int, default=2)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--embed_dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=6)
    ap.add_argument("--action_steps", type=int, default=4)
    ap.add_argument("--action_dim", type=int, default=16)
    ap.add_argument("--inverse_depth", type=int, default=4)
    ap.add_argument("--base_depth", type=int, default=2)
    ap.add_argument("--transition_depth", type=int, default=4)
    ap.add_argument("--flow_depth", type=int, default=8)
    ap.add_argument("--flow_heads", type=int, default=4)
    ap.add_argument("--n_context_tubelets", type=int, default=4)
    ap.add_argument("--alpha_min", type=float, default=0.02)
    ap.add_argument("--alpha_init", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed); random.seed(args.seed)

    grid_d = args.num_frames // args.tubelet_size
    grid_s = args.img_size // args.patch_size
    n_patch = grid_s * grid_s
    n_ctx_tubs = args.n_context_tubelets

    print(f"[eval] loading {args.ckpt}")
    online, target_enc, base, inv, tr, flow = load_ckpt(
        args.ckpt, device,
        num_frames=args.num_frames, tubelet_size=args.tubelet_size,
        patch_size=args.patch_size, img_size=args.img_size, embed_dim=args.embed_dim,
        depth=args.depth, num_heads=args.num_heads, action_steps=args.action_steps,
        action_dim=args.action_dim, inverse_depth=args.inverse_depth,
        base_depth=args.base_depth, transition_depth=args.transition_depth,
        flow_depth=args.flow_depth, flow_heads=args.flow_heads,
        alpha_min=args.alpha_min, alpha_init=args.alpha_init, n_patch=n_patch,
    )

    with open(args.index) as f:
        index = json.load(f)
    rng = random.Random(args.seed)
    rng.shuffle(index)
    span = (args.num_frames - 1) * args.frame_stride + 1
    eligible = [v for v in index if v["n_frames"] >= span * args.clips_per_video + 32]
    if not eligible:
        raise SystemExit("not enough long-enough videos")
    chosen = eligible[:args.n_videos]
    print(f"[eval] using {len(chosen)} videos, {args.clips_per_video} clips each "
          f"= {len(chosen) * args.clips_per_video} total clips")

    # Sample clips: V*C × T × H × W × 3.
    all_clips = []
    labels = []
    for v_id, spec in enumerate(chosen):
        n = spec["n_frames"]
        max_start = n - span - 1
        # Pick C anchor frames, well-spaced.
        anchors = sorted(rng.sample(range(max_start), args.clips_per_video))
        for a in anchors:
            try:
                arr = _decode_clip(spec["path"], a, args.num_frames,
                                    args.frame_stride, args.img_size)
            except Exception:
                arr = None
            if arr is None:
                continue
            t = torch.from_numpy(arr).float() / 127.5 - 1.0
            all_clips.append(t)
            labels.append(v_id)
    if not all_clips:
        raise SystemExit("no clips decoded")
    video_clips = torch.stack(all_clips, dim=0)  # [V*C, T, H, W, 3]
    labels = torch.tensor(labels, device=device)
    print(f"[eval] sampled {video_clips.shape[0]} clips")

    # === Get Z per clip ===
    BATCH = 32
    Z_list = []
    H_t_list = []
    H_target_list = []
    for i in range(0, video_clips.shape[0], BATCH):
        chunk = video_clips[i:i + BATCH].to(device)
        Z, H_t, H_target = get_Z_for_clips(target_enc, base, inv, chunk, grid_d,
                                            n_ctx_tubs, n_patch)
        Z_list.append(Z.cpu())
        H_t_list.append(H_t.cpu())
        H_target_list.append(H_target.cpu())
    Z_all = torch.cat(Z_list, dim=0)              # [N, S, A]
    Z_flat = Z_all.flatten(1)                      # [N, S*A]
    print(f"[eval] Z shape {tuple(Z_all.shape)}  flat {tuple(Z_flat.shape)}")
    print(f"[eval] Z std={Z_all.std().item():.4f}  Z mean abs={Z_all.abs().mean().item():.4f}")

    # === 1. Effective rank ===
    Zf = Z_flat.float()
    Zc = Zf - Zf.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Zc)
    p = (s ** 2) / (s ** 2).sum().clamp_min(1e-12)
    eff_rank = float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item())
    print(f"[eval] effective_rank(Z) = {eff_rank:.2f}  (max {Zf.shape[1]})")

    # === 2. Z → video_id probe ===
    Z_dev = Zf.to(device)
    probe_result = video_id_probe(Z_dev, labels,
                                    n_train_per_class=max(1, args.clips_per_video - 1),
                                    n_test_per_class=1)
    if probe_result:
        acc, chance = probe_result
        print(f"[eval] Z → video_id  probe acc = {acc:.3f}  (chance {chance:.3f}, "
              f"{acc/chance:.1f}× chance)")
    else:
        print("[eval] not enough clips per video for probe")

    # === 3. Within-video retrieval ===
    res = within_video_retrieval(target_enc, base, inv, tr, video_clips, grid_d,
                                   n_ctx_tubs, n_patch, args.clips_per_video)
    if res:
        wv_acc, wv_chance = res
        print(f"[eval] within-video retrieval acc = {wv_acc:.3f}  (chance {wv_chance:.3f}, "
              f"{wv_acc/wv_chance:.1f}× chance)")
    else:
        print("[eval] within-video retrieval failed")

    print("\n[eval] DONE")


if __name__ == "__main__":
    main()
