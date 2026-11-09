"""Offline probe for a v6 ckpt: Z → ground-truth action, per action dimension.

Reports R² separately for xyz (dims 0-2), rotation (3-5), gripper (6) —
the in-loop scalar averages all 7 and can hide xyz signal under rotation noise.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(THIS, "..")))
sys.path.insert(0, THIS)

from bridge_clips import build_bridge_manifest, sample_bridge_batch
from models.vit3d import VisionTransformer3D
from models.la_modules import InverseFromDeltaPerStep

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data_root", default="/root/bridge/scripted_raw")
ap.add_argument("--n_batches", type=int, default=32)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--frame_stride", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--action_dim", type=int, default=16)
ap.add_argument("--dim", type=int, default=384)
ap.add_argument("--enc_depth", type=int, default=12)
ap.add_argument("--heads", type=int, default=6)
ap.add_argument("--inv_heads", type=int, default=8)
args = ap.parse_args()

device = "cuda"
torch.manual_seed(args.seed); random.seed(args.seed)

T, IMG, CTX, TUB, PATCH = 16, 224, 8, 2, 16
DIM = args.dim
grid_d, grid_s = T // TUB, IMG // PATCH
P = grid_s * grid_s
n_ctx_tubs = CTX // TUB
S = grid_d - n_ctx_tubs
A = args.action_dim

target = VisionTransformer3D(num_frames=T, tubelet_size=TUB, patch_size=PATCH,
                             img_size=IMG, embed_dim=DIM, depth=args.enc_depth, num_heads=args.heads,
                             mlp_ratio=4.0).to(device).eval()
inv = InverseFromDeltaPerStep(state_dim=DIM, action_dim=A, n_steps=S, n_patch=P,
                              depth=4, heads=args.inv_heads).to(device).eval()
ck = torch.load(args.ckpt, map_location=device, weights_only=False)
target.load_state_dict(ck["target"]); inv.load_state_dict(ck["inv"])
print(f"[probe] loaded {args.ckpt} (step {ck['step']})")

manifest = build_bridge_manifest(args.data_root)
rng = random.Random(0)
rng.shuffle(manifest)
val = manifest[:max(1, int(len(manifest) * 0.05))]   # same split as trainer seed 0

Z_all, act_all = [], []
with torch.no_grad():
    for i in range(args.n_batches):
        try:
            fr, acts, _, _ = sample_bridge_batch(val, args.bs, T, IMG, CTX, TUB,
                                                  grid_s, with_flow=False,
                                                  frame_stride=args.frame_stride)
        except Exception as e:
            print("skip", e); continue
        clips = fr.permute(0, 4, 1, 2, 3).contiguous().to(device)
        H = target(clips, keep_indices=None).view(fr.shape[0], grid_d, P, DIM)
        sl = [H[:, n_ctx_tubs - 1]] + [H[:, n_ctx_tubs + k] for k in range(S)]
        deltas = torch.stack([sl[k + 1] - sl[k] for k in range(S)], dim=1)
        Z = inv(deltas)
        Z_all.append(Z.cpu()); act_all.append(acts)
Z = torch.cat(Z_all).flatten(0, 1).float()        # [N*S, A]
Y = torch.cat(act_all).flatten(0, 1).float()      # [N*S, 7]
print(f"[probe] {Z.shape[0]} (clip,step) samples")

def ridge_r2_multi(X, Y, seeds=5, lam=1e-3):
    out = []
    for s in range(seeds):
        g = torch.Generator().manual_seed(s)
        idx = torch.randperm(X.shape[0], generator=g)
        cut = int(len(idx) * 0.8)
        tr, te = idx[:cut], idx[cut:]
        W = torch.linalg.solve(X[tr].T @ X[tr] + lam * torch.eye(X.shape[1]),
                               X[tr].T @ Y[tr])
        pred = X[te] @ W
        ss_res = ((pred - Y[te]) ** 2).sum(0)
        ss_tot = ((Y[te] - Y[te].mean(0)) ** 2).sum(0).clamp_min(1e-9)
        out.append(1 - ss_res / ss_tot)
    return torch.stack(out).mean(0)               # [7]

r2 = ridge_r2_multi(Z, Y)
groups = {"xyz": r2[:3].mean().item(), "rot": r2[3:6].mean().item(),
          "gripper": r2[6].item(), "all": r2.mean().item()}
print("[probe] R2 per dim:", [round(float(v), 3) for v in r2])
print("[probe] groups:", json.dumps({k: round(v, 4) for k, v in groups.items()}))

# shuffled-Z control
perm = torch.randperm(Z.shape[0])
r2s = ridge_r2_multi(Z[perm], Y)
print(f"[probe] shuffled-Z control all-dim R2: {r2s.mean().item():.4f}")
