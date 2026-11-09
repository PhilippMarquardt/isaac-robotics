"""v6 trainer — from-scratch latent-action JEPA on BridgeData V2.

Implements METHOD.md:
  A: V-JEPA 3D multi-block masked prediction   -> trains encoder (only loss that does)
  B: chain latent action, teacher-forced       -> trains inverse + transition
  C: composition/rollout consistency           -> same modules, real multi-step
  D: classical-flow grounding head             -> anti appearance-leakage
  E: flow-matching DiT prior over Z            -> action prior (z stop-gradded)
  regs: VICReg variance + covariance on Z
  launch gate: U = 1 - L_B(z*)/L_B(z_shuffled) must exceed --gate_u after --gate_steps
  in-loop eval: U, eff_rank(Z), scene-leakage probe, GT-action probe R^2

Chain form: z_k explains s_{t+k-1} -> s_{t+k} (one tubelet = 0.4 s at 5 Hz).
All action-stack inputs come from the EMA target encoder, stop-gradded.
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
LAJ = os.path.normpath(os.path.join(THIS, ".."))
sys.path.insert(0, LAJ)
sys.path.insert(0, THIS)

from bridge_clips import build_bridge_manifest, sample_bridge_batch  # noqa: E402
from models.vit3d import VisionTransformer3D, Block, get_3d_sincos_pos_embed  # noqa: E402
from models.la_modules import (  # noqa: E402
    InverseFromDeltaPerStep, MultiStepTransitionPerStep, GoalConditionedDiTFlow,
)


# ---------------------------------------------------------------- JEPA parts

def sample_multiblock_mask(bs, grid_d, grid_h, grid_w, device,
                           n_blocks=(4, 8), target_ratio=(0.4, 0.6)):
    """Random 3D cuboid masks. Returns bool [B, N] True = MASKED."""
    N = grid_d * grid_h * grid_w
    masks = torch.zeros(bs, grid_d, grid_h, grid_w, dtype=torch.bool, device=device)
    for b in range(bs):
        want = random.uniform(*target_ratio) * N
        nb = random.randint(*n_blocks)
        while masks[b].sum() < want:
            dd = random.randint(max(1, grid_d // 4), grid_d)
            hh = random.randint(2, max(2, grid_h // 2))
            ww = random.randint(2, max(2, grid_w // 2))
            d0 = random.randint(0, grid_d - dd)
            h0 = random.randint(0, grid_h - hh)
            w0 = random.randint(0, grid_w - ww)
            masks[b, d0:d0 + dd, h0:h0 + hh, w0:w0 + ww] = True
            nb -= 1
            if nb <= 0 and masks[b].sum() >= want * 0.5:
                break
    return masks.flatten(1)  # [B, N]


class MaskedPredictor(nn.Module):
    """Narrow ViT: visible tokens + learned mask queries -> masked-token features."""

    def __init__(self, dim_in=384, dim=192, depth=6, heads=6, grid=(8, 14, 14)):
        super().__init__()
        self.proj_in = nn.Linear(dim_in, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        pos = get_3d_sincos_pos_embed(dim, grid[1], grid[0])
        self.register_buffer("pos", pos.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads, 4.0, 0.0) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, dim_in)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, h_vis, vis_idx, mask_idx_list):
        """h_vis [B, Nv, D_in]; vis_idx [B, Nv]; mask_idx_list: list of LongTensor per sample.
        Returns list of [n_mask_b, D_in] predictions per sample."""
        B = h_vis.shape[0]
        x = self.proj_in(h_vis)
        dim = x.shape[-1]
        pos = self.pos.expand(B, -1, -1)                       # [B, N, dim]
        pos_vis = pos.gather(1, vis_idx.unsqueeze(-1).expand(-1, -1, dim))
        x = x + pos_vis
        outs = []
        for b in range(B):
            mi = mask_idx_list[b]
            q = self.mask_token.expand(1, mi.shape[0], dim) + pos[b:b + 1, mi]
            seq = torch.cat([x[b:b + 1], q], dim=1)
            for blk in self.blocks:
                seq = blk(seq)
            seq = self.norm(seq)
            outs.append(self.proj_out(seq[0, x.shape[1]:]))
        return outs


class FMPredictor(nn.Module):
    """Generative JEPA predictor: conditional flow matching over masked target
    tokens instead of point regression. v_w(y_tau, tau, H_vis, pos) -> velocity.

    Same narrow-ViT skeleton as MaskedPredictor; the mask queries carry the
    noisy target token y_tau plus a tau embedding instead of a learned token.
    """

    def __init__(self, dim_in=384, dim=192, depth=6, heads=6, grid=(8, 14, 14)):
        super().__init__()
        self.proj_in = nn.Linear(dim_in, dim)
        self.y_proj = nn.Linear(dim_in, dim)
        self.tau_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        pos = get_3d_sincos_pos_embed(dim, grid[1], grid[0])
        self.register_buffer("pos", pos.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads, 4.0, 0.0) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, dim_in)
        self._dim = dim

    def _tau_embed(self, tau, device):
        # sinusoidal embedding of tau in [0,1]
        half = self._dim // 2
        freqs = torch.exp(torch.arange(half, device=device) * (-math.log(10000.0) / half))
        ang = tau.view(-1, 1) * freqs.view(1, -1) * 1000.0
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.tau_mlp(emb)                                   # [B, dim]

    def forward(self, h_vis, vis_idx, mask_idx_list, y_tau_list, tau):
        B = h_vis.shape[0]
        x = self.proj_in(h_vis)
        dim = x.shape[-1]
        pos = self.pos.expand(B, -1, -1)
        pos_vis = pos.gather(1, vis_idx.unsqueeze(-1).expand(-1, -1, dim))
        x = x + pos_vis
        t_emb = self._tau_embed(tau, h_vis.device)                 # [B, dim]
        outs = []
        for b in range(B):
            mi = mask_idx_list[b]
            q = self.y_proj(y_tau_list[b]).unsqueeze(0) + pos[b:b + 1, mi]                 + t_emb[b].view(1, 1, -1)
            seq = torch.cat([x[b:b + 1], q], dim=1)
            for blk in self.blocks:
                seq = blk(seq)
            seq = self.norm(seq)
            outs.append(self.proj_out(seq[0, x.shape[1]:]))
        return outs


class FlowHead(nn.Module):
    """(s_prev [B,P,D], z [B,A]) -> flow grid [B, gh, gw, 2]."""

    def __init__(self, state_dim=384, action_dim=16, grid=14, hidden=256):
        super().__init__()
        self.grid = grid
        self.z_proj = nn.Linear(action_dim, hidden)
        self.s_proj = nn.Linear(state_dim, hidden)
        self.mix = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, s_prev, z):
        h = self.s_proj(s_prev) + self.z_proj(z).unsqueeze(1)   # [B, P, hidden]
        f = self.mix(h)                                          # [B, P, 2]
        B = f.shape[0]
        return f.view(B, self.grid, self.grid, 2)


# ---------------------------------------------------------------- metrics

def eff_rank(x):
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x.float())
    p = (s ** 2) / (s ** 2).sum().clamp_min(1e-12)
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item())


def ridge_probe(X, y_int, n_classes, lam=1e-3):
    """Closed-form ridge to one-hot classes; returns train-split-free accuracy
    via 80/20 split."""
    n = X.shape[0]
    idx = torch.randperm(n, device=X.device)
    cut = int(n * 0.8)
    tr_i, te_i = idx[:cut], idx[cut:]
    Y = F.one_hot(y_int[tr_i], n_classes).float()
    Xtr = X[tr_i]
    W = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(X.shape[1], device=X.device),
                           Xtr.T @ Y)
    acc = ((X[te_i] @ W).argmax(-1) == y_int[te_i]).float().mean().item()
    return acc


def ridge_r2(X, Y, lam=1e-3):
    n = X.shape[0]
    idx = torch.randperm(n, device=X.device)
    cut = int(n * 0.8)
    tr_i, te_i = idx[:cut], idx[cut:]
    Xtr, Ytr = X[tr_i], Y[tr_i]
    W = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(X.shape[1], device=X.device),
                           Xtr.T @ Ytr)
    pred = X[te_i] @ W
    ss_res = ((pred - Y[te_i]) ** 2).sum(0)
    ss_tot = ((Y[te_i] - Y[te_i].mean(0)) ** 2).sum(0).clamp_min(1e-9)
    return float((1 - ss_res / ss_tot).mean().item())


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/root/bridge/scripted_raw")
    ap.add_argument("--out_dir", default="/root/v6/runs/v6_a")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--n_ctx_frames", type=int, default=8)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--tubelet", type=int, default=2)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--enc_depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--pred_depth", type=int, default=6)
    ap.add_argument("--jepa_fm", action="store_true",
                    help="objective A = conditional flow matching over masked tokens "
                         "instead of L1 regression (generative JEPA)")
    ap.add_argument("--action_dim", type=int, default=16)
    ap.add_argument("--inv_depth", type=int, default=4)
    ap.add_argument("--z_noise", type=float, default=0.0,
                    help="std of gaussian noise on z fed to transition (robustness "
                         "to imperfect prior samples at inference)")
    ap.add_argument("--inv_heads", type=int, default=8,
                    help="heads for action_dim-wide attention (must divide action_dim)")
    ap.add_argument("--tr_depth", type=int, default=4)
    ap.add_argument("--flow_depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--l_b", type=float, default=1.0)
    ap.add_argument("--l_c", type=float, default=0.5)
    ap.add_argument("--l_d", type=float, default=0.5)
    ap.add_argument("--l_e", type=float, default=0.5)
    ap.add_argument("--l_var", type=float, default=0.1)
    ap.add_argument("--l_cov", type=float, default=0.05)
    ap.add_argument("--flow_scale", type=float, default=8.0,
                    help="divide Farneback pixel flow by this before the loss")
    ap.add_argument("--gate_steps", type=int, default=400)
    ap.add_argument("--gate_u", type=float, default=0.2)
    ap.add_argument("--no_gate_abort", action="store_true")
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_clips", type=int, default=128)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--data_frac", type=float, default=1.0,
                    help="fraction of TRAIN trajectories to use (data-scaling runs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    device = "cuda"
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    manifest = build_bridge_manifest(args.data_root)
    n_scenes = max(m["scene_id"] for m in manifest) + 1
    rng = random.Random(args.seed)
    rng.shuffle(manifest)
    n_val = max(1, int(len(manifest) * args.val_frac))
    val_manifest, train_manifest = manifest[:n_val], manifest[n_val:]
    if args.data_frac < 1.0:
        keep = max(1, int(len(train_manifest) * args.data_frac))
        train_manifest = train_manifest[:keep]
        print(f"[v6] data_frac={args.data_frac} -> {keep} train trajectories")
    print(f"[v6] {len(manifest)} trajectories, {n_scenes} scenes  "
          f"(train={len(train_manifest)}, val={len(val_manifest)})", flush=True)

    grid_d = args.num_frames // args.tubelet
    grid_s = args.img_size // args.patch
    P = grid_s * grid_s
    N = grid_d * P
    n_ctx_tubs = args.n_ctx_frames // args.tubelet
    S = grid_d - n_ctx_tubs
    A = args.action_dim
    print(f"[v6] grid {grid_d}x{grid_s}x{grid_s}  N={N}  ctx_tubs={n_ctx_tubs}  S={S}  A={A}")

    online = VisionTransformer3D(
        num_frames=args.num_frames, tubelet_size=args.tubelet, patch_size=args.patch,
        img_size=args.img_size, embed_dim=args.dim, depth=args.enc_depth,
        num_heads=args.heads, mlp_ratio=4.0).to(device).train()
    target = copy.deepcopy(online).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    pred_cls = FMPredictor if args.jepa_fm else MaskedPredictor
    predictor = pred_cls(dim_in=args.dim, dim=192, depth=args.pred_depth,
                         heads=6, grid=(grid_d, grid_s, grid_s)).to(device).train()
    inv = InverseFromDeltaPerStep(state_dim=args.dim, action_dim=A, n_steps=S,
                                   n_patch=P, depth=args.inv_depth,
                                   heads=args.inv_heads).to(device).train()
    tr_step = MultiStepTransitionPerStep(state_dim=args.dim, action_dim=A, n_steps=1,
                                          n_patch=P, depth=args.tr_depth, heads=args.heads,
                                          alpha_min=0.02, alpha_init=0.05).to(device).train()
    flow_head = FlowHead(state_dim=args.dim, action_dim=A, grid=grid_s).to(device).train()
    prior = GoalConditionedDiTFlow(state_dim=args.dim, action_dim=A, n_steps=S,
                                    depth=args.flow_depth, heads=4).to(device).train()

    nparam = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    print(f"[v6] params  enc={nparam(online):.1f}M pred={nparam(predictor):.1f}M "
          f"inv={nparam(inv):.1f}M tr={nparam(tr_step):.1f}M flowH={nparam(flow_head):.2f}M "
          f"prior={nparam(prior):.1f}M", flush=True)

    params = (list(online.parameters()) + list(predictor.parameters())
              + list(inv.parameters()) + list(tr_step.parameters())
              + list(flow_head.parameters()) + list(prior.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    def encode_target(clips_bchw):
        with torch.no_grad():
            H = target(clips_bchw, keep_indices=None)
            return H.view(H.shape[0], grid_d, P, -1)

    def states_from(Hfull):
        s_list = [Hfull[:, n_ctx_tubs - 1]]                     # s_t
        for k in range(S):
            s_list.append(Hfull[:, n_ctx_tubs + k])             # s_{t+1..t+S}
        return s_list                                           # len S+1 of [B,P,D]

    def chain_z(s_list):
        deltas = torch.stack([s_list[k + 1] - s_list[k] for k in range(S)], dim=1)
        return inv(deltas)                                      # [B, S, A]

    def teacher_forced_preds(s_list, Z):
        preds = []
        for k in range(S):
            preds.append(tr_step(s_list[k], Z[:, k:k + 1])[:, 0])
        return torch.stack(preds, dim=1)                        # [B, S, P, D]

    def rollout_preds(s_list, Z):
        cur = s_list[0]
        preds = []
        for k in range(S):
            cur = tr_step(cur, Z[:, k:k + 1])[:, 0]
            preds.append(cur)
        return torch.stack(preds, dim=1)

    def cos_loss(pred, tgt):
        return 1.0 - F.cosine_similarity(
            pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1, tgt.shape[-1]), dim=-1).mean()

    def delta_loss(preds, s_list, tgt):
        """Compare layer-normalized CHANGES, not raw states. Raw adjacent-tubelet
        features are ~cos 0.999 similar, so the raw-state cosine saturates and Z
        gets no gradient (v6_a gate failure). Deltas have full dynamic range.
        preds/tgt: [B, S, P, D]; s_list: python list, s_list[k] is the state each
        step k was predicted FROM."""
        D = preds.shape[-1]
        losses = []
        for k in range(preds.shape[1]):
            d_pred = F.layer_norm(preds[:, k] - s_list[k], (D,))
            d_tgt = F.layer_norm(tgt[:, k] - s_list[k], (D,))
            losses.append(1.0 - F.cosine_similarity(
                d_pred.reshape(-1, D), d_tgt.reshape(-1, D), dim=-1).mean())
        return torch.stack(losses).mean()

    @torch.no_grad()
    def eval_pass(step):
        for m in (online, predictor, inv, tr_step, flow_head, prior):
            m.eval()
        Z_all, act_all, scene_all = [], [], []
        lb_real, lb_shuf = [], []
        n_batches = max(1, args.eval_clips // 32)
        for _ in range(n_batches):
            try:
                fr, acts, fls, scn = sample_bridge_batch(
                    val_manifest, 32, args.num_frames, args.img_size,
                    args.n_ctx_frames, args.tubelet, grid_s, with_flow=False,
                    frame_stride=args.frame_stride)
            except Exception as e:
                print(f"[eval] sample fail {e}")
                continue
            clips = fr.permute(0, 4, 1, 2, 3).contiguous().to(device)
            Hf = encode_target(clips)
            sl = states_from(Hf)
            Z = chain_z(sl)
            tgt = torch.stack(sl[1:], dim=1)
            pr = teacher_forced_preds(sl, Z)
            lb_real.append(delta_loss(pr, sl, tgt).item())
            perm = torch.randperm(Z.shape[0], device=device)
            pr_s = teacher_forced_preds(sl, Z[perm])
            lb_shuf.append(delta_loss(pr_s, sl, tgt).item())
            Z_all.append(Z.cpu()); act_all.append(acts); scene_all.append(scn)
        if not Z_all:
            return None
        Z = torch.cat(Z_all).to(device)                          # [n, S, A]
        acts = torch.cat(act_all).to(device)
        scn = torch.cat(scene_all).to(device)
        U = 1.0 - (sum(lb_real) / len(lb_real)) / max(sum(lb_shuf) / len(lb_shuf), 1e-9)
        er = eff_rank(Z.flatten(1))
        # saturation diagnostic: raw cosine between adjacent tubelet states
        with torch.no_grad():
            sat = F.cosine_similarity(sl[0].flatten(1), sl[1].flatten(1), dim=-1).mean().item()
        # leakage: Z -> scene id
        leak = ridge_probe(Z.flatten(1).float(), scn, n_scenes)
        chance = 1.0 / max(1, len(scn.unique()))
        # action probe: per-step z -> per-step GT action (per-group)
        Xf = Z.flatten(0, 1).float(); Yf = acts.flatten(0, 1).float()
        n_s = Xf.shape[0]
        idxp = torch.randperm(n_s, device=Xf.device)
        cutp = int(n_s * 0.8)
        Wp = torch.linalg.solve(
            Xf[idxp[:cutp]].T @ Xf[idxp[:cutp]] + 1e-3 * torch.eye(Xf.shape[1], device=Xf.device),
            Xf[idxp[:cutp]].T @ Yf[idxp[:cutp]])
        predp = Xf[idxp[cutp:]] @ Wp
        ssr = ((predp - Yf[idxp[cutp:]]) ** 2).sum(0)
        sst = ((Yf[idxp[cutp:]] - Yf[idxp[cutp:]].mean(0)) ** 2).sum(0).clamp_min(1e-9)
        r2_dims = (1 - ssr / sst)
        r2 = float(r2_dims.mean().item())
        r2_xyz = float(r2_dims[:3].mean().item())
        r2_grip = float(r2_dims[6].item())
        for m in (online, predictor, inv, tr_step, flow_head, prior):
            m.train()
        return {"step": step, "U": U, "eff_rank": er,
                "leak_acc": leak, "leak_chance": chance,
                "action_r2": r2, "r2_xyz": r2_xyz, "r2_grip": r2_grip,
                "state_sat_cos": sat,
                "lb_real": sum(lb_real) / len(lb_real),
                "lb_shuf": sum(lb_shuf) / len(lb_shuf)}

    hist = []
    print(f"\n[v6] === train {args.steps} steps  bs={args.bs} ===", flush=True)
    for step in range(args.steps):
        t0 = time.time()
        try:
            fr, acts, fls, scn = sample_bridge_batch(
                train_manifest, args.bs, args.num_frames, args.img_size,
                args.n_ctx_frames, args.tubelet, grid_s, with_flow=(args.l_d > 0),
                frame_stride=args.frame_stride)
        except Exception as e:
            print(f"[v6] sample fail: {e}")
            continue
        clips = fr.permute(0, 4, 1, 2, 3).contiguous().to(device)
        flows = (fls / args.flow_scale).to(device)

        # ---- A: masked JEPA (trains encoder) ----
        mask = sample_multiblock_mask(args.bs, grid_d, grid_s, grid_s, device)
        with torch.no_grad():
            H_tgt_full = target(clips, keep_indices=None)        # [B, N, D]
        vis_counts = (~mask).sum(-1)
        n_keep = int(vis_counts.min().item())
        vis_idx = torch.stack([(~mask[b]).nonzero(as_tuple=True)[0][:n_keep]
                               for b in range(args.bs)])
        mask_idx_list = [mask[b].nonzero(as_tuple=True)[0] for b in range(args.bs)]
        H_vis = online(clips, keep_indices=vis_idx)
        if args.jepa_fm:
            tau_A = torch.rand(args.bs, device=device).clamp(1e-3, 1 - 1e-3)
            y_tau_list, v_tgt_list = [], []
            for b in range(args.bs):
                y = H_tgt_full[b, mask_idx_list[b]]
                eps_A = torch.randn_like(y)
                y_tau_list.append((1 - tau_A[b]) * eps_A + tau_A[b] * y)
                v_tgt_list.append(y - eps_A)
            v_preds = predictor(H_vis, vis_idx, mask_idx_list, y_tau_list, tau_A)
            L_A = torch.stack([
                F.mse_loss(v_preds[b], v_tgt_list[b]) for b in range(args.bs)]).mean()
        else:
            preds = predictor(H_vis, vis_idx, mask_idx_list)
            L_A = torch.stack([
                (preds[b] - H_tgt_full[b, mask_idx_list[b]]).abs().mean()
                for b in range(args.bs)]).mean()

        # ---- action stack on sg(target features) ----
        Hf = H_tgt_full.view(args.bs, grid_d, P, -1)
        sl = states_from(Hf)
        Z = chain_z(sl)                                          # [B, S, A]
        Z_tr = Z + args.z_noise * torch.randn_like(Z) if args.z_noise > 0 else Z
        tgt = torch.stack(sl[1:], dim=1)
        L_B = delta_loss(teacher_forced_preds(sl, Z_tr), sl, tgt)
        L_C = delta_loss(rollout_preds(sl, Z_tr), sl, tgt)

        if args.l_d > 0:
            f_preds = torch.stack([flow_head(sl[k], Z[:, k]) for k in range(S)], dim=1)
            L_D = F.huber_loss(f_preds, flows)
        else:
            L_D = torch.zeros((), device=device)

        eps = torch.randn_like(Z)
        tau = torch.rand(args.bs, device=device).clamp(1e-3, 1 - 1e-3)
        z_tau = (1 - tau).view(-1, 1, 1) * eps + tau.view(-1, 1, 1) * Z.detach()
        v_tgt = Z.detach() - eps
        H_ctx = Hf[:, :n_ctx_tubs].flatten(1, 2)
        goal_mask = torch.zeros(args.bs, device=device)
        v_pred = prior(z_tau, tau, H_ctx.detach(), Hf[:, -1].detach(), goal_mask=goal_mask)
        L_E = F.mse_loss(v_pred, v_tgt)

        Zf = Z.flatten(0, 1)
        L_var = F.relu(1.0 - (Zf.var(0) + 1e-4).sqrt()).mean()
        zc = Zf - Zf.mean(0, keepdim=True)
        cov = (zc.T @ zc) / max(1, Zf.shape[0] - 1)
        off = cov - torch.diag(torch.diag(cov))
        L_cov = (off ** 2).sum() / A

        loss = (L_A + args.l_b * L_B + args.l_c * L_C + args.l_d * L_D
                + args.l_e * L_E + args.l_var * L_var + args.l_cov * L_cov)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        with torch.no_grad():
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.data.mul_(args.ema).add_(po.data, alpha=1 - args.ema)
            for bo, bt in zip(online.buffers(), target.buffers()):
                bt.data.copy_(bo.data)

        if (step + 1) % args.log_every == 0 or step == 0:
            print(f"  step {step+1:5d}/{args.steps}  A={L_A.item():.3f} B={L_B.item():.3f} "
                  f"C={L_C.item():.3f} D={L_D.item():.3f} E={L_E.item():.3f} "
                  f"var={L_var.item():.3f} cov={L_cov.item():.3f} "
                  f"t={time.time()-t0:.2f}s", flush=True)

        if (step + 1) % args.eval_every == 0:
            m = eval_pass(step + 1)
            if m:
                hist.append(m)
                with open(os.path.join(args.out_dir, "eval_history.json"), "w") as f:
                    json.dump(hist, f, indent=2)
                print(f"  [EVAL {step+1}]  U={m['U']:.3f}  eff_rank={m['eff_rank']:.2f}/"
                      f"{S*A}  leak={m['leak_acc']:.3f} (chance {m['leak_chance']:.3f}) "
                      f" R2xyz={m['r2_xyz']:.3f} R2grip={m['r2_grip']:.3f}  sat={m['state_sat_cos']:.4f}", flush=True)

        if (step + 1) == args.gate_steps and not args.no_gate_abort:
            m = eval_pass(step + 1)
            if m and m["U"] < args.gate_u:
                print(f"[v6] LAUNCH GATE FAILED  U={m['U']:.3f} < {args.gate_u} — aborting. "
                      f"Futures too predictable at this horizon/data.", flush=True)
                with open(os.path.join(args.out_dir, "GATE_FAILED"), "w") as f:
                    json.dump(m, f)
                return
            if m:
                print(f"[v6] LAUNCH GATE PASSED  U={m['U']:.3f} >= {args.gate_u}", flush=True)

        if (step + 1) % args.ckpt_every == 0:
            torch.save({"step": step + 1,
                        "online": online.state_dict(), "target": target.state_dict(),
                        "predictor": predictor.state_dict(), "inv": inv.state_dict(),
                        "tr": tr_step.state_dict(), "flow_head": flow_head.state_dict(),
                        "prior": prior.state_dict()},
                       os.path.join(args.out_dir, f"ckpt_step{step+1}.pt"))
            print(f"  [CKPT] step {step+1}", flush=True)

    print("[v6] DONE", flush=True)


if __name__ == "__main__":
    main()
