"""Goal-Conditioned Flow-Matched Latent Action JEPA — single-file training code.

Consolidates the current state of the prototype:

  * Stage 1 trainer  (frozen-ViT encoder + ResidualTransition + InverseAction)
    Reproduces the Stage 1C+ ckpt that gives val correct_Z = 0.97 on BBB.
  * Stage 2B goal-conditioned flow trainer with hard negatives
    Reproduces the goal-flow ckpt that gives val K=1 = 0.97 on BBB and
    K=1 = 0.99 on cross-video Sintel.

CLI subcommands:
    stage1          train Stage 1 transition + inverse (frozen encoder).
    goalflow        train goal-conditioned hard-neg flow given a Stage 1 ckpt.
    eval_baseline   non-parametric prior baselines (empZ / kNN / fgZ).
    eval_goalflow   single + best-of-K eval of a goal-flow ckpt.

Architecture defaults reproduce Stage 1C+:
  encoder = ViT-S/16 (timm pretrained), patch tokens from final block, no CLS.
  state reducer = parameter-free spatial-pool 14x14 -> 8x8 (64 tokens, dim 384).
  future reducer = spatial-pool 14x14 -> 4x4 (16 tokens, dim 384).
  inverse_encoder depth 4, transition depth 2, action 8 x 256.
  residual transition: H_pred = base_reducer(H_t) + alpha * delta(H_t, Z)
    alpha is learnable scalar, init 0.05, clamped to >= 0.02.
  losses = per-token cosine + symmetric InfoNCE (tau 0.1)
           + VICReg variance + VICReg covariance on Z* + tiny L2.
  k = 16 frame gap. Train block: stride 1, n_train 12000, train_start 200.
  Val block: stride 18, n_val 100, gap 80 frames after train end.

This file holds no Stage 2A (deterministic prior) or Stage 2C (reranker) since
those underperformed the non-parametric baselines on val. They are documented
in STAGE2_SUMMARY.md / STAGE2_RESULTS.md.
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def decode_indices(path, indices, size=224):
    """Decode specific frame indices into a normalized tensor [N,3,H,W]."""
    cap = cv2.VideoCapture(path)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"failed at idx={idx}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    arr = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr


def decode_contiguous_pairs(path, n_pairs, k, stride, start_skip, size=224):
    """Sample n_pairs anchors at (start_skip + i*stride); pair with anchor + k."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert stride >= 1
    anchors = [start_skip + i * stride for i in range(n_pairs)]
    assert anchors[-1] + k < total - 1, (anchors[-1], k, total, n_pairs, stride, start_skip)
    futures = [a + k for a in anchors]
    frames = decode_indices(path, anchors + futures, size=size)
    x_t = frames[:n_pairs]
    x_tp = frames[n_pairs:]
    return x_t, x_tp, torch.tensor(anchors), torch.tensor([k] * n_pairs)


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dim, mult=4, drop=0.0):
        super().__init__()
        h = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttnBlock(nn.Module):
    def __init__(self, dim, heads=8, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, drop=drop)

    def forward(self, x):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, heads=8, drop=0.0):
        super().__init__()
        self.nq = nn.LayerNorm(dim)
        self.nk = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.ns = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.nm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, drop=drop)

    def forward(self, q, kv):
        a, _ = self.attn(self.nq(q), self.nk(kv), self.nk(kv), need_weights=False)
        q = q + a
        s = self.ns(q)
        sa, _ = self.self_attn(s, s, s, need_weights=False)
        q = q + sa
        q = q + self.mlp(self.nm(q))
        return q


# ----------------------------------------------------------------------------
# Encoder + reducers
# ----------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    """Wrap timm ViT to return patch tokens [B,N,D]. layer_idx=None -> final."""

    def __init__(self, name="vit_small_patch16_224", pretrained=True, layer_idx=None):
        super().__init__()
        import timm
        self.vit = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.vit.embed_dim
        self.layer_idx = layer_idx
        self.n_blocks = len(self.vit.blocks)

    def _patch_embed_with_extras(self, x):
        v = self.vit
        x = v.patch_embed(x)
        x = v._pos_embed(x) if hasattr(v, "_pos_embed") else v.pos_drop(x + v.pos_embed)
        x = v.norm_pre(x) if hasattr(v, "norm_pre") else x
        return x

    def forward(self, x):
        if self.layer_idx is None:
            feats = self.vit.forward_features(x)
            if hasattr(self.vit, "cls_token") and self.vit.cls_token is not None:
                feats = feats[:, 1:, :]
            return feats
        x = self._patch_embed_with_extras(x)
        for i in range(self.layer_idx):
            x = self.vit.blocks[i](x)
        if hasattr(self.vit, "cls_token") and self.vit.cls_token is not None:
            x = x[:, 1:, :]
        return x


class FixedSpatialReducer(nn.Module):
    """Parameter-free: patch tokens [B,N,D] -> spatial grid -> adaptive avg pool
    to out_tokens (must be perfect square) -> tokens [B,K,D]. Dim D unchanged."""

    def __init__(self, out_tokens):
        super().__init__()
        side = int(math.isqrt(out_tokens))
        assert side * side == out_tokens, f"out_tokens={out_tokens} must be perfect square"
        self.out_h = side
        self.out_w = side

    def forward(self, tokens):
        b, n, d = tokens.shape
        side = int(math.isqrt(n))
        assert side * side == n, f"input N={n} must be perfect square"
        x = tokens.transpose(1, 2).reshape(b, d, side, side)
        x = F.adaptive_avg_pool2d(x, (self.out_h, self.out_w))
        return x.reshape(b, d, self.out_h * self.out_w).transpose(1, 2)


# ----------------------------------------------------------------------------
# Inverse / transition / flow
# ----------------------------------------------------------------------------

class InverseActionEncoder(nn.Module):
    """(H_t, H_tp) -> Z* via learned action queries cross-attending [H_t || H_tp]."""

    def __init__(self, state_dim, action_dim, action_tokens, depth=4, heads=8, drop=0.0):
        super().__init__()
        self.proj_t = nn.Linear(state_dim, action_dim)
        self.proj_tp = nn.Linear(state_dim, action_dim)
        self.t_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.tp_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.queries = nn.Parameter(torch.randn(1, action_tokens, action_dim) * 0.02)
        self.blocks = nn.ModuleList([CrossAttnBlock(action_dim, heads, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(action_dim)

    def forward(self, h_t, h_tp):
        b = h_t.shape[0]
        kv = torch.cat([self.proj_t(h_t) + self.t_marker, self.proj_tp(h_tp) + self.tp_marker], dim=1)
        q = self.queries.expand(b, -1, -1)
        for blk in self.blocks:
            q = blk(q, kv)
        return self.norm(q)


class TransitionModel(nn.Module):
    """(H_t, Z) -> H_pred via learned future queries cross-attending [H_t || Z_proj]."""

    def __init__(self, state_dim, action_dim, future_tokens, depth=6, heads=8, drop=0.0):
        super().__init__()
        self.z_proj = nn.Linear(action_dim, state_dim)
        self.h_marker = nn.Parameter(torch.zeros(1, 1, state_dim))
        self.z_marker = nn.Parameter(torch.zeros(1, 1, state_dim))
        self.future_q = nn.Parameter(torch.randn(1, future_tokens, state_dim) * 0.02)
        self.blocks = nn.ModuleList([CrossAttnBlock(state_dim, heads, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, h_t, z):
        b = h_t.shape[0]
        mem = torch.cat([h_t + self.h_marker, self.z_proj(z) + self.z_marker], dim=1)
        q = self.future_q.expand(b, -1, -1)
        for blk in self.blocks:
            q = blk(q, mem)
        return self.norm(q)


class ResidualTransitionModel(nn.Module):
    """H_pred = base(H_t) + alpha * delta(H_t, Z).

    base is parameter-free spatial pool of H_t to future_tokens.
    delta is a small TransitionModel.
    alpha is a learnable scalar (init alpha_init=0.05, clamped >= alpha_min=0.02).
    Note: softplus parameterization throttled the alpha gradient — DO NOT
    reintroduce. Direct clamp is the only path that worked.
    """

    def __init__(self, state_dim, action_dim, future_tokens, depth=2, heads=8, drop=0.0,
                 alpha_min=0.02, alpha_init=0.05):
        super().__init__()
        self.base = FixedSpatialReducer(future_tokens)
        self.delta = TransitionModel(state_dim, action_dim, future_tokens, depth=depth, heads=heads, drop=drop)
        self.alpha_min = float(alpha_min)
        self.raw_alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    @property
    def alpha(self):
        return self.raw_alpha.clamp(min=self.alpha_min)

    def forward(self, h_t, z):
        return self.base(h_t) + self.alpha * self.delta(h_t, z)


def sinusoidal_time_embed(tau, dim):
    """tau [B,1,1] -> [B,1,dim]."""
    device = tau.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
    args = tau.view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], dim - emb.shape[-1], device=device)], dim=-1)
    return emb.unsqueeze(1)


class FlowPrior(nn.Module):
    """Unconditional flow over Z given (z_tau, tau, H_t). Predicts velocity.
    Kept for completeness; not used in the goal-conditioned trainer below."""

    def __init__(self, state_dim, action_dim, depth=6, heads=8, drop=0.0):
        super().__init__()
        self.ctx_proj = nn.Linear(state_dim, action_dim)
        self.time_dim = action_dim
        self.blocks = nn.ModuleList([CrossAttnBlock(action_dim, heads, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(action_dim)
        self.head = nn.Linear(action_dim, action_dim)

    def forward(self, z_tau, tau, h_t):
        if tau.ndim == 1:
            tau = tau.view(-1, 1, 1)
        t_emb = sinusoidal_time_embed(tau, self.time_dim)
        x = z_tau + t_emb
        ctx = torch.cat([self.ctx_proj(h_t), t_emb], dim=1)
        for blk in self.blocks:
            x = blk(x, ctx)
        return self.head(self.norm(x))


class GoalConditionedFlowPrior(nn.Module):
    """Flow predicting velocity for (z_tau, tau) conditioned on H_t AND H_goal.

    At training time, H_goal is typically H_tp (true future) with a fraction
    of the batch using a SHUFFLED H_goal (hard negatives). Target Z is then
    inv(H_t, H_goal) — so the flow learns the general map (state, goal) -> action.
    """

    def __init__(self, state_dim, action_dim, depth=6, heads=8, drop=0.0):
        super().__init__()
        self.ctx_t_proj = nn.Linear(state_dim, action_dim)
        self.ctx_g_proj = nn.Linear(state_dim, action_dim)
        self.t_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.g_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.time_dim = action_dim
        self.blocks = nn.ModuleList([CrossAttnBlock(action_dim, heads, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(action_dim)
        self.head = nn.Linear(action_dim, action_dim)

    def forward(self, z_tau, tau, h_t, h_goal):
        if tau.ndim == 1:
            tau = tau.view(-1, 1, 1)
        t_emb = sinusoidal_time_embed(tau, self.time_dim)
        x = z_tau + t_emb
        ctx_t = self.ctx_t_proj(h_t) + self.t_marker
        ctx_g = self.ctx_g_proj(h_goal) + self.g_marker
        ctx = torch.cat([ctx_t, ctx_g, t_emb], dim=1)
        for blk in self.blocks:
            x = blk(x, ctx)
        return self.head(self.norm(x))


@torch.no_grad()
def euler_sample(flow, h_t, action_tokens, action_dim, steps=30, seed=None, h_goal=None):
    b = h_t.shape[0]
    g = torch.Generator(device=h_t.device).manual_seed(seed) if seed is not None else None
    z = torch.randn(b, action_tokens, action_dim, device=h_t.device, generator=g)
    for i in range(steps):
        tau = torch.full((b, 1, 1), i / steps, device=h_t.device)
        v = flow(z, tau, h_t, h_goal) if h_goal is not None else flow(z, tau, h_t)
        z = z + v / steps
    return z


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------

def cosine_distance(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return (2.0 - 2.0 * (pred * target).sum(-1)).mean()


def info_nce(q, k, tau=0.1, symmetric=True):
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = q @ k.T / tau
    labels = torch.arange(q.shape[0], device=q.device)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(k @ q.T / tau, labels))
    return loss


def vicreg_var(z):
    return F.relu(1 - z.flatten(0, 1).std(0)).mean()


def vicreg_cov(z):
    z_flat = z.flatten(0, 1)
    z_c = z_flat - z_flat.mean(0, keepdim=True)
    cov = (z_c.T @ z_c) / max(1, z_c.shape[0] - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return (off ** 2).sum() / cov.shape[0]


# ----------------------------------------------------------------------------
# Eval helpers
# ----------------------------------------------------------------------------

def effective_rank(x):
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x.float())
    p = (s ** 2) / (s ** 2).sum().clamp_min(1e-12)
    H = -(p * (p.clamp_min(1e-12).log())).sum()
    return torch.exp(H).item()


def retrieval(q, k, N, name=None):
    qn = F.normalize(q, dim=-1)
    kn = F.normalize(k, dim=-1)
    sim = qn @ kn.T
    r = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(N, device=q.device).unsqueeze(-1)
    top1 = (r[:, :1] == gt).any(-1).float().mean().item()
    top5 = (r[:, :5] == gt).any(-1).float().mean().item()
    mr = (r == gt).float().argmax(-1).float().mean().item() + 1
    if name:
        print(f"  retrieval[{name:20s}] top1={top1:.3f}  top5={top5:.3f}  mr={mr:.2f}")
    return top1, top5, mr


def best_of_k_retrieval(preds_list, H_tp, N, name=None):
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
    mr = best_rank.float().mean().item()
    if name:
        print(f"  bestK={len(preds_list):3d} {name:18s} top1={top1:.3f}  mr={mr:.2f}")
    return top1, mr


# ----------------------------------------------------------------------------
# Stage 1: frozen encoder, train inverse + residual transition
# ----------------------------------------------------------------------------

def encode_pair(enc, red_state, red_future, x_t, x_tp, chunk=64):
    Hts, Htps = [], []
    for s in range(0, x_t.shape[0], chunk):
        e = min(s + chunk, x_t.shape[0])
        Hts.append(red_state(enc(x_t[s:e])))
        Htps.append(red_future(enc(x_tp[s:e])))
    return torch.cat(Hts, 0), torch.cat(Htps, 0)


def train_stage1(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    state_dim = 384

    enc = ViTEncoder("vit_small_patch16_224", pretrained=True, layer_idx=None).to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    red_state = FixedSpatialReducer(64).to(device)
    red_future = FixedSpatialReducer(16).to(device)
    inv = InverseActionEncoder(state_dim, args.action_dim, args.action_tokens, depth=args.inverse_depth).to(device)
    tr = ResidualTransitionModel(state_dim, args.action_dim, args.future_tokens,
                                 depth=args.transition_depth).to(device)

    print(f"[stage1] action {args.action_tokens}x{args.action_dim}  inv_depth={args.inverse_depth}  tr_depth={args.transition_depth}")
    print(f"[stage1] params train: {sum(p.numel() for p in list(inv.parameters()) + list(tr.parameters()))/1e6:.2f} M")

    # ---- Decode (or load cached) pairs ----
    cache_train = args.cache_train
    cache_val = args.cache_val
    if cache_train and os.path.exists(cache_train):
        d = torch.load(cache_train, map_location="cpu", weights_only=False)
        x_t, x_tp = d["x_t"], d["x_tp"]
        train_end = int(d.get("train_end", args.train_start + args.n_train * args.stride_train + args.k))
    else:
        x_t, x_tp, ts, _ = decode_contiguous_pairs(args.video, n_pairs=args.n_train, k=args.k,
                                                  stride=args.stride_train, start_skip=args.train_start)
        train_end = int(ts[-1].item() + args.k)
        if cache_train:
            os.makedirs(os.path.dirname(cache_train) or ".", exist_ok=True)
            torch.save({"x_t": x_t, "x_tp": x_tp, "train_end": train_end}, cache_train)

    val_start = args.val_start if args.val_start >= 0 else train_end + args.block_gap
    print(f"[stage1] train block: [{args.train_start}, {train_end}]  stride {args.stride_train}")
    print(f"[stage1] val   block: [{val_start}, ?]  stride {args.stride_val}  gap {val_start - train_end}")

    if cache_val and os.path.exists(cache_val):
        d = torch.load(cache_val, map_location="cpu", weights_only=False)
        x_v_t, x_v_tp = d["x_t"], d["x_tp"]
    else:
        x_v_t, x_v_tp, _, _ = decode_contiguous_pairs(args.video, n_pairs=args.n_val, k=args.k,
                                                    stride=args.stride_val, start_skip=val_start)
        if cache_val:
            os.makedirs(os.path.dirname(cache_val) or ".", exist_ok=True)
            torch.save({"x_t": x_v_t, "x_tp": x_v_tp}, cache_val)

    print(f"[stage1] train pairs={x_t.shape[0]}  val pairs={x_v_t.shape[0]}  k={args.k}")

    x_t = x_t.to(device); x_tp = x_tp.to(device)
    x_v_t = x_v_t.to(device); x_v_tp = x_v_tp.to(device)

    print("[stage1] encoding all pairs once through frozen ViT-S...")
    H_t_train, H_tp_train = encode_pair(enc, red_state, red_future, x_t, x_tp)
    H_t_val, H_tp_val = encode_pair(enc, red_state, red_future, x_v_t, x_v_tp)
    print(f"[stage1] H_t_train {tuple(H_t_train.shape)}  H_tp_train {tuple(H_tp_train.shape)}")

    params = list(inv.parameters()) + list(tr.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    N_tr = H_t_train.shape[0]

    def eval_split(H_t, H_tp, tag):
        N = H_t.shape[0]
        inv.eval(); tr.eval()
        with torch.no_grad():
            Z = inv(H_t, H_tp)
            preds = {
                "correct_Z": tr(H_t, Z),
                "zero_Z": tr(H_t, torch.zeros_like(Z)),
                "random_Z": tr(H_t, torch.randn_like(Z) * Z.flatten(1).std(0).mean()),
                "shuffle_Z": tr(H_t, torch.roll(Z, 1, 0)),
            }
            print(f"\n[stage1 eval {tag}]  N={N}  alpha={tr.alpha.item():.4f}  std(Z)={Z.flatten(1).std(0).mean().item():.4f}  eff_rank(Z)={effective_rank(Z.flatten(1)):.1f}")
            for k_name, p in preds.items():
                retrieval(p.mean(1), H_tp.mean(1), N, k_name)
            retrieval(H_t.mean(1), H_tp.mean(1), N, "copy_Ht")
        inv.train(); tr.train()

    eval_split(H_t_val, H_tp_val, "val pre")

    inv.train(); tr.train()
    for step in range(args.steps):
        idx = torch.randint(0, N_tr, (args.bs,), device=device)
        h_t = H_t_train[idx]; h_tp = H_tp_train[idx]
        z = inv(h_t, h_tp)
        h_pred = tr(h_t, z)

        L_t = cosine_distance(h_pred, h_tp.detach())
        L_nce = info_nce(h_pred.mean(1), h_tp.mean(1).detach(), tau=args.nce_tau, symmetric=True)
        L_l2 = (z ** 2).mean()
        L_var = vicreg_var(z)
        L_cov = vicreg_cov(z)
        loss = (L_t
                + args.lambda_nce * L_nce
                + args.lambda_z_l2 * L_l2
                + args.lambda_z_var * L_var
                + args.lambda_z_cov * L_cov)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  L_t={L_t.item():.4f}  L_nce={L_nce.item():.4f}  "
                  f"L_var={L_var.item():.4f}  L_cov={L_cov.item():.4f}  alpha={tr.alpha.item():+.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    eval_split(H_t_train, H_tp_train, "train post")
    eval_split(H_t_val, H_tp_val, "val post")

    if args.save_ckpt:
        os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
        torch.save({
            "inverse_encoder": inv.state_dict(),
            "transition_model": tr.state_dict(),
            "config": {
                "state_dim": state_dim, "action_dim": args.action_dim,
                "action_tokens": args.action_tokens, "future_tokens": args.future_tokens,
                "inverse_depth": args.inverse_depth, "transition_depth": args.transition_depth,
                "encoder_name": "vit_small_patch16_224", "encoder_layer_idx": None,
                "state_tokens": 64, "k": args.k,
            },
            "args": vars(args),
        }, args.save_ckpt)
        print(f"[stage1] saved checkpoint -> {args.save_ckpt}")

    if args.save_z:
        os.makedirs(os.path.dirname(args.save_z) or ".", exist_ok=True)
        inv.eval()
        with torch.no_grad():
            Z_train = inv(H_t_train, H_tp_train).detach().cpu()
            Z_val = inv(H_t_val, H_tp_val).detach().cpu()
        torch.save({
            "H_t_train": H_t_train.cpu(), "H_tp_train": H_tp_train.cpu(),
            "Z_star_train": Z_train,
            "H_t_val": H_t_val.cpu(), "H_tp_val": H_tp_val.cpu(),
            "Z_star_val": Z_val,
        }, args.save_z)
        print(f"[stage1] saved feature/Z bundle -> {args.save_z}")


# ----------------------------------------------------------------------------
# Stage 2B: goal-conditioned flow with hard negatives
# ----------------------------------------------------------------------------

def train_goalflow(args):
    device = "cuda"
    torch.manual_seed(args.seed)

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
    H_t_train = bundle["H_t_train"].to(device); H_tp_train = bundle["H_tp_train"].to(device)
    H_t_val = bundle["H_t_val"].to(device); H_tp_val = bundle["H_tp_val"].to(device)
    Z_star_val = bundle["Z_star_val"].to(device)
    N_tr = H_t_train.shape[0]; N_va = H_t_val.shape[0]

    flow = GoalConditionedFlowPrior(cfg["state_dim"], cfg["action_dim"],
                                    depth=args.flow_depth, drop=args.flow_dropout).to(device)
    print(f"[goalflow] params: {sum(p.numel() for p in flow.parameters())/1e6:.2f} M  p_neg={args.p_neg}")

    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    flow.train()
    for step in range(args.steps):
        idx = torch.randint(0, N_tr, (args.bs,), device=device)
        h_t = H_t_train[idx]
        h_goal = H_tp_train[idx].clone()
        # hard-negative goals: fraction p_neg uses another sample's H_tp as goal
        if args.p_neg > 0:
            mask_shuf = (torch.rand(args.bs, device=device) < args.p_neg)
            if mask_shuf.any():
                shuf = torch.randperm(args.bs, device=device)
                h_goal[mask_shuf] = H_tp_train[idx][shuf][mask_shuf]
        with torch.no_grad():
            z_target = inv(h_t, h_goal)
        eps = torch.randn_like(z_target)
        tau = torch.rand(z_target.shape[0], 1, 1, device=device)
        z_tau = (1 - tau) * eps + tau * z_target
        v_target = z_target - eps
        v_pred = flow(z_tau, tau, h_t, h_goal)
        loss = F.mse_loss(v_pred, v_target)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  L_flow={loss.item():.4f}  lr={opt.param_groups[0]['lr']:.2e}")

    # ---- Eval ----
    flow.eval()
    print(f"\n[goalflow eval]  N={N_va}")
    with torch.no_grad():
        retrieval(tr(H_t_val, Z_star_val).mean(1), H_tp_val.mean(1), N_va, "oracle_Z")
        retrieval(H_t_val.mean(1), H_tp_val.mean(1), N_va, "copy_Ht")

        K_max = max(args.ks)
        samples = [euler_sample(flow, H_t_val, cfg["action_tokens"], cfg["action_dim"],
                                steps=args.sample_steps, seed=10 + k, h_goal=H_tp_val)
                   for k in range(K_max)]
        preds = [tr(H_t_val, z) for z in samples]
        retrieval(preds[0].mean(1), H_tp_val.mean(1), N_va, "goal_flow K=1")
        for k in args.ks:
            best_of_k_retrieval(preds[:k], H_tp_val, N_va, "goal_flow")

        # shuffled-goal sanity
        perm = torch.randperm(N_va, device=device)
        H_goal_shuf = H_tp_val[perm]
        z_sh = euler_sample(flow, H_t_val, cfg["action_tokens"], cfg["action_dim"],
                            steps=args.sample_steps, seed=999, h_goal=H_goal_shuf)
        pred_sh = tr(H_t_val, z_sh)
        retrieval(pred_sh.mean(1), H_tp_val[perm].mean(1), N_va, "shuf vs shuf_target")
        retrieval(pred_sh.mean(1), H_tp_val.mean(1), N_va, "shuf vs own_target")

    if args.save_ckpt:
        os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
        torch.save({"flow": flow.state_dict(), "args": vars(args)}, args.save_ckpt)
        print(f"[goalflow] saved -> {args.save_ckpt}")


# ----------------------------------------------------------------------------
# Eval: non-parametric prior baselines
# ----------------------------------------------------------------------------

def eval_baseline(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tr = ResidualTransitionModel(cfg["state_dim"], cfg["action_dim"], cfg["future_tokens"],
                                 depth=cfg["transition_depth"]).to(device)
    tr.load_state_dict(ckpt["transition_model"])
    for p in tr.parameters():
        p.requires_grad_(False)
    tr.eval()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t = bundle["H_t_val"].to(device); H_tp = bundle["H_tp_val"].to(device)
    Z_star_val = bundle["Z_star_val"].to(device)
    Z_pool = bundle["Z_star_train"].to(device)
    N = H_t.shape[0]
    print(f"[baseline] N={N}  Z_pool={Z_pool.shape[0]}  alpha={tr.alpha.item():.4f}")

    with torch.no_grad():
        retrieval(tr(H_t, Z_star_val).mean(1), H_tp.mean(1), N, "oracle_Z")
        retrieval(H_t.mean(1), H_tp.mean(1), N, "copy_Ht")
        retrieval(tr(H_t, torch.zeros_like(Z_star_val)).mean(1), H_tp.mean(1), N, "zero_Z")
        std_z = Z_star_val.flatten(1).std(0).mean().item()
        retrieval(tr(H_t, torch.randn_like(Z_star_val) * std_z).mean(1), H_tp.mean(1), N, "random_Z(stdZ)")

        K_max = max(args.ks)

        # kNN: top-K nearest train H_t neighbours; use their Z*'s
        train_keys = F.normalize(bundle["H_t_train"].to(device).mean(1), dim=-1)
        q = F.normalize(H_t.mean(1), dim=-1)
        sim = q @ train_keys.T
        topk_idx = sim.topk(K_max, dim=-1).indices

        print("\n[baseline] kNN best-of-K:")
        knn_preds = [tr(H_t, Z_pool[topk_idx[:, k]]) for k in range(K_max)]
        for k in args.ks:
            best_of_k_retrieval(knn_preds[:k], H_tp, N, f"kNN_top{k}")

        print("\n[baseline] empZ best-of-K (uniform sample from train Z*):")
        emp_preds = [tr(H_t, Z_pool[torch.randint(0, Z_pool.shape[0], (N,), device=device)])
                     for _ in range(K_max)]
        for k in args.ks:
            best_of_k_retrieval(emp_preds[:k], H_tp, N, f"empZ_{k}")

        print("\n[baseline] fgZ best-of-K (factored Gaussian on train moments):")
        z_mu = Z_pool.flatten(1).mean(0)
        z_std = Z_pool.flatten(1).std(0)
        fg_preds = [tr(H_t, (torch.randn(N, z_mu.numel(), device=device) * z_std + z_mu)
                              .view(N, cfg["action_tokens"], cfg["action_dim"]))
                    for _ in range(K_max)]
        for k in args.ks:
            best_of_k_retrieval(fg_preds[:k], H_tp, N, f"fgZ_{k}")


# ----------------------------------------------------------------------------
# Eval: goal-conditioned flow checkpoint (single + best-of-K)
# ----------------------------------------------------------------------------

def eval_goalflow(args):
    device = "cuda"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tr = ResidualTransitionModel(cfg["state_dim"], cfg["action_dim"], cfg["future_tokens"],
                                 depth=cfg["transition_depth"]).to(device)
    tr.load_state_dict(ckpt["transition_model"])
    for p in tr.parameters():
        p.requires_grad_(False)
    tr.eval()

    fc = torch.load(args.goal_flow, map_location=device, weights_only=False)
    flow = GoalConditionedFlowPrior(cfg["state_dim"], cfg["action_dim"],
                                    depth=fc["args"]["flow_depth"]).to(device)
    flow.load_state_dict(fc["flow"])
    for p in flow.parameters():
        p.requires_grad_(False)
    flow.eval()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    H_t = bundle["H_t_val"].to(device); H_tp = bundle["H_tp_val"].to(device)
    Z_star_val = bundle["Z_star_val"].to(device)
    N = H_t.shape[0]

    with torch.no_grad():
        retrieval(tr(H_t, Z_star_val).mean(1), H_tp.mean(1), N, "oracle_Z")
        retrieval(H_t.mean(1), H_tp.mean(1), N, "copy_Ht")
        K_max = max(args.ks)
        samples = [euler_sample(flow, H_t, cfg["action_tokens"], cfg["action_dim"],
                                steps=args.sample_steps, seed=10 + k, h_goal=H_tp)
                   for k in range(K_max)]
        preds = [tr(H_t, z) for z in samples]
        retrieval(preds[0].mean(1), H_tp.mean(1), N, "goal_flow K=1")
        for k in args.ks:
            best_of_k_retrieval(preds[:k], H_tp, N, f"goal_flow K={k}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_argparse():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- stage1 ----
    s1 = sub.add_parser("stage1", help="Stage 1: train inverse + residual transition (frozen encoder).")
    s1.add_argument("--video", required=True)
    s1.add_argument("--cache_train", default="")
    s1.add_argument("--cache_val", default="")
    s1.add_argument("--save_ckpt", default="")
    s1.add_argument("--save_z", default="")
    s1.add_argument("--seed", type=int, default=0)
    # data
    s1.add_argument("--k", type=int, default=16)
    s1.add_argument("--n_train", type=int, default=12000)
    s1.add_argument("--stride_train", type=int, default=1)
    s1.add_argument("--train_start", type=int, default=200)
    s1.add_argument("--n_val", type=int, default=100)
    s1.add_argument("--stride_val", type=int, default=18)
    s1.add_argument("--val_start", type=int, default=-1, help="-1 = auto train_end + block_gap")
    s1.add_argument("--block_gap", type=int, default=80)
    # model
    s1.add_argument("--action_tokens", type=int, default=8)
    s1.add_argument("--action_dim", type=int, default=256)
    s1.add_argument("--future_tokens", type=int, default=16)
    s1.add_argument("--inverse_depth", type=int, default=4)
    s1.add_argument("--transition_depth", type=int, default=2)
    # opt
    s1.add_argument("--steps", type=int, default=8000)
    s1.add_argument("--bs", type=int, default=64)
    s1.add_argument("--lr", type=float, default=3e-4)
    s1.add_argument("--wd", type=float, default=0.05)
    s1.add_argument("--log_every", type=int, default=200)
    # losses
    s1.add_argument("--nce_tau", type=float, default=0.1)
    s1.add_argument("--lambda_nce", type=float, default=1.0)
    s1.add_argument("--lambda_z_l2", type=float, default=0.001)
    s1.add_argument("--lambda_z_var", type=float, default=0.05)
    s1.add_argument("--lambda_z_cov", type=float, default=0.01)
    s1.set_defaults(func=train_stage1)

    # ---- goalflow ----
    gf = sub.add_parser("goalflow", help="Stage 2B: goal-conditioned hard-neg flow.")
    gf.add_argument("--ckpt", required=True, help="Stage 1 checkpoint path.")
    gf.add_argument("--bundle", required=True, help="oracle bundle (H_t / H_tp / Z*) from Stage 1.")
    gf.add_argument("--save_ckpt", default="")
    gf.add_argument("--seed", type=int, default=0)
    gf.add_argument("--flow_depth", type=int, default=6)
    gf.add_argument("--flow_dropout", type=float, default=0.0)
    gf.add_argument("--steps", type=int, default=6000)
    gf.add_argument("--bs", type=int, default=128)
    gf.add_argument("--lr", type=float, default=3e-4)
    gf.add_argument("--wd", type=float, default=0.05)
    gf.add_argument("--log_every", type=int, default=200)
    gf.add_argument("--sample_steps", type=int, default=30)
    gf.add_argument("--p_neg", type=float, default=0.5)
    gf.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    gf.set_defaults(func=train_goalflow)

    # ---- eval_baseline ----
    eb = sub.add_parser("eval_baseline", help="Non-parametric Z* priors (empZ / kNN / fgZ) best-of-K.")
    eb.add_argument("--ckpt", required=True)
    eb.add_argument("--bundle", required=True)
    eb.add_argument("--seed", type=int, default=0)
    eb.add_argument("--ks", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128])
    eb.set_defaults(func=eval_baseline)

    # ---- eval_goalflow ----
    eg = sub.add_parser("eval_goalflow", help="Single + best-of-K eval of a goal-flow checkpoint.")
    eg.add_argument("--ckpt", required=True)
    eg.add_argument("--goal_flow", required=True)
    eg.add_argument("--bundle", required=True)
    eg.add_argument("--sample_steps", type=int, default=30)
    eg.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    eg.set_defaults(func=eval_goalflow)

    return p


def main():
    p = build_argparse()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
