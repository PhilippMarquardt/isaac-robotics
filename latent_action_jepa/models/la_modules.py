"""Latent-action modules on top of frozen V-JEPA-2 features.

Notation. H_t: [B, P, D] (P=196 patch tokens of the most recent tubelet).
H_targets: [B, S, P, D] where S=8 future tubelets. H_context: [B, N_ctx, D]
flat (default N_ctx = 1568 for a 16-frame context clip).

All modules are small enough to train at bs=2-4 on 32 GB.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit3d import Block, get_3d_sincos_pos_embed
from .flow_prior import sinusoidal_time_embed


class FSQ(nn.Module):
    """Finite Scalar Quantization. No learned codebook, no collapse.

    Each dim is independently quantized to one of L_d levels. Codebook size
    = prod(levels). Round with straight-through estimator.
    """

    def __init__(self, levels):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32))
        self.D = len(levels)

    @property
    def codebook_size(self):
        return int(self.levels.prod().item())

    def forward(self, z):
        # z: [..., D] where D = len(levels)
        L = self.levels.to(z.device)
        half = (L - 1) / 2.0
        z_b = torch.tanh(z) * half                                  # bound to [-half, +half]
        z_q = z_b + (torch.round(z_b) - z_b).detach()                # STE
        return z_q

    @torch.no_grad()
    def to_index(self, z_q):
        """z_q: [..., D] integer-valued floats in [-half, half]. Returns flat index in [0, K)."""
        L = self.levels.to(z_q.device)
        half = (L - 1) / 2.0
        shifted = (z_q + half).long()                                # [..., D] integers in [0, L_d)
        # mixed-radix encoding
        idx = torch.zeros(z_q.shape[:-1], dtype=torch.long, device=z_q.device)
        radix = 1
        for d in range(self.D):
            idx = idx + shifted[..., d] * radix
            radix = radix * int(L[d].item())
        return idx


class FSQBottleneck(nn.Module):
    """Linear down to small dim, FSQ, linear back up to action_dim."""

    def __init__(self, action_dim=128, levels=(5, 5, 5, 5, 5)):
        super().__init__()
        self.proj_down = nn.Linear(action_dim, len(levels))
        self.fsq = FSQ(list(levels))
        self.proj_up = nn.Linear(len(levels), action_dim)
        self.codebook_size = self.fsq.codebook_size

    def forward(self, z):
        h = self.proj_down(z)
        h_q = self.fsq(h)
        idx = self.fsq.to_index(h_q.detach())
        out = self.proj_up(h_q)
        return out, idx


class VectorQuantizer(nn.Module):
    """EMA-updated VQ-VAE codebook with straight-through estimator.

    Input z [B, S, D]. Per-step the same codebook is used. Returns
    (z_q, indices, commit_loss). z_q has straight-through gradient to z.
    """

    def __init__(self, num_codes=256, code_dim=128, commit=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.K = num_codes
        self.D = code_dim
        self.commit = commit
        self.decay = decay
        self.eps = eps
        self.register_buffer("codes", torch.randn(num_codes, code_dim) * 0.02)
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_sum", torch.zeros(num_codes, code_dim))
        self.register_buffer("init_done", torch.tensor(False))

    @torch.no_grad()
    def _init_codes(self, flat):
        N = flat.shape[0]
        if N >= self.K:
            idx = torch.randperm(N, device=flat.device)[:self.K]
            self.codes.copy_(flat[idx])
        else:
            # sample with replacement
            idx = torch.randint(0, N, (self.K,), device=flat.device)
            self.codes.copy_(flat[idx])
        self.ema_count.fill_(1.0)
        self.ema_sum.copy_(self.codes)
        self.init_done.fill_(True)

    def forward(self, z):
        B, S, D = z.shape
        flat = z.reshape(-1, D)
        if self.training and not bool(self.init_done.item()):
            self._init_codes(flat.detach())
        # distances
        d = flat.pow(2).sum(-1, keepdim=True) - 2 * flat @ self.codes.T \
            + self.codes.pow(2).sum(-1)
        idx = d.argmin(dim=-1)                                     # [B*S]
        z_q_flat = self.codes[idx]                                 # [B*S, D]
        # EMA update of codebook
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.K).type_as(flat)
                n = onehot.sum(0)
                self.ema_count.mul_(self.decay).add_(n, alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(onehot.T @ flat, alpha=1 - self.decay)
                count_smooth = (self.ema_count + self.eps) / (
                    self.ema_count.sum() + self.K * self.eps) * self.ema_count.sum()
                self.codes.copy_(self.ema_sum / count_smooth.unsqueeze(-1))
        commit = F.mse_loss(flat, z_q_flat.detach())
        z_q_flat = flat + (z_q_flat - flat).detach()              # straight-through
        z_q = z_q_flat.reshape(B, S, D)
        return z_q, idx.reshape(B, S), commit


def _step_embed(n_steps, dim):
    """Sin-cos step embedding [n_steps, dim]."""
    half = dim // 2
    omega = torch.arange(half, dtype=torch.float64) / half
    omega = 1.0 / (10000 ** omega)
    pos = torch.arange(n_steps, dtype=torch.float64)
    args = pos[:, None] * omega[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1).float()


class EffectNormalizer(nn.Module):
    """Strip film/style content from the inverse's input residual.

    Modes:
      none     : pass-through (baseline / debugging).
      layernorm: LayerNorm over the feature axis D per (sample, step, patch).
      patchstd : Per (sample, step) standardize across patches:
                 (delta - mean(P)) / (std(P) + eps).  Removes global step bias.
      dirnorm  : Unit-L2 along feature axis (cosine direction only). Strips scale.
      patch_then_dir: patchstd then dirnorm. Strongest "effect" extraction.
    """

    def __init__(self, mode="patchstd", state_dim=384, eps=1e-5):
        super().__init__()
        self.mode = mode
        self.eps = eps
        if mode == "layernorm":
            self.ln = nn.LayerNorm(state_dim, eps=eps)
        elif mode == "patchstd_affine":
            # learned scale/shift after patchstd
            self.scale = nn.Parameter(torch.ones(state_dim))
            self.shift = nn.Parameter(torch.zeros(state_dim))

    def forward(self, delta):
        # delta: [B, S, P, D]
        if self.mode == "none":
            return delta
        if self.mode == "layernorm":
            return self.ln(delta)
        if self.mode == "patchstd":
            mean = delta.mean(dim=2, keepdim=True)
            std = delta.std(dim=2, keepdim=True).clamp_min(self.eps)
            return (delta - mean) / std
        if self.mode == "patchstd_affine":
            mean = delta.mean(dim=2, keepdim=True)
            std = delta.std(dim=2, keepdim=True).clamp_min(self.eps)
            out = (delta - mean) / std
            return out * self.scale + self.shift
        if self.mode == "dirnorm":
            return F.normalize(delta, dim=-1, eps=self.eps)
        if self.mode == "patch_then_dir":
            mean = delta.mean(dim=2, keepdim=True)
            std = delta.std(dim=2, keepdim=True).clamp_min(self.eps)
            x = (delta - mean) / std
            return F.normalize(x, dim=-1, eps=self.eps)
        raise ValueError(f"unknown EffectNormalizer mode: {self.mode}")


class CrossBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.nq = nn.LayerNorm(dim)
        self.nk = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.ns = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.nm = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(h, dim), nn.Dropout(drop))

    def forward(self, q, kv):
        a, _ = self.attn(self.nq(q), self.nk(kv), self.nk(kv), need_weights=False)
        q = q + a
        s = self.ns(q)
        sa, _ = self.self_attn(s, s, s, need_weights=False)
        q = q + sa
        q = q + self.mlp(self.nm(q))
        return q


class StepBase(nn.Module):
    """Per-step base predictor.

    H_t [B, P, D] -> base [B, S, P, D]  where base_i = "predicted H_{t+i} under
    no action".  Implemented as a small per-step MLP applied per-patch:

        base_i(p) = MLP_i(H_t(p))

    This gives the inverse a clean motion residual signal:
        delta_i = stopgrad(H_target_i) - stopgrad(base_i)
    while leaving the transition free to learn its own H_pred independently.
    """

    def __init__(self, state_dim=384, n_steps=4, hidden=None, depth=2):
        super().__init__()
        h = hidden or state_dim
        # one tiny MLP per step (cheap; n_steps <= 16 normally)
        def mk_mlp():
            layers = [nn.Linear(state_dim, h), nn.GELU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(h, h), nn.GELU()]
            layers += [nn.Linear(h, state_dim)]
            return nn.Sequential(*layers)
        self.steps = nn.ModuleList([mk_mlp() for _ in range(n_steps)])

    def forward(self, H_t):
        # H_t [B, P, D] -> [B, S, P, D]
        return torch.stack([s(H_t) for s in self.steps], dim=1)


class InverseFromDelta(nn.Module):
    """Z* = Inverse(delta_1..S).

    Inverse sees ONLY the motion residual `delta = H_target - base(H_t)`.
    No raw state pair. Forces Z to encode motion / change, not state /
    appearance. F12 lesson applied at the encoder-output level.
    """

    def __init__(self, state_dim=384, action_dim=128, n_steps=4, n_patch=196,
                 depth=4, heads=4, drop=0.0):
        super().__init__()
        self.n_steps = n_steps
        self.proj = nn.Linear(state_dim, action_dim)
        self.action_q = nn.Parameter(torch.randn(1, n_steps, action_dim) * 0.02)
        self.register_buffer("step_embed",
                             _step_embed(n_steps, action_dim).unsqueeze(0).unsqueeze(2),
                             persistent=False)
        self.blocks = nn.ModuleList([CrossBlock(action_dim, heads=heads, drop=drop)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(action_dim)

    def forward(self, deltas):
        # deltas: [B, S, P, D] -> Z [B, S, action_dim]
        B, S, P, D = deltas.shape
        d = self.proj(deltas) + self.step_embed[:, :S]              # [B, S, P, A]
        kv = d.reshape(B, S * P, -1)
        q = self.action_q[:, :S].expand(B, -1, -1).contiguous()
        for blk in self.blocks:
            q = blk(q, kv)
        return self.norm(q)                                          # [B, S, A]


class InverseFromDeltaPerStep(nn.Module):
    """Per-step inverse: Z_i computed ONLY from delta_i.

    Each Z_i is forced to be a function of its own step's motion residual.
    Shared inverse module across steps (weight sharing), but kv at step i is
    only delta_i. This GUARANTEES time-shuffle changes the inputs to the
    transition step routing — Z_i carries step-i-specific information.
    """

    def __init__(self, state_dim=384, action_dim=128, n_steps=4, n_patch=196,
                 depth=4, heads=4, drop=0.0):
        super().__init__()
        self.n_steps = n_steps
        self.proj = nn.Linear(state_dim, action_dim)
        # one query per step (distinct so step embeds break ties even with shared kv)
        self.action_q = nn.Parameter(torch.randn(1, n_steps, action_dim) * 0.02)
        self.blocks = nn.ModuleList([CrossBlock(action_dim, heads=heads, drop=drop)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(action_dim)

    def forward(self, deltas):
        # deltas: [B, S, P, D] -> Z [B, S, action_dim]
        B, S, P, D = deltas.shape
        d = self.proj(deltas)                                         # [B, S, P, A]
        # per-step routing: queries Z_i cross-attend ONLY to delta_i
        kv = d.reshape(B * S, P, -1)
        q = self.action_q[:, :S].expand(B, -1, -1).reshape(B * S, 1, -1).contiguous()
        for blk in self.blocks:
            q = blk(q, kv)
        return self.norm(q).reshape(B, S, -1)                         # [B, S, action_dim]


class InverseActionMultiStep(nn.Module):
    """(H_t [B,P,D], H_targets [B,S,P,D]) -> Z* [B,S,action_dim]."""

    def __init__(self, state_dim=384, action_dim=128, n_steps=8, n_patch=196,
                 depth=4, heads=4, drop=0.0):
        super().__init__()
        self.n_steps = n_steps
        self.proj_t = nn.Linear(state_dim, action_dim)
        self.proj_tgt = nn.Linear(state_dim, action_dim)
        self.t_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.tgt_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.queries = nn.Parameter(torch.randn(1, n_steps, action_dim) * 0.02)
        self.register_buffer("step_embed", _step_embed(n_steps, action_dim).unsqueeze(0),
                             persistent=False)
        # also add temporal embedding to target tokens so the inverse can read time order
        self.register_buffer("target_step_embed",
                             _step_embed(n_steps, action_dim).unsqueeze(0).unsqueeze(2),
                             persistent=False)
        self.blocks = nn.ModuleList([CrossBlock(action_dim, heads=heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(action_dim)

    def forward(self, H_t, H_targets):
        B = H_t.shape[0]
        S = H_targets.shape[1]
        ht = self.proj_t(H_t) + self.t_marker
        htgt = self.proj_tgt(H_targets) + self.tgt_marker
        htgt = htgt + self.target_step_embed[:, :S]
        htgt = htgt.reshape(B, S * H_targets.shape[2], -1)
        kv = torch.cat([ht, htgt], dim=1)
        q = self.queries.expand(B, -1, -1) + self.step_embed[:, :S]
        for blk in self.blocks:
            q = blk(q, kv)
        return self.norm(q)


class MultiStepTransition(nn.Module):
    """(H_t [B,P,D], Z [B,S,action_dim]) -> H_pred [B,S,P,D].

    base = H_t broadcast per step (gives a copy floor),
    delta = cross-attn from learned future queries (with step embed) to
            [H_t || projected Z].
    H_pred = base + alpha * delta. alpha per step, clamped >= 0.02.
    """

    def __init__(self, state_dim=384, action_dim=128, n_steps=8, n_patch=196,
                 depth=4, heads=6, drop=0.0, alpha_min=0.02, alpha_init=0.05):
        super().__init__()
        self.n_steps = n_steps
        self.n_patch = n_patch
        self.z_proj = nn.Linear(action_dim, state_dim)
        self.h_marker = nn.Parameter(torch.zeros(1, 1, state_dim))
        self.z_marker = nn.Parameter(torch.zeros(1, 1, state_dim))
        self.future_q = nn.Parameter(torch.randn(1, n_steps, n_patch, state_dim) * 0.02)
        self.register_buffer("step_state_embed",
                             _step_embed(n_steps, state_dim).unsqueeze(0).unsqueeze(2),
                             persistent=False)
        self.register_buffer("z_step_embed", _step_embed(n_steps, state_dim).unsqueeze(0),
                             persistent=False)
        self.blocks = nn.ModuleList([CrossBlock(state_dim, heads=heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(state_dim)
        self.alpha_min = float(alpha_min)
        self.raw_alpha = nn.Parameter(torch.full((n_steps,), float(alpha_init)))

    @property
    def alpha(self):
        return self.raw_alpha.clamp(min=self.alpha_min)

    def forward(self, H_t, Z):
        B, P, D = H_t.shape
        S = Z.shape[1]
        # memory = H_t + Z_proj (per step with embeds)
        z_proj = self.z_proj(Z) + self.z_marker + self.z_step_embed[:, :S]  # [B,S,D]
        ht = H_t + self.h_marker                                              # [B,P,D]
        mem = torch.cat([ht, z_proj], dim=1)
        q = self.future_q[:, :S].expand(B, -1, -1, -1).clone()                # [B,S,P,D]
        q = q + self.step_state_embed[:, :S]
        q = q.reshape(B, S * P, D)
        for blk in self.blocks:
            q = blk(q, mem)
        q = self.norm(q).reshape(B, S, P, D)
        # alpha-gated delta on top of base (broadcast H_t)
        base = H_t.unsqueeze(1).expand(-1, S, -1, -1)
        alpha = self.alpha.view(1, S, 1, 1)
        return base + alpha * q


class MultiStepTransitionPerStep(nn.Module):
    """Per-step routing: output step i sees only Z[:, i] in the action stream.

    Implementation tiles H_t per step and concatenates each Z[:, i] alone.
    Forces Z to behave as a temporal action chunk: shuffling Z's time order
    necessarily changes which action is applied at which output step.
    """

    def __init__(self, state_dim=384, action_dim=128, n_steps=8, n_patch=196,
                 depth=4, heads=6, drop=0.0, alpha_min=0.02, alpha_init=0.05):
        super().__init__()
        self.n_steps = n_steps
        self.n_patch = n_patch
        self.z_proj = nn.Linear(action_dim, state_dim)
        self.h_marker = nn.Parameter(torch.zeros(1, 1, state_dim))
        self.z_marker = nn.Parameter(torch.zeros(1, 1, state_dim))
        self.future_q = nn.Parameter(torch.randn(1, n_steps, n_patch, state_dim) * 0.02)
        self.register_buffer("step_state_embed",
                             _step_embed(n_steps, state_dim).unsqueeze(0).unsqueeze(2),
                             persistent=False)
        # NOTE: no step embedding on Z — Z[i] is implicit-action-at-step-i via routing.
        self.blocks = nn.ModuleList([CrossBlock(state_dim, heads=heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(state_dim)
        self.alpha_min = float(alpha_min)
        self.raw_alpha = nn.Parameter(torch.full((n_steps,), float(alpha_init)))

    @property
    def alpha(self):
        return self.raw_alpha.clamp(min=self.alpha_min)

    def forward(self, H_t, Z):
        B, P, D = H_t.shape
        S = Z.shape[1]
        z_proj = self.z_proj(Z) + self.z_marker            # [B, S, D]
        ht = H_t + self.h_marker                             # [B, P, D]
        # tile H_t per step and concat single Z[i] per step
        ht_tiled = ht.unsqueeze(1).expand(B, S, P, D).reshape(B * S, P, D)
        z_tiled = z_proj.reshape(B * S, 1, D)
        mem = torch.cat([ht_tiled, z_tiled], dim=1)          # [B*S, P+1, D]
        q = (self.future_q[:, :S] + self.step_state_embed[:, :S]).expand(B, -1, -1, -1)
        q = q.reshape(B * S, P, D)
        for blk in self.blocks:
            q = blk(q, mem)
        q = self.norm(q).reshape(B, S, P, D)
        base = H_t.unsqueeze(1).expand(-1, S, -1, -1)
        alpha = self.alpha.view(1, S, 1, 1)
        return base + alpha * q


class SingleStepInverse(nn.Module):
    """(H_t [B,P,D], H_next [B,P,D]) -> z [B, action_dim]."""

    def __init__(self, state_dim=384, action_dim=128, depth=3, heads=4, drop=0.0):
        super().__init__()
        self.proj_t = nn.Linear(state_dim, action_dim)
        self.proj_n = nn.Linear(state_dim, action_dim)
        self.t_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.n_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.q = nn.Parameter(torch.randn(1, 1, action_dim) * 0.02)
        self.blocks = nn.ModuleList([CrossBlock(action_dim, heads=heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(action_dim)

    def forward(self, H_t, H_next):
        B = H_t.shape[0]
        ht = self.proj_t(H_t) + self.t_marker
        hn = self.proj_n(H_next) + self.n_marker
        kv = torch.cat([ht, hn], dim=1)
        q = self.q.expand(B, -1, -1).contiguous()
        for blk in self.blocks:
            q = blk(q, kv)
        return self.norm(q).squeeze(1)              # [B, action_dim]


class SingleStepTransition(nn.Module):
    """(H_t [B,P,D], z [B, action_dim]) -> H_pred [B,P,D].

    base = H_t (broadcast). delta from cross-attn(future_q + 1-token z_proj).
    H_pred = base + delta (no alpha gate; delta has its own scale via the head).
    """

    def __init__(self, state_dim=384, action_dim=128, n_patch=196,
                 depth=4, heads=6, drop=0.0):
        super().__init__()
        self.n_patch = n_patch
        self.z_proj = nn.Linear(action_dim, state_dim)
        self.h_marker = nn.Parameter(torch.zeros(1, 1, state_dim))
        self.z_marker = nn.Parameter(torch.zeros(1, 1, state_dim))
        self.future_q = nn.Parameter(torch.randn(1, n_patch, state_dim) * 0.02)
        self.blocks = nn.ModuleList([CrossBlock(state_dim, heads=heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(state_dim)
        self.head = nn.Linear(state_dim, state_dim)

    def forward(self, H_t, z):
        # z: [B, action_dim]
        B, P, D = H_t.shape
        z_tok = self.z_proj(z).unsqueeze(1) + self.z_marker          # [B,1,D]
        ht = H_t + self.h_marker
        mem = torch.cat([ht, z_tok], dim=1)
        q = self.future_q.expand(B, -1, -1).contiguous()
        for blk in self.blocks:
            q = blk(q, mem)
        q = self.norm(q)
        delta = self.head(q)
        return H_t + delta


class MultiStepTransitionNoBase(nn.Module):
    """No base broadcast. Pure cross-attn predictor.

    H_pred = norm(crossattn(future_q + step_embed, [H_t || Z_proj])) -> head.
    Forces Z to carry future content (no skip from H_t that biases toward copy).
    """

    def __init__(self, state_dim=384, action_dim=128, n_steps=8, n_patch=196,
                 depth=4, heads=6, drop=0.0):
        super().__init__()
        self.n_steps = n_steps
        self.n_patch = n_patch
        self.z_proj = nn.Linear(action_dim, state_dim)
        self.h_marker = nn.Parameter(torch.zeros(1, 1, state_dim))
        self.z_marker = nn.Parameter(torch.zeros(1, 1, state_dim))
        self.future_q = nn.Parameter(torch.randn(1, n_steps, n_patch, state_dim) * 0.02)
        self.register_buffer("step_state_embed",
                             _step_embed(n_steps, state_dim).unsqueeze(0).unsqueeze(2),
                             persistent=False)
        self.register_buffer("z_step_embed", _step_embed(n_steps, state_dim).unsqueeze(0),
                             persistent=False)
        self.blocks = nn.ModuleList([CrossBlock(state_dim, heads=heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(state_dim)
        self.head = nn.Linear(state_dim, state_dim)

    @property
    def alpha(self):
        return torch.ones(self.n_steps, device=self.z_proj.weight.device)

    def forward(self, H_t, Z):
        B, P, D = H_t.shape
        S = Z.shape[1]
        z_proj = self.z_proj(Z) + self.z_marker + self.z_step_embed[:, :S]
        ht = H_t + self.h_marker
        mem = torch.cat([ht, z_proj], dim=1)
        q = (self.future_q[:, :S] + self.step_state_embed[:, :S]).expand(B, -1, -1, -1)
        q = q.reshape(B, S * P, D)
        for blk in self.blocks:
            q = blk(q, mem)
        q = self.norm(q).reshape(B, S, P, D)
        return self.head(q)


class MaskedGoalMultiStepFlow(nn.Module):
    """Flow over Z[B,S,A] conditioned on H_context [B,N,D] and a partially-visible
    H_goal [B,P,D]. Hidden goal tokens are replaced by a learned [MASK] in
    state-dim then projected. Each goal token also gets a visible/hidden marker.
    """

    def __init__(self, state_dim=384, action_dim=128, n_steps=8, n_patch=196,
                 depth=6, heads=4, drop=0.0):
        super().__init__()
        self.n_steps = n_steps
        self.n_patch = n_patch
        self.ctx_proj = nn.Linear(state_dim, action_dim)
        self.goal_proj = nn.Linear(state_dim, action_dim)
        self.ctx_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.goal_vis_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.goal_hidden_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        # learned [MASK] in raw state space so projection has structure
        self.mask_token = nn.Parameter(torch.randn(1, 1, state_dim) * 0.02)
        self.register_buffer("step_embed", _step_embed(n_steps, action_dim).unsqueeze(0),
                             persistent=False)
        self.time_dim = action_dim
        self.blocks = nn.ModuleList([CrossBlock(action_dim, heads=heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(action_dim)
        self.head = nn.Linear(action_dim, action_dim)

    def forward(self, z_tau, tau, H_context, H_goal, visible_mask):
        """
        H_context: [B, N_ctx, state_dim]
        H_goal:    [B, P, state_dim]
        visible_mask: [B, P] bool (True = goal token revealed)
        """
        B, S, _ = z_tau.shape
        if tau.ndim == 1:
            tau = tau.view(-1, 1, 1)
        m = visible_mask.float().unsqueeze(-1)                                    # [B,P,1]
        G_in = m * H_goal + (1.0 - m) * self.mask_token                            # broadcast learned [MASK]
        g_proj = self.goal_proj(G_in)                                              # [B,P,A]
        markers = m * self.goal_vis_marker + (1.0 - m) * self.goal_hidden_marker
        g_proj = g_proj + markers
        ctx = self.ctx_proj(H_context) + self.ctx_marker
        t_emb = sinusoidal_time_embed(tau, self.time_dim)                          # [B,1,A]
        kv = torch.cat([ctx, g_proj, t_emb], dim=1)
        x = z_tau + t_emb + self.step_embed[:, :S]
        for blk in self.blocks:
            x = blk(x, kv)
        return self.head(self.norm(x))


class DiTCrossBlock(nn.Module):
    """DiT-style block with adaLN modulation + cross-attention to context.

    Three sub-branches (self-attn, cross-attn, mlp). Each has its own
    (shift, scale, gate) modulation derived from the conditioning vector c.
    Final-linear in adaLN MLP zero-inited so block starts as identity.
    """

    def __init__(self, dim, heads=4, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm_q = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, dim), nn.Dropout(drop))
        # 9 = (shift, scale, gate) x 3 branches (sa, ca, mlp)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, kv, c):
        params = self.ada(c).chunk(9, dim=-1)
        ssa, csa, gsa, sca, cca, gca, smlp, cmlp, gmlp = [p.unsqueeze(1) for p in params]
        # self-attn
        h = self.norm_sa(x) * (1 + csa) + ssa
        a, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + gsa * a
        # cross-attn (queries from x, k/v from kv)
        q = self.norm_q(x) * (1 + cca) + sca
        k = self.norm_kv(kv)
        ca, _ = self.cross_attn(q, k, k, need_weights=False)
        x = x + gca * ca
        # mlp
        h2 = self.norm_mlp(x) * (1 + cmlp) + smlp
        x = x + gmlp * self.mlp(h2)
        return x


class ContextDiTFlow(nn.Module):
    """DiT-style flow over Z [B,S,action_dim]. Conditioning:
    - adaLN params per block from (t_emb + pooled context).
    - Cross-attention from Z queries to full H_context tokens (token-level info preserved).
    Head + adaLN final layers zero-inited (standard DiT init).
    """

    def __init__(self, state_dim=384, action_dim=128, n_steps=4,
                 depth=6, heads=4, drop=0.0):
        super().__init__()
        self.n_steps = n_steps
        self.action_dim = action_dim
        self.ctx_proj = nn.Linear(state_dim, action_dim)
        self.ctx_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.step_embed = nn.Parameter(torch.randn(1, n_steps, action_dim) * 0.02)
        self.cond_mlp = nn.Sequential(
            nn.Linear(2 * action_dim, action_dim),
            nn.SiLU(),
            nn.Linear(action_dim, action_dim),
        )
        self.blocks = nn.ModuleList([DiTCrossBlock(action_dim, heads, mlp_ratio=4.0, drop=drop)
                                     for _ in range(depth)])
        self.final_norm = nn.LayerNorm(action_dim, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(action_dim, 2 * action_dim))
        nn.init.zeros_(self.final_mod[-1].weight)
        nn.init.zeros_(self.final_mod[-1].bias)
        self.head = nn.Linear(action_dim, action_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z_tau, tau, H_context):
        B, S, _ = z_tau.shape
        if tau.ndim == 1:
            tau = tau.view(-1, 1, 1)
        t_emb = sinusoidal_time_embed(tau, self.action_dim).squeeze(1)        # [B, A]
        ctx_proj = self.ctx_proj(H_context) + self.ctx_marker                  # [B, N, A]
        ctx_pool = ctx_proj.mean(1)                                             # [B, A]
        c = self.cond_mlp(torch.cat([t_emb, ctx_pool], dim=-1))                 # [B, A]
        x = z_tau + self.step_embed[:, :S]
        for blk in self.blocks:
            x = blk(x, ctx_proj, c)
        shift, scale = self.final_mod(c).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.head(x)


class GoalConditionedDiTFlow(nn.Module):
    """DiT planner: noisy compact action tokens + context + goal -> velocity.

    Inputs:
      z_tau     [B, S, action_dim]
      tau       [B] or [B,1,1]
      H_context [B, Nctx, state_dim]   visible online encoder output
      H_goal    [B, Ng,   state_dim]   target encoder on goal tubelet
      goal_mask [B] or [B,1]           1.0 = use real goal, 0.0 = null goal

    Memory:
      context tokens with ctx marker + spatial pos
      goal tokens with goal marker + spatial pos
      when goal_mask=0, swap goal tokens with learned null token.

    adaLN modulation conditioning vector c built from
      (t_emb + pooled context + pooled goal_or_null)
    so the per-block modulation also gets a "goal-on / goal-off" signal.

    Head + final adaLN linear zero-inited (DiT standard).
    """

    def __init__(self, state_dim=384, action_dim=128, n_steps=4,
                 depth=6, heads=4, drop=0.0):
        super().__init__()
        self.n_steps = n_steps
        self.action_dim = action_dim
        self.ctx_proj = nn.Linear(state_dim, action_dim)
        self.goal_proj = nn.Linear(state_dim, action_dim)
        self.ctx_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.goal_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        # learned null token in action_dim used when goal_mask = 0
        self.null_goal = nn.Parameter(torch.randn(1, 1, action_dim) * 0.02)
        self.step_embed = nn.Parameter(torch.randn(1, n_steps, action_dim) * 0.02)
        self.cond_mlp = nn.Sequential(
            nn.Linear(3 * action_dim, action_dim),
            nn.SiLU(),
            nn.Linear(action_dim, action_dim),
        )
        self.blocks = nn.ModuleList([DiTCrossBlock(action_dim, heads, mlp_ratio=4.0, drop=drop)
                                     for _ in range(depth)])
        self.final_norm = nn.LayerNorm(action_dim, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(action_dim, 2 * action_dim))
        nn.init.zeros_(self.final_mod[-1].weight); nn.init.zeros_(self.final_mod[-1].bias)
        self.head = nn.Linear(action_dim, action_dim)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, z_tau, tau, H_context, H_goal, goal_mask=None):
        B, S, _ = z_tau.shape
        Ng = H_goal.shape[1]
        if tau.ndim == 1:
            tau = tau.view(-1, 1, 1)
        if goal_mask is None:
            goal_mask = torch.ones(B, device=z_tau.device)
        goal_mask = goal_mask.view(B, 1, 1)

        ctx_proj = self.ctx_proj(H_context) + self.ctx_marker
        ctx_pool = ctx_proj.mean(1)

        goal_proj = self.goal_proj(H_goal) + self.goal_marker
        null = self.null_goal.expand(B, Ng, -1)
        goal_tokens = goal_mask * goal_proj + (1.0 - goal_mask) * null
        goal_pool = goal_tokens.mean(1)

        t_emb = sinusoidal_time_embed(tau, self.action_dim).squeeze(1)        # [B, A]
        c = self.cond_mlp(torch.cat([t_emb, ctx_pool, goal_pool], dim=-1))     # [B, A]

        memory = torch.cat([ctx_proj, goal_tokens], dim=1)                     # [B, Nctx+Ng, A]
        x = z_tau + self.step_embed[:, :S]
        for blk in self.blocks:
            x = blk(x, memory, c)
        shift, scale = self.final_mod(c).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.head(x)


class ContextOnlyMultiStepFlow(nn.Module):
    """Flow over Z [B,S,action_dim] conditioned on context tokens ONLY (no goal).

    Use for end-to-end JEPA+LA where the prior should generate Z from past
    context (not from a future-goal hint).
    """

    def __init__(self, state_dim=384, action_dim=128, n_steps=4,
                 depth=6, heads=4, drop=0.0):
        super().__init__()
        self.n_steps = n_steps
        self.ctx_proj = nn.Linear(state_dim, action_dim)
        self.ctx_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.register_buffer("step_embed", _step_embed(n_steps, action_dim).unsqueeze(0),
                             persistent=False)
        self.time_dim = action_dim
        self.blocks = nn.ModuleList([CrossBlock(action_dim, heads=heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(action_dim)
        self.head = nn.Linear(action_dim, action_dim)

    def forward(self, z_tau, tau, H_context):
        B, S, _ = z_tau.shape
        if tau.ndim == 1:
            tau = tau.view(-1, 1, 1)
        t_emb = sinusoidal_time_embed(tau, self.time_dim)
        x = z_tau + t_emb + self.step_embed[:, :S]
        ctx = self.ctx_proj(H_context) + self.ctx_marker
        kv = torch.cat([ctx, t_emb], dim=1)
        for blk in self.blocks:
            x = blk(x, kv)
        return self.head(self.norm(x))


class GoalConditionedMultiStepFlow(nn.Module):
    """Flow over Z [B,S,action_dim] conditioned on H_context and H_goal.

    H_context: [B, N_ctx, state_dim]
    H_goal:    [B, P_goal, state_dim]
    """

    def __init__(self, state_dim=384, action_dim=128, n_steps=8,
                 depth=6, heads=4, drop=0.0):
        super().__init__()
        self.n_steps = n_steps
        self.ctx_proj = nn.Linear(state_dim, action_dim)
        self.goal_proj = nn.Linear(state_dim, action_dim)
        self.ctx_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.goal_marker = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.register_buffer("step_embed", _step_embed(n_steps, action_dim).unsqueeze(0),
                             persistent=False)
        self.time_dim = action_dim
        self.blocks = nn.ModuleList([CrossBlock(action_dim, heads=heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(action_dim)
        self.head = nn.Linear(action_dim, action_dim)

    def forward(self, z_tau, tau, H_context, H_goal):
        B = z_tau.shape[0]; S = z_tau.shape[1]
        if tau.ndim == 1:
            tau = tau.view(-1, 1, 1)
        t_emb = sinusoidal_time_embed(tau, self.time_dim)
        x = z_tau + t_emb + self.step_embed[:, :S]
        ctx = self.ctx_proj(H_context) + self.ctx_marker
        goal = self.goal_proj(H_goal) + self.goal_marker
        kv = torch.cat([ctx, goal, t_emb], dim=1)
        for blk in self.blocks:
            x = blk(x, kv)
        return self.head(self.norm(x))
