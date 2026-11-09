"""V-JEPA-2 multi-block 3D mask collator.

For each sample, for each mask spec, sample `num_blocks` 3D blocks and zero
their positions in the encoder visibility map. Return (visible_indices,
target_indices) lists per mask spec, truncated to a min-length per batch
(the standard V-JEPA-2 collator uses min lengths across the batch).

Spec format:
    {"num_blocks": 8, "aspect_ratio": (0.75, 1.5), "spatial_scale": (0.15, 0.15),
     "temporal_scale": (1.0, 1.0)}
"""
import math
import random
import torch


def sample_block(T, H, W, spec):
    """Return (t_start, t_size, h_start, h_size, w_start, w_size) for one block."""
    t_lo, t_hi = spec["temporal_scale"]
    sp_lo, sp_hi = spec["spatial_scale"]
    ar_lo, ar_hi = spec["aspect_ratio"]
    t_scale = random.uniform(t_lo, t_hi)
    sp_scale = random.uniform(sp_lo, sp_hi)
    ar = random.uniform(ar_lo, ar_hi)
    t = max(1, int(round(T * t_scale)))
    spatial_keep = max(1, int(round(H * W * sp_scale)))
    h = max(1, min(H, int(round(math.sqrt(spatial_keep * ar)))))
    w = max(1, min(W, int(round(math.sqrt(spatial_keep / ar)))))
    t_start = random.randint(0, T - t)
    h_start = random.randint(0, H - h)
    w_start = random.randint(0, W - w)
    return t_start, t, h_start, h, w_start, w


def make_mask(T, H, W, specs):
    """Construct a 3D visibility map of shape [T, H, W]; 1 = keep (encoder sees),
    0 = mask (target to predict). All blocks across specs are combined.
    Returns (vis_idx, pred_idx) as 1D tensors over T*H*W.
    """
    vis = torch.ones(T, H, W, dtype=torch.bool)
    for spec in specs:
        for _ in range(spec["num_blocks"]):
            ts, t, hs, h, ws, w = sample_block(T, H, W, spec)
            vis[ts:ts + t, hs:hs + h, ws:ws + w] = False
    flat = vis.reshape(-1)
    vis_idx = torch.nonzero(flat, as_tuple=False).squeeze(-1)
    pred_idx = torch.nonzero(~flat, as_tuple=False).squeeze(-1)
    return vis_idx, pred_idx


def batched_mask(B, T, H, W, specs, device="cpu"):
    """Sample B masks. Truncate each to per-batch minimums (V-JEPA-2 behavior)."""
    vis_list, pred_list = [], []
    for _ in range(B):
        vi, pi = make_mask(T, H, W, specs)
        vis_list.append(vi); pred_list.append(pi)
    if all(v.numel() == 0 for v in vis_list) or all(p.numel() == 0 for p in pred_list):
        raise RuntimeError("Degenerate mask: every block covers entire volume or none.")
    n_vis = min(v.numel() for v in vis_list)
    n_pred = min(p.numel() for p in pred_list)
    # truncate (drop trailing) so batch dimensions match
    vis_idx = torch.stack([v[:n_vis] for v in vis_list], dim=0).to(device)
    pred_idx = torch.stack([p[:n_pred] for p in pred_list], dim=0).to(device)
    return vis_idx, pred_idx


# Default specs from V-JEPA-2 vit_large pretrain-256px-16f.yaml
DEFAULT_SPECS = [
    {"num_blocks": 8, "aspect_ratio": (0.75, 1.5), "spatial_scale": (0.15, 0.15),
     "temporal_scale": (1.0, 1.0)},
    {"num_blocks": 2, "aspect_ratio": (0.75, 1.5), "spatial_scale": (0.7, 0.7),
     "temporal_scale": (1.0, 1.0)},
]
