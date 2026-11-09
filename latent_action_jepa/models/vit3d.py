"""V-JEPA-2-style ViT with 3D tubelet patch embedding and 3D sin-cos pos embed.

Reproduces facebookresearch/vjepa2 src/models/vision_transformer.py at ViT-S scale:
    embed_dim=384, depth=12, num_heads=6, mlp_ratio=4
plus tubelet_size=2, patch_size=16. Operates on clips [B, 3, T, H, W].

Optional masking: forward(x, mask_indices=...) keeps only specified token indices
(per V-JEPA-2 `apply_masks`).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3D sin-cos position embedding (matches V-JEPA-2 non-uniform_power default).
# ---------------------------------------------------------------------------

def _get_1d_sincos(dim, pos):
    """1D sincos: pos [N] -> [N, dim]."""
    assert dim % 2 == 0
    half = dim // 2
    omega = torch.arange(half, dtype=torch.float64) / half
    omega = 1.0 / (10000 ** omega)
    pos = pos.to(torch.float64)
    args = pos[:, None] * omega[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1).float()


def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth):
    """Returns [T*H*W, embed_dim]. Splits dims as D/2 temporal, D/4 height, rest width."""
    H = W = grid_size
    T = grid_depth
    dim_d = (embed_dim // 4) * 2          # ensure even
    dim_h = ((embed_dim - dim_d) // 4) * 2
    dim_w = embed_dim - dim_d - dim_h     # rest
    if dim_w % 2 != 0:
        dim_w -= 1
        dim_d += 1
    pos_d = torch.arange(T)
    pos_h = torch.arange(H)
    pos_w = torch.arange(W)
    emb_d = _get_1d_sincos(dim_d, pos_d)   # [T, dim_d]
    emb_h = _get_1d_sincos(dim_h, pos_h)   # [H, dim_h]
    emb_w = _get_1d_sincos(dim_w, pos_w)   # [W, dim_w]
    out = torch.zeros(T, H, W, embed_dim)
    out[..., :dim_d] = emb_d.view(T, 1, 1, dim_d)
    out[..., dim_d:dim_d + dim_h] = emb_h.view(1, H, 1, dim_h)
    out[..., dim_d + dim_h:] = emb_w.view(1, 1, W, dim_w)
    return out.reshape(T * H * W, embed_dim)


# ---------------------------------------------------------------------------
# Tubelet patch embedding (Conv3D non-overlapping)
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=3, embed_dim=384, tubelet_size=2, patch_size=16):
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        # x: [B, 3, T, H, W]
        x = self.proj(x)                       # [B, D, T/tube, H/patch, W/patch]
        return x.flatten(2).transpose(1, 2)    # [B, N, D]


# ---------------------------------------------------------------------------
# Transformer blocks (standard pre-norm)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        h = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x


# ---------------------------------------------------------------------------
# 3D Vision Transformer (encoder)
# ---------------------------------------------------------------------------

class VisionTransformer3D(nn.Module):
    def __init__(self, num_frames=16, tubelet_size=2, patch_size=16, img_size=224, in_chans=3,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_depth = num_frames // tubelet_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.n_tokens = self.grid_depth * self.grid_h * self.grid_w
        self.patch_embed = PatchEmbed3D(in_chans, embed_dim, tubelet_size, patch_size)
        pos_embed = get_3d_sincos_pos_embed(embed_dim, self.grid_h, self.grid_depth)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, keep_indices=None):
        """
        x: [B, 3, T, H, W]
        keep_indices: optional [B, N_keep] long — keep only these patch positions.
        Returns: [B, N_keep_or_N, embed_dim].
        """
        x = self.patch_embed(x)                        # [B, N, D]
        x = x + self.pos_embed                          # add 3D pos
        if keep_indices is not None:
            B, _, D = x.shape
            idx = keep_indices.unsqueeze(-1).expand(-1, -1, D)
            x = x.gather(1, idx)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
