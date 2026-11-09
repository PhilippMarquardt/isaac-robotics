"""V-JEPA-2 predictor module.

Project encoder tokens into predictor space, add 3D sin-cos pos embed at
visible positions, append learned mask tokens at target positions (also with
pos embed), run a stack of standard transformer blocks, project back to
encoder embed_dim. Outputs predictions only at the target positions.
"""
import torch
import torch.nn as nn

from .vit3d import Block, get_3d_sincos_pos_embed


class VJEPAPredictor(nn.Module):
    def __init__(self, embed_dim=384, pred_embed_dim=384, pred_depth=12, pred_num_heads=12,
                 grid_depth=8, grid_size=14, mlp_ratio=4.0, num_mask_tokens=1, drop=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.pred_embed_dim = pred_embed_dim
        self.grid_depth = grid_depth
        self.grid_size = grid_size
        self.n_tokens = grid_depth * grid_size * grid_size
        self.embed = nn.Linear(embed_dim, pred_embed_dim)
        # one learnable mask token per mask spec (default 1)
        self.mask_tokens = nn.Parameter(torch.randn(num_mask_tokens, pred_embed_dim) * 0.02)
        pos_embed = get_3d_sincos_pos_embed(pred_embed_dim, grid_size, grid_depth)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList([Block(pred_embed_dim, pred_num_heads, mlp_ratio, drop)
                                     for _ in range(pred_depth)])
        self.norm = nn.LayerNorm(pred_embed_dim)
        self.proj = nn.Linear(pred_embed_dim, embed_dim)

    def forward(self, ctx_tokens, ctx_indices, pred_indices, mask_token_id=0):
        """
        ctx_tokens:   [B, N_ctx, embed_dim]  — encoder output at visible positions
        ctx_indices:  [B, N_ctx]             — patch ids of visible positions
        pred_indices: [B, N_pred]            — patch ids to predict
        mask_token_id: which learnable mask token to use (0..num_mask_tokens-1)
        Returns: [B, N_pred, embed_dim]
        """
        B, N_ctx, _ = ctx_tokens.shape
        D = self.pred_embed_dim
        # project context to predictor space, add pos
        x = self.embed(ctx_tokens)
        ctx_pos = self.pos_embed.expand(B, -1, -1).gather(
            1, ctx_indices.unsqueeze(-1).expand(-1, -1, D))
        x = x + ctx_pos
        # build target queries: mask token + pos
        N_pred = pred_indices.shape[1]
        q = self.mask_tokens[mask_token_id].view(1, 1, D).expand(B, N_pred, D).contiguous()
        q_pos = self.pos_embed.expand(B, -1, -1).gather(
            1, pred_indices.unsqueeze(-1).expand(-1, -1, D))
        q = q + q_pos
        # concat (ctx, queries) and self-attend
        seq = torch.cat([x, q], dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        out = seq[:, N_ctx:]                            # take target slot outputs
        return self.proj(out)
