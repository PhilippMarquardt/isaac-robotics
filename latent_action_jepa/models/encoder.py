import torch
import torch.nn as nn
import timm


class ViTEncoder(nn.Module):
    """Wrap timm ViT. Returns patch tokens [B,N,D] from a chosen block (1-indexed).

    layer_idx=None -> final layer (post norm).
    layer_idx=k    -> output of vit.blocks[k-1], i.e., after k transformer blocks (no final norm).
    """

    def __init__(self, name="vit_base_patch16_224", pretrained=True, layer_idx=None):
        super().__init__()
        self.vit = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.vit.embed_dim
        self.layer_idx = layer_idx
        self.n_blocks = len(self.vit.blocks)

    def _patch_embed_with_extras(self, x):
        v = self.vit
        x = v.patch_embed(x)
        # timm ViT adds cls + pos via _pos_embed
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
        # drop cls if present
        if hasattr(self.vit, "cls_token") and self.vit.cls_token is not None:
            x = x[:, 1:, :]
        return x
