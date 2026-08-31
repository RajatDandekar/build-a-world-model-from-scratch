"""Minimal ViT encoder + I-JEPA predictor, faithful to facebookresearch/ijepa.

Key fidelity points (verified against the official repo before it was archived):
- 2D sin-cos positional embeddings, no cls token (I-JEPA average-pools patch tokens)
- encoder can run on a SUBSET of patches (context tokens only), like MAE
- predictor is a narrow ViT: projects encoder dim down, adds mask tokens carrying
  the positional embedding of each TARGET location, predicts per target block
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_sincos_pos_embed(embed_dim: int, grid: int) -> torch.Tensor:
    """(1, grid*grid, embed_dim) fixed sin-cos position embedding."""
    assert embed_dim % 4 == 0
    coords = torch.arange(grid, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)
    omega = 1.0 / (10000 ** omega)
    out = []
    for pos in (yy, xx):
        p = pos.reshape(-1)[:, None] * omega[None, :]
        out.extend([torch.sin(p), torch.cos(p)])
    return torch.cat(out, dim=1)[None]  # (1, N, D)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """ViT over patch tokens; optionally only a visible subset of patches."""

    def __init__(self, img_size=96, patch_size=8, in_chans=3,
                 embed_dim=384, depth=12, heads=6):
        super().__init__()
        self.grid = img_size // patch_size
        self.num_patches = self.grid ** 2
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.register_buffer(
            "pos_embed", build_2d_sincos_pos_embed(embed_dim, self.grid),
            persistent=False)
        self.blocks = nn.ModuleList([Block(embed_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, imgs, keep_idx=None):
        """imgs (B,3,H,W); keep_idx (B,K) long — indices of visible patches.
        Returns (B, N or K, D) patch tokens."""
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)  # (B,N,D)
        x = x + self.pos_embed
        if keep_idx is not None:
            x = torch.gather(
                x, 1, keep_idx[..., None].expand(-1, -1, x.size(-1)))
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class Predictor(nn.Module):
    """Narrow ViT predictor: context tokens + mask tokens at target positions."""

    def __init__(self, num_patches, encoder_dim=384, pred_dim=192,
                 depth=6, heads=6, grid=12):
        super().__init__()
        self.input_proj = nn.Linear(encoder_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer(
            "pos_embed", build_2d_sincos_pos_embed(pred_dim, grid),
            persistent=False)
        self.blocks = nn.ModuleList([Block(pred_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)
        self.output_proj = nn.Linear(pred_dim, encoder_dim)

    def forward(self, ctx_tokens, ctx_idx, tgt_idx):
        """ctx_tokens (B,K,De); ctx_idx (B,K); tgt_idx (B,M) — one target block.
        Returns (B,M,De) predictions at the target positions."""
        B = ctx_tokens.size(0)
        x = self.input_proj(ctx_tokens)
        pos = self.pos_embed.expand(B, -1, -1)
        x = x + torch.gather(pos, 1, ctx_idx[..., None].expand(-1, -1, x.size(-1)))
        m = self.mask_token.expand(B, tgt_idx.size(1), -1)
        m = m + torch.gather(pos, 1, tgt_idx[..., None].expand(-1, -1, m.size(-1)))
        n_ctx = x.size(1)
        x = torch.cat([x, m], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, n_ctx:])
        return self.output_proj(x)


class PixelDecoder(nn.Module):
    """MAE-style decoder: full token grid (visible tokens + mask tokens) -> pixels."""

    def __init__(self, num_patches, patch_size, encoder_dim=384,
                 dec_dim=192, depth=4, heads=6, grid=12, in_chans=3):
        super().__init__()
        self.input_proj = nn.Linear(encoder_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer(
            "pos_embed", build_2d_sincos_pos_embed(dec_dim, grid),
            persistent=False)
        self.blocks = nn.ModuleList([Block(dec_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dec_dim)
        self.head = nn.Linear(dec_dim, patch_size * patch_size * in_chans)
        self.num_patches = num_patches

    def forward(self, vis_tokens, keep_idx):
        """vis_tokens (B,K,De); keep_idx (B,K). Returns (B,N,P*P*3) pixel preds."""
        B, K, _ = vis_tokens.shape
        x = self.input_proj(vis_tokens)
        full = self.mask_token.expand(B, self.num_patches, -1).clone().to(x.dtype)
        full = full.scatter(1, keep_idx[..., None].expand(-1, -1, x.size(-1)), x)
        full = full + self.pos_embed
        for blk in self.blocks:
            full = blk(full)
        return self.head(self.norm(full))


def patchify(imgs, patch_size):
    """(B,3,H,W) -> (B,N,P*P*3)"""
    B, C, H, W = imgs.shape
    g = H // patch_size
    x = imgs.reshape(B, C, g, patch_size, g, patch_size)
    x = torch.einsum("bcgphq->bghpqc", x)
    return x.reshape(B, g * g, patch_size * patch_size * C)


def unpatchify(patches, patch_size, img_size):
    B, N, _ = patches.shape
    g = img_size // patch_size
    x = patches.reshape(B, g, g, patch_size, patch_size, 3)
    x = torch.einsum("bghpqc->bcgphq", x)
    return x.reshape(B, 3, img_size, img_size)
