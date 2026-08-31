"""Five self-supervised objectives over the SAME ViT encoder.

arch options:
  jea          naive joint embedding: pull two augmented views together.
               No stop-grad, no EMA, no predictor -> representation collapse.
  mae          generative baseline: mask 75% random patches, reconstruct pixels.
  genmask      generative with I-JEPA's masking: same context/target blocks as
               ijepa but the loss is in PIXEL space. Isolates "predict in
               representation space" from "block masking".
  ijepa        full I-JEPA: predict target-block representations from context,
               targets from an EMA target encoder (stop-grad), narrow predictor.
  ijepa_nostop ablation: targets from the ONLINE encoder with gradients
               (no EMA, no stop-grad). Shows JEPA inherits JEA's collapse
               without the guard.

Fidelity notes vs the official repo:
- target representations are layer-normed (F.layer_norm) before the loss —
  undocumented in the paper but load-bearing in the code
- loss is smooth L1 (the paper says L2; the code ships smooth L1)
- EMA momentum ramps linearly 0.996 -> 1.0 over training
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import ViTEncoder, Predictor, PixelDecoder, patchify
from .masks import MultiBlockMaskGenerator, random_mask


class SSLModel(nn.Module):
    def __init__(self, arch, img_size=96, patch_size=8, embed_dim=384,
                 depth=12, heads=6, pred_dim=192, pred_depth=6,
                 mae_mask_ratio=0.75):
        super().__init__()
        self.arch = arch
        self.patch_size = patch_size
        self.img_size = img_size
        self.mae_mask_ratio = mae_mask_ratio
        self.encoder = ViTEncoder(img_size, patch_size, 3, embed_dim, depth, heads)
        grid = self.encoder.grid
        n = self.encoder.num_patches
        self.mask_gen = MultiBlockMaskGenerator(grid)

        if arch in ("ijepa", "ijepa_nostop"):
            self.predictor = Predictor(n, embed_dim, pred_dim, pred_depth,
                                       heads=heads, grid=grid)
        if arch in ("mae", "genmask"):
            self.decoder = PixelDecoder(n, patch_size, embed_dim,
                                        dec_dim=pred_dim, depth=4, heads=heads,
                                        grid=grid)
        if arch == "ijepa":
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def ema_update(self, momentum):
        if self.arch != "ijepa":
            return
        for q, k in zip(self.encoder.parameters(),
                        self.target_encoder.parameters()):
            k.mul_(momentum).add_(q, alpha=1.0 - momentum)

    def forward(self, batch):
        if self.arch == "jea":
            return self._forward_jea(batch)
        if self.arch == "mae":
            return self._forward_mae(batch)
        if self.arch == "genmask":
            return self._forward_genmask(batch)
        return self._forward_ijepa(batch)

    # ---- naive joint embedding -------------------------------------------
    def _forward_jea(self, batch):
        v1, v2 = batch  # two augmented views
        z1 = self.encoder(v1).mean(dim=1)
        z2 = self.encoder(v2).mean(dim=1)
        # negative cosine similarity, symmetric, NO stop-grad anywhere
        loss = -0.5 * (F.cosine_similarity(z1, z2, dim=-1).mean()
                       + F.cosine_similarity(z2, z1, dim=-1).mean())
        return loss, {"pooled": z1.detach()}

    # ---- generative: MAE-style random masking ----------------------------
    def _forward_mae(self, batch):
        imgs = batch
        B = imgs.size(0)
        n = self.encoder.num_patches
        keep_idx, masked_idx = random_mask(B, n, self.mae_mask_ratio, imgs.device)
        vis = self.encoder(imgs, keep_idx=keep_idx)
        pred = self.decoder(vis, keep_idx)  # (B,N,P*P*3)
        target = patchify(imgs, self.patch_size)
        # norm-pix loss (MAE paper): normalize each patch by its own mean/var
        mu = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mu) / (var + 1e-6).sqrt()
        loss_all = (pred - target).pow(2).mean(dim=-1)  # (B,N)
        mask = torch.zeros(B, n, device=imgs.device)
        mask.scatter_(1, masked_idx, 1.0)
        loss = (loss_all * mask).sum() / mask.sum()
        with torch.no_grad():
            pooled = self.encoder(imgs).mean(dim=1)
        return loss, {"pooled": pooled, "pred_patches": pred.detach(),
                      "keep_idx": keep_idx}

    # ---- generative with I-JEPA masking ----------------------------------
    def _forward_genmask(self, batch):
        imgs = batch
        B = imgs.size(0)
        n = self.encoder.num_patches
        ctx_idx, tgt_list = self.mask_gen.collate(B)
        ctx_idx = ctx_idx.to(imgs.device)
        tgt_list = [t.to(imgs.device) for t in tgt_list]
        vis = self.encoder(imgs, keep_idx=ctx_idx)
        pred = self.decoder(vis, ctx_idx)  # (B,N,P*P*3)
        target = patchify(imgs, self.patch_size)
        mu = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mu) / (var + 1e-6).sqrt()
        loss_all = (pred - target).pow(2).mean(dim=-1)
        mask = torch.zeros(B, n, device=imgs.device)
        for t in tgt_list:
            mask.scatter_(1, t, 1.0)
        loss = (loss_all * mask).sum() / mask.sum()
        with torch.no_grad():
            pooled = self.encoder(imgs).mean(dim=1)
        return loss, {"pooled": pooled}

    # ---- I-JEPA -----------------------------------------------------------
    def _forward_ijepa(self, batch):
        imgs = batch
        B = imgs.size(0)
        ctx_idx, tgt_list = self.mask_gen.collate(B)
        ctx_idx = ctx_idx.to(imgs.device)
        tgt_list = [t.to(imgs.device) for t in tgt_list]

        if self.arch == "ijepa":
            with torch.no_grad():
                h = self.target_encoder(imgs)              # full image
                h = F.layer_norm(h, (h.size(-1),))          # the undocumented LN
        else:  # ijepa_nostop: gradients flow into the target branch
            h = self.encoder(imgs)
            h = F.layer_norm(h, (h.size(-1),))

        ctx = self.encoder(imgs, keep_idx=ctx_idx)
        loss = 0.0
        for t_idx in tgt_list:
            pred = self.predictor(ctx, ctx_idx, t_idx)      # (B,M,D)
            tgt = torch.gather(
                h, 1, t_idx[..., None].expand(-1, -1, h.size(-1)))
            loss = loss + F.smooth_l1_loss(pred, tgt)
        loss = loss / len(tgt_list)
        with torch.no_grad():
            pooled = self.encoder(imgs).mean(dim=1)
        return loss, {"pooled": pooled}


@torch.no_grad()
def collapse_metrics(pooled: torch.Tensor):
    """Diagnostics on pooled (B,D) features.
    feat_std: mean per-dim std of l2-normalized features (SimSiam's collapse
              gauge — 1/sqrt(D) is healthy, ~0 is collapse).
    eff_rank: entropy-based effective rank of the covariance spectrum.
    """
    z = F.normalize(pooled.float(), dim=-1)
    feat_std = z.std(dim=0).mean().item()
    zc = pooled.float() - pooled.float().mean(0, keepdim=True)
    try:
        s = torch.linalg.svdvals(zc)
        p = (s ** 2) / (s ** 2).sum().clamp_min(1e-12)
        eff_rank = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
    except Exception:
        eff_rank = float("nan")
    return feat_std, eff_rank
