"""CPU unit tests for the core model code (no torchvision needed).
Run: python tests/test_core.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modal_app"))

import torch
from ijepa.vit import (ViTEncoder, Predictor, PixelDecoder, patchify,
                       unpatchify, build_2d_sincos_pos_embed)
from ijepa.masks import MultiBlockMaskGenerator, random_mask
from ijepa.models import SSLModel, collapse_metrics

PASS = 0


def ok(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1
    print(f"  ok: {name}")


# ---- masking ---------------------------------------------------------------
g = MultiBlockMaskGenerator(grid=12)
for _ in range(50):
    ctx, tgts = g.sample()
    union = set().union(*[set(t) for t in tgts])
    ok_ctx_disjoint = len(set(ctx) & union) == 0
    assert ok_ctx_disjoint
    assert len(tgts) == 4
    for t in tgts:
        assert 0.10 * 144 <= len(t) <= 0.30 * 144, len(t)  # ~15-20% + rounding
    assert len(ctx) >= 4
ok("multiblock: context disjoint from targets, sizes sane", True)

ctx_b, tgt_b = g.collate(8)
ok("multiblock collate shapes", ctx_b.shape[0] == 8 and len(tgt_b) == 4
   and all(t.shape[0] == 8 for t in tgt_b))
ok("collate indices in range", int(ctx_b.max()) < 144 and int(ctx_b.min()) >= 0)

keep, masked = random_mask(4, 144, 0.75, "cpu")
ok("random_mask 75%: keeps 36", keep.shape == (4, 36) and masked.shape == (4, 108))
all_idx = torch.cat([keep, masked], 1).sort(1).values
ok("random_mask partitions all patches",
   bool((all_idx == torch.arange(144)[None].expand(4, -1)).all()))

# ---- patchify --------------------------------------------------------------
imgs = torch.randn(2, 3, 48, 48)
p = patchify(imgs, 8)
ok("patchify shape", p.shape == (2, 36, 192))
ok("patchify roundtrip", torch.allclose(unpatchify(p, 8, 48), imgs, atol=1e-6))

pe = build_2d_sincos_pos_embed(64, 6)
ok("pos embed shape", pe.shape == (1, 36, 64))

# ---- encoder / predictor / decoder ----------------------------------------
enc = ViTEncoder(img_size=48, patch_size=8, embed_dim=64, depth=2, heads=4)
full = enc(imgs)
ok("encoder full shape", full.shape == (2, 36, 64))
keep_idx = torch.tensor([[0, 5, 7, 11, 20], [1, 2, 3, 30, 35]])
sub = enc(imgs, keep_idx=keep_idx)
ok("encoder subset shape", sub.shape == (2, 5, 64))

pred = Predictor(36, encoder_dim=64, pred_dim=32, depth=2, heads=4, grid=6)
tgt_idx = torch.tensor([[12, 13, 14], [21, 22, 23]])
out = pred(sub, keep_idx, tgt_idx)
ok("predictor shape", out.shape == (2, 3, 64))

dec = PixelDecoder(36, 8, encoder_dim=64, dec_dim=32, depth=2, heads=4, grid=6)
px = dec(sub, keep_idx)
ok("decoder shape", px.shape == (2, 36, 192))

# ---- objectives: forward + backward, gradient routing ----------------------
kw = dict(img_size=48, patch_size=8, embed_dim=64, depth=2, heads=4,
          pred_dim=32, pred_depth=2)
x = torch.randn(4, 3, 48, 48)

for arch in ["jea", "mae", "genmask", "ijepa", "ijepa_nostop"]:
    torch.manual_seed(0)
    m = SSLModel(arch, **kw)
    batch = (torch.randn(4, 3, 48, 48), torch.randn(4, 3, 48, 48)) \
        if arch == "jea" else x
    loss, aux = m(batch)
    loss.backward()
    ok(f"{arch}: finite loss", bool(torch.isfinite(loss)))
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in m.encoder.parameters())
    ok(f"{arch}: encoder receives gradient", has_grad)
    if arch == "ijepa":
        tg = [p.grad for p in m.target_encoder.parameters()
              if p.grad is not None]
        ok("ijepa: target encoder gets NO gradient", len(tg) == 0)
        before = [p.clone() for p in m.target_encoder.parameters()]
        with torch.no_grad():
            for p in m.encoder.parameters():
                p.add_(1.0)
        m.ema_update(0.5)
        moved = any(not torch.allclose(a, b) for a, b in
                    zip(before, m.target_encoder.parameters()))
        ok("ijepa: EMA update moves target encoder", moved)

# ---- collapse metrics ------------------------------------------------------
healthy = torch.randn(256, 64)
fs_h, er_h = collapse_metrics(healthy)
# constant collapse: every sample maps to the same vector -> feat_std ~ 0
constant = torch.ones(256, 64) + 1e-6 * torch.randn(256, 64)
fs_c, _ = collapse_metrics(constant)
ok("feat_std detects constant collapse", fs_h > 10 * fs_c)
# dimensional collapse: variance lives in only 3 directions -> eff_rank ~ 3
basis = torch.randn(3, 64)
lowrank = torch.randn(256, 3) @ basis + 1e-4 * torch.randn(256, 64)
_, er_l = collapse_metrics(lowrank)
ok("eff_rank detects dimensional collapse", er_h > 40 and er_l < 6)
ok("healthy feat_std near 1/sqrt(D)", abs(fs_h - 64 ** -0.5) < 0.05)

print(f"\nALL {PASS} CHECKS PASSED")
