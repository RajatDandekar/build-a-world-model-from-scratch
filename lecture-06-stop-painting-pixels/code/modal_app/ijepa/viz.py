"""Dump raw arrays for the deck's figures: multi-block mask examples on real
images, and MAE reconstructions from a trained checkpoint. Figures themselves
are drawn locally; this only exports data.
"""
import os
import random
import numpy as np
import torch

from .masks import MultiBlockMaskGenerator, random_mask
from .vit import ViTEncoder, PixelDecoder, patchify, unpatchify
from .data import get_probe_datasets, STL_MEAN, STL_STD, IN_MEAN, IN_STD


def _denorm(x, mean, std):
    m = torch.tensor(mean)[:, None, None]
    s = torch.tensor(std)[:, None, None]
    return (x * s + m).clamp(0, 1)


def dump_mask_examples(data_root="/data", out_root="/results",
                       n_images=8, seed=7):
    """For n images: the image + context mask + 4 target masks (patch level)."""
    out = {}
    for name, (mean, std, img_size, patch) in {
        "stl10": (STL_MEAN, STL_STD, 96, 8),
        "in100": (IN_MEAN, IN_STD, 224, 16),
    }.items():
        try:
            _, te = get_probe_datasets(name, data_root)
        except Exception as e:
            print(f"skip {name}: {e}")
            continue
        rng = random.Random(seed)
        gen = MultiBlockMaskGenerator(img_size // patch, rng=rng)
        idxs = rng.sample(range(len(te)), n_images)
        imgs, ctxs, tgts = [], [], []
        for i in idxs:
            x, _ = te[i]
            ctx, tg = gen.sample()
            imgs.append(_denorm(x, mean, std).numpy())
            g = img_size // patch
            cm = np.zeros(g * g, dtype=bool)
            cm[list(ctx)] = True
            tm = np.zeros((4, g * g), dtype=bool)
            for k, t in enumerate(tg):
                tm[k, list(t)] = True
            ctxs.append(cm)
            tgts.append(tm)
        out[name] = dict(imgs=np.stack(imgs), ctx=np.stack(ctxs),
                         tgt=np.stack(tgts), grid=img_size // patch,
                         patch=patch)
    os.makedirs(os.path.join(out_root, "viz"), exist_ok=True)
    for name, d in out.items():
        np.savez_compressed(
            os.path.join(out_root, "viz", f"masks_{name}.npz"), **d)
    return list(out.keys())


@torch.no_grad()
def dump_mae_recon(run_name="stl10_mae_s0", data_root="/data",
                   out_root="/results", n_images=8, seed=7):
    """MAE reconstructions: original, masked input, reconstruction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(f"/results/{run_name}/ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    meta = ck["meta"]
    img_size, patch = meta["img_size"], meta["patch_size"]
    enc = ViTEncoder(img_size, patch).to(device).eval()
    enc.load_state_dict(ck["encoder"])
    # decoder weights were not saved with the encoder checkpoint; retrain-free
    # reconstructions need the full model checkpoint — fall back to full state
    full = ck.get("full_state")
    dec = PixelDecoder(enc.num_patches, patch, enc.embed_dim,
                       dec_dim=192, depth=4, heads=6,
                       grid=enc.grid).to(device).eval()
    if full is not None:
        dec.load_state_dict({k[len("decoder."):]: v for k, v in full.items()
                             if k.startswith("decoder.")})
        enc.load_state_dict({k[len("encoder."):]: v for k, v in full.items()
                             if k.startswith("encoder.")})
    _, te = get_probe_datasets("stl10", data_root)
    rng = random.Random(seed)
    idxs = rng.sample(range(len(te)), n_images)
    x = torch.stack([te[i][0] for i in idxs]).to(device)
    keep, masked = random_mask(x.size(0), enc.num_patches, 0.75, device)
    vis = enc(x, keep_idx=keep)
    pred = dec(vis, keep)  # (B,N,P*P*3), normalized-pixel space
    target = patchify(x, patch)
    mu = target.mean(-1, keepdim=True)
    sd = (target.var(-1, keepdim=True) + 1e-6).sqrt()
    pred_px = pred * sd + mu  # un-normalize with true patch stats (viz only)
    # paste visible patches from the original
    keepmask = torch.zeros(x.size(0), enc.num_patches, 1, device=device)
    keepmask.scatter_(1, keep[..., None], 1.0)
    recon = pred_px * (1 - keepmask) + target * keepmask
    masked_img = target * keepmask + 0.0 * (1 - keepmask)
    d = dict(
        orig=_denorm(x.cpu(), STL_MEAN, STL_STD).numpy(),
        recon=_denorm(unpatchify(recon.cpu(), patch, img_size),
                      STL_MEAN, STL_STD).numpy(),
        masked=_denorm(unpatchify(masked_img.cpu(), patch, img_size),
                       STL_MEAN, STL_STD).numpy(),
        keepfrac=float(keepmask.mean().item()),
    )
    os.makedirs(os.path.join(out_root, "viz"), exist_ok=True)
    np.savez_compressed(os.path.join(out_root, "viz", "mae_recon.npz"), **d)
    return "ok"
