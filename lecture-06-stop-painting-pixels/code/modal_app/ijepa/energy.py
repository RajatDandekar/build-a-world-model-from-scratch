"""Energy-landscape measurements on the frozen E1 checkpoints.

For each architecture we define the energy E(x, y) its own training loss
implies — the score it assigns to "target y goes with context x":

  ijepa / ijepa_nostop : smooth-L1 between predictor(context, target-pos)
                         and target-encoder tokens at that position
                         (per-sample mean) — representation space
  jea                  : 1 - cos( enc(x-view), enc(y-view) ) — the training
                         objective, with y as the second "view"
  mae / genmask        : mean squared error on the HIDDEN patches of the
                         decoder's reconstruction vs y — pixel space

Three measurements (the "first 3 points"):
  1. corruption curves — E as the true target is progressively corrupted
     (noise ramp, patch shuffle, impostor target, wrong position)
  2. 2D energy contours — E on a grid spanned by two directions in target
     space (toward an impostor, and a noise direction)
  3. energy-gap AUC — can E rank the true target above impostors?

Everything is dumped as raw npz/json to /results/energy/ for local plotting.
"""
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

from .vit import ViTEncoder, Predictor, PixelDecoder, patchify
from .masks import MultiBlockMaskGenerator
from .models import SSLModel
from .data import get_probe_datasets

ARCHS = ["jea", "mae", "genmask", "ijepa", "ijepa_nostop"]


def load_model(arch, results_root="/results", device="cuda"):
    ck = torch.load(f"{results_root}/stl10_{arch}_s0/ckpt_final.pt",
                    map_location="cpu", weights_only=False)
    meta = ck["meta"]
    m = SSLModel(arch, img_size=meta["img_size"], patch_size=meta["patch_size"])
    m.load_state_dict(ck["full_state"])
    return m.to(device).eval()


@torch.no_grad()
def energy(model, x, y, ctx_idx, tgt_idx):
    """Per-sample energy for a batch. x,y (B,3,H,W)."""
    arch = model.arch
    if arch in ("ijepa", "ijepa_nostop"):
        enc_t = model.target_encoder if arch == "ijepa" else model.encoder
        h = enc_t(y)
        h = F.layer_norm(h, (h.size(-1),))
        tgt = torch.gather(h, 1, tgt_idx[..., None].expand(-1, -1, h.size(-1)))
        ctx = model.encoder(x, keep_idx=ctx_idx)
        pred = model.predictor(ctx, ctx_idx, tgt_idx)
        return F.smooth_l1_loss(pred, tgt, reduction="none").mean(dim=(1, 2))
    if arch == "jea":
        zx = model.encoder(x).mean(dim=1)
        zy = model.encoder(y).mean(dim=1)
        return 1.0 - F.cosine_similarity(zx, zy, dim=-1)
    # mae / genmask: reconstruct hidden patches of y FROM CONTEXT OF x.
    # For a compatibility energy we give the decoder x's visible patches and
    # score against y's pixels on the hidden set.
    vis = model.encoder(x, keep_idx=ctx_idx)
    px = model.decoder(vis, ctx_idx)                     # (B,N,P*P*3)
    target = patchify(y, model.patch_size)
    mu = target.mean(-1, keepdim=True)
    sd = (target.var(-1, keepdim=True) + 1e-6).sqrt()
    target = (target - mu) / sd
    err = (px - target).pow(2).mean(-1)                  # (B,N)
    B, N = err.shape
    hidden = torch.ones(B, N, device=err.device)
    hidden.scatter_(1, ctx_idx, 0.0)
    return (err * hidden).sum(1) / hidden.sum(1)


def shuffle_patches(y, patch, g, rng):
    B = y.size(0)
    p = patchify(y, patch)
    idx = torch.stack([torch.randperm(g * g, generator=rng) for _ in range(B)])
    p = torch.gather(p, 1, idx.to(y.device)[..., None].expand(-1, -1, p.size(-1)))
    from .vit import unpatchify
    return unpatchify(p, patch, y.size(-1))


def run_energy_study(results_root="/results", data_root="/data",
                     n_images=256, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(results_root, "energy")
    os.makedirs(out_dir, exist_ok=True)
    rng = torch.Generator().manual_seed(seed)

    _, te = get_probe_datasets("stl10", data_root)
    pick = torch.randperm(len(te), generator=rng)[:n_images * 2]
    xs = torch.stack([te[i][0] for i in pick[:n_images]]).to(device)
    impostors = torch.stack([te[i][0] for i in pick[n_images:]]).to(device)

    # one fixed context/target mask family for all models & corruptions
    gen = MultiBlockMaskGenerator(grid=12, rng=__import__("random").Random(seed))
    ctx_idx, tgt_list = gen.collate(n_images)
    ctx_idx = ctx_idx.to(device)
    tgt_idx = tgt_list[0].to(device)
    wrong_tgt = tgt_list[1].to(device)   # a different block = wrong position

    noise_levels = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    results = {}
    for arch in ARCHS:
        model = load_model(arch, results_root, device)
        r = {}
        # 1) corruption curves
        curve = []
        for s in noise_levels:
            yc = xs + s * torch.randn_like(xs)
            curve.append(energy(model, xs, yc, ctx_idx, tgt_idx).cpu().numpy())
        r["noise_levels"] = noise_levels
        r["noise_curve"] = np.stack(curve)                    # (L, B)
        r["e_true"] = curve[0]
        r["e_shuffle"] = energy(model, xs,
                                shuffle_patches(xs, 8, 12, rng).to(device),
                                ctx_idx, tgt_idx).cpu().numpy()
        r["e_impostor"] = energy(model, xs, impostors,
                                 ctx_idx, tgt_idx).cpu().numpy()
        r["e_wrongpos"] = energy(model, xs, xs, ctx_idx,
                                 wrong_tgt).cpu().numpy()
        # 3) impostor AUC: true target should have LOWER energy
        e_t, e_i = r["e_true"], r["e_impostor"]
        pairs = (e_t[:, None] < e_i[None, :]).mean()
        ties = (e_t[:, None] == e_i[None, :]).mean()
        r["auc"] = float(pairs + 0.5 * ties)

        # 2) 2D contour: y(a,b) = normalize(y + a*(impostor-y) + b*noise)
        k = 24
        aa = np.linspace(-0.25, 1.25, k)
        bb = np.linspace(-1.0, 1.0, k)
        sub = 32
        x_s, y_s = xs[:sub], xs[:sub]
        imp_s = impostors[:sub]
        nz = torch.randn_like(y_s)
        nz = nz / nz.flatten(1).norm(dim=1)[:, None, None, None] \
            * y_s.flatten(1).norm(dim=1).mean()
        grid = np.zeros((k, k))
        for i, a in enumerate(aa):
            for j, b in enumerate(bb):
                yy = y_s + float(a) * (imp_s - y_s) + float(b) * nz
                grid[i, j] = float(energy(model, x_s, yy, ctx_idx[:sub],
                                          tgt_idx[:sub]).mean())
        r["contour_grid"] = grid
        r["contour_a"] = aa
        r["contour_b"] = bb

        np.savez_compressed(os.path.join(out_dir, f"energy_{arch}.npz"),
                            **{k2: v for k2, v in r.items()})
        results[arch] = {"auc": r["auc"],
                         "e_true_mean": float(np.mean(r["e_true"])),
                         "e_impostor_mean": float(np.mean(r["e_impostor"]))}
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results
