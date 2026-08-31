"""Data dumps for the Lecture-6 revision pass.

1. fingerprints: 8 fixed STL-10 test images encoded by all five E1 encoders
   (pooled embeddings) -> the per-run "embedding fingerprint" figures.
2. montage: training images of ONE ImageNet-100 class + its count -> the
   "what does 1% of labels mean" figure.
3. neighbors: specific ImageNet-100 validation images by index -> the
   nearest-neighbor retrieval figure (indices computed locally from feats).
"""
import json
import os
import numpy as np
import torch

from .vit import ViTEncoder
from .data import get_probe_datasets, STL_MEAN, STL_STD, IN_MEAN, IN_STD

FP_INDICES = [3, 801, 1602, 2403, 3204, 4005, 4806, 5607]
ARCHS = ["jea", "mae", "genmask", "ijepa", "ijepa_nostop"]


def _denorm(x, mean, std):
    m = torch.tensor(mean)[:, None, None]
    s = torch.tensor(std)[:, None, None]
    return (x * s + m).clamp(0, 1)


def run(val_indices, results_root="/results", data_root="/data"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(results_root, "review")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. fingerprints
    _, te = get_probe_datasets("stl10", data_root)
    xs = torch.stack([te[i][0] for i in FP_INDICES])
    ys = [int(te[i][1]) for i in FP_INDICES]
    embs = {}
    for arch in ARCHS:
        ck = torch.load(f"{results_root}/stl10_{arch}_s0/ckpt_final.pt",
                        map_location="cpu", weights_only=False)
        enc = ViTEncoder(ck["meta"]["img_size"], ck["meta"]["patch_size"])
        enc.load_state_dict(ck["encoder"])
        enc = enc.to(device).eval()
        with torch.no_grad():
            embs[arch] = enc(xs.to(device)).mean(dim=1).cpu().numpy()
        del enc
    np.savez_compressed(
        os.path.join(out_dir, "fingerprints.npz"),
        imgs=_denorm(xs, STL_MEAN, STL_STD).numpy(),
        labels=np.array(ys),
        **{f"emb_{a}": v for a, v in embs.items()})

    # ---- 2. montage of one class (train split)
    from datasets import load_dataset
    from torchvision import transforms
    cache = os.path.join(data_root, "hf")
    tr = load_dataset("clane9/imagenet-100", split="train", cache_dir=cache)
    labels = np.array(tr["label"])
    cls = 23  # arbitrary fixed class
    idx = np.where(labels == cls)[0]
    t = transforms.Compose([transforms.Resize(72), transforms.CenterCrop(64)])
    thumbs = np.stack([
        np.asarray(t(tr[int(i)]["image"].convert("RGB")), dtype=np.uint8)
        for i in idx[:60]])
    np.savez_compressed(
        os.path.join(out_dir, "montage.npz"),
        thumbs=thumbs, class_count=len(idx),
        class_name=str(tr.features["label"].int2str(cls)))

    # ---- 3. neighbor val images
    va = load_dataset("clane9/imagenet-100", split="validation",
                      cache_dir=cache)
    t2 = transforms.Compose([transforms.Resize(112),
                             transforms.CenterCrop(96)])
    imgs, keys = [], []
    for i in val_indices:
        imgs.append(np.asarray(t2(va[int(i)]["image"].convert("RGB")),
                               dtype=np.uint8))
        keys.append(int(i))
    np.savez_compressed(
        os.path.join(out_dir, "neighbor_imgs.npz"),
        imgs=np.stack(imgs), indices=np.array(keys),
        labels=np.array([int(va[int(i)]["label"]) for i in val_indices]),
        names=np.array([str(va.features["label"].int2str(int(va[int(i)]["label"])))
                        for i in val_indices]))
    return {"fingerprints": len(FP_INDICES), "montage": int(len(idx)),
            "neighbors": len(val_indices)}
