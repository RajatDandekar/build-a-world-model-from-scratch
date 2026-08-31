"""E2: linear probes on ImageNet-100 with OFFICIAL checkpoints.

Models:
  facebook/ijepa_vith14_1k   the paper's ViT-H/14 pretrained on IN-1k
  facebook/vit-mae-huge      generative architecture at matched scale
  random_vith                same ViT-H/14 architecture, random init
  google/vit-base-patch16-224  supervised-with-labels reference (smaller model,
                             for the "what do labels buy you" discussion)

All probed identically: average-pooled last-layer patch tokens, frozen.
"""
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .probe import linear_probe, knn_probe, lowshot_indices


class HFProcDataset(Dataset):
    def __init__(self, hf_split, processor):
        self.ds = hf_split
        self.proc = processor

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex["image"].convert("RGB")
        px = self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        return px, ex["label"]


def load_model(model_key, device):
    from transformers import AutoModel, AutoConfig, AutoImageProcessor
    if model_key == "ijepa_vith14":
        mid = "facebook/ijepa_vith14_1k"
        model = AutoModel.from_pretrained(mid, torch_dtype=torch.float16)
        proc = AutoImageProcessor.from_pretrained(mid)
        pool = "mean_all"
    elif model_key == "mae_vith":
        mid = "facebook/vit-mae-huge"
        cfg = AutoConfig.from_pretrained(mid)
        cfg.mask_ratio = 0.0  # CRITICAL: ViTMAE masks 75% by default, even in eval
        model = AutoModel.from_pretrained(mid, config=cfg,
                                          torch_dtype=torch.float16)
        proc = AutoImageProcessor.from_pretrained(mid)
        pool = "mean_patches"  # drop cls token
    elif model_key == "random_vith":
        cfg = AutoConfig.from_pretrained("facebook/ijepa_vith14_1k")
        torch.manual_seed(1234)
        model = AutoModel.from_config(cfg)
        model = model.to(torch.float16)
        proc = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k")
        pool = "mean_all"
    elif model_key == "supervised_vitb":
        mid = "google/vit-base-patch16-224"
        model = AutoModel.from_pretrained(mid, torch_dtype=torch.float16)
        proc = AutoImageProcessor.from_pretrained(mid)
        pool = "mean_patches"
    else:
        raise ValueError(model_key)
    return model.to(device).eval(), proc, pool


@torch.no_grad()
def extract(model, pool, dataset, device, batch_size=128, num_workers=16):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    feats, labels = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True).to(torch.float16)
        out = model(pixel_values=x)
        h = out.last_hidden_state
        if pool == "mean_patches":
            h = h[:, 1:]
        feats.append(h.mean(dim=1).float().cpu())
        labels.append(torch.as_tensor(y))
    return torch.cat(feats), torch.cat(labels)


def probe_pretrained(model_key, data_root="/data", out_root="/results",
                     fracs=(1.0, 0.1, 0.01)):
    from datasets import load_dataset
    device = "cuda"
    model, proc, pool = load_model(model_key, device)
    cache = os.path.join(data_root, "hf")
    tr = load_dataset("clane9/imagenet-100", split="train", cache_dir=cache)
    va = load_dataset("clane9/imagenet-100", split="validation", cache_dir=cache)
    tr_f, tr_y = extract(model, pool, HFProcDataset(tr, proc), device)
    te_f, te_y = extract(model, pool, HFProcDataset(va, proc), device)
    del model
    torch.cuda.empty_cache()

    results = {}
    for frac in fracs:
        sub = (torch.arange(len(tr_y)) if frac >= 1.0
               else lowshot_indices(tr_y, frac))
        results[f"linear_{int(frac * 100)}pct"] = linear_probe(
            tr_f[sub], tr_y[sub], te_f, te_y, device=device)
    results["knn"] = knn_probe(tr_f, tr_y, te_f, te_y, device=device)
    results["feat_dim"] = tr_f.size(1)

    out_dir = os.path.join(out_root, "probes")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"pretrained_{model_key}.json"), "w") as f:
        json.dump({"model": model_key, "dataset": "in100", **results},
                  f, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, f"pretrained_{model_key}_testfeats.npz"),
        feats=te_f.numpy().astype(np.float16), labels=te_y.numpy())
    return results
