"""Frozen-feature evaluation: linear probe (paper protocol: average-pooled
last-layer patch tokens), kNN, and low-shot probes. Also dumps features for
t-SNE / figure generation.
"""
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .vit import ViTEncoder
from .data import get_probe_datasets


@torch.no_grad()
def extract_features(encoder, dataset, device, batch_size=256, num_workers=12):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    feats, labels = [], []
    encoder.eval()
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            z = encoder(x).mean(dim=1)
        feats.append(z.float().cpu())
        labels.append(torch.as_tensor(y))
    return torch.cat(feats), torch.cat(labels)


def linear_probe(tr_f, tr_y, te_f, te_y, epochs=100, lr=0.01, device="cuda"):
    """Train a linear classifier on frozen, standardized features."""
    mu, sd = tr_f.mean(0, keepdim=True), tr_f.std(0, keepdim=True) + 1e-6
    tr = ((tr_f - mu) / sd).to(device)
    te = ((te_f - mu) / sd).to(device)
    tr_y, te_y = tr_y.to(device), te_y.to(device)
    n_cls = int(tr_y.max().item()) + 1
    clf = torch.nn.Linear(tr.size(1), n_cls).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = 1024
    for ep in range(epochs):
        perm = torch.randperm(tr.size(0), device=device)
        for i in range(0, tr.size(0), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(clf(tr[idx]), tr_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        acc = (clf(te).argmax(1) == te_y).float().mean().item()
    return acc


@torch.no_grad()
def knn_probe(tr_f, tr_y, te_f, te_y, k=20, T=0.07, device="cuda"):
    tr = F.normalize(tr_f, dim=1).to(device)
    te = F.normalize(te_f, dim=1).to(device)
    tr_y = tr_y.to(device)
    n_cls = int(tr_y.max().item()) + 1
    correct, total = 0, 0
    for i in range(0, te.size(0), 512):
        sim = te[i:i + 512] @ tr.T
        top, idx = sim.topk(k, dim=1)
        votes = torch.zeros(idx.size(0), n_cls, device=device)
        votes.scatter_add_(1, tr_y[idx], (top / T).exp())
        pred = votes.argmax(1)
        correct += (pred == te_y[i:i + 512].to(device)).sum().item()
        total += idx.size(0)
    return correct / total


def lowshot_indices(labels, frac, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = []
    for c in labels.unique():
        ci = (labels == c).nonzero(as_tuple=True)[0]
        n = max(1, int(round(len(ci) * frac)))
        pick = ci[torch.randperm(len(ci), generator=g)[:n]]
        idx.append(pick)
    return torch.cat(idx)


def evaluate_checkpoint(ckpt_path, dataset="stl10", data_root="/data",
                        out_root="/results", random_init=False,
                        fracs=(1.0, 0.1, 0.01), tag=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    enc = ViTEncoder(meta["img_size"], meta["patch_size"])
    if random_init:
        torch.manual_seed(1234)
        # fresh random weights: just don't load
    else:
        enc.load_state_dict(ck["encoder"])
    enc = enc.to(device)

    tr_ds, te_ds = get_probe_datasets(dataset, data_root)
    tr_f, tr_y = extract_features(enc, tr_ds, device)
    te_f, te_y = extract_features(enc, te_ds, device)

    results = {}
    for frac in fracs:
        if frac >= 1.0:
            sub = torch.arange(len(tr_y))
        else:
            sub = lowshot_indices(tr_y, frac)
        acc = linear_probe(tr_f[sub], tr_y[sub], te_f, te_y, device=device)
        results[f"linear_{int(frac * 100)}pct"] = acc
    results["knn"] = knn_probe(tr_f, tr_y, te_f, te_y, device=device)
    results["feat_dim"] = tr_f.size(1)
    results["n_train"] = len(tr_y)
    results["n_test"] = len(te_y)

    name = tag or (os.path.basename(os.path.dirname(ckpt_path))
                   + ("_randominit" if random_init else ""))
    out_dir = os.path.join(out_root, "probes")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump({"ckpt": ckpt_path, "dataset": dataset,
                   "random_init": random_init, **results}, f, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, f"{name}_testfeats.npz"),
        feats=te_f.numpy().astype(np.float16), labels=te_y.numpy())
    return results
