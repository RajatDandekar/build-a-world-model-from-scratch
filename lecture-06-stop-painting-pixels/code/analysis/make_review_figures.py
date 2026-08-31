#!/usr/bin/env python3
"""Figures for the Lecture-6 revision pass (Raj's review).

  fig_loss_<arch>.png         one loss curve + speech-bubble callouts
  fig_fingerprint_<arch>.png  8 real test images + their embedding barcodes
  fig_collapse_explainer.png  what "feature std" is, from real embeddings
  fig_labels_fraction.png     what 100% / 10% / 1% of labels means, one class
  fig_neighbors.png           nearest-neighbour retrieval, I-JEPA vs MAE
Appends new keys to figs/headlines.json (never removes existing ones).
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import style
from style import (ARCH_COLORS, ARCH_LABELS, TEAL, TEAL_DEEP, GOLD, GOLD_DEEP,
                   CLAY, CLAY_DEEP, MUTED, INK, INK2, LINE, PAPER, GOOD, BLUE)

style.apply()
ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figs")

ARCHS = ["jea", "mae", "genmask", "ijepa", "ijepa_nostop"]
STL_CLASSES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse",
               "monkey", "ship", "truck"]

H = json.load(open(os.path.join(FIGS, "headlines.json")))


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)
    print("wrote", name)


def bubble(ax, text, xy, xytext, color):
    ax.annotate(text, xy=xy, xytext=xytext, textcoords="axes fraction",
                xycoords="axes fraction", fontsize=9.5, color=INK,
                ha="center", va="center", fontweight="500",
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=color,
                          lw=1.6),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5,
                                connectionstyle="arc3,rad=0.15"))


def load_log(run):
    path = os.path.join(RES, run, "log.jsonl")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    return {k: np.array([r[k] for r in recs]) for k in recs[0]}


# --------------------------------------------------- per-run loss curves
BUBBLES = {
    "jea": [("Falls to −1 in the first\nfew hundred steps —\na PERFECT loss",
             (0.08, 0.15), (0.45, 0.55), CLAY),
            ("…because it stopped\nlooking at the images",
             (0.75, 0.06), (0.72, 0.45), CLAY_DEEP)],
    "mae": [("Steady descent —\npainting slowly improves",
             (0.25, 0.45), (0.6, 0.75), GOLD_DEEP),
            ("Never reaches zero:\npixels keep surprising you",
             (0.9, 0.28), (0.62, 0.12), GOLD_DEEP)],
    "genmask": [("Same shape as MAE —\nthe mask pattern barely\nchanges pixel learning",
                 (0.5, 0.4), (0.55, 0.78), GOLD_DEEP)],
    "ijepa": [("Dips early while the\nteacher is still random…",
               (0.06, 0.25), (0.35, 0.72), TEAL_DEEP),
              ("…then RISES as the teacher\nimproves: the target keeps\nmoving. Healthy!",
               (0.55, 0.62), (0.68, 0.2), TEAL_DEEP)],
    "ijepa_nostop": [("Racing to zero —\nlooks like a triumph",
                     (0.12, 0.2), (0.5, 0.65), CLAY),
                    ("Actual value 1.0×10⁻⁶.\nStudent and teacher agreed\nto say the same constant",
                     (0.85, 0.05), (0.63, 0.3), CLAY_DEEP)],
}

for a in ARCHS:
    l = load_log(f"stl10_{a}_s0")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(l["step"], l["loss"], color=ARCH_COLORS[a], lw=2.0)
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss")
    for text, xy, xytext, color in BUBBLES[a]:
        bubble(ax, text, xy, xytext, color)
    save(fig, f"fig_loss_{a}.png")

# --------------------------------------------------- fingerprints
d = np.load(os.path.join(RES, "review", "fingerprints.npz"))
imgs, labels = d["imgs"], d["labels"]
n = imgs.shape[0]
for a in ARCHS:
    emb = d[f"emb_{a}"]                    # (8, 384)
    # shared color scale across the 8 fingerprints of THIS run
    vmin, vmax = np.percentile(emb, 2), np.percentile(emb, 98)
    # how distinct are they? mean pairwise cosine
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cos = e @ e.T
    off = cos[~np.eye(n, dtype=bool)]
    H[f"fp_cos_{a}"] = round(float(off.mean()), 4)
    fig, axes = plt.subplots(2, n, figsize=(1.55 * n, 3.6),
                             constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.5, 1.0]})
    for j in range(n):
        axes[0, j].imshow(np.transpose(imgs[j], (1, 2, 0)))
        axes[0, j].set_title(STL_CLASSES[labels[j]], fontsize=8.5)
        axes[1, j].imshow(emb[j].reshape(16, 24), cmap="RdBu_r",
                          vmin=vmin, vmax=vmax, aspect="auto")
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[1, 0].set_ylabel("its 384-number\nfingerprint", fontsize=8)
    save(fig, f"fig_fingerprint_{a}.png")

# --------------------------------------------------- collapse explainer
fig = plt.figure(figsize=(11.8, 4.2), constrained_layout=True)
gs = fig.add_gridspec(2, 4, width_ratios=[2.2, 1.0, 2.2, 1.0])
sub = 48
for row, (a, tcol) in enumerate([("ijepa", TEAL_DEEP), ("jea", CLAY_DEEP)]):
    f = np.load(os.path.join(RES, "probes",
                             f"stl10_{a}_s0_testfeats.npz"))["feats"]
    f = f[:sub, :64].astype(np.float32)
    ax = fig.add_subplot(gs[row, 0])
    lim = np.percentile(np.abs(f), 98)
    ax.imshow(f, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_ylabel(f"{ARCH_LABELS[a]}\n48 images (rows)", fontsize=8.5,
                  color=tcol)
    ax.set_xlabel("first 64 of 384 dimensions", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    std = np.load(os.path.join(RES, "probes",
                               f"stl10_{a}_s0_testfeats.npz"))["feats"]
    stdv = (std.astype(np.float32)
            / np.linalg.norm(std.astype(np.float32), axis=1, keepdims=True)
            ).std(axis=0)
    ax2 = fig.add_subplot(gs[row, 1])
    ax2.bar(np.arange(64), stdv[:64], color=tcol, width=1.0)
    ax2.set_ylim(0, 0.11)
    ax2.axhline(384 ** -0.5, color=MUTED, ls="--", lw=1)
    ax2.set_xticks([])
    ax2.set_ylabel("std per dim", fontsize=8)
    if row == 0:
        ax2.set_title("spread of each column", fontsize=9)
    H[f"expl_std_{a}"] = round(float(stdv.mean()), 6)
# right half: the sentence, drawn
axr = fig.add_subplot(gs[:, 2:])
axr.axis("off")
axr.text(0.5, 0.78, "Read each column of the matrix:", ha="center",
         fontsize=12, color=INK, fontweight="600")
axr.text(0.5, 0.60, "does this dimension VARY across images?",
         ha="center", fontsize=12, color=INK)
axr.text(0.5, 0.38, "I-JEPA: rows differ, so columns have spread: std ≈ 1/√D",
         ha="center", fontsize=10.5, color=TEAL_DEEP)
axr.text(0.5, 0.26, "Collapsed: every row identical, zero spread: std ≈ 0",
         ha="center", fontsize=10.5, color=CLAY_DEEP)
axr.text(0.5, 0.06, '"Collapse is one number" = the average of those bars.',
         ha="center", fontsize=11.5, color=INK, fontweight="600")
save(fig, "fig_collapse_explainer.png")

# --------------------------------------------------- labels fraction
d = np.load(os.path.join(RES, "review", "montage.npz"), allow_pickle=True)
thumbs, count = d["thumbs"], int(d["class_count"])
cname = str(d["class_name"])
fracs = [(1.0, "all labels"), (0.1, "10% of labels"), (0.01, "1% of labels")]
cols, rows = 10, 5
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), constrained_layout=True)
rng = np.random.default_rng(0)
for ax, (fr, lab) in zip(axes, fracs):
    grid = np.ones((rows * 64, cols * 64, 3), dtype=np.float32)
    n_show = rows * cols
    lit = set(rng.choice(n_show, max(1, int(round(n_show * fr))),
                         replace=False).tolist())
    for k in range(n_show):
        r, c = divmod(k, cols)
        t = thumbs[k % len(thumbs)].astype(np.float32) / 255.0
        if k not in lit:
            t = 0.15 * t + 0.85          # washed out = label thrown away
        grid[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = t
    ax.imshow(grid)
    kept = max(1, int(round(count * fr)))
    ax.set_title(f"{lab}:  {kept:,} of {count:,} labelled", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
H["labels_class_count"] = count
H["labels_class_name"] = cname
save(fig, "fig_labels_fraction.png")

# --------------------------------------------------- neighbours (landscape)
spec = json.load(open(os.path.join(RES, "neighbors_spec.json")))
nd = np.load(os.path.join(RES, "review", "neighbor_imgs.npz"),
             allow_pickle=True)
img_by_idx = {int(i): nd["imgs"][k] for k, i in enumerate(nd["indices"])}
lab_by_idx = {int(i): str(nd["names"][k]).split(",")[0]
              for k, i in enumerate(nd["indices"])}
queries = [spec["queries"][1], spec["queries"][0], spec["queries"][2]]  # goose, shaker, jean
models = [("ijepa_vith14", "I-JEPA", TEAL_DEEP), ("mae_vith", "MAE", GOLD_DEEP)]
ncols = len(queries) * 5 + (len(queries) - 1)   # spacer cols between blocks
fig, axes = plt.subplots(2, ncols, figsize=(15.6, 3.1),
                         constrained_layout=True)
for qi, q in enumerate(queries):
    qlab = lab_by_idx[q]
    base = qi * 6
    for mi, (key, mname, mcol) in enumerate(models):
        ax = axes[mi, base]
        ax.imshow(img_by_idx[q])
        ax.set_ylabel(mname, fontsize=10, color=mcol, fontweight="600")
        if mi == 0:
            ax.set_title("query", fontsize=8.5, color=INK2)
        if mi == 1:
            ax.set_xlabel(qlab, fontsize=10, fontweight="600", color=INK)
        for c, nb in enumerate(spec["neighbors"][key][str(q)]):
            axn = axes[mi, base + 1 + c]
            axn.imshow(img_by_idx[nb])
            ok = lab_by_idx[nb] == qlab
            for sp in axn.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(GOOD if ok else CLAY)
                sp.set_linewidth(3)
            axn.set_title(lab_by_idx[nb], fontsize=7.2,
                          color=GOOD if ok else CLAY_DEEP)
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for r in range(2):
    for sp_c in range(5, ncols, 6):
        axes[r, sp_c].axis("off")        # spacer columns
save(fig, "fig_neighbors.png")

with open(os.path.join(FIGS, "headlines.json"), "w") as f:
    json.dump(H, f, indent=2, sort_keys=True)
print("headlines updated:", {k: v for k, v in H.items()
                             if k.startswith(("fp_", "expl_", "labels_"))})
