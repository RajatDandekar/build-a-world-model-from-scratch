#!/usr/bin/env python3
"""Generate every results figure for Lecture 6 from the raw Modal outputs.

Inputs  (synced from the ijepa-l6-results volume into results/):
  results/<run>/log.jsonl            training logs (loss, feat_std, eff_rank)
  results/probes/<name>.json         probe accuracies
  results/probes/<name>_testfeats.npz  frozen test features for t-SNE
  results/viz/masks_*.npz            mask examples on real images
  results/viz/mae_recon.npz          MAE reconstructions

Outputs figs/*.png (220 dpi, warm-paper theme).

RULE (learned the hard way in the previous L6): every number that appears on a
slide is recomputed here from the raw logs — never typed by hand — and runs are
only ever compared at the largest step both reached.
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(__file__))
import style
from style import (ARCH_COLORS, ARCH_LABELS, TEAL, TEAL_DEEP, GOLD, GOLD_DEEP,
                   CLAY, CLAY_DEEP, MUTED, INK, INK2, LINE, PAPER, GOOD, BLUE)

style.apply()
ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)

ARCHS = ["jea", "mae", "genmask", "ijepa", "ijepa_nostop"]


def load_log(run):
    path = os.path.join(RES, run, "log.jsonl")
    if not os.path.exists(path):
        return None
    recs = [json.loads(l) for l in open(path) if l.strip()]
    return {k: np.array([r[k] for r in recs]) for k in recs[0]}


def load_probe(name):
    path = os.path.join(RES, "probes", f"{name}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def common_max_step(logs):
    """Largest step every provided run reached — the honest comparison point."""
    return int(min(l["step"].max() for l in logs.values() if l is not None))


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)
    print("wrote", name)


headlines = {}

# ---------------------------------------------------------------- E1 curves
logs = {a: load_log(f"stl10_{a}_s0") for a in ARCHS}
have_logs = {a: l for a, l in logs.items() if l is not None}

if have_logs:
    cms = common_max_step(have_logs)

    # collapse curves: feat_std
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for a, l in have_logs.items():
        m = l["step"] <= cms
        ax.plot(l["step"][m], l["feat_std"][m], color=ARCH_COLORS[a],
                lw=2.2, label=ARCH_LABELS[a])
    ax.axhline(384 ** -0.5, color=MUTED, ls="--", lw=1.2)
    ax.annotate("healthy level  1/√D", xy=(0.99, 384 ** -0.5),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=9, color=MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("feature std (log)")
    ax.legend(fontsize=9, loc="lower left")
    save(fig, "fig_collapse_feat_std.png")

    # effective rank
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for a, l in have_logs.items():
        m = l["step"] <= cms
        ax.plot(l["step"][m], l["eff_rank"][m], color=ARCH_COLORS[a],
                lw=2.2, label=ARCH_LABELS[a])
    ax.set_xlabel("training step")
    ax.set_ylabel("effective rank of features (max 384)")
    ax.legend(fontsize=9)
    save(fig, "fig_eff_rank.png")

    # the deception plot: loss per arch (own panel, own scale)
    fig, axes = plt.subplots(1, len(have_logs), figsize=(13.4, 3.0),
                             constrained_layout=True)
    if len(have_logs) == 1:
        axes = [axes]
    for ax, (a, l) in zip(axes, have_logs.items()):
        m = l["step"] <= cms
        ax.plot(l["step"][m], l["loss"][m], color=ARCH_COLORS[a], lw=1.8)
        ax.set_title(ARCH_LABELS[a], fontsize=10)
        ax.set_xlabel("step", fontsize=9)
        final = l["loss"][m][-1]
        # never print a tiny value as 0.0000 — the previous lecture shipped
        # that exact lie; use scientific notation below 1e-3
        lab = f"{final:.1e}" if 0 < abs(final) < 1e-3 else f"{final:.4f}"
        ax.annotate(lab, xy=(0.97, 0.88), xycoords="axes fraction",
                    ha="right", fontsize=10, fontweight="bold",
                    color=ARCH_COLORS[a])
        headlines[f"final_loss_{a}"] = float(final)
    axes[0].set_ylabel("training loss")
    save(fig, "fig_loss_deception.png")

    for a, l in have_logs.items():
        m = l["step"] <= cms
        headlines[f"final_feat_std_{a}"] = float(l["feat_std"][m][-1])
        headlines[f"final_eff_rank_{a}"] = float(l["eff_rank"][m][-1])
    headlines["common_max_step"] = cms

    # teaser: two anonymized loss curves (normalized to [0,1] so neither scale
    # gives the answer away). Run A = ijepa_nostop (broken), Run B = ijepa.
    if "ijepa" in have_logs and "ijepa_nostop" in have_logs:
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        for a, lab, col in [("ijepa_nostop", "Run A", CLAY_DEEP),
                            ("ijepa", "Run B", TEAL_DEEP)]:
            l = have_logs[a]
            m = l["step"] <= cms
            y = l["loss"][m]
            y = (y - y.min()) / max(1e-9, (y.max() - y.min()))
            ax.plot(l["step"][m], y, lw=2.4, color=col, label=lab)
        ax.set_xlabel("training step")
        ax.set_ylabel("loss (normalized)")
        ax.legend(fontsize=11)
        save(fig, "fig_teaser_two_losses.png")

# ---------------------------------------------------------------- E1 probes
probes = {a: load_probe(f"stl10_{a}_s0") for a in ARCHS}
probes["random"] = load_probe("stl10_ijepa_s0_randominit")
have_probes = {a: p for a, p in probes.items() if p is not None}

if have_probes:
    order = [a for a in
             ["random", "jea", "ijepa_nostop", "mae", "genmask", "ijepa"]
             if a in have_probes]
    accs = [100 * have_probes[a]["linear_100pct"] for a in order]
    knns = [100 * have_probes[a]["knn"] for a in order]
    ys = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    b1 = ax.barh(ys - 0.19, accs, height=0.36,
                 color=[ARCH_COLORS[a] for a in order], label="linear probe")
    b2 = ax.barh(ys + 0.19, knns, height=0.36, alpha=0.45,
                 color=[ARCH_COLORS[a] for a in order], label="kNN (k=20)")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_width()
            ax.annotate(f"{v:.1f}", xy=(v + 0.8, b.get_y() + b.get_height() / 2),
                        va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([ARCH_LABELS[a] for a in order])
    ax.axvline(10.0, color=MUTED, ls=":", lw=1.2)
    ax.annotate("chance", xy=(10.6, len(order) - 0.55), fontsize=8.5,
                color=MUTED)
    ax.set_xlim(0, max(accs) * 1.16)
    ax.set_xlabel("STL-10 accuracy on frozen features (%)")
    ax.invert_yaxis()
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "fig_probe_bars_stl10.png")
    for a in order:
        headlines[f"probe_{a}"] = 100 * have_probes[a]["linear_100pct"]
        headlines[f"knn_{a}"] = 100 * have_probes[a]["knn"]

    # t-SNE panels
    try:
        from sklearn.manifold import TSNE
        panel = [a for a in ["jea", "mae", "ijepa"] if a in have_probes]
        fig, axes = plt.subplots(1, len(panel), figsize=(4.3 * len(panel), 4.3),
                                 constrained_layout=True)
        for ax, a in zip(np.atleast_1d(axes), panel):
            name = f"stl10_{a}_s0"
            d = np.load(os.path.join(RES, "probes", f"{name}_testfeats.npz"))
            f, y = d["feats"].astype(np.float32), d["labels"]
            sub = np.random.RandomState(0).choice(len(f), 2000, replace=False)
            emb = TSNE(n_components=2, init="pca", perplexity=30,
                       random_state=0).fit_transform(f[sub])
            ax.scatter(emb[:, 0], emb[:, 1], c=y[sub], cmap="tab10", s=4,
                       alpha=0.75, linewidths=0)
            ax.set_title(ARCH_LABELS[a], fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        save(fig, "fig_tsne_panels.png")
    except Exception as e:
        print("tsne skipped:", e)

# ---------------------------------------------------------------- E2 official
KEYMAP = {
    "ijepa_vith14": ("I-JEPA ViT-H/14\n(official weights)", TEAL),
    "mae_vith": ("MAE ViT-H/14\n(official weights)", GOLD),
    "random_vith": ("Random\nViT-H/14", MUTED),
    "supervised_vitb": ("Supervised ViT-B\n(1.28M labels)", BLUE),
}
pre = {k: load_probe(f"pretrained_{k}") for k in KEYMAP}
have_pre = {k: p for k, p in pre.items() if p is not None}
if have_pre:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    keys = [k for k in ["random_vith", "mae_vith", "ijepa_vith14",
                        "supervised_vitb"] if k in have_pre]
    x = np.arange(len(keys))
    w = 0.26
    for j, (frac, lab, alpha) in enumerate(
            [("linear_1pct", "1% of labels", 0.45),
             ("linear_10pct", "10% of labels", 0.7),
             ("linear_100pct", "all labels", 1.0)]):
        vals = [100 * have_pre[k].get(frac, np.nan) for k in keys]
        cols = [KEYMAP[k][1] for k in keys]
        ax.bar(x + (j - 1) * w, vals, w, color=cols, alpha=alpha, label=lab)
        for xi, v in zip(x + (j - 1) * w, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.1f}", xy=(xi, v + 0.8), ha="center",
                            fontsize=8.4, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([KEYMAP[k][0] for k in keys], fontsize=9.5)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1)
    ax.set_ylabel("ImageNet-100 linear-probe accuracy (%)")
    ax.legend(fontsize=9)
    save(fig, "fig_official_probes_in100.png")
    for k in keys:
        for frac in ["linear_1pct", "linear_10pct", "linear_100pct", "knn"]:
            if frac in have_pre[k]:
                headlines[f"pre_{k}_{frac}"] = 100 * have_pre[k][frac]

    # t-SNE of official-checkpoint features (20 classes for legibility)
    try:
        from sklearn.manifold import TSNE
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6),
                                 constrained_layout=True)
        for ax, key, lab in [(axes[0], "mae_vith", "MAE ViT-H/14 (official)"),
                             (axes[1], "ijepa_vith14",
                              "I-JEPA ViT-H/14 (official)")]:
            d = np.load(os.path.join(RES, "probes",
                                     f"pretrained_{key}_testfeats.npz"))
            f, y = d["feats"].astype(np.float32), d["labels"]
            keep = y < 20
            f, y = f[keep], y[keep]
            emb = TSNE(n_components=2, init="pca", perplexity=30,
                       random_state=0).fit_transform(f)
            ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab20", s=6,
                       alpha=0.8, linewidths=0)
            ax.set_title(lab, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        save(fig, "fig_tsne_official.png")
    except Exception as e:
        print("official tsne skipped:", e)

# ---------------------------------------------------------------- E3 scratch
sc = load_probe("in100_ijepa_s0")
sc_r = load_probe("in100_ijepa_s0_randominit")
sc_log = load_log("in100_ijepa_s0")
if sc:
    headlines["scratch_in100_probe"] = 100 * sc["linear_100pct"]
    if sc_r:
        headlines["scratch_in100_random"] = 100 * sc_r["linear_100pct"]
    if sc_log is not None:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.plot(sc_log["step"], sc_log["loss"], color=TEAL, lw=1.8)
        ax.set_xlabel("training step")
        ax.set_ylabel("smooth-L1 loss in representation space")
        save(fig, "fig_scratch_in100_loss.png")

# ---------------------------------------------------------------- E4 masks
for ds in ["in100", "stl10"]:
    path = os.path.join(RES, "viz", f"masks_{ds}.npz")
    if not os.path.exists(path):
        continue
    d = np.load(path)
    imgs, ctx, tgt, g, p = d["imgs"], d["ctx"], d["tgt"], int(d["grid"]), int(d["patch"])
    n = min(4, len(imgs))
    tcols = [TEAL, GOLD, CLAY, BLUE]
    fig, axes = plt.subplots(3, n, figsize=(2.9 * n, 8.6),
                             constrained_layout=True)
    for j in range(n):
        img = np.transpose(imgs[j], (1, 2, 0))
        # row 0: original
        axes[0, j].imshow(img)
        # row 1: the four target blocks outlined
        axes[1, j].imshow(img, alpha=0.92)
        for k in range(4):
            mask2d = tgt[j, k].reshape(g, g)
            ys, xs = np.where(mask2d)
            if len(ys) == 0:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            axes[1, j].add_patch(Rectangle(
                (x0 * p - .5, y0 * p - .5), (x1 - x0 + 1) * p, (y1 - y0 + 1) * p,
                fill=False, edgecolor=tcols[k], lw=2.6))
        # row 2: context = image with target+dropped patches greyed
        shown = img.copy()
        cm = ctx[j].reshape(g, g)
        for yy in range(g):
            for xx in range(g):
                if not cm[yy, xx]:
                    shown[yy * p:(yy + 1) * p, xx * p:(xx + 1) * p] = \
                        0.25 * shown[yy * p:(yy + 1) * p, xx * p:(xx + 1) * p] \
                        + 0.75 * np.array([0.965, 0.955, 0.925])
        axes[2, j].imshow(shown)
    for r, lab in enumerate(["the image", "4 target blocks (what to predict)",
                             "the context (what the model sees)"]):
        axes[r, 0].set_ylabel(lab, fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    save(fig, f"fig_masks_{ds}.png")

# ---------------------------------------------------------------- MAE recon
path = os.path.join(RES, "viz", "mae_recon.npz")
if os.path.exists(path):
    d = np.load(path)
    n = min(6, len(d["orig"]))
    fig, axes = plt.subplots(3, n, figsize=(1.9 * n, 6.0),
                             constrained_layout=True)
    for j in range(n):
        for r, key in enumerate(["orig", "masked", "recon"]):
            axes[r, j].imshow(np.transpose(d[key][j], (1, 2, 0)))
    for r, lab in enumerate(["original", "input (75% masked)",
                             "reconstruction"]):
        axes[r, 0].set_ylabel(lab, fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    save(fig, "fig_mae_recon.png")

with open(os.path.join(FIGS, "headlines.json"), "w") as f:
    json.dump(headlines, f, indent=2, sort_keys=True)
print(json.dumps(headlines, indent=2, sort_keys=True))
