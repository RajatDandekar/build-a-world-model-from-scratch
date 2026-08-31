#!/usr/bin/env python3
"""Per-architecture energy figures for the rebuilt 6a deck.

  fig_energy_intro.png     E made concrete: one real masked context, three
                           candidate targets, their measured mean energies
  fig_contour_<arch>.png   that architecture's 2D energy contour, solo
  fig_ebars_<arch>.png     its four-corruption energy bars, solo
Also merges the needed keys into figs_motivation/headlines.json.
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
FIGS = os.path.join(ROOT, "figs_motivation")

ARCHS = ["jea", "ijepa_nostop", "mae", "genmask", "ijepa"]
H = json.load(open(os.path.join(FIGS, "headlines.json")))
S = json.load(open(os.path.join(RES, "energy", "summary.json")))


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- intro
d = np.load(os.path.join(RES, "viz", "masks_in100.npz"))
img = np.transpose(d["imgs"][1], (1, 2, 0))          # the dog
imp = np.transpose(d["imgs"][3], (1, 2, 0))          # a different image
rng = np.random.default_rng(0)
noise = np.clip(img + rng.normal(0, 0.7, img.shape), 0, 1)
e = S["ijepa"]
noise_curve = np.load(os.path.join(RES, "energy",
                                   "energy_ijepa.npz"))["noise_curve"]
cands = [("the TRUE hidden content", img, e["e_true_mean"], TEAL_DEEP),
         ("content from a DIFFERENT image", imp, e["e_impostor_mean"],
          CLAY_DEEP),
         ("pure noise", noise, float(noise_curve[-1].mean()), MUTED)]
fig = plt.figure(figsize=(11.8, 3.6), constrained_layout=True)
gs = fig.add_gridspec(2, 4, height_ratios=[2.4, 1.0])
axq = fig.add_subplot(gs[:, 0])
ctxim = img.copy()
g, p = int(d["grid"]), int(d["patch"])
cm = d["ctx"][1].reshape(g, g)
for yy in range(g):
    for xx in range(g):
        if not cm[yy, xx]:
            ctxim[yy * p:(yy + 1) * p, xx * p:(xx + 1) * p] = \
                0.25 * ctxim[yy * p:(yy + 1) * p, xx * p:(xx + 1) * p] + 0.72
axq.imshow(ctxim)
axq.set_title("the context x\n(what the model sees)", fontsize=10)
axq.set_xticks([]); axq.set_yticks([]); axq.grid(False)
emax = max(c[2] for c in cands)
for k, (lab, im, ev, col) in enumerate(cands):
    ax = fig.add_subplot(gs[0, k + 1])
    ax.imshow(im)
    ax.set_title(f'candidate y{k + 1}:\n{lab}', fontsize=9, color=col)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axb = fig.add_subplot(gs[1, k + 1])
    axb.barh([0], [ev], color=col, height=0.55)
    axb.set_xlim(0, emax * 1.15)
    axb.set_yticks([])
    axb.annotate(f"score: {ev:.2f}", xy=(ev + emax * 0.02, 0), va="center",
                 fontsize=10, fontweight="bold", color=col)
    axb.set_xticks([])
    for sp in axb.spines.values():
        sp.set_visible(False)
save(fig, "fig_energy_intro.png")
H["intro_e_true"] = round(e["e_true_mean"], 2)
H["intro_e_impostor"] = round(e["e_impostor_mean"], 2)
H["intro_e_noise"] = round(float(noise_curve[-1].mean()), 2)

# ------------------------------------------------- per-arch contours & bars
kinds = [("e_true", "true\ntarget"), ("e_shuffle", "patches\nshuffled"),
         ("e_impostor", "different\nimage"), ("e_wrongpos", "wrong\nposition")]
for a in ARCHS:
    dd = np.load(os.path.join(RES, "energy", f"energy_{a}.npz"))
    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    ax.contourf(dd["contour_b"], dd["contour_a"], dd["contour_grid"],
                levels=14, cmap="YlOrBr")
    ax.plot(0, 0, "o", color=TEAL_DEEP, ms=11, mec="white", mew=1.8)
    ax.annotate("true target", xy=(0.03, -0.02), fontsize=10,
                color=TEAL_DEEP, fontweight="600")
    ax.plot(0, 1, "X", color=CLAY_DEEP, ms=12, mec="white", mew=1.4)
    ax.annotate("a different image", xy=(0.03, 0.98), fontsize=10,
                color=CLAY_DEEP, fontweight="600")
    ax.set_xlabel("noise direction")
    ax.set_ylabel("toward a different image")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    save(fig, f"fig_contour_{a}.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    vals = [float(np.mean(dd[k])) for k, _ in kinds]
    bars = ax.bar(range(4), vals, color=ARCH_COLORS[a],
                  alpha=None)
    for b, v in zip(bars, vals):
        lab = f"{v:.2f}" if v > 1e-3 else f"{v:.0e}"
        ax.annotate(lab, xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold", color=INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels([lab for _, lab in kinds], fontsize=9)
    ax.set_ylabel("energy")
    if vals[0] < 1e-5:
        ax.annotate("all four ≈ 0 — the model\ncannot tell them apart",
                    xy=(0.5, 0.6), xycoords="axes fraction", ha="center",
                    fontsize=11, color=CLAY_DEEP, fontweight="bold")
    save(fig, f"fig_ebars_{a}.png")

# merge keys used by 6a anchors (from the lecture-6 headline store)
H6 = json.load(open(os.path.join(ROOT, "figs", "headlines.json")))
for k in ["fp_cos_jea", "fp_cos_ijepa", "fp_cos_mae",
          "pre_ijepa_vith14_knn", "pre_mae_vith_knn"]:
    if k in H6:
        H[k] = H6[k]
with open(os.path.join(FIGS, "headlines.json"), "w") as f:
    json.dump(H, f, indent=2, sort_keys=True)
print("headlines merged")
