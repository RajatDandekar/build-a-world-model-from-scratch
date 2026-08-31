#!/usr/bin/env python3
"""Figures for the "Motivation for JEPA" prequel deck.

Inputs: results/energy/*.npz (E1 checkpoints)  ·  results/toy/*.npz
Outputs: figs_motivation/*.png + figs_motivation/headlines.json
Same rule as always: every number on a slide comes from here.
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
os.makedirs(FIGS, exist_ok=True)

H = {}
ARCHS = ["jea", "ijepa_nostop", "mae", "genmask", "ijepa"]

TOY_LABELS = {
    "gen_det": "Generative (deterministic)",
    "ae": "Autoencoder (on y)",
    "jea": "Joint embedding (no guard)",
    "jepa_z": "JEPA + latent z",
    "jepa_noz": "JEPA, no latent",
}
TOY_COLORS = {
    "gen_det": GOLD, "ae": PURPLE if (PURPLE := "#8D6FC0") else GOLD,
    "jea": CLAY, "jepa_z": TEAL, "jepa_noz": TEAL_DEEP,
}


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)
    print("wrote", name)


# ================================================================ E1 energy
edata = {a: np.load(os.path.join(RES, "energy", f"energy_{a}.npz"))
         for a in ARCHS if os.path.exists(
             os.path.join(RES, "energy", f"energy_{a}.npz"))}

if edata:
    # --- corruption wells: bar groups per corruption type, log scale
    kinds = [("e_true", "true target"), ("e_shuffle", "patches shuffled"),
             ("e_impostor", "different image"), ("e_wrongpos", "wrong position")]
    fig, axes = plt.subplots(1, len(edata), figsize=(13.6, 3.4),
                             constrained_layout=True)
    for ax, a in zip(axes, ARCHS):
        d = edata[a]
        vals = [float(np.mean(d[k])) for k, _ in kinds]
        ax.bar(range(len(kinds)), vals, color=ARCH_COLORS[a],
               alpha=[1.0, 0.75, 0.55, 0.4][0:1] * 0 or None)
        ax.set_title(ARCH_LABELS[a], fontsize=9.5)
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels([lab for _, lab in kinds], rotation=28,
                           ha="right", fontsize=7.5)
        ax.set_yscale("log")
        if float(np.mean(d["e_true"])) < 1e-5:
            ax.annotate("flat:\nall ≈ 0", xy=(0.5, 0.55),
                        xycoords="axes fraction", ha="center", fontsize=10,
                        color=CLAY_DEEP, fontweight="bold")
    axes[0].set_ylabel("energy (log)")
    save(fig, "fig_energy_bars_real.png")

    # --- noise-ramp curves
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for a in ARCHS:
        d = edata[a]
        m = d["noise_curve"].mean(axis=1)
        m = m / max(m.max(), 1e-12)          # normalize per model for shape
        ax.plot(d["noise_levels"], m, "-o", ms=4, lw=2.0,
                color=ARCH_COLORS[a], label=ARCH_LABELS[a])
    ax.set_xlabel("noise added to the target  (σ)")
    ax.set_ylabel("energy, normalized to each model's max")
    ax.legend(fontsize=8.5)
    save(fig, "fig_energy_noise_ramp.png")

    # --- 2D contours
    fig, axes = plt.subplots(1, len(edata), figsize=(13.6, 3.1),
                             constrained_layout=True)
    for ax, a in zip(axes, ARCHS):
        d = edata[a]
        g = d["contour_grid"]
        im = ax.contourf(d["contour_b"], d["contour_a"], g, levels=14,
                         cmap="YlOrBr")
        ax.plot(0, 0, "o", color=TEAL_DEEP, ms=8, mec="white", mew=1.5)
        ax.plot(0, 1, "X", color=CLAY_DEEP, ms=9, mec="white", mew=1.2)
        ax.set_title(ARCH_LABELS[a], fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[0].set_ylabel("toward a different image")
    axes[0].set_xlabel("noise direction")
    save(fig, "fig_energy_contours_real.png")

    # --- AUC bars
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    aucs = [float(json.load(open(os.path.join(RES, "energy",
                                              "summary.json")))[a]["auc"])
            for a in ARCHS]
    bars = ax.barh([ARCH_LABELS[a] for a in ARCHS], aucs,
                   color=[ARCH_COLORS[a] for a in ARCHS], height=0.6)
    for b, v in zip(bars, aucs):
        ax.annotate(f"{v:.4f}" if v < 0.9995 else f"{v:.4f}".rstrip("0"),
                    xy=(min(v + 0.012, 1.02),
                                    b.get_y() + b.get_height() / 2),
                    va="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.axvline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.annotate("coin flip", xy=(0.503, len(ARCHS) - 0.5), fontsize=8.5,
                color=MUTED)
    ax.set_xlim(0.4, 1.06)
    ax.set_xlabel("can the energy rank the true target above impostors? (AUC)")
    ax.invert_yaxis()
    save(fig, "fig_energy_auc.png")
    for a, v in zip(ARCHS, aucs):
        H[f"auc_{a}"] = round(v, 4)
    s = json.load(open(os.path.join(RES, "energy", "summary.json")))
    H["ijepa_gap_ratio"] = round(s["ijepa"]["e_impostor_mean"]
                                 / s["ijepa"]["e_true_mean"], 2)

# ================================================================ toy: fork
p = os.path.join(RES, "toy", "fork_models.npz")
if os.path.exists(p):
    d = np.load(p)
    # --- the world itself: context frames + the two futures
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.6),
                             constrained_layout=True)
    for i in range(3):
        axes[i].imshow(d["ctx"][i], cmap="gray_r")
        axes[i].set_title(f"context  t={i}", fontsize=10)
    axes[3].imshow(d["true0"], cmap="gray_r")
    axes[3].set_title("future A  (t=8)", fontsize=10, color=TEAL_DEEP)
    axes[4].imshow(d["true1"], cmap="gray_r")
    axes[4].set_title("future B  (t=8)", fontsize=10, color=CLAY_DEEP)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    save(fig, "fig_fork_world.png")

    # --- deterministic generative: the blur
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), constrained_layout=True)
    for ax, img, t, c in [
            (axes[0], d["true0"], "future A", TEAL_DEEP),
            (axes[1], d["true1"], "future B", CLAY_DEEP),
            (axes[2], d["pred_blur"], "the network's one prediction",
             GOLD_DEEP)]:
        ax.imshow(img, cmap="gray_r")
        ax.set_title(t, fontsize=10.5, color=c)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    save(fig, "fig_fork_blur.png")

    # --- energy landscapes over ball position, per architecture.
    # All grids are smoothed IDENTICALLY (gaussian, sigma=1 grid cell) for
    # display only — disclosed on the slide. The per-panel range annotation
    # uses the RAW grid.
    from scipy.ndimage import gaussian_filter
    names = ["gen_det", "ae", "jea", "jepa_noz", "jepa_z"]
    fig, axes = plt.subplots(1, len(names), figsize=(13.6, 3.2),
                             constrained_layout=True)
    for ax, n in zip(axes, names):
        g = d[f"grid_{n}"]
        rng_raw = float(g.max() - g.min())
        gs = gaussian_filter(g, 1.0)
        ax.contourf(d["xs"], d["ys"], gs, levels=16, cmap="YlOrBr")
        for (mx, my), c in zip(d["mode_positions"], [TEAL_DEEP, CLAY_DEEP]):
            ax.plot(mx, my, "*", color="white", ms=14, mec=c, mew=1.6)
        ax.set_title(TOY_LABELS[n], fontsize=9)
        lab = f"range Δ = {rng_raw:.3f}" if rng_raw > 5e-3             else f"range Δ = {rng_raw:.0e}  (≈ flat)"
        ax.set_xlabel(lab, fontsize=8.5,
                      color=CLAY_DEEP if rng_raw < 5e-3 else INK2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        ax.invert_yaxis()
        H[f"toy_range_{n}"] = rng_raw
    save(fig, "fig_energy_landscapes_toy.png")

    # --- 1D slice through both outcomes (x = 40 column)
    j = int(np.argmin(np.abs(d["xs"] - 40)))
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for n in ["gen_det", "ae", "jea", "jepa_noz", "jepa_z"]:
        col = d[f"grid_{n}"][:, j].astype(float)
        rngc = col.max() - col.min()
        col = (col - col.min()) / max(rngc, 1e-12)
        ax.plot(d["ys"], col, lw=2.0, color=TOY_COLORS[n],
                label=TOY_LABELS[n] + ("  (Δ≈0)" if rngc < 1e-3 else ""))
    for (mx, my), lab, c in zip(d["mode_positions"],
                                ["future A", "future B"],
                                [TEAL_DEEP, CLAY_DEEP]):
        ax.axvline(my, color=c, ls=":", lw=1.3)
        ax.annotate(lab, xy=(my + 0.6, 1.03), fontsize=9, color=c)
    ax.set_xlabel("vertical position of the ball in the target frame")
    ax.set_ylabel("energy (normalized per model)")
    ax.legend(fontsize=8, loc="lower center")
    save(fig, "fig_fork_slice.png")

    # --- collapse trace
    tr = d["jea_trace"]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(tr[:, 0], tr[:, 1], color=CLAY, lw=2.2)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("feature std (log)")
    save(fig, "fig_toy_jea_collapse.png")

    ts = json.load(open(os.path.join(RES, "toy", "summary.json")))
    H["fork_z_purity"] = round(100 * ts["fork_z_purity"], 1)
    H["camera_mae_px"] = round(ts["camera_mae_px"], 2)
    H["toy_jea_feat_std"] = float(ts["fork_jea_final_feat_std"])

# ================================================================ toy: camera
# 3-seed sweep: show ALL seeds — the inconsistency is the finding
import glob as _glob
seed_files = sorted(_glob.glob(os.path.join(RES, "toy", "camera_seed*.npz")))
if seed_files:
    maes = []
    for f in seed_files:
        dd = np.load(f)
        maes.append(float(np.abs(dd["inferred"] - dd["true"]).mean()))
    H["camera_mae_px_median"] = round(sorted(maes)[len(maes) // 2], 2)
    H["camera_mae_px_best"] = round(min(maes), 2)
    H["camera_mae_px_worst"] = round(max(maes), 2)
    d0 = np.load(seed_files[0])
    fig = plt.figure(figsize=(11.2, 3.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1.1, 1.1, 1.1])
    ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1])
    ax0.imshow(d0["scene"], cmap="gray_r"); ax0.set_title("frame t", fontsize=9.5)
    ax1.imshow(d0["shifted_example"], cmap="gray_r")
    ax1.set_title("frame t+1 (camera moved)", fontsize=9.5)
    for a in (ax0, ax1):
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for k, f in enumerate(seed_files):
        dd = np.load(f)
        ax = fig.add_subplot(gs[2 + k])
        ax.scatter(dd["true"][:, 1], dd["inferred"][:, 1], s=7, alpha=0.45,
                   color=[TEAL, GOLD_DEEP, CLAY][k], linewidths=0)
        ax.plot([-6, 6], [-6, 6], color=MUTED, lw=1.1, ls="--")
        ax.set_title(f"seed {k} · error {maes[k]:.1f} px", fontsize=9)
        ax.set_xlabel("true dy (px)", fontsize=8)
        if k == 0:
            ax.set_ylabel("inferred z (dy)", fontsize=8)
        ax.set_aspect("equal")
    save(fig, "fig_camera_latent.png")

# ================================================================ toy: audio
p = os.path.join(RES, "toy", "audio.npz")
if os.path.exists(p):
    d = np.load(p)
    # standalone modality figure for Part 2 (context + the two continuations)
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.7), constrained_layout=True)
    for ax, (key, t, c) in zip(axes, [
            ("ctx", "context: note A, then silence", INK2),
            ("true_c", "continuation 1: note C", TEAL_DEEP),
            ("true_g", "continuation 2: note G", CLAY_DEEP)]):
        ax.imshow(d[key], cmap="gray_r", origin="lower", aspect="auto")
        ax.set_title(t, fontsize=9.5, color=c)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0].set_ylabel("frequency")
    save(fig, "fig_audio_world.png")

    fig = plt.figure(figsize=(11.6, 3.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.6])
    for i, (key, t) in enumerate([("ctx", "context: first note"),
                                  ("true_c", "continuation 1: note C"),
                                  ("true_g", "continuation 2: note G")]):
        ax = fig.add_subplot(gs[i])
        ax.imshow(d[key], cmap="gray_r", origin="lower", aspect="auto")
        ax.set_title(t, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        if i == 0:
            ax.set_ylabel("frequency")
    ax = fig.add_subplot(gs[3])
    for n, lab, col in [("gen_det", "generative", GOLD),
                        ("jepa_noz", "JEPA, no latent", TEAL_DEEP),
                        ("jepa_z", "JEPA + latent z", TEAL)]:
        c = d[f"curve_{n}"]
        c = (c - c.min()) / max(1e-9, c.max() - c.min())
        ax.plot(d["rows"], c, lw=2.0, color=col, label=lab)
    for r, lab in zip(d["note_rows"], ["C", "G"]):
        ax.axvline(r, color=MUTED, ls=":", lw=1.1)
        ax.annotate(f"note {lab}", xy=(r + 0.4, 1.02), fontsize=9,
                    color=MUTED)
    ax.set_xlabel("pitch of the continuation (spectrogram row)")
    ax.set_ylabel("energy (normalized)")
    ax.legend(fontsize=8.5, loc="upper center")
    save(fig, "fig_audio_energy.png")

with open(os.path.join(FIGS, "headlines.json"), "w") as f:
    json.dump(H, f, indent=2, sort_keys=True)
print(json.dumps(H, indent=2, sort_keys=True))
