"""Shared matplotlib style for Lecture 6 figures — warm-paper serif theme
matching the deck (paper #FCFBF7, teal/gold/clay accents)."""
import matplotlib as mpl

PAPER = "#FCFBF7"
PANEL = "#FFFFFF"
INK = "#16130D"
INK2 = "#3A352B"
MUTED = "#6D665A"
LINE = "#E7E2D5"
TEAL = "#2E8F8F"
TEAL_DEEP = "#1E6B6B"
GOLD = "#DD9F3E"
GOLD_DEEP = "#B37F28"
CLAY = "#C96442"
CLAY_DEEP = "#A94F31"
GOOD = "#4A9467"
PURPLE = "#8D6FC0"
BLUE = "#4A7FA8"

# one color per architecture, used consistently across every figure
ARCH_COLORS = {
    "jea": CLAY,             # the collapse story = clay/red
    "ijepa_nostop": CLAY_DEEP,
    "mae": GOLD,             # generative = gold
    "genmask": GOLD_DEEP,
    "ijepa": TEAL,           # the hero = teal
    "random": MUTED,
}
ARCH_LABELS = {
    "jea": "Joint Embedding (naive)",
    "ijepa_nostop": "I-JEPA, no EMA/stop-grad",
    "mae": "Generative (MAE, pixels)",
    "genmask": "Generative + JEPA masks",
    "ijepa": "I-JEPA",
    "random": "Random encoder",
}


def apply():
    mpl.rcParams.update({
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "text.color": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": LINE,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "serif",
        "font.serif": ["Source Serif 4", "Georgia", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "600",
        "legend.frameon": False,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })
