#!/usr/bin/env python3
"""Parts 1 & 2 of Lecture 5, built from real news text.

Everything here runs on a CPU in a couple of minutes and uses 20 Newsgroups,
which ships with scikit-learn — no download, no API key, no setup. A student
can rerun every figure.

  Part 1  RNN vs transformer      how a model reads a sequence
  Part 2  continuous vs discrete  what a latent variable can be

  python3.13 text_demos.py            # writes ../public/img/fig_t*.png
"""
import os
import re
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "img")
os.makedirs(OUT, exist_ok=True)

PAPER, INK, MUTED = "#FBF9F1", "#16130D", "#6D665A"
TEAL, GOLD, CLAY, BLUE = "#2E8F8F", "#DD9F3E", "#C96442", "#4A7FA8"
rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif", "axes.titlesize": 13, "axes.titleweight": "bold",
})
rng = np.random.default_rng(0)


# ════════════════════════════════════════════════════ the corpus
def load_corpus():
    from sklearn.datasets import fetch_20newsgroups
    cats = ["rec.sport.baseball", "rec.sport.hockey", "talk.politics.misc",
            "sci.space", "comp.graphics", "sci.med"]
    d = fetch_20newsgroups(subset="train", categories=cats,
                           remove=("headers", "footers", "quotes"))
    keep = [(t, y) for t, y in zip(d.data, d.target) if len(t.split()) > 60]
    texts = [t for t, _ in keep]
    labels = np.array([y for _, y in keep])
    names = [c.split(".")[-1] for c in d.target_names]
    print(f"corpus: {len(texts)} articles across {len(names)} topics {names}")
    return texts, labels, names


# ═══════════════════════════════════ PART 2 · continuous vs discrete
def fig_continuous_map(texts, labels, names):
    """Demo C+D — articles as points on a map, and the walk between two of them."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    vec = TfidfVectorizer(max_features=8000, stop_words="english", min_df=3)
    X = vec.fit_transform(texts)
    emb = TruncatedSVD(n_components=64, random_state=0).fit_transform(X)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    xy = TruncatedSVD(n_components=2, random_state=0).fit_transform(emb)

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.0))
    ax = axes[0]
    cols = [TEAL, "#1E6B6B", CLAY, GOLD, BLUE, "#8D6FC0"]
    for i, nm in enumerate(names):
        m = labels == i
        ax.scatter(xy[m, 0], xy[m, 1], s=9, alpha=0.55, color=cols[i], label=nm)
    ax.legend(frameon=False, fontsize=10, loc="best")
    ax.set_title("every article is a point — meaning varies smoothly", loc="left")
    ax.set_xticks([]); ax.set_yticks([])

    # the walk: from the baseball region to the space region, decoded at each
    # step into how much each topic it resembles. Nothing snaps — it slides.
    centro = np.stack([emb[labels == i].mean(0) for i in range(len(names))])
    centro /= (np.linalg.norm(centro, axis=1, keepdims=True) + 1e-9)
    a_i, b_i = names.index("baseball"), names.index("space")
    steps = np.linspace(0, 1, 41)
    sims = []
    for al in steps:
        p = (1 - al) * centro[a_i] + al * centro[b_i]
        p /= np.linalg.norm(p) + 1e-9
        sims.append(centro @ p)
    sims = np.array(sims)

    ax2 = axes[1]
    for i, nm in enumerate(names):
        ax2.plot(steps, sims[:, i], lw=2.6 if i in (a_i, b_i) else 1.4,
                 color=cols[i], alpha=1.0 if i in (a_i, b_i) else 0.45, label=nm)
    ax2.set_xlabel("position along the straight line from baseball to space")
    ax2.set_ylabel("how much the point resembles each topic")
    ax2.set_title("walk between two topics — nothing snaps, everything slides",
                  loc="left")
    ax2.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper center")
    ax2.set_ylim(-0.05, 1.25)
    plt.tight_layout(rect=[0, 0.115, 1, 0.96])
    fig.text(0.735, 0.015, "at 0.5 the point is genuinely half of each —\n"
                           "a real, usable state that is neither article",
             ha="center", va="bottom", fontsize=11.5, color=CLAY, style="italic",
             linespacing=1.35)
    fig.text(0.26, 0.015, "nobody labelled these regions;\n"
                          "closeness means similarity",
             ha="center", va="bottom", fontsize=11.5, color=MUTED, style="italic",
             linespacing=1.35)
    plt.savefig(f"{OUT}/fig_t_continuous.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("fig_t_continuous.png")
    return vec, emb


def fig_discrete_vocabulary(texts, vec):
    """Demo E+F — the same corpus as a vocabulary, and the gap between two words."""
    vocab = np.array(vec.get_feature_names_out())
    fig = plt.figure(figsize=(14.6, 6.4))
    axl = fig.add_axes([0.03, 0.10, 0.47, 0.76]); axl.axis("off")
    NR, NC = 16, 6
    axl.set_title(f"the ENTIRE vocabulary is {len(vocab):,} words — "
                  f"here are {NR*NC} of them, at random", loc="left", fontsize=13)
    show = sorted(rng.choice(vocab, NR * NC, replace=False))
    for r in range(NR):
        axl.text(0.0, 0.95 - r * 0.059,
                 " ".join(f"{w:<12s}" for w in show[r*NC:(r+1)*NC]),
                 fontsize=9.4, family="monospace", color=INK)
    axl.text(0.0, -0.02, "a discrete space is enumerable. a continuous one is not.",
             fontsize=12, color=CLAY, style="italic")

    axr = fig.add_axes([0.58, 0.10, 0.39, 0.76]); axr.axis("off")
    axr.text(0, 0.92, "what is halfway between two words?", fontsize=13,
             fontweight="bold", color=INK)
    axr.text(0.06, 0.74, "baseball", fontsize=20, color=TEAL, family="monospace")
    axr.text(0.06, 0.44, "hockey", fontsize=20, color=TEAL, family="monospace")
    axr.add_patch(FancyArrowPatch((0.17, 0.72), (0.17, 0.50), arrowstyle="<|-|>",
                                  mutation_scale=18, lw=2.2, color=MUTED,
                                  transform=axr.transAxes))
    axr.text(0.30, 0.60, "?", fontsize=44, color=CLAY, fontweight="bold")
    axr.text(0, 0.26, "Nothing. There is no word there.\n"
                      "A discrete space does not interpolate —\n"
                      "and that is the price it charges.",
             fontsize=12.5, color=INK)
    plt.savefig(f"{OUT}/fig_t_discrete.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("fig_t_discrete.png")


def fig_ambiguous_word(texts, vec, emb):
    """Demo G — the pivotal figure: averaging vs choosing."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    # word vectors from co-occurrence, so "average of words" is a real object
    win = 6
    words = ["rain", "snow", "strike", "injury", "fog", "match", "game",
             "weather", "player", "season", "team", "storm"]
    toks = [re.findall(r"[a-z]+", t.lower()) for t in texts[:2500]]
    vocab = {w: i for i, w in enumerate(sorted({w for t in toks for w in t
                                                if len(w) > 2}))}
    co = np.zeros((len(vocab), 300), np.float32)
    common = [w for w, _ in sorted(vocab.items(), key=lambda kv: kv[1])[:300]]
    cidx = {w: i for i, w in enumerate(common)}
    for t in toks:
        for i, w in enumerate(t):
            if w not in vocab:
                continue
            for j in range(max(0, i-win), min(len(t), i+win)):
                if i != j and t[j] in cidx:
                    co[vocab[w], cidx[t[j]]] += 1
    co = np.log1p(co)
    W = TruncatedSVD(n_components=32, random_state=0).fit_transform(co)
    W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)

    have = [w for w in words if w in vocab]
    cand = [w for w in ["rain", "snow", "strike"] if w in vocab]
    fig = plt.figure(figsize=(14.6, 6.6))
    fig.suptitle('"The match was cancelled because of the ____"',
                 fontsize=17, y=0.98)

    # left: the continuous answer — one point, the average
    axl = fig.add_axes([0.05, 0.12, 0.40, 0.70])
    xy = TruncatedSVD(n_components=2, random_state=0).fit_transform(W)
    # Place the three highlighted words FIRST, then drop any grey label that
    # would land on top of one — otherwise "rain" prints over "match".
    span = xy[[vocab[w] for w in have]]
    scale = float(np.abs(span).max()) + 1e-9
    placed = []
    if len(cand) >= 2:
        pts = np.array([xy[vocab[w]] for w in cand])
        for w, p in zip(cand, pts):
            axl.scatter([p[0]], [p[1]], s=120, color=TEAL, zorder=4)
            axl.text(p[0], p[1] + 0.02 * scale, w, fontsize=13, ha="center",
                     color=TEAL, fontweight="bold")
            placed.append((p[0], p[1] + 0.02 * scale))
    for w in have:
        if w in cand:
            continue
        x, y = xy[vocab[w]]
        ly = y + 0.012 * scale
        if any(abs(x - px) < 0.13 * scale and abs(ly - py) < 0.055 * scale
               for px, py in placed):
            axl.scatter([x], [y], s=40, color=MUTED, zorder=3)   # dot only
            continue
        axl.scatter([x], [y], s=40, color=MUTED, zorder=3)
        axl.text(x, ly, w, fontsize=11, ha="center", color=MUTED)
        placed.append((x, ly))
    if len(cand) >= 2:
        mean = pts.mean(0)
        axl.scatter([mean[0]], [mean[1]], s=260, marker="X", color=CLAY, zorder=5)
        axl.annotate("the average", (mean[0], mean[1]),
                     textcoords="offset points", xytext=(0, -26), ha="center",
                     va="top", fontsize=13, color=CLAY, fontweight="bold")
    axl.set_xticks([]); axl.set_yticks([])
    axl.set_title("CONTINUOUS · one Gaussian must output a point", loc="left",
                  fontsize=12.5, color=INK)
    axl.text(0.5, -0.10, "decode that X and you get a blur of a word —\n"
                         "an answer that is none of the three",
             transform=axl.transAxes, ha="center", fontsize=12, color=CLAY)

    # right: the discrete answer — a distribution over real words
    axr = fig.add_axes([0.56, 0.20, 0.38, 0.56])
    probs = [0.50, 0.30, 0.20][:len(cand)]
    axr.barh(range(len(cand)), probs, color=[TEAL, GOLD, CLAY][:len(cand)],
             height=0.52)
    axr.set_yticks(range(len(cand)))
    axr.set_yticklabels(cand, fontsize=15)
    axr.invert_yaxis(); axr.set_xlim(0, 0.62)
    for i, p in enumerate(probs):
        axr.text(p + 0.015, i, f"{p:.2f}", va="center", fontsize=12.5, color=INK)
    axr.set_title("DISCRETE · a distribution over the vocabulary", loc="left",
                  fontsize=12.5, color=INK)
    axr.text(0, -0.30, "honest, multimodal, and every outcome it names\n"
                       "is a real word. nothing in between is invented.",
             transform=axr.transAxes, fontsize=12, color=TEAL)
    plt.savefig(f"{OUT}/fig_t_ambiguous.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("fig_t_ambiguous.png")


# ═══════════════════════════════════════ PART 1 · RNN vs transformer
def long_range_experiment():
    """Demo B — accuracy vs how far back the answer lives. RNN vs attention.

    A synthetic-but-newspaper-shaped task: a passage names a person early, then
    N filler sentences, then asks who was named. Both models see the same data.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    NAMES = ["patel", "okafor", "novak", "silva", "haddad", "ibarra",
             "kaur", "moreau"]
    FILLER = ("the committee met again the following week and the report was "
              "circulated to members before the vote was scheduled").split()
    V = sorted(set(NAMES + FILLER + ["minister", "named", "was", "who",
                                     "resigned", "?"]))
    idx = {w: i for i, w in enumerate(V)}

    def make(n_fill, batch):
        xs, ys = [], []
        for _ in range(batch):
            who = NAMES[np.random.randint(len(NAMES))]
            seq = ["minister", who, "was", "named"]
            for _ in range(n_fill):
                seq.append(FILLER[np.random.randint(len(FILLER))])
            seq += ["who", "resigned", "?"]
            xs.append([idx[w] for w in seq]); ys.append(NAMES.index(who))
        return torch.tensor(xs), torch.tensor(ys)

    class RNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.e = nn.Embedding(len(V), 64); s.r = nn.GRU(64, 64, batch_first=True)
            s.o = nn.Linear(64, len(NAMES))
        def forward(s, x):
            h, _ = s.r(s.e(x)); return s.o(h[:, -1])

    class ATT(nn.Module):
        def __init__(s):
            super().__init__()
            s.e = nn.Embedding(len(V), 64)
            s.p = nn.Parameter(torch.zeros(1, 400, 64))
            s.a = nn.MultiheadAttention(64, 4, batch_first=True)
            s.o = nn.Linear(64, len(NAMES))
        def forward(s, x):
            h = s.e(x) + s.p[:, :x.shape[1]]
            z, _ = s.a(h, h, h, need_weights=False)
            return s.o(z[:, -1])

    fills = [4, 12, 24, 48, 96, 160]
    res = {"rnn": [], "att": []}
    for n_fill in fills:
        for tag, Model in (("rnn", RNN), ("att", ATT)):
            torch.manual_seed(0)
            m = Model(); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            for _ in range(320):
                x, y = make(n_fill, 32)
                loss = nn.functional.cross_entropy(m(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                x, y = make(n_fill, 400)
                acc = float((m(x).argmax(1) == y).float().mean())
            res[tag].append(acc)
            print(f"  gap {n_fill:4d} words · {tag} acc {acc:.2f}", flush=True)

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.plot(fills, res["rnn"], "-o", lw=2.6, color=CLAY, label="RNN · one running summary")
    ax.plot(fills, res["att"], "-o", lw=2.6, color=TEAL, label="attention · looks back")
    ax.axhline(1/8, ls=":", color=MUTED, lw=1.4)
    ax.text(fills[-1], 1/8 + 0.02, "chance", fontsize=10, color=MUTED, ha="right")
    ax.set_xlabel("how many words back the answer was stated")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=11)
    ax.set_title("The bottleneck, measured: \"who was named?\"", loc="left")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_t_longrange.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("fig_t_longrange.png")
    return fills, res


def fig_rnn_summary(texts):
    """Demo A — the RNN's hidden vector as a heatmap, one column per word.
    Deliberately the same visual as Lecture 4's h heatmap on the robot."""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    # a small GRU language model, actually TRAINED on the corpus — an untrained
    # hidden state is noise and would teach nothing.
    corpus = [re.findall(r"[a-zA-Z']+", t.lower()) for t in texts[:600]]
    counts = {}
    for c in corpus:
        for w in c:
            counts[w] = counts.get(w, 0) + 1
    V = ["<unk>"] + sorted([w for w, n in counts.items() if n >= 5])
    idx = {w: i for i, w in enumerate(V)}
    seqs = [[idx.get(w, 0) for w in c] for c in corpus if len(c) > 60]
    e = nn.Embedding(len(V), 48); r = nn.GRU(48, 48, batch_first=True)
    head = nn.Linear(48, len(V))
    opt = torch.optim.Adam(list(e.parameters()) + list(r.parameters())
                           + list(head.parameters()), lr=3e-3)
    L = 48
    for step in range(1500):
        b = []
        for _ in range(32):
            s = seqs[np.random.randint(len(seqs))]
            t = np.random.randint(0, len(s) - L - 1)
            b.append(s[t:t+L+1])
        b = torch.tensor(b)
        h, _ = r(e(b[:, :-1]))
        loss = nn.functional.cross_entropy(
            head(h).reshape(-1, len(V)), b[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"  GRU language model trained · next-word CE {float(loss):.2f}")

    words = [w for w in re.findall(r"[a-zA-Z']+", texts[1].lower())[:44]]
    x = torch.tensor([[idx.get(w, 0) for w in words]])
    with torch.no_grad():
        h, _ = r(e(x))
    H = h[0].numpy().T

    fig = plt.figure(figsize=(15.0, 5.0))
    ax = fig.add_axes([0.06, 0.30, 0.90, 0.56])
    ax.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_yticks([]); ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=90, fontsize=8.5, color=MUTED)
    ax.set_ylabel("the running summary\n(48 numbers)", fontsize=11)
    fig.text(0.06, 0.02, "Everything the model will ever know about word 1 has to survive "
             "inside this one column of numbers until the end of the article.",
             fontsize=12.5, color=INK)
    plt.savefig(f"{OUT}/fig_t_rnn_summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("fig_t_rnn_summary.png")


def fig_attention_example():
    """§1.5 — where a TRAINED attention model actually looks.

    Same task as the long-range experiment. We train the attention model at a
    24-word gap (where the RNN is already at chance), then read out which word
    the question token attends to. Nothing here is a schematic.
    """
    import torch
    import torch.nn as nn
    np.random.seed(0); torch.manual_seed(0)

    NAMES = ["patel", "okafor", "novak", "silva", "haddad", "ibarra",
             "kaur", "moreau"]
    FILLER = ("the committee met again the following week and the report was "
              "circulated to members before the vote was scheduled").split()
    V = sorted(set(NAMES + FILLER + ["minister", "named", "was", "who",
                                     "resigned", "?"]))
    idx = {w: i for i, w in enumerate(V)}
    N_FILL = 24

    def make(batch):
        xs, ys, ws = [], [], []
        for _ in range(batch):
            who = NAMES[np.random.randint(len(NAMES))]
            seq = ["minister", who, "was", "named"]
            for _ in range(N_FILL):
                seq.append(FILLER[np.random.randint(len(FILLER))])
            seq += ["who", "resigned", "?"]
            xs.append([idx[w] for w in seq]); ys.append(NAMES.index(who))
            ws.append(seq)
        return torch.tensor(xs), torch.tensor(ys), ws

    class ATT(nn.Module):
        def __init__(s):
            super().__init__()
            s.e = nn.Embedding(len(V), 64)
            s.p = nn.Parameter(torch.zeros(1, 400, 64))
            s.a = nn.MultiheadAttention(64, 4, batch_first=True)
            s.o = nn.Linear(64, len(NAMES))
        def forward(s, x, need_w=False):
            h = s.e(x) + s.p[:, :x.shape[1]]
            z, w = s.a(h, h, h, need_weights=need_w, average_attn_weights=True)
            return s.o(z[:, -1]), w

    m = ATT(); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(320):
        x, y, _ = make(32)
        loss = nn.functional.cross_entropy(m(x)[0], y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x, y, ws = make(256)
        acc = float((m(x)[0].argmax(1) == y).float().mean())
    print(f"  trained attention model · accuracy {acc:.2f} at a {N_FILL}-word gap")

    with torch.no_grad():
        xs, ys, ws = make(1)
        _, W = m(xs, need_w=True)
    w_last = W[0, -1].numpy()               # what the "?" token looks at
    seq = ws[0]
    name_pos = 1

    fig = plt.figure(figsize=(15.4, 5.6))
    ax = fig.add_axes([0.05, 0.34, 0.92, 0.50])
    cols_bar = [CLAY if i == name_pos else "#C9C3B4" for i in range(len(seq))]
    ax.bar(range(len(seq)), w_last, color=cols_bar, width=0.72)
    ax.set_xticks(range(len(seq)))
    ax.set_xticklabels(seq, rotation=90, fontsize=9.5,
                       color=MUTED)
    ax.get_xticklabels()[name_pos].set_color(CLAY)
    ax.get_xticklabels()[name_pos].set_fontweight("bold")
    ax.set_ylabel("attention weight", fontsize=11)
    ax.axhline(1/len(seq), ls=":", lw=1.3, color=MUTED)
    ax.text(len(seq)-0.5, -float(w_last.max())*0.055,
            "the dotted line is what an even spread would look like",
            ha="right", va="top", fontsize=11, color=MUTED)
    ax.set_ylim(0, float(w_last.max()) * 1.30)
    ax.annotate(f"the name — {w_last[name_pos]*100:.0f}% of the attention",
                xy=(name_pos + 0.4, w_last[name_pos] * 1.02),
                xytext=(name_pos + 4.0, float(w_last.max()) * 1.20),
                fontsize=13, color=CLAY, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", color=CLAY, lw=1.8,
                                connectionstyle="arc3,rad=-0.2"))
    fig.suptitle('Where the question token "?" actually looks — '
                 "a trained attention model", fontsize=15.5, y=0.98)
    fig.text(0.05, 0.045,
             f"The name sits 27 words back, past {N_FILL} words of filler. The model puts "
             f"{w_last[name_pos]*100:.0f}% of its attention on it and answers correctly "
             f"{acc*100:.0f}% of the time.\n"
             "Nothing was compressed away, so nothing had to be remembered — "
             "it simply looked.", fontsize=12, color=INK)
    plt.savefig(f"{OUT}/fig_t_attention.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("fig_t_attention.png")


if __name__ == "__main__":
    t0 = time.time()
    texts, labels, names = load_corpus()
    print("\n— Part 1 · how a model reads a sequence")
    fig_rnn_summary(texts)
    fig_attention_example()
    long_range_experiment()
    print("\n— Part 2 · what a latent can be")
    vec, emb = fig_continuous_map(texts, labels, names)
    fig_discrete_vocabulary(texts, vec)
    fig_ambiguous_word(texts, vec, emb)
    print(f"\nall text figures written in {time.time()-t0:.0f}s")
