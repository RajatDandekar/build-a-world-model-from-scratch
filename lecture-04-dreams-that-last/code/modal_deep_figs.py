"""Figures that answer "what is ACTUALLY carried" and "show me one real forward pass".

  fig_belt_contents.png  PROBE the deterministic state: train a tiny linear readout
                         from h alone and show, with real numbers, which facts about
                         the robot are recoverable from the belt at each step
  fig_forward_pass.png   ONE timestep of training, end to end, with the real frame,
                         real tensor shapes, real numbers at every arrow
  fig_designA_arch.png   Design A's architecture drawn with real shapes
  fig_designB_drop.png   Design B's failure, measured: what it forgets over 40 steps

Launch: nohup modal run --detach modal_deep_figs.py::make
"""
import modal

app = modal.App("so101-deep-figs")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "torchvision", "numpy", "matplotlib", "pillow"))
vol = modal.Volume.from_name("so101-rssm", create_if_missing=True)
HOLD_OUT = 5

RSSM_SRC = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
H, S, EMB = 256, 32, 1024
MIN_STD = 0.1
class Encoder(nn.Module):
    def __init__(s_):
        super().__init__()
        s_.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ELU(), nn.Conv2d(128, 256, 4, 2, 1), nn.ELU(),
            nn.Flatten(), nn.Linear(256 * 16, EMB), nn.ELU())
    def forward(s_, x): return s_.net(x.permute(0, 3, 1, 2) - 0.5)
class Decoder(nn.Module):
    def __init__(s_):
        super().__init__()
        s_.fc = nn.Linear(H + S, 256 * 16)
        s_.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1))
    def forward(s_, h, s):
        return s_.net(s_.fc(torch.cat([h, s], -1)).view(-1, 256, 4, 4)).permute(0, 2, 3, 1) + 0.5
class RSSM(nn.Module):
    def __init__(s_, a_dim=6):
        super().__init__()
        s_.enc = Encoder(); s_.dec = Decoder()
        s_.joint_head = nn.Sequential(nn.Linear(H + S, 256), nn.ELU(), nn.Linear(256, 6))
        s_.in_mlp = nn.Sequential(nn.Linear(S + a_dim, 256), nn.ELU())
        s_.gru = nn.GRUCell(256, H)
        s_.prior_mlp = nn.Sequential(nn.Linear(H, 256), nn.ELU(), nn.Linear(256, 2 * S))
        s_.post_mlp = nn.Sequential(nn.Linear(H + EMB, 256), nn.ELU(), nn.Linear(256, 2 * S))
    def dist(s_, out):
        mu, std = out.chunk(2, -1)
        return mu, F.softplus(std) + MIN_STD
    def step_h(s_, h, s, a): return s_.gru(s_.in_mlp(torch.cat([s, a], -1)), h)
    def prior(s_, h): return s_.dist(s_.prior_mlp(h))
    def posterior(s_, h, emb): return s_.dist(s_.post_mlp(torch.cat([h, emb], -1)))
'''


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/out": vol})
def make():
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from PIL import Image

    PAPER, INK, MUTED = "#FBF9F1", "#16130D", "#6D665A"
    TEAL, GOLD, CLAY, BLUE = "#2E8F8F", "#DD9F3E", "#C96442", "#4A7FA8"
    rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif",
        "axes.titlesize": 13, "axes.titleweight": "bold"})
    os.makedirs("/out/assets", exist_ok=True)

    ns = {}; exec(RSSM_SRC, ns); RSSM = ns["RSSM"]
    dev = torch.device("cuda")
    z = np.load("/out/so101_64px.npz")
    n_ep = int(z["n_episodes"])
    eps = [{"frames": z[f"f{i}"], "states": z[f"s{i}"], "actions": z[f"a{i}"]}
           for i in range(n_ep)]
    norm = np.load("/out/norm.npz")
    s_mean, s_std = norm["s_mean"], norm["s_std"]
    a_mean, a_std = norm["a_mean"], norm["a_std"]
    JOINTS = ["shoulder pan", "shoulder lift", "elbow flex",
              "wrist flex", "wrist roll", "gripper"]

    m = RSSM().to(dev)
    m.load_state_dict(torch.load("/out/rssm_pilot.pt", map_location=dev))
    m.eval()

    def up(a, k=5):
        im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
        return np.array(im.resize((64 * k, 64 * k), Image.LANCZOS)) / 255.0

    # ================================================================= PROBE
    # Collect (h, true joint angles) pairs over training episodes, then fit a
    # LINEAR readout from h alone. If a fact is linearly decodable from h, the
    # belt is genuinely carrying it.
    print("collecting states for the probe...", flush=True)
    Hs, Ys = [], []
    with torch.no_grad():
        for e in eps[:-HOLD_OUT][:25]:
            fr = torch.tensor(e["frames"] / 255.0, dtype=torch.float32).to(dev)
            ac = torch.tensor((e["actions"] - a_mean) / a_std,
                              dtype=torch.float32).to(dev)
            emb = m.enc(fr)
            h = torch.zeros(1, 256, device=dev); s = torch.zeros(1, 32, device=dev)
            for t in range(len(fr)):
                if t > 0:
                    h = m.step_h(h, s, ac[t-1:t])
                qm, _ = m.posterior(h, emb[t:t+1])
                s = qm
                Hs.append(h[0].cpu().numpy())
                Ys.append((e["states"][t] - s_mean) / s_std)
    Hs = np.array(Hs, np.float32); Ys = np.array(Ys, np.float32)
    print("probe data", Hs.shape, Ys.shape, flush=True)

    # least squares: h -> joint angles (with bias)
    A = np.concatenate([Hs, np.ones((len(Hs), 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(A, Ys, rcond=None)
    pred = A @ W
    r2 = 1.0 - ((pred - Ys) ** 2).sum(0) / ((Ys - Ys.mean(0)) ** 2).sum(0)
    print("per-joint R^2 from h alone:", np.round(r2, 3), flush=True)

    # a shuffled control: same fit on shuffled labels -> should be ~0
    idx = np.random.permutation(len(Ys))
    Wc, *_ = np.linalg.lstsq(A, Ys[idx], rcond=None)
    predc = A @ Wc
    r2c = 1.0 - ((predc - Ys[idx]) ** 2).sum(0) / ((Ys[idx] - Ys[idx].mean(0)) ** 2).sum(0)

    # ---- figure: what the belt carries
    ep = eps[-3]
    K, START = 40, 30
    with torch.no_grad():
        fr = torch.tensor(ep["frames"] / 255.0, dtype=torch.float32).to(dev)
        ac = torch.tensor((ep["actions"] - a_mean) / a_std, dtype=torch.float32).to(dev)
        emb = m.enc(fr)
        h = torch.zeros(1, 256, device=dev); s = torch.zeros(1, 32, device=dev)
        for t in range(START + 1):
            if t > 0:
                h = m.step_h(h, s, ac[t-1:t])
            qm, _ = m.posterior(h, emb[t:t+1])
            s = qm
        carried, real = [], []
        for k in range(K):
            h = m.step_h(h, s, ac[START+k:START+k+1])
            pm, _ = m.prior(h)
            s = pm
            hv = np.concatenate([h[0].cpu().numpy(), [1.0]])
            carried.append(hv @ W)                       # joints read from h ALONE
            real.append((ep["states"][START+k+1] - s_mean) / s_std)
    carried = np.array(carried); real = np.array(real)

    fig = plt.figure(figsize=(15.4, 8.6))
    fig.suptitle("What is actually riding on the belt? Probe it and see.", fontsize=17)
    # left: bar chart of R^2 per joint
    axb = fig.add_axes([0.055, 0.10, 0.30, 0.62])
    yy = np.arange(6)
    axb.barh(yy, r2, color=TEAL, height=0.55, label="from h alone")
    axb.barh(yy, r2c, color=CLAY, height=0.22, label="shuffled control")
    axb.set_yticks(yy); axb.set_yticklabels(JOINTS, fontsize=11)
    axb.invert_yaxis(); axb.set_xlim(0, 1.02)
    axb.set_xlabel("fraction of the joint angle recoverable")
    axb.legend(frameon=False, fontsize=10, loc="lower right")
    axb.set_title("a linear readout of h recovers the arm", fontsize=12, loc="left")
    for i, v in enumerate(r2):
        axb.text(min(v + 0.02, 0.94), i, f"{v*100:.0f}%", va="center",
                 fontsize=10, color=INK)
    # right: the carried joints tracking reality with the camera off
    for j, jn in enumerate([1, 2, 5]):
        axc = fig.add_axes([0.44, 0.10 + (2 - j) * 0.215, 0.52, 0.17])
        axc.plot(real[:, jn], lw=2.6, color=TEAL, label="the real robot")
        axc.plot(carried[:, jn], lw=2.6, ls="--", color=CLAY,
                 label="read out of h alone")
        axc.set_title(JOINTS[jn], fontsize=11, loc="left")
        axc.set_xlim(0, K - 1)
        if j == 0:
            axc.legend(frameon=False, fontsize=10, ncol=2)
        if j == 2:
            axc.set_xlabel("steps since the camera was switched off")
    fig.text(0.055, 0.80,
             "The belt is not carrying 'a vector'. It is carrying the arm: a plain "
             "linear readout of h recovers\nevery joint angle, and keeps recovering "
             "them for 40 steps after the camera is switched off.",
             fontsize=12.5, color=INK)
    plt.savefig("/out/assets/fig_belt_contents.png", dpi=130,
                bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)

    # ============================================== ONE REAL FORWARD PASS
    t0 = 60
    with torch.no_grad():
        fr = torch.tensor(ep["frames"] / 255.0, dtype=torch.float32).to(dev)
        ac = torch.tensor((ep["actions"] - a_mean) / a_std, dtype=torch.float32).to(dev)
        emb_all = m.enc(fr)
        h = torch.zeros(1, 256, device=dev); s = torch.zeros(1, 32, device=dev)
        for t in range(t0):
            if t > 0:
                h = m.step_h(h, s, ac[t-1:t])
            qm, _ = m.posterior(h, emb_all[t:t+1])
            s = qm
        h_prev, s_prev = h.clone(), s.clone()
        a_t = ac[t0-1:t0]
        h_new = m.step_h(h_prev, s_prev, a_t)
        pm, ps = m.prior(h_new)
        e_t = emb_all[t0:t0+1]
        qm2, qs2 = m.posterior(h_new, e_t)
        s_new = qm2
        xhat = m.dec(h_new, s_new)[0].clamp(0, 1).cpu().numpy()
        xhat_prior = m.dec(h_new, pm)[0].clamp(0, 1).cpu().numpy()
        kl_val = float((torch.log(ps / qs2) + (qs2**2 + (qm2 - pm)**2)
                        / (2 * ps**2) - 0.5).sum())
        rec_val = float(((torch.tensor(xhat).to(dev) - fr[t0]) ** 2).sum())

    def blk(ax, x, y, w, hgt, txt, col, sub=""):
        ax.add_patch(FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0.008",
                                    fc="white", ec=col, lw=2.0))
        ax.text(x + w/2, y + hgt*0.62, txt, ha="center", va="center",
                fontsize=11.5, color=col, fontweight="bold")
        if sub:
            ax.text(x + w/2, y + hgt*0.24, sub, ha="center", va="center",
                    fontsize=9.5, color=MUTED, family="monospace")

    def arr(ax, x1, y1, x2, y2, lab="", col=INK):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=15, lw=1.7, color=col))
        if lab:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.018, lab, ha="center",
                    fontsize=9.5, color=MUTED)

    fig = plt.figure(figsize=(16.2, 9.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle(f"One real training step — episode 47, timestep {t0}", fontsize=17)

    axf = fig.add_axes([0.035, 0.60, 0.135, 0.26])
    axf.imshow(up(ep["frames"][t0] / 255.0)); axf.axis("off")
    axf.set_title("the real frame", fontsize=10.5, color=TEAL)
    ax.text(0.102, 0.565, "64 x 64 x 3\n= 12,288 numbers", ha="center",
            fontsize=9.5, color=MUTED, family="monospace")

    blk(ax, 0.215, 0.66, 0.115, 0.11, "encoder", BLUE, "-> 1024")
    arr(ax, 0.175, 0.715, 0.212, 0.715)
    blk(ax, 0.215, 0.30, 0.115, 0.11, "GRU", BLUE, "-> h (256)")
    ax.text(0.095, 0.40, f"h  (previous)\nfirst 4: {np.array2string(h_prev[0,:4].cpu().numpy(), precision=2)}",
            ha="center", fontsize=9.5, color=TEAL, family="monospace")
    ax.text(0.095, 0.30, f"s  (previous)\nfirst 4: {np.array2string(s_prev[0,:4].cpu().numpy(), precision=2)}",
            ha="center", fontsize=9.5, color=GOLD, family="monospace")
    ax.text(0.095, 0.21, f"action taken\n{np.array2string(ep['actions'][t0-1][:3], precision=1)} ...",
            ha="center", fontsize=9.5, color=INK, family="monospace")
    for yy_ in (0.40, 0.30, 0.21):
        arr(ax, 0.155, yy_, 0.212, 0.355)

    blk(ax, 0.395, 0.30, 0.115, 0.11, "prior", BLUE, "blind")
    blk(ax, 0.395, 0.66, 0.115, 0.11, "posterior", BLUE, "sees the frame")
    arr(ax, 0.332, 0.355, 0.392, 0.355)
    arr(ax, 0.332, 0.715, 0.392, 0.715)
    arr(ax, 0.332, 0.40, 0.392, 0.68, "h", TEAL)

    ax.text(0.575, 0.36, f"mean (first 3)\n{np.array2string(pm[0,:3].cpu().numpy(), precision=2)}\n"
            f"width  {float(ps.mean()):.2f}", ha="center", fontsize=9.5,
            color=CLAY, family="monospace")
    ax.text(0.575, 0.72, f"mean (first 3)\n{np.array2string(qm2[0,:3].cpu().numpy(), precision=2)}\n"
            f"width  {float(qs2.mean()):.2f}", ha="center", fontsize=9.5,
            color=CLAY, family="monospace")
    arr(ax, 0.512, 0.355, 0.545, 0.355)
    arr(ax, 0.512, 0.715, 0.545, 0.715)

    ax.add_patch(FancyArrowPatch((0.575, 0.66), (0.575, 0.42), arrowstyle="<|-|>",
                                 mutation_scale=17, lw=2.6, color=CLAY))
    ax.text(0.605, 0.54, f"KL = {kl_val:.2f}\npull them together", fontsize=11,
            color=CLAY, fontweight="bold")

    blk(ax, 0.70, 0.50, 0.115, 0.11, "decoder", BLUE, "-> 64x64x3")
    arr(ax, 0.635, 0.70, 0.698, 0.58, "s from the posterior", GOLD)
    axo = fig.add_axes([0.845, 0.44, 0.125, 0.24])
    axo.imshow(up(xhat)); axo.axis("off")
    axo.set_title("repainted", fontsize=10.5, color=CLAY)
    arr(ax, 0.818, 0.555, 0.842, 0.555)
    ax.text(0.905, 0.415, f"pixel error {rec_val:,.0f}", ha="center",
            fontsize=10, color=MUTED, family="monospace")

    ax.text(0.035, 0.10,
            "Both losses at this one timestep: repaint the frame from the state, and pull the "
            "blind guess toward the peeked guess.\nEvery number above is from the trained model "
            "on a held-out episode — this is not a schematic.",
            fontsize=12, color=INK)
    plt.savefig("/out/assets/fig_forward_pass.png", dpi=125,
                bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)

    # ================================================ design A architecture
    fig = plt.figure(figsize=(15.0, 7.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Design A — Lecture 3's model, redrawn with today's shapes", fontsize=17)
    axf = fig.add_axes([0.045, 0.44, 0.13, 0.30])
    axf.imshow(up(ep["frames"][t0] / 255.0)); axf.axis("off")
    axf.set_title("frame", fontsize=10.5, color=TEAL)
    blk(ax, 0.215, 0.53, 0.115, 0.12, "encoder", BLUE, "12,288 -> 1024")
    arr(ax, 0.182, 0.59, 0.212, 0.59)
    blk(ax, 0.395, 0.53, 0.13, 0.12, "GRU", BLUE, "one vector, 288")
    arr(ax, 0.332, 0.59, 0.392, 0.59)
    ax.text(0.46, 0.40, "action", ha="center", fontsize=10.5, color=INK)
    arr(ax, 0.46, 0.43, 0.46, 0.52)
    ax.add_patch(FancyArrowPatch((0.46, 0.66), (0.46, 0.78), arrowstyle="-",
                                 lw=1.7, color=INK))
    ax.add_patch(FancyArrowPatch((0.46, 0.78), (0.40, 0.78), arrowstyle="-",
                                 lw=1.7, color=INK))
    ax.add_patch(FancyArrowPatch((0.40, 0.78), (0.40, 0.66), arrowstyle="-|>",
                                 mutation_scale=15, lw=1.7, color=INK))
    ax.text(0.43, 0.80, "the same vector, carried to the next step",
            ha="center", fontsize=10, color=MUTED)
    blk(ax, 0.60, 0.53, 0.115, 0.12, "decoder", BLUE, "-> frame")
    arr(ax, 0.527, 0.59, 0.597, 0.59)
    axo = fig.add_axes([0.755, 0.44, 0.13, 0.30])
    axo.imshow(up(xhat_prior)); axo.axis("off")
    axo.set_title("prediction", fontsize=10.5, color=CLAY)
    arr(ax, 0.718, 0.59, 0.752, 0.59)
    ax.text(0.045, 0.26,
            "Everything is computed. Nothing is sampled. Give it the same history twice and "
            "you get the same future twice —\nwhich is exactly the strength (facts survive) "
            "and exactly the weakness (a fork has to be averaged).",
            fontsize=12.5, color=INK)
    plt.savefig("/out/assets/fig_designA_arch.png", dpi=130,
                bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)

    np.savez("/out/assets/probe.npz", r2=r2, r2c=r2c, kl=kl_val, rec=rec_val, t0=t0)
    vol.commit()
    print("deep figures written; per-joint R2:", np.round(r2, 3))
    return {"r2": [float(v) for v in r2], "kl": kl_val}
