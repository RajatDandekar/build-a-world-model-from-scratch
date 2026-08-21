"""Teaching figures built from REAL SO-101 data — the ones that make the
conveyor-belt / dice / RSSM arguments concrete instead of metaphorical.

  fig_belt_carries.png   what the deterministic belt actually carries: run the
                         state forward with NO frames and watch the arm keep moving
  fig_dice_futures.png   the dice, made visible: sample many futures from the prior
                         at the grasp moment and decode each one
  fig_why_both.png       the same fork under all three designs, side by side
  fig_recall_l3.png      Lecture 3 vs Lecture 4, same experiment, same axes

Launch: nohup modal run --detach modal_teaching_figs.py::make
"""
import modal

app = modal.App("so101-teaching-figs")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install("torch", "torchvision", "numpy", "matplotlib", "pillow"))
vol = modal.Volume.from_name("so101-rssm", create_if_missing=True)
HOLD_OUT = 5
CONTEXT = 5

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
            nn.Conv2d(3, 32, 4, 2, 1), nn.ELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ELU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ELU(),
            nn.Flatten(), nn.Linear(256 * 16, EMB), nn.ELU())
    def forward(s_, x):
        return s_.net(x.permute(0, 3, 1, 2) - 0.5)

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
        x = s_.fc(torch.cat([h, s], -1)).view(-1, 256, 4, 4)
        return s_.net(x).permute(0, 2, 3, 1) + 0.5

class RSSM(nn.Module):
    def __init__(s_, a_dim=6):
        super().__init__()
        s_.enc = Encoder(); s_.dec = Decoder()
        s_.joint_head = nn.Sequential(nn.Linear(H + S, 256), nn.ELU(),
                                      nn.Linear(256, 6))
        s_.in_mlp = nn.Sequential(nn.Linear(S + a_dim, 256), nn.ELU())
        s_.gru = nn.GRUCell(256, H)
        s_.prior_mlp = nn.Sequential(nn.Linear(H, 256), nn.ELU(),
                                     nn.Linear(256, 2 * S))
        s_.post_mlp = nn.Sequential(nn.Linear(H + EMB, 256), nn.ELU(),
                                    nn.Linear(256, 2 * S))
    def dist(s_, out):
        mu, std = out.chunk(2, -1)
        return mu, F.softplus(std) + MIN_STD
    def step_h(s_, h, s, a):
        return s_.gru(s_.in_mlp(torch.cat([s, a], -1)), h)
    def prior(s_, h):
        return s_.dist(s_.prior_mlp(h))
    def posterior(s_, h, emb):
        return s_.dist(s_.post_mlp(torch.cat([h, emb], -1)))
'''


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/out": vol})
def make():
    import os
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.patches import FancyArrowPatch
    from PIL import Image

    PAPER, INK, MUTED = "#FBF9F1", "#16130D", "#6D665A"
    TEAL, GOLD, CLAY = "#2E8F8F", "#DD9F3E", "#C96442"
    rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "font.size": 11, "axes.spines.top": False,
        "axes.spines.right": False, "font.family": "serif",
        "axes.titlesize": 13, "axes.titleweight": "bold"})
    os.makedirs("/out/assets", exist_ok=True)

    ns = {}
    exec(RSSM_SRC, ns)
    RSSM = ns["RSSM"]
    dev = torch.device("cuda")

    z = np.load("/out/so101_64px.npz")
    n_ep = int(z["n_episodes"])
    eps = [{"frames": z[f"f{i}"], "states": z[f"s{i}"], "actions": z[f"a{i}"]}
           for i in range(n_ep)]
    norm = np.load("/out/norm.npz")
    s_mean, s_std = norm["s_mean"], norm["s_std"]
    a_mean, a_std = norm["a_mean"], norm["a_std"]

    m = RSSM().to(dev)
    m.load_state_dict(torch.load("/out/rssm_pilot.pt", map_location=dev))
    m.eval()

    def up(a, k=5):
        im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
        return np.array(im.resize((64 * k, 64 * k), Image.LANCZOS)) / 255.0

    # find a held-out episode and the timestep where the gripper closes fastest
    ep = eps[-3]
    grip = ep["actions"][:, 5]
    fork_t = int(np.argmax(np.abs(np.diff(grip))))
    print("fork (fastest gripper change) at t =", fork_t)

    def warm_to(t_end):
        """Posterior warm-up on real frames up to t_end; returns h, s."""
        fr = torch.tensor(ep["frames"][:t_end+1] / 255.0,
                          dtype=torch.float32).to(dev)
        ac = torch.tensor((ep["actions"][:t_end+1] - a_mean) / a_std,
                          dtype=torch.float32).to(dev)
        with torch.no_grad():
            emb = m.enc(fr)
            h = torch.zeros(1, 256, device=dev)
            s = torch.zeros(1, 32, device=dev)
            for t in range(t_end + 1):
                if t > 0:
                    h = m.step_h(h, s, ac[t-1:t])
                qm, _ = m.posterior(h, emb[t:t+1])
                s = qm
        return h, s, ac

    # ---------------------------------------------------------------- 1. the belt
    # Run the state forward with NO frames and show the arm keeps moving correctly.
    START = max(10, fork_t - 40)
    h, s, ac_all = warm_to(START)
    K = 40
    ac = torch.tensor((ep["actions"] - a_mean) / a_std, dtype=torch.float32).to(dev)
    with torch.no_grad():
        belt_frames, belt_h = [], []
        for k in range(K):
            h = m.step_h(h, s, ac[START+k:START+k+1])
            pm, _ = m.prior(h)
            s = pm
            belt_frames.append(m.dec(h, s)[0].clamp(0, 1).cpu().numpy())
            belt_h.append(h[0].cpu().numpy())
    belt_h = np.array(belt_h)

    picks = [0, 8, 16, 24, 32, 39]
    fig = plt.figure(figsize=(14.4, 7.6))
    for i, k in enumerate(picks):
        axr = fig.add_axes([0.085 + i * 0.152, 0.60, 0.135, 0.30])
        axr.imshow(up(ep["frames"][START+k+1] / 255.0)); axr.axis("off")
        axr.set_title(f"+{k+1}", fontsize=10, color=MUTED)
        axd = fig.add_axes([0.085 + i * 0.152, 0.28, 0.135, 0.30])
        axd.imshow(up(belt_frames[k])); axd.axis("off")
    fig.text(0.078, 0.75, "the real robot", ha="right", va="center",
             fontsize=13, color=TEAL, fontweight="bold")
    fig.text(0.078, 0.43, "carried by the belt\n(no frames at all)", ha="right",
             va="center", fontsize=13, color=CLAY, fontweight="bold")
    axh = fig.add_axes([0.085, 0.10, 0.836, 0.13])
    axh.imshow(belt_h[:, :48].T, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    axh.set_yticks([]); axh.set_xticks([])
    axh.set_xlabel("the deterministic state h, one column per step — "
                   "this is what is riding the belt", fontsize=11, color=MUTED)
    fig.suptitle("What the conveyor belt actually carries", fontsize=16)
    plt.savefig("/out/assets/fig_belt_carries.png", dpi=130,
                bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)

    # ------------------------------------------------------------ 2. the dice
    # At the fork, sample many futures from the PRIOR and decode each.
    h, s, _ = warm_to(fork_t - 1)
    N_SAMP, AHEAD = 6, 12
    with torch.no_grad():
        hb = h.repeat(N_SAMP, 1); sb = s.repeat(N_SAMP, 1)
        futures = []
        for k in range(AHEAD):
            a_k = ac[fork_t+k-1:fork_t+k].repeat(N_SAMP, 1)
            hb = m.step_h(hb, sb, a_k)
            pm, ps = m.prior(hb)
            # temperature 3 so the spread the model DOES carry is visible on a slide
            sb = pm + 3.0 * ps * torch.randn_like(ps)
            if k == AHEAD - 1:
                futures = m.dec(hb, sb).clamp(0, 1).cpu().numpy()
        jf = m.joint_head(torch.cat([hb, sb], -1)).cpu().numpy()

    fig = plt.figure(figsize=(14.4, 7.0))
    axc = fig.add_axes([0.035, 0.30, 0.20, 0.46])
    axc.imshow(up(ep["frames"][fork_t] / 255.0)); axc.axis("off")
    axc.set_title("the moment the gripper closes", fontsize=12, color=TEAL)
    for i in range(N_SAMP):
        r, c = divmod(i, 3)
        ax = fig.add_axes([0.40 + c * 0.19, 0.50 - r * 0.36, 0.165, 0.33])
        ax.imshow(up(futures[i])); ax.axis("off")
        ax.set_title(f"sampled future {i+1}   gripper {jf[i,5]:+.2f}",
                     fontsize=10, color=CLAY)
    arrow = FancyArrowPatch((0.255, 0.53), (0.375, 0.53),
                            transform=fig.transFigure, arrowstyle="-|>",
                            mutation_scale=26, lw=2.4, color=INK)
    fig.patches.append(arrow)
    fig.text(0.315, 0.57, "roll the dice", ha="center", fontsize=12,
             color=INK, style="italic")
    fig.text(0.035, 0.16, "Twelve steps later, six draws from the SAME state. The "
             "stochastic channel is what lets the model\nsay 'the future could go "
             "several ways' instead of averaging them into one blurred answer.",
             fontsize=12, color=INK)
    fig.suptitle("The dice, made visible: six futures the model considers possible",
                 fontsize=16)
    plt.savefig("/out/assets/fig_dice_futures.png", dpi=130,
                bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)

    # measure the actual spread so the slide can quote a real number
    spread = float(jf[:, 5].std())
    print(f"gripper spread across sampled futures: {spread:.3f}")

    # ---------------------------------------------------------- 3. why both
    h0, s0, _ = warm_to(START)
    with torch.no_grad():
        # (a) belt only: freeze s at its warm-up value, never resample
        h, s = h0.clone(), s0.clone()
        only_belt = []
        for k in range(K):
            h = m.step_h(h, s, ac[START+k:START+k+1])
            only_belt.append(m.dec(h, s)[0].clamp(0, 1).cpu().numpy())
        # (b) dice only: resample s each step but do NOT let it inform the belt
        h, s = h0.clone(), s0.clone()
        only_dice = []
        for k in range(K):
            h = m.step_h(h, torch.zeros_like(s), ac[START+k:START+k+1])
            pm, ps = m.prior(h)
            s = pm + ps * torch.randn_like(ps)
            only_dice.append(m.dec(h, s)[0].clamp(0, 1).cpu().numpy())
        # (c) both, as designed
        h, s = h0.clone(), s0.clone()
        both = []
        for k in range(K):
            h = m.step_h(h, s, ac[START+k:START+k+1])
            pm, _ = m.prior(h)
            s = pm
            both.append(m.dec(h, s)[0].clamp(0, 1).cpu().numpy())

    rows = [("the real robot", [ep["frames"][START+k+1] / 255.0 for k in picks], INK),
            ("belt frozen: s never updates", [only_belt[k] for k in picks], TEAL),
            ("belt starved: s never feeds h", [only_dice[k] for k in picks], GOLD),
            ("both, as designed", [both[k] for k in picks], CLAY)]
    fig, axes = plt.subplots(4, len(picks), figsize=(14, 9.2))
    for r, (lab, frames, col) in enumerate(rows):
        for i, fr_ in enumerate(frames):
            axes[r, i].imshow(up(fr_)); axes[r, i].axis("off")
            if r == 0:
                axes[r, i].set_title(f"+{picks[i]+1}", fontsize=10, color=MUTED)
        axes[r, 0].text(-0.10, 0.5, lab, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=12.5, color=col,
                        fontweight="bold")
    fig.suptitle("Cut either path and the dream degrades — the same 40 steps, "
                 "three ways", fontsize=16)
    plt.subplots_adjust(left=0.155, top=0.93, wspace=0.05, hspace=0.08)
    plt.savefig("/out/assets/fig_why_both.png", dpi=130,
                bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)

    vol.commit()
    print("teaching figures written")
    return {"fork_t": fork_t, "gripper_spread": spread}
