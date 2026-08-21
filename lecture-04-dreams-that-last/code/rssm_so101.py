# %% [markdown]
# # Dreams That Last — building an RSSM on a real robot
#
# **Companion notebook for Lecture 4 of *Build a World Model from Scratch* (Vizuara AI).**
#
# In Lecture 3 we built a world model that could predict the next frame almost
# perfectly — and then watched its dream fall apart the moment it had to feed on its
# own predictions. This notebook fixes that, and it does it on **real robot data**:
# 50 tele-operated pick-and-place episodes from an SO-101 arm.
#
# By the end you will have a model that, given five frames of context and then
# **nothing but the joint commands**, imagines two full seconds of robot motion —
# and stays locked to reality the whole way.
#
# ![the model dreaming](https://raw.githubusercontent.com/RajatDandekar/build-a-world-model-from-scratch/main/lecture-04-dreams-that-last/assets/opener.gif)
#
# ---
#
# ### How to use this notebook
#
# * **Runtime → Run all** works end to end. On a free Colab **GPU** the training
#   section takes about 15 minutes; on CPU, skip training and load our checkpoint
#   (the notebook does this automatically).
# * It is written as a **guide**: each section explains what we are about to do and
#   *why*, then shows the code, then tells you what to look for.
# * Everything downloads itself — the data subset, the trained checkpoint, and the
#   lecture figures. No accounts, no keys, no setup.
#
# ### The three questions (the lecture's structure, kept)
#
# 1. **What does the model track?** Not a state — a *belief*, a distribution over states.
# 2. **How is the machine built?** Two designs that fail, then the one that works: the RSSM.
# 3. **How is it trained?** Two loss terms, and neither one is optional.

# %% [markdown]
# ## Setup — grab the data, the checkpoint and the figures
#
# We use a 12-episode subset of `lerobot/svla_so101_pickplace` (the full dataset is
# 50 episodes; the last few here are ones the model never trained on, so our
# evaluations stay honest).

# %%
import os
import urllib.request

BASE = ("https://raw.githubusercontent.com/RajatDandekar/"
        "build-a-world-model-from-scratch/main/lecture-04-dreams-that-last/")

for path in ["data/so101_mini.npz", "data/so101_norm.npz", "data/rssm_so101.pt"]:
    if not os.path.exists(os.path.basename(path)):
        print("downloading", path, "...")
        urllib.request.urlretrieve(BASE + path, os.path.basename(path))
print("ready")

# %%
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rcParams

SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

PAPER, INK, MUTED = "#FBF9F1", "#16130D", "#6D665A"
TEAL, GOLD, CLAY = "#2E8F8F", "#DD9F3E", "#C96442"
rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif", "axes.titlesize": 13, "axes.titleweight": "bold",
})

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]

z = np.load("so101_mini.npz")
EPISODES = [{"frames": z[f"f{i}"], "states": z[f"s{i}"], "actions": z[f"a{i}"]}
            for i in range(int(z["n_episodes"]))]
norm = np.load("so101_norm.npz")
S_MEAN, S_STD = norm["s_mean"], norm["s_std"]
A_MEAN, A_STD = norm["a_mean"], norm["a_std"]
HOLD_OUT = 4                       # the last 4 episodes are never trained on
print(f"{len(EPISODES)} episodes, "
      f"{sum(len(e['frames']) for e in EPISODES):,} frames, "
      f"frame shape {EPISODES[0]['frames'].shape[1:]}")

# %% [markdown]
# ## Part 1 · What the model tracks
#
# ### One row of the dataset — this is a *state*, and this is an *observation*
#
# Lecture 1 taught this distinction abstractly. Here it is, concretely, in one row of
# a real robot dataset. Every timestep stores exactly three things:
#
# | what | shape | meaning |
# |---|---|---|
# | `observation.images.up` | 64 × 64 × 3 = 12,288 numbers | what the camera saw |
# | `observation.state` | 6 numbers | the arm's true joint angles |
# | `action` | 6 numbers | the joint commands that were sent |
#
# **The state is six numbers. The observation is twelve thousand** — pixels pointed
# *at* those six numbers, plus a cube, a table, and lighting the robot does not
# control. A world model's job is to recover the small thing from the huge one.

# %%
ep = EPISODES[-3]
t0 = 60
fig = plt.figure(figsize=(12, 4.4))
ax = fig.add_axes([0.02, 0.08, 0.34, 0.82])
ax.imshow(ep["frames"][t0] / 255.0); ax.axis("off")
ax.set_title("the OBSERVATION\n64 x 64 x 3 = 12,288 numbers", color=TEAL, fontsize=12)
axt = fig.add_axes([0.42, 0.08, 0.56, 0.82]); axt.axis("off")
y = 0.94
for title, vals, col in [("observation.state  (6 numbers)", ep["states"][t0], TEAL),
                         ("action  (6 numbers)", ep["actions"][t0], CLAY)]:
    axt.text(0, y, title, fontsize=12.5, color=col, fontweight="bold"); y -= 0.10
    for name, v in zip(JOINTS, vals):
        axt.text(0.03, y, f"{name:<16s}", fontsize=11, family="monospace")
        axt.text(0.45, y, f"{v:8.2f}", fontsize=11, family="monospace",
                 color=col, fontweight="bold")
        y -= 0.072
    y -= 0.03
fig.suptitle("ONE row of the dataset", fontsize=14)
plt.show()

# %% [markdown]
# ### And now the part that is in *neither*
#
# Watch the gripper close on the cube. For about half a second the cube is **behind
# the hand** — its position is not in the frame, and no joint angle contains it
# either. The model has to *hold* it.
#
# And "held" is not one number: was the cube nudged as the fingers closed? Did it
# settle a millimetre left? The honest answer is a **region**, not a point.
#
# > **This is the jump the whole lecture rests on.** A world model does not carry a
# > state. It carries a **distribution over states** — and the width of that
# > distribution is itself information. A model that cannot represent width cannot
# > say "I don't know yet", so it will invent a confident wrong answer instead.

# %%
grip = ep["actions"][:, 5]
fork = int(np.argmax(np.abs(np.diff(grip))))
offs = [-8, -4, 0, 4, 8]
fig, axes = plt.subplots(1, len(offs), figsize=(13, 3.0))
for ax, o in zip(axes, offs):
    t = int(np.clip(fork + o, 0, len(ep["frames"]) - 1))
    ax.imshow(ep["frames"][t] / 255.0); ax.axis("off")
    ax.set_title("the gripper closes" if o == 0 else f"{o:+d} steps", fontsize=10,
                 color=CLAY if o == 0 else MUTED,
                 fontweight="bold" if o == 0 else "normal")
fig.suptitle("The fork: from here the future genuinely has more than one answer", y=1.06)
plt.show()
print(f"the gripper closes fastest at timestep {fork}")

# %% [markdown]
# ## Part 2 · The machine
#
# ### Design A — one vector, carried forward, never sampled
#
# This is Lecture 3's model. An encoder squeezes the frame; a GRU holds **one
# vector** and updates it from (that vector, the frame, the action); a decoder
# paints the next frame from it. Every arrow is a computation — nothing is sampled.
#
# ![design A](https://raw.githubusercontent.com/RajatDandekar/build-a-world-model-from-scratch/main/lecture-04-dreams-that-last/assets/fig_designA_arch.png)
#
# **Its strength:** facts loaded into that vector survive indefinitely, because no
# noise ever touches them. We will *measure* this below rather than assert it.
#
# **Its ceiling:** at the fork above, it must output one number. Under squared error
# the safest single answer is the *average* of the possible futures — half-gripped,
# half-slipped, a frame that never occurs. (Measured on our robot: 60-step joint
# error **0.409**.)
#
# ### Design B — sample the state instead
#
# So make the state a **draw from a distribution**: the network outputs a centre and
# a width, and the state is sampled from it. Now a two-answer question can get two
# answers — each one sharp and physically plausible, instead of one blur.
#
# ![the dice](https://raw.githubusercontent.com/RajatDandekar/build-a-world-model-from-scratch/main/lecture-04-dreams-that-last/assets/fig_dice_futures.png)
#
# **Its ceiling:** if the state is redrawn every timestep, every fact must survive a
# fresh dice roll every step. Fifteen occluded frames is fifteen consecutive rolls.
# It won't. Measured on our robot, this design's joint error *rose* during training
# (0.26 → 0.39) and reached **2.598** over a 60-step dream — the worst of the three.
#
# ### The RSSM — carry both
#
# ![one state, two parts](https://raw.githubusercontent.com/RajatDandekar/build-a-world-model-from-scratch/main/lecture-04-dreams-that-last/assets/fig_h_and_s.png)
#
# At every timestep the model carries **one state with two halves**:
#
# * **h** (256 numbers) — updated by a GRU, **never sampled**. Facts live here.
# * **s** (32 numbers) — **sampled every step** from a distribution computed *from h*.
#   Doubt lives here.
#
# They are literally concatenated: the decoder receives `[h, s]`, a 288-number
# vector. And they feed each other — the memory decides how uncertain to be, and the
# sample flows into the next memory, so **a doubt once resolved becomes a remembered
# fact**. That last property is what keeps a dream self-consistent.

# %%
H, S, EMB = 256, 32, 1024
MIN_STD = 0.1


class Encoder(nn.Module):
    """frame (64x64x3) -> 1024 numbers"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ELU(),      # 32x32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ELU(),     # 16x16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ELU(),    # 8x8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ELU(),   # 4x4
            nn.Flatten(), nn.Linear(256 * 16, EMB), nn.ELU())

    def forward(self, x):
        return self.net(x.permute(0, 3, 1, 2) - 0.5)


class Decoder(nn.Module):
    """[h, s] -> frame. Note it reads BOTH halves of the state."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(H + S, 256 * 16)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1))

    def forward(self, h, s):
        x = self.fc(torch.cat([h, s], -1)).view(-1, 256, 4, 4)
        return self.net(x).permute(0, 2, 3, 1) + 0.5


class RSSM(nn.Module):
    def __init__(self, a_dim=6):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()
        # a small head that reads the six joint angles back out of the state.
        # it is not needed for prediction — it exists to make the latent legible.
        self.joint_head = nn.Sequential(nn.Linear(H + S, 256), nn.ELU(),
                                        nn.Linear(256, 6))
        self.in_mlp = nn.Sequential(nn.Linear(S + a_dim, 256), nn.ELU())
        self.gru = nn.GRUCell(256, H)                      # the deterministic belt
        self.prior_mlp = nn.Sequential(nn.Linear(H, 256), nn.ELU(),
                                       nn.Linear(256, 2 * S))       # blind guesser
        self.post_mlp = nn.Sequential(nn.Linear(H + EMB, 256), nn.ELU(),
                                      nn.Linear(256, 2 * S))        # peeking guesser

    def dist(self, out):
        mu, std = out.chunk(2, -1)
        return mu, F.softplus(std) + MIN_STD

    def step_h(self, h, s, a):
        """the belt moves: h_t = GRU(h_{t-1}, s_{t-1}, a_{t-1}) — no randomness"""
        return self.gru(self.in_mlp(torch.cat([s, a], -1)), h)

    def prior(self, h):
        """guess the state from the memory ALONE — never sees the frame"""
        return self.dist(self.prior_mlp(h))

    def posterior(self, h, emb):
        """guess the same state, but allowed to look at the frame"""
        return self.dist(self.post_mlp(torch.cat([h, emb], -1)))


model = RSSM().to(device)
print(f"RSSM parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## Part 3 · How do you train this?
#
# Here is the difficulty: **the dataset contains no state labels.** Nowhere in those
# rows is a column saying "the state should be these 288 numbers" — and nobody could
# write one, because the state is the model's own private summary.
#
# So we build the training signal out of two things we *do* have.
#
# ![the training step](https://raw.githubusercontent.com/RajatDandekar/build-a-world-model-from-scratch/main/lecture-04-dreams-that-last/assets/fig_training_step.png)
#
# ### Loss 1 — repaint the frame
#
# Decode the state back into an image and compare it, pixel by pixel, with what the
# camera actually saw.
#
# *Why we need it:* without this the state could be anything at all, including
# nothing. The only way to repaint the arm, the cube and the table is to have kept
# them. **Loss 1 fills the state with content.**
#
# *Why it is not enough:* Lecture 3 had exactly this term and still failed. A state
# can be perfectly repaintable and still be impossible to roll forward.
#
# ### Loss 2 — ask the same question twice
#
# At each timestep we compute the state **twice**:
#
# * the **posterior** — given the memory, the action, *and the camera frame*
# * the **prior** — given only the memory and the action (this is what dreaming feels like)
#
# **Loss 2 is the distance between those two answers** (a KL divergence between the
# two distributions). Minimise it and you are training the blind guess to match the
# informed one — and there is no way to win that game except to genuinely carry
# forward whatever determines the next frame. That *is* prediction.
#
# And notice it pulls both ways: the posterior is also punished for encoding things
# the prior could never anticipate, so **perception is pushed toward representations
# that are predictable in the first place**. In Lecture 3 we hand-rolled a
# "smoothness" penalty guessing at this principle. Here it falls out of the objective.
#
# > **Loss 1 fills the state. Loss 2 makes what is in it forecastable.** Drop either
# > one and you are back to a model whose dreams dissolve.

# %%
def kl(m1, s1, m2, s2):
    """KL( N(m1,s1) || N(m2,s2) ), summed over dimensions"""
    return (torch.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5).sum(-1)


def sample_batch(B=8, L=24):
    train_eps = EPISODES[:-HOLD_OUT]
    fr = np.zeros((B, L, 64, 64, 3), np.float32)
    st = np.zeros((B, L, 6), np.float32)
    ac = np.zeros((B, L, 6), np.float32)
    for b in range(B):
        e = train_eps[np.random.randint(len(train_eps))]
        t = np.random.randint(0, len(e["frames"]) - L)
        fr[b] = e["frames"][t:t+L] / 255.0
        st[b] = (e["states"][t:t+L] - S_MEAN) / S_STD
        ac[b] = (e["actions"][t:t+L] - A_MEAN) / A_STD
    return (torch.tensor(fr).to(device), torch.tensor(st).to(device),
            torch.tensor(ac).to(device))


def train_steps(model, steps=600, B=8, L=24, free_nats=1.0, log_every=100):
    """One training step = roll the sequence, accumulate both losses, update."""
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    history = []
    t_start = time.time()
    for step in range(steps):
        fr, st, ac = sample_batch(B, L)
        emb = model.enc(fr.reshape(B * L, 64, 64, 3)).view(B, L, -1)
        h = torch.zeros(B, H, device=device)
        s = torch.zeros(B, S, device=device)
        l_rec = l_kl = l_joint = 0.0
        for t in range(L):
            if t > 0:                                   # 1. the belt moves
                h = model.step_h(h, s, ac[:, t - 1])
            pm, ps = model.prior(h)                     # 2a. the blind guess
            qm, qs = model.posterior(h, emb[:, t])      # 2b. the peeked guess
            s = qm + qs * torch.randn_like(qs)          #     sample from the posterior
            # ---- loss 1: repaint the frame from [h, s]
            l_rec = l_rec + ((model.dec(h, s) - fr[:, t]) ** 2).sum(dim=(1, 2, 3)).mean()
            l_joint = l_joint + ((model.joint_head(torch.cat([h, s], -1))
                                  - st[:, t]) ** 2).sum(-1).mean()
            # ---- loss 2: pull the blind guess toward the peeked guess.
            # "KL balancing": 0.8 of the gradient trains the prior, 0.2 the posterior.
            # "free nats": stop pushing once they are already close enough.
            k = (0.8 * kl(qm.detach(), qs.detach(), pm, ps).mean()
                 + 0.2 * kl(qm, qs, pm.detach(), ps.detach()).mean())
            l_kl = l_kl + torch.clamp(k, min=free_nats)
        loss = (l_rec + 10.0 * l_joint + 1.0 * l_kl) / L
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()
        history.append((float(l_rec) / L, float(l_joint) / L, float(l_kl) / L))
        if step % log_every == 0:
            print(f"step {step:5d}  loss1 (repaint) {history[-1][0]:8.1f}   "
                  f"joints {history[-1][1]:6.3f}   loss2 (KL) {history[-1][2]:5.2f}   "
                  f"({time.time()-t_start:.0f}s)")
    return np.array(history)


# A short run so you can watch both losses fall. On a Colab GPU this is ~10 minutes;
# on CPU it is slow, so we skip it and use the pretrained checkpoint below.
RUN_TRAINING = torch.cuda.is_available()
if RUN_TRAINING:
    hist = train_steps(model, steps=600)
else:
    hist = None
    print("no GPU detected — skipping the live training run "
          "(the pretrained checkpoint below gives the full results)")

# %%
if hist is not None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, col, lab, c in zip(axes, hist.T,
                               ["loss 1 · repaint the frame", "joint-angle readout",
                                "loss 2 · make the guesses agree"], [TEAL, GOLD, CLAY]):
        ax.plot(col, lw=1.6, color=c); ax.set_title(lab, fontsize=11)
        ax.set_xlabel("training step")
    fig.suptitle("Both losses falling together — the model is getting sharper AND "
                 "more predictable", y=1.04)
    plt.show()

# %% [markdown]
# ### Load the fully-trained model
#
# 600 steps is enough to see the losses move, but the results in the lecture come
# from a 21,000-step run on one GPU (about 50 minutes). Let's load that checkpoint
# and put it through the real tests.

# %%
ck = torch.load("rssm_so101.pt", map_location=device, weights_only=False)
model = RSSM().to(device)
model.load_state_dict(ck)          # the checkpoint is a plain state_dict
model.eval()
print("loaded the trained RSSM")

# %% [markdown]
# ## Test 1 · What is the belt actually carrying?
#
# People say a recurrent state "carries information forward" and move on. We can do
# better: we can **open the vector and ask it questions**. The technique is a
# *linear probe* and it takes ten lines.
#
# 1. Run the model over episodes and record, at every timestep, the 256-number
#    memory `h` **and** the six true joint angles the robot actually had.
# 2. Fit a single **linear** map from `h` to those angles — no hidden layers. If a
#    plain linear readout can recover a fact, that fact is genuinely *present*.
# 3. Fit the same map against **shuffled** labels as a control. Real structure
#    should score high; the control should score near zero.

# %%
Hs, Ys = [], []
with torch.no_grad():
    for e in EPISODES[:-HOLD_OUT]:
        fr = torch.tensor(e["frames"] / 255.0, dtype=torch.float32).to(device)
        ac = torch.tensor((e["actions"] - A_MEAN) / A_STD, dtype=torch.float32).to(device)
        emb = model.enc(fr)
        h = torch.zeros(1, H, device=device); s = torch.zeros(1, S, device=device)
        for t in range(len(fr)):
            if t > 0:
                h = model.step_h(h, s, ac[t-1:t])
            qm, _ = model.posterior(h, emb[t:t+1])
            s = qm
            Hs.append(h[0].cpu().numpy()); Ys.append((e["states"][t] - S_MEAN) / S_STD)
Hs, Ys = np.array(Hs, np.float32), np.array(Ys, np.float32)

A_ = np.concatenate([Hs, np.ones((len(Hs), 1), np.float32)], 1)
W, *_ = np.linalg.lstsq(A_, Ys, rcond=None)
r2 = 1 - ((A_ @ W - Ys) ** 2).sum(0) / ((Ys - Ys.mean(0)) ** 2).sum(0)
idx = np.random.permutation(len(Ys))
Wc, *_ = np.linalg.lstsq(A_, Ys[idx], rcond=None)
r2c = 1 - ((A_ @ Wc - Ys[idx]) ** 2).sum(0) / ((Ys[idx] - Ys[idx].mean(0)) ** 2).sum(0)

fig, ax = plt.subplots(figsize=(8.4, 3.4))
yy = np.arange(6)
ax.barh(yy, r2, color=TEAL, height=0.55, label="from the memory h alone")
ax.barh(yy, r2c, color=CLAY, height=0.22, label="shuffled control")
ax.set_yticks(yy); ax.set_yticklabels(JOINTS); ax.invert_yaxis(); ax.set_xlim(0, 1.02)
ax.set_xlabel("fraction of the joint angle recovered by a LINEAR readout")
ax.legend(frameon=False, fontsize=10, loc="lower right")
plt.show()
print("per-joint R^2:", np.round(r2, 4))

# %% [markdown]
# **The belt is not carrying "a vector" — it is carrying the arm.** A plain linear
# readout recovers essentially all of every joint angle, while the shuffled control
# sits at zero. Nobody trained `h` to contain joint angles; it built that
# representation because predicting frames required it.

# %% [markdown]
# ## Test 2 · The dream — sixty steps with the camera off
#
# The real test. Give the model five frames of context, then **switch the camera
# off** and feed it nothing but the joint commands that were actually sent. Ask it
# to imagine the next sixty steps — two full seconds of robot motion — on an episode
# it has never seen.

# %%
CONTEXT, HORIZON = 5, 60


def dream(ep, start, horizon=HORIZON):
    fr = torch.tensor(ep["frames"][start:start+CONTEXT+horizon] / 255.0,
                      dtype=torch.float32).to(device)
    st = (ep["states"][start:start+CONTEXT+horizon] - S_MEAN) / S_STD
    ac = torch.tensor((ep["actions"][start:start+CONTEXT+horizon] - A_MEAN) / A_STD,
                      dtype=torch.float32).to(device)
    with torch.no_grad():
        emb = model.enc(fr)
        h = torch.zeros(1, H, device=device); s = torch.zeros(1, S, device=device)
        for t in range(CONTEXT):                      # warm up on REAL frames
            if t > 0:
                h = model.step_h(h, s, ac[t-1:t])
            qm, _ = model.posterior(h, emb[t:t+1])    # the posterior may look
            s = qm
        frames, joints = [], []
        for k in range(horizon):                      # now the camera is OFF
            h = model.step_h(h, s, ac[CONTEXT+k-1:CONTEXT+k])
            pm, _ = model.prior(h)                    # only the blind guesser is left
            s = pm
            frames.append(model.dec(h, s)[0].clamp(0, 1).cpu().numpy())
            joints.append(model.joint_head(torch.cat([h, s], -1))[0].cpu().numpy())
    return fr.cpu().numpy(), np.array(frames), np.array(joints), st


ep = EPISODES[-2]
start = max(0, len(ep["frames"]) // 3 - CONTEXT)
real, dreamed, joints, st = dream(ep, start)

picks = [0, 11, 23, 35, 47, 59]
fig, axes = plt.subplots(2, len(picks), figsize=(13, 4.6))
for i, k in enumerate(picks):
    axes[0, i].imshow(real[CONTEXT + k]); axes[0, i].axis("off")
    axes[0, i].set_title(f"+{k+1} steps", fontsize=10, color=MUTED)
    axes[1, i].imshow(dreamed[k]); axes[1, i].axis("off")
axes[0, 0].text(-0.12, 0.5, "the real robot", transform=axes[0, 0].transAxes,
                ha="right", va="center", fontsize=12, color=TEAL, fontweight="bold")
axes[1, 0].text(-0.12, 0.5, "imagined\n(camera off)", transform=axes[1, 0].transAxes,
                ha="right", va="center", fontsize=12, color=CLAY, fontweight="bold")
fig.suptitle("Sixty steps of pure imagination on a held-out episode", y=1.02)
plt.subplots_adjust(left=0.13, wspace=0.05, hspace=0.08)
plt.show()

# %% [markdown]
# ### And the same dream, read as numbers
#
# Pixels can hide a lot. The joint-angle readout gives us six clean curves — watch
# the **gripper** panel in particular: the model predicts the contact event at the
# right moment, having never seen a frame of it.

# %%
fig, axes = plt.subplots(2, 3, figsize=(13, 5.4))
for j in range(6):
    ax = axes[j // 3, j % 3]
    ax.plot(st[CONTEXT:CONTEXT+HORIZON, j], lw=2.4, color=TEAL, label="real")
    ax.plot(joints[:, j], lw=2.4, ls="--", color=CLAY, label="dreamed")
    ax.set_title(JOINTS[j], fontsize=11)
    if j >= 3:
        ax.set_xlabel("dream step")
axes[0, 0].legend(frameon=False, fontsize=10)
fig.suptitle("Dreamed joint angles vs the real robot — 60 open-loop steps", y=1.01)
plt.tight_layout()
plt.show()

err = ((joints - st[CONTEXT:CONTEXT+HORIZON]) ** 2).mean(-1)
print(f"joint error at step 1: {err[0]:.4f}   step 30: {err[29]:.4f}   "
      f"step 60: {err[59]:.4f}")

# %% [markdown]
# **It does not blow up.** In Lecture 3 the error compounded until the imagined ball
# dissolved. Here it stays flat across the whole horizon — and on some episodes the
# error at step 60 is *lower* than at step 30, meaning the dream re-converges toward
# reality rather than running away.

# %% [markdown]
# ## Test 3 · Cut either path and watch it break
#
# The sharpest test in the lecture, and you can run it here in a few lines. We take
# **one trained model** and run it three ways — same weights, same episode, same
# actions. Only the wiring changes.

# %%
def dream_variant(ep, start, mode, horizon=40):
    """mode: 'both' (as designed) | 'freeze_s' | 'starve_h'"""
    fr = torch.tensor(ep["frames"][start:start+CONTEXT+horizon] / 255.0,
                      dtype=torch.float32).to(device)
    ac = torch.tensor((ep["actions"][start:start+CONTEXT+horizon] - A_MEAN) / A_STD,
                      dtype=torch.float32).to(device)
    with torch.no_grad():
        emb = model.enc(fr)
        h = torch.zeros(1, H, device=device); s = torch.zeros(1, S, device=device)
        for t in range(CONTEXT):
            if t > 0:
                h = model.step_h(h, s, ac[t-1:t])
            qm, _ = model.posterior(h, emb[t:t+1]); s = qm
        out = []
        for k in range(horizon):
            if mode == "starve_h":       # the sample never reaches the memory
                h = model.step_h(h, torch.zeros_like(s), ac[CONTEXT+k-1:CONTEXT+k])
            else:
                h = model.step_h(h, s, ac[CONTEXT+k-1:CONTEXT+k])
            if mode != "freeze_s":       # 'freeze_s' keeps the warm-up sample forever
                pm, ps = model.prior(h)
                s = pm if mode == "both" else pm + ps * torch.randn_like(ps)
            out.append(model.dec(h, s)[0].clamp(0, 1).cpu().numpy())
    return fr.cpu().numpy(), out


picks2 = [0, 8, 16, 24, 32, 39]
rows = [("the real robot", None), ("s frozen", "freeze_s"),
        ("s starved from h", "starve_h"), ("both, as designed", "both")]
fig, axes = plt.subplots(4, len(picks2), figsize=(13, 8.4))
real40, _ = dream_variant(ep, start, "both")
for r, (lab, mode) in enumerate(rows):
    frames = ([real40[CONTEXT + k] for k in picks2] if mode is None
              else [dream_variant(ep, start, mode)[1][k] for k in picks2])
    for i, f in enumerate(frames):
        axes[r, i].imshow(np.clip(f, 0, 1)); axes[r, i].axis("off")
        if r == 0:
            axes[r, i].set_title(f"+{picks2[i]+1}", fontsize=10, color=MUTED)
    axes[r, 0].text(-0.12, 0.5, lab, transform=axes[r, 0].transAxes, ha="right",
                    va="center", fontsize=12, fontweight="bold",
                    color=[INK, TEAL, GOLD, CLAY][r])
fig.suptitle("One trained model, three wirings — the two paths are load-bearing "
             "for each other", y=1.01)
plt.subplots_adjust(left=0.15, wspace=0.05, hspace=0.08)
plt.show()

# %% [markdown]
# ## What the full ablation showed
#
# In the lecture we went further and trained **three separate models** — same data,
# same 16,000 steps, same seed, matched parameter counts — changing only the state
# design. The code is in `code/modal_ablation.py`; the result:
#
# | design | parameters | pixel error @60 | **joint error @60** |
# |---|---|---|---|
# | deterministic only | 7,718,313 | 0.0106 | **0.409** |
# | stochastic only | 7,543,881 | 0.0155 | **2.598** |
# | **both — the RSSM** | 7,665,993 | **0.0050** | **0.009** |
#
# The RSSM is **45× better than deterministic-only and 288× better than
# stochastic-only** at matched capacity. Note also that pixel error separates the
# three far less than joint error does — pixels are dominated by the static table
# and background, while the *robot* is where the designs actually differ. Always
# check that your metric measures the thing you care about.
#
# ![the ablation](https://raw.githubusercontent.com/RajatDandekar/build-a-world-model-from-scratch/main/lecture-04-dreams-that-last/assets/fig_ablation_strips.png)

# %% [markdown]
# ## What we measured, and what we did not
#
# One claim we expected to make and could not. We predicted that the model's
# **uncertainty would spike at contact** — the grasp is the moment the future
# genuinely forks. Averaged over the state's dimensions, it does not.
#
# What we *did* see is real and worth keeping: at the very first timestep, with no
# history at all, uncertainty is at its maximum — and **one frame collapses it**.
# The belief-narrowing story, measured on a real robot.
#
# Why the contact spike is missing is an open question: either the forks live in a
# few dimensions that the average washes out, or 50 episodes of a reliably
# successful grasp simply contain little for the model to be unsure about. Both are
# testable — see the exercises.

# %%
e = EPISODES[-1]
with torch.no_grad():
    fr = torch.tensor(e["frames"] / 255.0, dtype=torch.float32).to(device)
    ac = torch.tensor((e["actions"] - A_MEAN) / A_STD, dtype=torch.float32).to(device)
    emb = model.enc(fr)
    h = torch.zeros(1, H, device=device); s = torch.zeros(1, S, device=device)
    sig = []
    for t in range(len(fr)):
        if t > 0:
            h = model.step_h(h, s, ac[t-1:t])
        qm, qs = model.posterior(h, emb[t:t+1]); s = qm
        sig.append(float(qs.mean()))
fig, ax = plt.subplots(figsize=(11, 3.0))
ax.plot(sig, lw=2, color=CLAY)
ax.set_xlabel("timestep"); ax.set_ylabel("uncertainty (mean width)")
ax.set_title("Maximum uncertainty at t=0 — one frame collapses it")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## What to take away
#
# 1. **A world model tracks a belief, not a state.** A centre *and* a width. A model
#    that cannot express doubt will invent a confident wrong answer.
# 2. **Two paths, two jobs.** A deterministic memory for facts you are sure of;
#    sampled dice for the moments the world forks. Merge them into one channel and
#    you lose both — we measured 45× and 288× penalties for dropping either.
# 3. **Prediction is taught by asking the same question twice.** With the frame and
#    without it, then pulling the answers together. That single term is what turns an
#    autoencoder into a world model.
# 4. **Train the pieces together.** A representation learned without the prediction
#    task will not survive the prediction task.
#
# ### Exercises
#
# 1. **Break the asymmetry.** Let the prior see the encoded frame too (pass `emb`
#    into `prior_mlp`). Loss 2 will collapse to near zero — and the dream will fall
#    apart. Why is a loss you can trivially minimise a useless loss?
# 2. **Shrink the memory.** Set `H = 32` and retrain. Which test degrades first —
#    the probe, the dream, or the reconstruction?
# 3. **Remove the stochastic path** (`S = 0`, decoder reads `h` alone). Reproduce the
#    deterministic-only arm of the ablation and check you get roughly 0.4.
# 4. **Hunt the missing spike.** Instead of averaging the width over all 32
#    dimensions, plot the *per-dimension* width around the grasp. Does any single
#    dimension spike at contact?
# 5. **Dream past 60 steps.** Push the horizon to 200. Where does it finally break,
#    and what breaks first — the arm, or the cube?

# %% [markdown]
# ---
# *Vizuara AI · Build a World Model from Scratch · Lecture 4 companion.*
# *Dataset: [lerobot/svla_so101_pickplace](https://huggingface.co/datasets/lerobot/svla_so101_pickplace) (Apache-2.0).*
# *Method: Hafner et al., ["Learning Latent Dynamics for Planning from Pixels"](https://arxiv.org/abs/1811.04551), 2019.*
