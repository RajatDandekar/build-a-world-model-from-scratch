# %% [markdown]
# # Building a World Model from Scratch — on MiniPong
#
# **Companion notebook for Lecture 3 of *Build a World Model from Scratch* (Vizuara AI).**
#
# In this notebook we build ONE thing, completely and from scratch: a **world model** —
# a neural network that learns how a small Pong game works, purely by watching recorded
# play. By the end, the network will paint the game's next frame *before the game shows
# it*, and then run as a closed loop with the real game switched off entirely.
#
# **No controller, no planner, no reward.** Acting well comes later in the series.
# Today's single question: *how does a network learn a world?*
#
# ---
#
# ### How to use this notebook
#
# * **Runtime → Run all** works: every cell runs top to bottom with no setup, no
#   installs, no GPU needed (a free Colab CPU takes ~10 minutes end to end; most of
#   that is training network V).
# * But this notebook is written as a **guide** — each section first explains, in plain
#   words, what we are about to do and *why*, then shows the code, then tells you what
#   to look for in the output. You will get far more out of reading along than out of
#   pressing Run All and scrolling to the bottom.
# * Everything you see on the lecture slides — every number, every plot — is this
#   notebook's real output. Change a line, rerun, and watch the world change.
#
# ### The roadmap (matching the lecture exactly)
#
# **Part 1 — Training**
# 1. **Collect the dataset** — 200 episodes of recorded play: frames + actions, nothing else.
# 2. **Network V (the encoder-decoder)** — compress every 3,072-pixel frame into a
#    12-number code, and decode it back. *"What does the world look like?"*
# 3. **Network M (memory + prediction)** — read the codes and actions in order,
#    maintain a memory across timesteps, and predict the NEXT code. *"What happens next?"*
#
# **Part 2 — Deployment**
# 4. Switch the game engine OFF and run the model in a **closed loop**: its own
#    predictions become its next inputs. This is where world models get hard —
#    and where the next lecture begins.
#
# ---
#
# ### First: the world we will learn
#
# The cell below is the entire game — about 60 lines. Read it once; it repays the
# attention. Three things to notice:
#
# * **The observation is pixels.** `observe()` returns a 32×32×3 image: a teal paddle,
#   a glowing yellow ball, dark background. The model will never see positions or
#   velocities as numbers — only this image, exactly like a human player.
# * **The catch that shapes everything:** a single frame shows *where* the ball is, but
#   never *where it is going*. Velocity lives only in the *difference between frames*.
#   Keep this in mind — it is the reason network M needs a memory.
# * **Two deliberate design choices** (both were hard-won lessons, discussed at the end):
#   the physics is **deterministic** (no randomness after reset — so a predictor can in
#   principle be exact), and motion is **continuous with anti-aliased rendering** (the
#   ball's sub-pixel position shows up smoothly in pixel intensities, so nearby states
#   get nearby codes).

# %% [markdown]
# ### Setup
#
# Standard imports, a fixed random seed (so your numbers match ours), and a warm paper
# plot style. Nothing here is world-model-specific.

# %%
import math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pong_env import MiniPong

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
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

# %% [markdown]
# ## Part 1, Step 1 · Collect the dataset
#
# A world model learns from **recorded experience**. So step one is simply to play the
# game and write everything down.
#
# We record **200 episodes × 120 timesteps = 24,000 moments** of MiniPong. At every
# timestep we store exactly two things:
#
# | what | shape | meaning |
# |---|---|---|
# | the frame | 32 × 32 × 3 = 3,072 numbers | what the screen showed |
# | the action | one integer (0/1/2) | the key held at that moment |
#
# That's the entire dataset. **No labels, no rewards, no positions, no velocities** —
# if the model wants to know where the ball is going, it will have to figure that out
# from the pixels alone.
#
# **Two decisions worth pausing on:**
#
# * **Why a RANDOM player?** The world model's job is to learn *how the world works* —
#   how the ball flies, bounces, and reacts to the paddle. Physics doesn't care whether
#   the player is skilled, and random play visits a rich variety of situations. Skill
#   is not required to learn the rules.
# * **Why does the random player HOLD each key** (the `rng.random() < 0.25` line — it
#   switches keys only occasionally)? Because at deployment, a *human* player holds
#   keys: long sweeps to the left, paddle pressed against a wall for many steps.
#   Training data must visit the situations deployment will visit. A player that
#   twitched randomly every frame would never pin the paddle to a wall — and the model
#   would face those states cold. This one line fixed a real failure we hit.
#
# **What to look for below:** eight raw frames from the dataset. Bright ball, bright
# paddle, dark world. Try to tell, from any single frame, which way the ball is
# moving. You can't — and neither can the model. Remember that.

# %%
env = MiniPong(seed=SEED)
EPISODES, T = 200, 120
frames = np.zeros((EPISODES, T, 32, 32, 3), np.float32)
actions = np.zeros((EPISODES, T), np.int64)
rng = np.random.default_rng(SEED)
for e in range(EPISODES):
    obs = env.reset()
    a = int(rng.integers(0, 3))
    for t in range(T):
        if rng.random() < 0.25:            # switch keys only occasionally
            a = int(rng.integers(0, 3))
        frames[e, t], actions[e, t] = obs, a
        obs, done = env.step(a)
print(f"dataset: {EPISODES} episodes × {T} steps = {EPISODES*T:,} frames "
      f"({frames.nbytes/1e6:.0f} MB)")

fig, axes = plt.subplots(1, 8, figsize=(13, 1.9))
for i in range(8):
    axes[i].imshow(frames[i, i * 13]); axes[i].axis("off")
fig.suptitle("MiniPong — eight raw observations from the dataset", y=1.12)
plt.tight_layout(); plt.savefig("plot_env_frames.png", dpi=150, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## Part 1, Step 2 · Network V — the encoder-decoder
#
# **New concept — the autoencoder.** A frame is 3,072 numbers, but the *situation* is
# tiny: where is the ball, where is the paddle. An autoencoder is a pair of networks
# trained together: an **encoder** squeezes the frame down into a small **code z**
# (here: 12 numbers), and a **decoder** proves nothing important was lost by redrawing
# the whole frame from the code alone. If the redrawn frame matches the original, those
# 12 numbers must contain everything that matters.
#
# Why compress at all? Because in Step 3 another network will learn to *predict* codes.
# Predicting 12 numbers is a far kinder task than predicting 3,072 pixels — this is the
# central design idea of Ha & Schmidhuber's 2018 *World Models* paper, and of every
# world model since.
#
# > 🎥 **Want the full depth on this idea?** Network V is built on the variational
# > autoencoder. Vizuara has a dedicated from-scratch lecture on it — intuition, math,
# > and coding: [Variational Autoencoder (VAE) from scratch | Intuition +
# > Coding](https://www.youtube.com/watch?v=VUwAGLM6K_8). This notebook uses a lean
# > variant of the same machinery, tuned for what a world model needs.
#
# **The architecture, in one breath:** two convolution layers scan the image for local
# shapes (a glowing blob, a teal bar) and shrink it 32→16→8; a linear layer maps the
# result to the 12-number code; the decoder mirrors the same path in reverse, ending in
# a sigmoid so every output pixel lands in [0, 1].
#
# **Four small decisions in the training loop carry most of the intelligence.** Each
# exists because the naive version failed in a specific, instructive way:
#
# 1. **Weight the ball's pixels up** (`w = 1 + 40·redness`). The ball is a handful of
#    pixels out of 1,024. To a plain reconstruction loss, dropping the ball entirely
#    costs almost nothing — and that is exactly what our first V did: perfect paddle,
#    no ball. Compression keeps *what the loss pays for*, nothing else. We weight by
#    the red channel because the ball is the only strongly red thing on screen.
#    Watch for this failure in every model you ever train.
# 2. **The code is deterministic** (`z` IS `mu` — no sampling). The same frame must
#    always get the same code: network M can only predict the next code as exactly as
#    V computes it. A code that is a noisy sample around the frame puts a floor under
#    M's accuracy that no amount of training can break.
# 3. **...but the decoder trains on fuzzed codes** (`+ 0.1·noise`). M's predictions
#    will land *near* real codes, never exactly on them — so we train the decoder in a
#    small neighbourhood around each code, teaching it that "near a real code" must
#    still paint a sane frame.
# 4. **Consecutive frames get nearby codes** (the `smooth` term — note the loader
#    samples consecutive *pairs*). The world moves a pixel at a time, so the codes
#    should drift gently too. Without this, V is free to scatter neighbouring states
#    to far-apart corners of code space — perfectly decodable, but a nightmare for the
#    prediction network. We make the code space *smooth to move through* because
#    someone is about to move through it.
#
# **What to look for below:** the loss falls steeply for ~5 epochs and then grinds
# slowly — that grind is V learning the ball, the hardest few pixels on screen.
# This is the slowest cell in the notebook (~5 minutes on a Colab CPU). 93,787
# parameters total.

# %%
Z = 12

class VNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(                       # 3 x 32 x 32
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),      # 16 x 16 x 16
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),     # 32 x 8 x 8
            nn.Flatten())                               # 2048
        self.mu = nn.Linear(2048, Z)
        self.logvar = nn.Linear(2048, Z)
        self.fc = nn.Linear(Z, 2048)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x.permute(0, 3, 1, 2))
        return self.mu(h), self.logvar(h).clamp(-6, 2)

    def decode(self, z):
        return self.dec(self.fc(z).view(-1, 32, 8, 8)).permute(0, 2, 3, 1)

vnet = VNet().to(device)
opt = torch.optim.Adam(vnet.parameters(), lr=1e-3)
flat = torch.tensor(frames.reshape(-1, 32, 32, 3))
# train on CONSECUTIVE PAIRS of frames — we need them for the smoothness term below
pair_idx = torch.tensor([e * T + t for e in range(EPISODES) for t in range(T - 1)])
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pair_idx),
                                     batch_size=256, shuffle=True)
t0 = time.time()
for epoch in range(20):
    tot = 0.0
    for (bi,) in loader:
        x = flat[bi].to(device)             # frame t
        xn = flat[bi + 1].to(device)        # frame t+1 (same episode)
        mu, logvar = vnet.encode(x)
        mun, _ = vnet.encode(xn)
        # deterministic code: z IS mu. The same frame always gets the same code — network
        # M can only predict the next code as exactly as V computes it, so the code must
        # be an exact function of the frame, not a noisy sample around it.
        z = mu + 0.1 * torch.randn_like(mu)  # fuzz for the DECODER only: predicted codes
                                             # will land NEAR real ones, so train it there
        xhat = vnet.decode(z)
        # the ball is a handful of pixels out of 1,024 — weight RED pixels up (only
        # the ball is red), or compression happily drops the most important thing
        w = 1.0 + 40.0 * x[..., :1]        # weight by REDNESS: only the ball is red
        recon = (w * F.binary_cross_entropy(xhat, x, reduction="none")).sum() / len(x)
        # SMOOTHNESS: the world moves one pixel at a time, so consecutive frames must
        # get nearby codes. Without this, V is free to scatter neighbouring states to
        # far-apart codes — decodable, but impossible for network M to predict.
        smooth = ((mu - mun) ** 2).sum(1).mean()
        # plus a light pull toward zero to keep the code scale bounded
        loss = recon + 1.0 * smooth + 0.01 * (mu ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(x)
    print(f"epoch {epoch+1:2d}  loss {tot/len(flat):8.1f}")
print(f"V trained in {time.time()-t0:.0f}s   ({sum(p.numel() for p in vnet.parameters()):,} parameters)")

# %% [markdown]
# ### V's report card
#
# The only test that matters: take frames V has never been graded on, squeeze each one
# through the 12-number bottleneck, and redraw it.
#
# **What to check, in order of importance:** ① is the **ball** present in every redrawn
# frame, at the right spot? (that's the weighted loss earning its keep) ② is the paddle
# at the right position and width? ③ the redrawn frames will look slightly softer than
# the originals — that's normal; 12 numbers cannot carry every pixel's exact value, and
# we only need them to carry the *situation*.

# %%
vnet.eval()
with torch.no_grad():
    test = flat[5000:24000:2713][:7].to(device)
    mu, _ = vnet.encode(test)
    rec = vnet.decode(mu)
fig, axes = plt.subplots(2, 7, figsize=(12, 3.6))
for i in range(7):
    axes[0, i].imshow(test[i].cpu()); axes[0, i].axis("off")
    axes[1, i].imshow(rec[i].cpu().clamp(0, 1)); axes[1, i].axis("off")
axes[0, 0].set_title("original frame", loc="left", color=TEAL)
axes[1, 0].set_title("redrawn from the 12-number code", loc="left", color=CLAY)
fig.suptitle("Network V — 3,072 pixels → 12 numbers → 3,072 pixels", y=1.02)
plt.tight_layout(); plt.savefig("plot_v_recon.png", dpi=150, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## Part 1, Step 3 · Network M — memory + prediction
#
# Now the heart of the world model: the network that learns *what happens next*.
#
# **New concept — a recurrent memory.** Recall the catch from Step 1: one frame shows
# *where* the ball is, never *where it is going*. So a network that reads one code and
# guesses the next is doomed — it cannot know the direction of travel. The fix is a
# **memory**: a vector **h of 128 numbers that persists across timesteps**. At every
# step the memory is *updated* — blended with the new evidence — rather than recomputed
# from scratch. After the memory has seen two ball positions, it can hold their
# difference. That difference IS the velocity. No frame ever showed it; the memory
# *derived* it, because prediction is impossible without it.
#
# The updating machinery is a **GRU** (gated recurrent unit) — a standard recurrent
# layer whose internal gates learn *what to keep* in the memory and *what to overwrite*.
# You met the idea in the lecture; here it is one line: `nn.GRU(...)`.
#
# **The exact dataflow at one timestep t** (this answers all the bookkeeping questions
# — fix it in your mind):
#
# * IN: the memory `h(t)` · the current code `z(t)` (from V) · the action `a(t)` (one-hot)
# * **network 1 (the GRU)** updates the memory: `h(t+1) = GRU(h(t), z(t), a(t))`
# * **network 2 (a small 2-layer head)** reads the updated memory and predicts the
#   next code: `ẑ(t+1) = head(h(t+1))`
# * The **loss** is the distance between the prediction `ẑ(t+1)` and the ACTUAL next
#   code `z(t+1)` — the code V assigns to the real next frame. That real code is the
#   answer key. One loss, both networks trained together by it, at every timestep of
#   every episode: 24,000 small exams per epoch.
#
# **Three engineering decisions in this cell — each one, again, bought with a failure:**
#
# 1. **Normalise every code dimension to unit scale** (`Z_MEAN`, `Z_STD`). The paddle
#    is big and busy, so the code dimensions describing it have large variance; the
#    ball's dimensions are quiet. An unnormalised loss obsesses over the paddle and
#    ignores the ball. After normalisation, every dimension matters equally. (We undo
#    the scaling before decoding.)
# 2. **Predict the CHANGE in the code, not the code itself** (`z_in + pred`). Subtle
#    but crucial: when a network is unsure, its safest MSE answer drifts toward the
#    dataset average — and the "average frame" has a smeared, invisible ball. If it
#    predicts *changes* instead, the safe answer under uncertainty is "nothing moves",
#    which keeps the ball painted where it was. Same loss, same data — very different
#    failure mode.
# 3. **Add a little noise to the input codes** (`+ 0.05·noise`). At deployment
#    (Part 2), M will eat its own slightly-imperfect predictions. Training with noisy
#    inputs teaches it to *correct* a slightly-wrong code instead of amplifying the
#    error. We are inoculating it against its own future mistakes.
#
# **What to look for below:** prediction error falling epoch over epoch — and note the
# speed: M trains in seconds, because it works on 12-number codes instead of
# 3,072-pixel frames. That speed is what Step 2 bought us. 73,740 parameters.

# %%
H = 128

class MNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(Z + 3, H, batch_first=True)   # network 1: memory update
        self.head = nn.Sequential(                       # network 2: prediction
            nn.Linear(H, 128), nn.ELU(), nn.Linear(128, Z))

    def forward(self, z_seq, a_seq, h0=None):
        inp = torch.cat([z_seq, a_seq], -1)
        out, hN = self.gru(inp, h0)
        return self.head(out), hN

with torch.no_grad():
    zs = []
    for i in range(0, len(flat), 4096):
        mu, _ = vnet.encode(flat[i:i+4096].to(device))
        zs.append(mu.cpu())
    zs = torch.cat(zs).view(EPISODES, T, Z)
# normalise each code dimension to unit scale for M's training — otherwise the
# dimensions describing the (big, busy) paddle dominate the loss and the (tiny)
# ball's dimensions get ignored. We undo this before decoding.
Z_MEAN = zs.reshape(-1, Z).mean(0)
Z_STD = zs.reshape(-1, Z).std(0).clamp_min(1e-4)
zs = (zs - Z_MEAN) / Z_STD
a1h = F.one_hot(torch.tensor(actions), 3).float()

mnet = MNet().to(device)
opt = torch.optim.Adam(mnet.parameters(), lr=1e-3)
t0 = time.time()
for epoch in range(25):
    perm, tot = torch.randperm(EPISODES), 0.0
    for i in range(0, EPISODES, 16):
        idx = perm[i:i+16]
        z = zs[idx].to(device); a = a1h[idx].to(device)
        # two robustness tricks that make the closed loop survivable:
        #   (a) the head predicts the CHANGE in the code, not the code itself —
        #       under uncertainty the safest change is "nothing moves", which keeps
        #       the ball painted instead of fading it toward an invisible average;
        #   (b) we add a little noise to the input codes, so the network practices
        #       CORRECTING a slightly-wrong code instead of amplifying the error.
        z_in = z[:, :-1] + 0.05 * torch.randn_like(z[:, :-1])
        pred, _ = mnet(z_in, a[:, :-1])
        loss = F.mse_loss(z_in + pred, z[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(idx)
    if (epoch + 1) % 5 == 0:
        print(f"epoch {epoch+1:2d}  prediction error {tot/EPISODES:8.4f}")
print(f"M trained in {time.time()-t0:.0f}s   ({sum(p.numel() for p in mnet.parameters()):,} parameters)")

# %% [markdown]
# ### Practising the closed loop — before we play it for real
#
# **New concept — open loop vs closed loop.** So far, M has trained in **open loop**:
# at every step it received the *real* frame's code, so a small mistake at step 5 could
# not hurt step 6 — step 6 got the truth anyway. At deployment there is no truth: M's
# own prediction becomes its next input, mistakes and all. Errors **compound**. A model
# that is excellent one step ahead can still fall apart when it eats its own output —
# training and deployment are different sports.
#
# So before deployment, we make M **practise the deployment condition**: short rollouts
# where it must survive on its own predictions — first 5 steps, then 15 (a curriculum,
# easy to hard). Two details:
#
# * At every rollout step we also **decode the predicted code back to pixels** and
#   demand the painted frame match the real one (with the same ball-weighting as V's
#   loss). Matching in *code* space is not enough — this pixel check is what keeps the
#   imagined ball a sharp dot instead of letting it diffuse into invisible mist.
# * V is **frozen** here (`requires_grad_(False)`) — we are teaching M to live in V's
#   code space, not letting the two renegotiate it.
#
# We save the trained pair to `pong_wm.pt` at the end — the lecture's demo GIFs are
# made from exactly this checkpoint.

# %%
# fine-tune on short CLOSED-LOOP rollouts: the model must survive eating its own
# predictions for 5 steps. Crucially, we also DECODE each predicted code and demand
# that the painted frame matches the real one — this keeps the imagined ball a
# sharp dot instead of letting it diffuse into invisible mist.
Z_MEAN_d, Z_STD_d = Z_MEAN.to(device), Z_STD.to(device)
for p in vnet.parameters():
    p.requires_grad_(False)
t0 = time.time()
STAGES = [(5, 1e-3, 25), (15, 3e-4, 15)]   # curriculum: survive 5 steps, then 15
for K, lr, n_epochs in STAGES:
  for g in opt.param_groups:
      g["lr"] = lr
  for epoch in range(n_epochs):
    perm, tot = torch.randperm(EPISODES), 0.0
    for i in range(0, EPISODES, 16):
        idx = perm[i:i+16]
        z = zs[idx].to(device); a = a1h[idx].to(device)
        x = torch.tensor(frames[idx.numpy()]).to(device)
        t_start = int(torch.randint(4, T - K - 1, (1,)))
        _, h = mnet(z[:, :t_start], a[:, :t_start])
        zc = z[:, t_start:t_start+1]
        loss = 0.0
        for k in range(K):
            pred, h = mnet(zc, a[:, t_start+k:t_start+k+1], h)
            znext = zc + pred                                # code + predicted change
            loss = loss + F.mse_loss(znext[:, 0], z[:, t_start+k+1])
            xk = x[:, t_start+k+1]
            xhat = vnet.decode(znext[:, 0] * Z_STD_d + Z_MEAN_d)
            wpix = 1.0 + 40.0 * xk[..., :1]
            loss = loss + 0.002 * (wpix * F.binary_cross_entropy(xhat, xk, reduction="none")).sum() / len(idx)
            zc = znext
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(idx)
    if (epoch + 1) % 5 == 0:
        print(f"rollout K={K:2d} epoch {epoch+1:2d}  loss {tot/EPISODES:8.3f}")
print(f"closed-loop fine-tune in {time.time()-t0:.0f}s")
torch.save({"vnet": vnet.state_dict(), "mnet": mnet.state_dict(),
            "z_mean": Z_MEAN, "z_std": Z_STD}, "pong_wm.pt")

# %% [markdown]
# ### The experiment that proves the memory story
#
# Time to test the claim we have been carrying since Step 1. If the memory story is
# true, then prediction error should follow a very specific pattern:
#
# * **at t = 1: LARGE.** The memory has seen only one frame. Velocity is unknowable —
#   the model literally cannot know which way the ball is going, so it must hedge.
# * **from t = 2–3 onward: small, and staying small.** The memory has now seen two
#   positions and can hold their difference. From here on it knows the velocity and
#   just carries it forward.
#
# A falling *cliff* between t=1 and t=3, then a flat floor — that exact shape is the
# fingerprint of a memory doing its job.
#
# **What to look for below:** our run gives error **≈0.094 at t=1 → ≈0.041 by t=3 →
# ≈0.021 at t=10** (yours will match, same seed). Nobody told the network about
# velocity. It *discovered* that keeping the difference between two positions is the
# most useful thing a memory can hold — because prediction is what the loss pays for.

# %%
mnet.eval()
with torch.no_grad():
    z = zs[:100].to(device); a = a1h[:100].to(device)
    pred, _ = mnet(z[:, :-1], a[:, :-1])
    err = ((z[:, :-1] + pred - z[:, 1:]) ** 2).mean(-1).cpu().numpy()   # (100, T-1)
fig, ax = plt.subplots(figsize=(8.2, 3.6))
steps = np.arange(1, 16)
ax.plot(steps, err[:, :15].mean(0), color=TEAL, lw=2.5, marker="o", ms=5)
ax.set_xticks(steps)
ax.set_xlabel("timestep being predicted")
ax.set_ylabel("prediction error")
ax.set_title("The memory needs exactly two frames — then prediction snaps into place")
plt.tight_layout(); plt.savefig("plot_memory_snap.png", dpi=150, bbox_inches="tight"); plt.show()
print(f"prediction error at t=1: {err[:,0].mean():.4f}   at t=3: {err[:,2].mean():.4f}   "
      f"at t=10: {err[:,9].mean():.4f}")

# %% [markdown]
# ### The payoff — the model paints the next frame, before it happens
#
# This is the demo the lecture opens with. At every step of a fresh episode we hand the
# model everything a player would know — the frames so far and the key being pressed —
# and ask it to **paint what the next frame will look like**, *before the game shows
# it*. The model has never seen this episode.
#
# **What to look for below:** top row is reality, bottom row is the model's painting of
# that same frame, made one step in advance. Ball position, ball direction, wall
# bounces, the paddle following the keys — all correct, at every step. Two small
# networks, trained in minutes, genuinely contain how this world works.

# %%
env = MiniPong(seed=8)
obs = env.reset()
real = [obs]
script = [2]*6 + [0]*7 + [1]*4 + [2]*5
for a in [1, 1] + script:
    obs, _ = env.step(a)
    real.append(obs)
with torch.no_grad():
    real_t = torch.tensor(np.stack(real), dtype=torch.float32).to(device)
    mu, _ = vnet.encode(real_t)
    zt = (mu - Z_MEAN.to(device)) / Z_STD.to(device)
    acts = [1, 1] + script
    a_all = F.one_hot(torch.tensor([acts]), 3).float().to(device)
    pred_seq, _ = mnet(zt[None, :-1], a_all[:, :len(zt) - 1])
    znext = zt[None, :-1] + pred_seq                  # predicted next codes
    painted = vnet.decode(znext[0] * Z_STD.to(device) + Z_MEAN.to(device)).cpu().numpy()

offs = 2                                              # skip the 2 warm-up steps
glyph = {0: "left", 1: "stay", 2: "right"}
show = list(range(0, len(script), 3))
fig, axes = plt.subplots(2, len(show), figsize=(13, 4.0))
for i, t in enumerate(show):
    axes[0, i].imshow(np.clip(real[offs + 1 + t], 0, 1)); axes[0, i].axis("off")
    axes[0, i].set_title(f"step {t+1} · key {glyph[script[t]]}", fontsize=8, color=MUTED)
    axes[1, i].imshow(np.clip(painted[offs + t], 0, 1)); axes[1, i].axis("off")
fig.text(0.075, 0.68, "what really\nhappened", fontsize=11, color=TEAL, ha="right", fontweight="bold")
fig.text(0.075, 0.27, "painted by the model\n— before it happened", fontsize=11, color=CLAY, ha="right", fontweight="bold")
fig.suptitle("At every step, the network paints the NEXT frame before the game shows it", y=0.99)
plt.subplots_adjust(left=0.10, top=0.85, bottom=0.03, wspace=0.06, hspace=0.16)
plt.savefig("plot_pred_next.png", dpi=150, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## Part 2 · Deployment — the closed loop, for real
#
# Now the full deployment condition. Switch the game engine OFF, warm the model up on
# just 3 real frames, and then loop with **no help from the game at all**:
#
# 1. the player presses a key (left / stay / right)
# 2. M updates its memory and predicts the next code ẑ
# 3. V's decoder paints ẑ as a frame — that's what the player sees
# 4. **the predicted code is fed back in as if it were real**; repeat
#
# Step 4 is the whole difference from the previous demo. There, every input was the
# truth; here, every input is the model's own previous guess. Around and around — the
# model is now the game.
#
# One sneaky detail in the code (it cost us a real bug): the warm-up consumes codes
# 0..2, so the first input of the closed loop must be the real code at index 3 — feed
# the GRU a code it has already eaten and its sense of time silently desynchronises.
#
# **What to look for below — read it honestly:** the paddle obeys every key, for as
# long as you play. The ball is painted correctly for the first steps, then fades and
# re-forms as the dream drifts. That is not a bug in this notebook — it is THE central
# open problem of world models, and exactly where this series goes next.

# %%
def dream(actions_list, warm_ep=2, warm_steps=3):
    """Run the model closed-loop with a given list of actions; return decoded frames.
    Alignment matters: the warm-up consumes codes 0..warm-1, so the first input of
    the closed loop is the REAL code at index `warm_steps` — never a code the GRU
    has already eaten."""
    with torch.no_grad():
        _, h = mnet(zs[warm_ep:warm_ep+1, :warm_steps].to(device),
                    a1h[warm_ep:warm_ep+1, :warm_steps].to(device))
        z = zs[warm_ep:warm_ep+1, warm_steps:warm_steps+1].to(device)
        out = []
        for a in actions_list:
            ah = F.one_hot(torch.tensor([[a]]), 3).float().to(device)
            pred, h = mnet(z, ah, h)
            z = z + pred[:, -1:]                             # code + predicted change
            z_raw = z[:, 0] * Z_STD.to(device) + Z_MEAN.to(device)   # back to V's scale
            out.append(vnet.decode(z_raw)[0].cpu())
        return torch.stack(out)

# a scripted "player": hold right, hold left, then stay — and run the SAME keys
# through the real engine, so we can put reality and the dream side by side
script = [2]*5 + [0]*5 + [1]*4
env = MiniPong(seed=11)
obs = env.reset()
real = [obs]
for a in [1, 1, 1] + script:                      # 3 warm-up frames, then the script
    obs, _ = env.step(a)
    real.append(obs)
real_t = torch.tensor(np.stack(real), dtype=torch.float32).to(device)
with torch.no_grad():                             # dream from the same 3-frame warm-up
    mu, _ = vnet.encode(real_t)
    zt = ((mu - Z_MEAN.to(device)) / Z_STD.to(device))
    _, h = mnet(zt[None, :3], F.one_hot(torch.tensor([[1, 1, 1]]), 3).float().to(device))
    z = zt[None, 3:4]
    dreamed = []
    for a in script:
        pred, h = mnet(z, F.one_hot(torch.tensor([[a]]), 3).float().to(device), h)
        z = z + pred[:, -1:]
        dreamed.append(vnet.decode(z[:, 0] * Z_STD.to(device) + Z_MEAN.to(device))[0].cpu())

fig, axes = plt.subplots(2, 7, figsize=(13, 4.0))
show = list(range(0, len(script), 2))
glyph = {0: "left", 1: "stay", 2: "right"}
for i, t in enumerate(show):
    axes[0, i].imshow(np.clip(real[4 + t], 0, 1)); axes[0, i].axis("off")
    axes[0, i].set_title(f"step {t+1} · key {glyph[script[t]]}", fontsize=8, color=MUTED)
    axes[1, i].imshow(dreamed[t].clamp(0, 1)); axes[1, i].axis("off")
fig.text(0.075, 0.68, "real game\n(engine on)", fontsize=11, color=TEAL, ha="right", fontweight="bold")
fig.text(0.075, 0.27, "the dream\n(engine off)", fontsize=11, color=CLAY, ha="right", fontweight="bold")
fig.suptitle("The same keys, two worlds — the bottom one is the network's imagination", y=0.99)
plt.subplots_adjust(left=0.10, top=0.85, bottom=0.03, wspace=0.06, hspace=0.16)
plt.savefig("plot_dream_play.png", dpi=150, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## What just happened — and what to take with you
#
# Look back at what these two small networks (167,527 parameters, trained in under ten
# minutes) actually did:
#
# * **V** learned to compress 3,072 pixels into 12 numbers and back — and the ball only
#   survived compression because we made the loss care about it.
# * **M** learned the physics. Its memory accumulates what single frames cannot show —
#   you *measured* the velocity being discovered, as the error cliff between t=1 and
#   t=3. Its prediction head paints the next frame before the game shows it, on
#   episodes it has never seen.
# * Run closed-loop, the two networks ARE a small imagined Pong — the paddle obeys
#   indefinitely, and the ball holds for the first steps before the dream drifts.
#   One-step prediction is near-perfect, yet the loop still degrades: tiny errors
#   compound every time the model eats its own output. **This compounding-error
#   problem is the central problem of world models.** Our V and M were trained
#   *separately* — V never knew its codes would be used for prediction. Models that
#   train them *together*, with latents built for prediction from the ground up, is
#   exactly where the next lecture picks up.
#
# ### Four things to keep
#
# 1. **A world model = compress + remember + predict.** Three jobs, two small networks,
#    one loss.
# 2. **Compression keeps what the loss pays for.** The ball vanished until the loss
#    cared. Watch for this in every model you ever train.
# 3. **Memory earns its keep in exactly two frames.** Velocity lives in the difference
#    between frames, and the memory learned to keep it unprompted.
# 4. **Deployment is a different sport from training.** Practise the closed loop;
#    respect what is genuinely unpredictable; make the world learnable.
#
# ### Exercises (each is a one-line change — rerun and observe)
#
# ① **Shrink the code to `Z = 3`.** What breaks first — the paddle or the ball? Why
#    that one? *(Hint: which object needs more numbers to describe, and which one does
#    the loss weight harder?)*
# ② **Delete the memory:** in `MNet.forward`, pass `h0=None` AND reset `h` between
#    every single step in the `dream` loop. Prediction error at t=3 should now stay as
#    bad as t=1 — exactly Lecture 1's claim that state ≠ observation.
# ③ **Remove the redness weighting** (`w = 1.0`). Retrain V and look at the report
#    card: where did the ball go?
# ④ **Remove the smoothness term** (`smooth = 0`). V's report card will look just as
#    good — but M's prediction error and the dream get worse. A code space can be
#    perfectly decodable and still hostile to prediction.
# ⑤ **Lengthen the dream** to 50+ steps and watch how it drifts. Then make the world
#    *stochastic* — in `pong_env.py`, re-add a random respawn after the ball passes the
#    paddle — retrain, and watch the dreamed ball turn into a smear. A deterministic
#    head facing true randomness predicts the average of all futures — a future that
#    never happens. Richer prediction heads (mixtures, categorical codes) are how real
#    world models answer this; that story starts next lecture.

# %% [markdown]
# ---
# *Vizuara AI · Build a World Model from Scratch · Lecture 3 companion.*
# *All numbers on the lecture slides are this notebook's output at seed 0.*
