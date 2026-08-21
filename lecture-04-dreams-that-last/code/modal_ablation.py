"""The Lecture-4 ablation: deterministic-only vs stochastic-only vs RSSM.

Same SO-101 data, same budget, same seed — only the state design changes:

  * "deterministic"  — h only (a GRU carrying a vector; no sampled state at all).
                       This is essentially Lecture 3's design, trained jointly.
  * "stochastic"     — s only (a sampled state each step; no deterministic path).
                       PlaNet's SSM: honest dice, leaky memory.
  * "rssm"           — both paths (already trained; retrained here for fairness).

Each arm is evaluated identically: 60-step open-loop dreams on the 5 held-out
episodes, scored in pixel space AND joint-angle space, plus dream strips.

Launch: nohup modal run --detach modal_ablation.py::run_all
"""
import modal

app = modal.App("so101-rssm-ablation")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install("torch", "torchvision", "numpy", "matplotlib", "pillow"))
vol = modal.Volume.from_name("so101-rssm", create_if_missing=True)
HOLD_OUT = 5
CONTEXT, HORIZON = 5, 60

MODEL_SRC = '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EMB = 1024
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
    def __init__(s_, in_dim):
        super().__init__()
        s_.fc = nn.Linear(in_dim, 256 * 16)
        s_.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1))
    def forward(s_, feat):
        x = s_.fc(feat).view(-1, 256, 4, 4)
        return s_.net(x).permute(0, 2, 3, 1) + 0.5

class Model(nn.Module):
    """arm in {deterministic, stochastic, rssm}; total state width kept equal."""
    def __init__(s_, arm, a_dim=6, width=288):
        super().__init__()
        s_.arm = arm
        if arm == "deterministic":
            s_.H, s_.S = width, 0
        elif arm == "stochastic":
            s_.H, s_.S = 0, width
        else:
            s_.H, s_.S = 256, 32
        feat = s_.H + s_.S
        s_.enc = Encoder()
        s_.dec = Decoder(feat)
        s_.joint_head = nn.Sequential(nn.Linear(feat, 256), nn.ELU(),
                                      nn.Linear(256, 6))
        if s_.H:                                  # the deterministic belt
            s_.in_mlp = nn.Sequential(nn.Linear(s_.S + a_dim, 256), nn.ELU())
            s_.gru = nn.GRUCell(256, s_.H)
        if s_.S:                                  # the stochastic channel
            prior_in = s_.H if s_.H else s_.S + a_dim
            s_.prior_mlp = nn.Sequential(nn.Linear(prior_in, 256), nn.ELU(),
                                         nn.Linear(256, 2 * s_.S))
            s_.post_mlp = nn.Sequential(nn.Linear(prior_in + EMB, 256), nn.ELU(),
                                        nn.Linear(256, 2 * s_.S))
        if not s_.S:                              # deterministic needs an obs path
            s_.obs_mlp = nn.Sequential(nn.Linear(s_.H + EMB, 256), nn.ELU(),
                                       nn.Linear(256, s_.H))

    def dist(s_, out):
        mu, std = out.chunk(2, -1)
        return mu, F.softplus(std) + MIN_STD

    def step_h(s_, h, s, a):
        return s_.gru(s_.in_mlp(torch.cat([s, a], -1)), h)

    def feat(s_, h, s):
        if s_.arm == "deterministic":
            return h
        if s_.arm == "stochastic":
            return s
        return torch.cat([h, s], -1)
'''


@app.function(image=image, gpu="A10G", timeout=14400, volumes={"/out": vol})
def train_arm(arm: str, steps: int = 16000):
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F

    ns = {}
    exec(MODEL_SRC, ns)
    Model = ns["Model"]
    dev = torch.device("cuda")
    torch.manual_seed(0); np.random.seed(0)

    z = np.load("/out/so101_64px.npz")
    n_ep = int(z["n_episodes"])
    eps = [{"frames": z[f"f{i}"], "states": z[f"s{i}"], "actions": z[f"a{i}"]}
           for i in range(n_ep)]
    train_eps = eps[:-HOLD_OUT]
    norm = np.load("/out/norm.npz")
    s_mean, s_std = norm["s_mean"], norm["s_std"]
    a_mean, a_std = norm["a_mean"], norm["a_std"]

    B, L = 16, 32
    def batch():
        fr = np.zeros((B, L, 64, 64, 3), np.float32)
        st = np.zeros((B, L, 6), np.float32)
        ac = np.zeros((B, L, 6), np.float32)
        for b in range(B):
            e = train_eps[np.random.randint(len(train_eps))]
            t0 = np.random.randint(0, len(e["frames"]) - L)
            fr[b] = e["frames"][t0:t0+L] / 255.0
            st[b] = (e["states"][t0:t0+L] - s_mean) / s_std
            ac[b] = (e["actions"][t0:t0+L] - a_mean) / a_std
        return (torch.tensor(fr).to(dev), torch.tensor(st).to(dev),
                torch.tensor(ac).to(dev))

    m = Model(arm).to(dev)
    n_par = sum(p.numel() for p in m.parameters())
    print(f"[{arm}] parameters: {n_par:,}  (H={m.H}, S={m.S})", flush=True)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4, eps=1e-5)

    def kl(m1, s1, m2, s2):
        return (torch.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2)
                / (2 * s2 ** 2) - 0.5).sum(-1)

    t0 = time.time()
    for step in range(steps):
        fr, st, ac = batch()
        emb = m.enc(fr.reshape(B * L, 64, 64, 3)).view(B, L, -1)
        h = torch.zeros(B, m.H, device=dev) if m.H else None
        s = torch.zeros(B, m.S, device=dev) if m.S else None
        l_rec = l_st = l_kl = 0.0
        for t in range(L):
            if t > 0:
                if m.arm == "deterministic":
                    h = m.step_h(h, torch.zeros(B, 0, device=dev), ac[:, t-1])
                elif m.arm == "stochastic":
                    pass                       # prior is computed from (s, a) below
                else:
                    h = m.step_h(h, s, ac[:, t-1])
            if m.arm == "deterministic":
                # observations enter by correcting h directly (no sampling anywhere)
                h = h + m.obs_mlp(torch.cat([h, emb[:, t]], -1))
                feat = m.feat(h, None)
            else:
                cond = h if m.H else torch.cat(
                    [s, ac[:, t-1] if t > 0 else torch.zeros(B, 6, device=dev)], -1)
                pm, ps = m.dist(m.prior_mlp(cond))
                qm, qs = m.dist(m.post_mlp(torch.cat([cond, emb[:, t]], -1)))
                s = qm + qs * torch.randn_like(qs)
                l_kl = l_kl + torch.clamp(
                    0.8 * kl(qm.detach(), qs.detach(), pm, ps).mean()
                    + 0.2 * kl(qm, qs, pm.detach(), ps.detach()).mean(), min=1.0)
                feat = m.feat(h, s)
            l_rec = l_rec + ((m.dec(feat) - fr[:, t]) ** 2).sum(dim=(1, 2, 3)).mean()
            l_st = l_st + ((m.joint_head(feat) - st[:, t]) ** 2).sum(-1).mean()
        loss = (l_rec + 10.0 * l_st + 1.0 * l_kl) / L
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 100.0)
        opt.step()
        if step % 2000 == 0:
            klv = float(l_kl) / L if m.S else 0.0
            print(f"[{arm}] step {step:6d}  rec {float(l_rec)/L:8.1f}  "
                  f"joint {float(l_st)/L:6.3f}  kl {klv:5.2f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    torch.save({"sd": m.state_dict(), "arm": arm, "n_par": n_par},
               f"/out/ablation_{arm}.pt")
    vol.commit()
    print(f"[{arm}] done in {(time.time()-t0)/60:.0f} min", flush=True)
    return arm


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/out": vol})
def compare():
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
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

    ns = {}
    exec(MODEL_SRC, ns)
    Model = ns["Model"]
    dev = torch.device("cuda")

    z = np.load("/out/so101_64px.npz")
    n_ep = int(z["n_episodes"])
    eps = [{"frames": z[f"f{i}"], "states": z[f"s{i}"], "actions": z[f"a{i}"]}
           for i in range(n_ep)]
    norm = np.load("/out/norm.npz")
    s_mean, s_std = norm["s_mean"], norm["s_std"]
    a_mean, a_std = norm["a_mean"], norm["a_std"]

    ARMS = [("deterministic", "deterministic only", TEAL),
            ("stochastic", "stochastic only", GOLD),
            ("rssm", "RSSM (both)", CLAY)]
    results, strips = {}, {}

    for arm, label, _c in ARMS:
        ck = torch.load(f"/out/ablation_{arm}.pt", map_location=dev)
        m = Model(arm).to(dev); m.load_state_dict(ck["sd"]); m.eval()
        pix_all, jnt_all = [], []
        for ei, e in enumerate(eps[-HOLD_OUT:]):
            if len(e["frames"]) < CONTEXT + HORIZON + 10:
                continue
            start = max(0, len(e["frames"]) // 3 - CONTEXT)
            fr = torch.tensor(e["frames"][start:start+CONTEXT+HORIZON] / 255.0,
                              dtype=torch.float32).to(dev)
            st = (e["states"][start:start+CONTEXT+HORIZON] - s_mean) / s_std
            ac = torch.tensor((e["actions"][start:start+CONTEXT+HORIZON] - a_mean)
                              / a_std, dtype=torch.float32).to(dev)
            with torch.no_grad():
                emb = m.enc(fr)
                h = torch.zeros(1, m.H, device=dev) if m.H else None
                s = torch.zeros(1, m.S, device=dev) if m.S else None
                for t in range(CONTEXT):                     # warm-up with frames
                    if t > 0:
                        if arm == "deterministic":
                            h = m.step_h(h, torch.zeros(1, 0, device=dev), ac[t-1:t])
                        elif arm == "rssm":
                            h = m.step_h(h, s, ac[t-1:t])
                    if arm == "deterministic":
                        h = h + m.obs_mlp(torch.cat([h, emb[t:t+1]], -1))
                    else:
                        cond = h if m.H else torch.cat(
                            [s, ac[t-1:t] if t > 0 else torch.zeros(1, 6, device=dev)], -1)
                        qm, _ = m.dist(m.post_mlp(torch.cat([cond, emb[t:t+1]], -1)))
                        s = qm
                pix, jnt, frames = [], [], []
                for k in range(HORIZON):                     # open-loop dream
                    if arm == "deterministic":
                        h = m.step_h(h, torch.zeros(1, 0, device=dev),
                                     ac[CONTEXT+k-1:CONTEXT+k])
                    elif arm == "rssm":
                        h = m.step_h(h, s, ac[CONTEXT+k-1:CONTEXT+k])
                    if arm != "deterministic":
                        cond = h if m.H else torch.cat(
                            [s, ac[CONTEXT+k-1:CONTEXT+k]], -1)
                        pm, _ = m.dist(m.prior_mlp(cond))
                        s = pm
                    feat = m.feat(h, s)
                    xhat = m.dec(feat)[0].clamp(0, 1)
                    jhat = m.joint_head(feat)[0].cpu().numpy()
                    frames.append(xhat.cpu().numpy())
                    pix.append(float(((xhat - fr[CONTEXT+k]) ** 2).mean()))
                    jnt.append(float(((jhat - st[CONTEXT+k]) ** 2).mean()))
            pix_all.append(pix); jnt_all.append(jnt)
            if ei == 0:
                strips[arm] = (frames, fr[CONTEXT:CONTEXT+HORIZON].cpu().numpy())
        results[arm] = (np.array(pix_all), np.array(jnt_all), ck["n_par"])
        print(f"{label:20s} params {ck['n_par']:,}  "
              f"pix@60 {np.array(pix_all)[:, -1].mean():.4f}  "
              f"joint@60 {np.array(jnt_all)[:, -1].mean():.4f}", flush=True)

    # ---- error-vs-horizon comparison
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6))
    x = np.arange(1, HORIZON + 1)
    for arm, label, c in ARMS:
        pix, jnt, _ = results[arm]
        axes[0].plot(x, pix.mean(0), lw=3, color=c, label=label)
        axes[1].plot(x, jnt.mean(0), lw=3, color=c, label=label)
    axes[0].set_title("pixel error"); axes[1].set_title("joint-angle error")
    for a in axes:
        a.set_xlabel("dream step (no camera input)")
    axes[0].legend(frameon=False)
    fig.suptitle("Same data, same budget — only the state design differs",
                 fontsize=15)
    plt.tight_layout()
    plt.savefig("/out/assets/fig_ablation_error.png", dpi=140,
                bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)

    # ---- visual comparison: real on top, then one row per arm
    def up(a, k=5):
        im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
        return np.array(im.resize((64 * k, 64 * k), Image.LANCZOS)) / 255.0
    picks = [0, 11, 23, 35, 47, 59]
    fig, axes = plt.subplots(4, len(picks), figsize=(14, 9.4))
    real = strips["rssm"][1]
    for i, k in enumerate(picks):
        axes[0, i].imshow(up(real[k])); axes[0, i].axis("off")
        axes[0, i].set_title(f"+{k+1}", fontsize=10, color=MUTED)
    for r, (arm, label, c) in enumerate(ARMS, start=1):
        for i, k in enumerate(picks):
            axes[r, i].imshow(up(strips[arm][0][k])); axes[r, i].axis("off")
    labels = ["the real robot"] + [l for _a, l, _c in ARMS]
    colors = [INK] + [c for _a, _l, c in ARMS]
    for r, (lab, col) in enumerate(zip(labels, colors)):
        axes[r, 0].text(-0.12, 0.5, lab, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=13, color=col,
                        fontweight="bold")
    fig.suptitle("Sixty steps of open-loop dreaming — three state designs",
                 fontsize=16)
    plt.subplots_adjust(left=0.13, top=0.93, wspace=0.05, hspace=0.08)
    plt.savefig("/out/assets/fig_ablation_strips.png", dpi=130,
                bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)

    summary = {a: {"pix60": float(results[a][0][:, -1].mean()),
                   "joint60": float(results[a][1][:, -1].mean()),
                   "pix_mean": float(results[a][0].mean()),
                   "joint_mean": float(results[a][1].mean()),
                   "params": int(results[a][2])} for a, _l, _c in ARMS}
    import json
    json.dump(summary, open("/out/assets/ablation_summary.json", "w"), indent=1)
    vol.commit()
    print("ablation figures written")
    return summary


@app.function(image=image, timeout=43200, volumes={"/out": vol})
def run_all():
    for arm in ("deterministic", "stochastic", "rssm"):
        train_arm.remote(arm, steps=16000)
    return compare.remote()
