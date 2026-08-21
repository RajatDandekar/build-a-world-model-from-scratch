"""RSSM pilot on the SO-101 pick-and-place dataset (lerobot/svla_so101_pickplace).

Validates the Lecture 4 premise before any slides exist:
  1. prep():  download the dataset, decode the 'up' camera to 64x64, save tensors
  2. train(): train a small Dreamer-style RSSM (h=256 deterministic GRU + s=32
              Gaussian stochastic) with pixel + joint-angle reconstruction and
              KL(posterior || prior), free bits + KL balancing
  3. evaluate(): open-loop dreams on held-out episodes -> strips, joint curves,
              per-step error, and the posterior-sigma trace

Launch:  modal run --detach modal_rssm_pilot.py
Results in the 'so101-rssm' volume under /out.
"""
import modal

app = modal.App("so101-rssm-pilot")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         # no torch pin: let lerobot resolve a torch/torchcodec pair with
         # matching ABI (a pinned torch broke torchcodec's shared library)
         .pip_install("lerobot", "matplotlib"))
vol = modal.Volume.from_name("so101-rssm", create_if_missing=True)

DATASET = "lerobot/svla_so101_pickplace"
IMG_SIZE = 64
HOLD_OUT = 5          # last N episodes reserved for evaluation


# ----------------------------------------------------------------- data prep
@app.function(image=image, timeout=3600, volumes={"/out": vol},
              memory=16384, cpu=8)
def prep():
    import numpy as np
    import torch
    import torch.nn.functional as Fn
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(DATASET)
    print("dataset:", ds.num_episodes, "episodes,", ds.num_frames, "frames")
    print("camera keys:", ds.meta.video_keys)

    loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=8,
                                         shuffle=False)
    frames, states, actions, ep_idx = [], [], [], []
    for i, batch in enumerate(loader):
        img = batch["observation.images.up"]            # (B,3,480,640) in [0,1]
        img = Fn.interpolate(img, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                             align_corners=False)
        frames.append((img.permute(0, 2, 3, 1) * 255).to(torch.uint8).numpy())
        states.append(batch["observation.state"].numpy())
        actions.append(batch["action"].numpy())
        ep_idx.append(batch["episode_index"].numpy())
        if i % 20 == 0:
            print(f"  batch {i}/{len(loader)}")
    frames = np.concatenate(frames)
    states = np.concatenate(states).astype(np.float32)
    actions = np.concatenate(actions).astype(np.float32)
    ep_idx = np.concatenate(ep_idx)

    # split the flat arrays into per-episode lists (episodes have varying length)
    episodes = []
    for e in np.unique(ep_idx):
        m = ep_idx == e
        episodes.append({"frames": frames[m], "states": states[m],
                         "actions": actions[m]})
    lens = [len(e["frames"]) for e in episodes]
    print(f"{len(episodes)} episodes, len min/mean/max = "
          f"{min(lens)}/{np.mean(lens):.0f}/{max(lens)}")

    np.savez_compressed("/out/so101_64px.npz",
                        n_episodes=len(episodes),
                        **{f"f{i}": ep["frames"] for i, ep in enumerate(episodes)},
                        **{f"s{i}": ep["states"] for i, ep in enumerate(episodes)},
                        **{f"a{i}": ep["actions"] for i, ep in enumerate(episodes)})
    vol.commit()
    print("saved /out/so101_64px.npz")


# ----------------------------------------------------------------- the model
RSSM_SRC = """
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H, S, EMB = 256, 32, 1024
MIN_STD = 0.1

class Encoder(nn.Module):
    def __init__(s_):
        super().__init__()
        s_.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ELU(),     # 32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ELU(),    # 16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ELU(),   # 8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ELU(),  # 4
            nn.Flatten(), nn.Linear(256 * 16, EMB), nn.ELU())
    def forward(s_, x):                                # x: (B,64,64,3) in [0,1]
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
        return s_.net(x).permute(0, 2, 3, 1) + 0.5     # (B,64,64,3)

class RSSM(nn.Module):
    def __init__(s_, a_dim=6):
        super().__init__()
        s_.enc = Encoder()
        s_.dec = Decoder()
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
"""


@app.function(image=image, gpu="A10G", timeout=14400, volumes={"/out": vol})
def train(steps: int = 20000):
    import io, time
    import numpy as np
    import torch
    import torch.nn.functional as F

    ns = {}
    exec(RSSM_SRC, ns)
    RSSM = ns["RSSM"]
    dev = torch.device("cuda")
    torch.manual_seed(0); np.random.seed(0)

    z = np.load("/out/so101_64px.npz")
    n_ep = int(z["n_episodes"])
    eps = [{"frames": z[f"f{i}"], "states": z[f"s{i}"], "actions": z[f"a{i}"]}
           for i in range(n_ep)]
    train_eps = eps[:-HOLD_OUT]

    # normalisation stats for states/actions (computed on train episodes)
    all_s = np.concatenate([e["states"] for e in train_eps])
    all_a = np.concatenate([e["actions"] for e in train_eps])
    s_mean, s_std = all_s.mean(0), all_s.std(0) + 1e-4
    a_mean, a_std = all_a.mean(0), all_a.std(0) + 1e-4
    np.savez("/out/norm.npz", s_mean=s_mean, s_std=s_std, a_mean=a_mean, a_std=a_std)

    B, L = 16, 32
    def sample_batch():
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

    model = RSSM().to(dev)
    print(f"RSSM parameters: {sum(p.numel() for p in model.parameters()):,}")
    import os
    if os.path.exists("/out/rssm_pilot.pt"):
        model.load_state_dict(torch.load("/out/rssm_pilot.pt", map_location=dev))
        print("resumed from /out/rssm_pilot.pt")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    FREE_NATS = 1.0
    t0 = time.time()
    for step in range(steps):
        fr, st, ac = sample_batch()
        emb = model.enc(fr.reshape(B * L, 64, 64, 3)).view(B, L, -1)
        h = torch.zeros(B, 256, device=dev)
        s = torch.zeros(B, 32, device=dev)
        losses_rec, losses_st, losses_kl = 0, 0, 0
        for t in range(L):
            if t > 0:
                h = model.step_h(h, s, ac[:, t - 1])
            pm, ps = model.prior(h)
            qm, qs = model.posterior(h, emb[:, t])
            s = qm + qs * torch.randn_like(qs)
            xhat = model.dec(h, s)
            sthat = model.joint_head(torch.cat([h, s], -1))
            losses_rec = losses_rec + ((xhat - fr[:, t]) ** 2).sum(dim=(1, 2, 3)).mean()
            losses_st = losses_st + ((sthat - st[:, t]) ** 2).sum(-1).mean()
            # KL with balancing (0.8 toward training the prior)
            def kl(m1, s1, m2, s2):
                return (torch.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2)
                        / (2 * s2 ** 2) - 0.5).sum(-1)
            kl_prior = kl(qm.detach(), qs.detach(), pm, ps).mean()
            kl_post = kl(qm, qs, pm.detach(), ps.detach()).mean()
            k = 0.8 * kl_prior + 0.2 * kl_post
            losses_kl = losses_kl + torch.clamp(k, min=FREE_NATS)
        loss = (losses_rec + 10.0 * losses_st + 1.0 * losses_kl) / L
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()
        if step % 500 == 0:
            print(f"step {step:6d}  rec {losses_rec.item()/L:9.1f}  "
                  f"joint {losses_st.item()/L:7.3f}  kl {losses_kl.item()/L:6.2f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if step > 0 and step % 1000 == 0:
            torch.save(model.state_dict(), "/out/rssm_pilot.pt")
            vol.commit()
    torch.save(model.state_dict(), "/out/rssm_pilot.pt")
    vol.commit()
    print(f"trained {steps} steps in {(time.time()-t0)/60:.0f} min")


# ----------------------------------------------------------------- evaluation
@app.function(image=image, gpu="A10G", timeout=1800, volumes={"/out": vol})
def evaluate():
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    model = RSSM().to(dev)
    model.load_state_dict(torch.load("/out/rssm_pilot.pt", map_location=dev))
    model.eval()

    CONTEXT, HORIZON = 5, 60
    for ei, e in enumerate(eps[-HOLD_OUT:]):
        T = len(e["frames"])
        start = max(0, T // 3 - CONTEXT)
        fr = torch.tensor(e["frames"][start:start+CONTEXT+HORIZON] / 255.0,
                          dtype=torch.float32).to(dev)
        st = (e["states"][start:start+CONTEXT+HORIZON] - s_mean) / s_std
        ac = torch.tensor((e["actions"][start:start+CONTEXT+HORIZON] - a_mean)
                          / a_std, dtype=torch.float32).to(dev)
        if len(fr) < CONTEXT + HORIZON:
            continue
        with torch.no_grad():
            emb = model.enc(fr)
            h = torch.zeros(1, 256, device=dev)
            s = torch.zeros(1, 32, device=dev)
            sigmas = []
            # posterior warm-up on the context
            for t in range(CONTEXT):
                if t > 0:
                    h = model.step_h(h, s, ac[t - 1:t])
                qm, qs = model.posterior(h, emb[t:t+1])
                s = qm
                sigmas.append(float(qs.mean()))
            # open-loop dream with the real actions
            dream_frames, dream_joints, pix_err, joint_err = [], [], [], []
            for k in range(HORIZON):
                h = model.step_h(h, s, ac[CONTEXT + k - 1:CONTEXT + k])
                pm, ps = model.prior(h)
                s = pm                       # mean rollout for the pilot
                xhat = model.dec(h, s)[0].clamp(0, 1)
                jhat = model.joint_head(torch.cat([h, s], -1))[0]
                dream_frames.append(xhat.cpu().numpy())
                dream_joints.append(jhat.cpu().numpy())
                pix_err.append(float(((xhat - fr[CONTEXT + k]) ** 2).mean()))
                joint_err.append(float(((jhat.cpu().numpy() - st[CONTEXT + k]) ** 2).mean()))

        # strip: real vs dream
        picks = [0, 9, 19, 29, 44, 59]
        fig, axes = plt.subplots(2, len(picks), figsize=(14, 4.4))
        for i, k in enumerate(picks):
            axes[0, i].imshow(fr[CONTEXT + k].cpu().numpy()); axes[0, i].axis("off")
            axes[0, i].set_title(f"+{k+1} steps", fontsize=9)
            axes[1, i].imshow(dream_frames[k]); axes[1, i].axis("off")
        axes[0, 0].set_ylabel("real")
        fig.suptitle(f"held-out episode {ei}: real (top) vs open-loop dream (bottom), "
                     f"context {CONTEXT} frames")
        plt.tight_layout()
        plt.savefig(f"/out/eval_strip_{ei}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        # joint curves
        dream_joints = np.array(dream_joints)
        real_joints = st[CONTEXT:CONTEXT + HORIZON]
        fig, axes = plt.subplots(2, 3, figsize=(13, 5.5))
        names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                 "wrist_flex", "wrist_roll", "gripper"]
        for j in range(6):
            ax = axes[j // 3, j % 3]
            ax.plot(real_joints[:, j], lw=2, label="real")
            ax.plot(dream_joints[:, j], lw=2, ls="--", label="dreamed")
            ax.set_title(names[j], fontsize=10)
        axes[0, 0].legend()
        fig.suptitle(f"episode {ei}: joint angles, real vs open-loop dream (normalised)")
        plt.tight_layout()
        plt.savefig(f"/out/eval_joints_{ei}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"episode {ei}: pix MSE @1/@30/@60 = {pix_err[0]:.4f}/{pix_err[29]:.4f}/"
              f"{pix_err[59]:.4f}   joint MSE @1/@30/@60 = {joint_err[0]:.3f}/"
              f"{joint_err[29]:.3f}/{joint_err[59]:.3f}")

    # sigma trace across one full held-out episode (posterior, teacher-forced)
    e = eps[-1]
    fr = torch.tensor(e["frames"] / 255.0, dtype=torch.float32).to(dev)
    ac = torch.tensor((e["actions"] - a_mean) / a_std, dtype=torch.float32).to(dev)
    with torch.no_grad():
        emb = model.enc(fr)
        h = torch.zeros(1, 256, device=dev)
        s = torch.zeros(1, 32, device=dev)
        sig = []
        for t in range(len(fr)):
            if t > 0:
                h = model.step_h(h, s, ac[t - 1:t])
            qm, qs = model.posterior(h, emb[t:t+1])
            s = qm
            sig.append(float(qs.mean()))
    grip = e["actions"][:, 5]
    fig, ax1 = plt.subplots(figsize=(11, 3.4))
    ax1.plot(sig, lw=1.8, label="posterior sigma (mean)")
    ax2 = ax1.twinx()
    ax2.plot(grip, lw=1.2, alpha=0.5, color="tab:orange", label="gripper action")
    ax1.set_xlabel("timestep"); ax1.set_ylabel("sigma"); ax2.set_ylabel("gripper")
    fig.suptitle("does uncertainty spike at contact events?")
    plt.tight_layout()
    plt.savefig("/out/eval_sigma.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    vol.commit()
    print("evaluation artifacts saved")


@app.function(image=image, gpu="A10G", timeout=21600, volumes={"/out": vol})
def train_and_eval(steps: int = 16000):
    """One function so `modal run --detach` keeps the whole job alive."""
    train.local(steps=steps)
    evaluate.local()


@app.local_entrypoint()
def main():
    prep.remote()
    train_and_eval.remote(steps=16000)
    print("pilot complete — fetch /out from the so101-rssm volume")
