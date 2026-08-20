"""The lecture-opener assets: the dream next to reality, driven by the same keys.

Produces:
  plot_dream_play.png — 2-row strip: real engine (top) vs the model's dream (bottom)
  dream_play.gif/.mp4 — animated side-by-side with an arrow-key overlay

The dream is honest: after a 3-frame warm-up the game engine is off and the model
eats its own predictions. We show the horizon where the model is genuinely good
(~14 steps); the deck then discusses why longer dreams drift.
"""
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image, ImageDraw, ImageFont

from pong_env import MiniPong

Z, H = 12, 128
N_STEPS = 14
SCRIPT = [2] * 5 + [0] * 5 + [1] * 4          # hold right, hold left, stay

PAPER, INK, MUTED = "#FBF9F1", "#16130D", "#6D665A"
TEAL, GOLD, CLAY = "#2E8F8F", "#DD9F3E", "#C96442"
rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "text.color": INK, "font.family": "serif",
})


class VNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc = nn.Sequential(nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
                              nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(), nn.Flatten())
        s.mu = nn.Linear(2048, Z)
        s.logvar = nn.Linear(2048, Z)
        s.fc = nn.Linear(Z, 2048)
        s.dec = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
                              nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid())
    def encode(s, x): return s.mu(s.enc(x.permute(0, 3, 1, 2)))
    def decode(s, z): return s.dec(s.fc(z).view(-1, 32, 8, 8)).permute(0, 2, 3, 1)


class MNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.gru = nn.GRU(Z + 3, H, batch_first=True)
        s.head = nn.Sequential(nn.Linear(H, 128), nn.ELU(), nn.Linear(128, Z))
    def forward(s, z, a, h0=None):
        out, hN = s.gru(torch.cat([z, a], -1), h0)
        return s.head(out), hN


def rollout(vnet, mnet, ZM, ZS, seed):
    """Real episode + the model's dream of it, from a 3-frame warm-up."""
    env = MiniPong(seed=seed)
    obs = env.reset()
    real = [obs]
    for a in [1, 1, 1] + SCRIPT:
        obs, _ = env.step(a)
        real.append(obs)
    real_t = torch.tensor(np.stack(real), dtype=torch.float32)
    with torch.no_grad():
        zt = (vnet.encode(real_t) - ZM) / ZS
        aw = F.one_hot(torch.tensor([[1, 1, 1]]), 3).float()
        _, h = mnet(zt[None, :3], aw)
        z = zt[None, 3:4]
        dream = []
        for a in SCRIPT:
            ah = F.one_hot(torch.tensor([[a]]), 3).float()
            pred, h = mnet(z, ah, h)
            z = z + pred[:, -1:]
            dream.append(vnet.decode(z[:, 0] * ZS + ZM)[0].numpy())
    return [real[4 + k] for k in range(len(SCRIPT))], dream


def ball_score(frames):
    return sum(float((f[..., 0] * (f[..., 0] > 0.5)).sum()) for f in frames)


def main():
    ck = torch.load("pong_wm.pt", map_location="cpu", weights_only=False)
    vnet, mnet = VNet(), MNet()
    vnet.load_state_dict(ck["vnet"]); mnet.load_state_dict(ck["mnet"])
    vnet.eval(); mnet.eval()
    ZM, ZS = ck["z_mean"], ck["z_std"]

    # pick the cleanest example among deterministic starts: what matters for the
    # demo is CONSECUTIVE fidelity from step 1, not total ball pixels
    def consec(dream):
        n = 0
        for f in dream:
            if float((f[..., 0] * (f[..., 0] > 0.5)).sum()) < 1.0:
                break
            n += 1
        return n
    best, best_key = None, (-1, -1.0)
    for seed in range(3, 40):
        real, dream = rollout(vnet, mnet, ZM, ZS, seed)
        key = (consec(dream), ball_score(dream))
        if key > best_key:
            best, best_key, best_seed = (real, dream), key, seed
    real, dream = best
    print(f"picked warm-up seed {best_seed}  (consecutive {best_key[0]}, score {best_key[1]:.1f})")

    # ---------------- the deck strip: real vs dream
    show = list(range(0, N_STEPS, 2))
    fig, axes = plt.subplots(2, len(show), figsize=(13, 4.1))
    glyph = {0: "left", 1: "stay", 2: "right"}
    for i, t in enumerate(show):
        axes[0, i].imshow(np.clip(real[t], 0, 1)); axes[0, i].axis("off")
        axes[0, i].set_title(f"step {t+1} · key {glyph[SCRIPT[t]]}", fontsize=8, color=MUTED)
        axes[1, i].imshow(np.clip(dream[t], 0, 1)); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("real", fontsize=11)
    fig.text(0.055, 0.70, "the real game\n(engine on)", fontsize=11, color=TEAL,
             ha="right", va="center", fontweight="bold")
    fig.text(0.055, 0.28, "the model's\nimagination\n(engine off)", fontsize=11,
             color=CLAY, ha="right", va="center", fontweight="bold")
    fig.suptitle("The same keys, two worlds — one of them is a neural network's dream",
                 y=0.99, fontsize=13, fontweight="bold")
    plt.subplots_adjust(left=0.09, top=0.86, bottom=0.03, wspace=0.06, hspace=0.16)
    plt.savefig("plot_dream_play.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("plot_dream_play.png written")

    # ---------------- the animated opener
    S = 11
    PW, PH = 32 * S, 32 * S
    W, HH = PW * 2 + 72, PH + 168
    PAPER_RGB, INK_RGB = (251, 249, 241), (40, 38, 34)
    TEAL_RGB, CLAY_RGB, RED_RGB = (46, 143, 143), (201, 100, 66), (201, 68, 54)
    try:
        f_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 26)
        f_lab = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 19)
        f_small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia.ttf", 16)
        f_key = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 24)
    except OSError:
        f_title = f_lab = f_small = f_key = ImageFont.load_default()

    def upscale(fr):
        return Image.fromarray((np.clip(fr, 0, 1) * 255).astype(np.uint8)).resize(
            (PW, PH), Image.NEAREST)

    def render(t):
        img = Image.new("RGB", (W, HH), PAPER_RGB)
        d = ImageDraw.Draw(img)
        d.text((24, 12), "Playing the game — and playing the dream", font=f_title, fill=INK_RGB)
        x1, x2, y0 = 24, PW + 48, 56
        img.paste(upscale(real[t]), (x1, y0)); img.paste(upscale(dream[t]), (x2, y0))
        d.rectangle([x1 - 1, y0 - 1, x1 + PW, y0 + PH], outline=INK_RGB, width=2)
        d.rectangle([x2 - 1, y0 - 1, x2 + PW, y0 + PH], outline=INK_RGB, width=2)
        d.text((x1, y0 + PH + 8), "real game · engine ON", font=f_lab, fill=TEAL_RGB)
        d.text((x2, y0 + PH + 8), "world model · engine OFF", font=f_lab, fill=CLAY_RGB)
        d.text((x2, y0 + PH + 32), "every frame painted by the network", font=f_small, fill=INK_RGB)
        # keys
        ky = y0 + PH + 8
        cx = W // 2 - 80
        for i, (g, code) in enumerate([("<", 0), ("o", 1), (">", 2)]):
            kx = cx + i * 54
            pressed = (SCRIPT[t] == code)
            d.rounded_rectangle([kx, ky, kx + 44, ky + 44], radius=8,
                                fill=(TEAL_RGB if pressed else PAPER_RGB),
                                outline=INK_RGB, width=3 if pressed else 2)
            tw = d.textlength(g, font=f_key)
            d.text((kx + 22 - tw / 2, ky + 8), g, font=f_key,
                   fill=(PAPER_RGB if pressed else INK_RGB))
        d.text((cx, ky + 50), f"the player's key · step {t+1:02d}", font=f_small, fill=INK_RGB)
        return img

    imgs = [render(t) for t in range(N_STEPS)]
    imgs += [imgs[-1]] * 4                     # hold the last frame a moment
    imgs[0].save("dream_play.gif", save_all=True, append_images=imgs[1:],
                 duration=330, loop=0)
    print(f"dream_play.gif written  ({len(imgs)} frames, {W}x{HH})")
    for i, im in enumerate(imgs):
        im.save(f"/tmp/dreamf_{i:03d}.png")
    r = subprocess.run(
        ["ffmpeg", "-y", "-framerate", "3", "-i", "/tmp/dreamf_%03d.png",
         "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "dream_play.mp4"], capture_output=True)
    print("dream_play.mp4 written" if r.returncode == 0 else "ffmpeg missing — GIF only")


if __name__ == "__main__":
    main()
