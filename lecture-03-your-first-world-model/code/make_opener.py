"""Opener assets: the model paints the NEXT frame, at every step, before it happens.

Given the real frames so far and the player's key, network M predicts the next
code and V's decoder paints it — one step ahead of reality, at every timestep.

Produces plot_pred_next.png (deck strip) and dream_play.gif/.mp4 (animated opener).
"""
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image, ImageDraw, ImageFont

from pong_env import MiniPong
from make_final_plots import VNet, MNet, PAPER, INK, MUTED, TEAL, GOLD, CLAY

rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "text.color": INK, "font.family": "serif",
})

N = 22
SCRIPT = [2] * 6 + [0] * 7 + [1] * 4 + [2] * 5


def main():
    ck = torch.load("pong_wm.pt", map_location="cpu", weights_only=False)
    vnet, mnet = VNet(), MNet()
    vnet.load_state_dict(ck["vnet"]); mnet.load_state_dict(ck["mnet"])
    vnet.eval(); mnet.eval()
    ZM, ZS = ck["z_mean"], ck["z_std"]

    env = MiniPong(seed=8)
    obs = env.reset()
    real = [obs]
    for a in [1, 1] + SCRIPT:
        obs, _ = env.step(a)
        real.append(obs)

    with torch.no_grad():
        real_t = torch.tensor(np.stack(real), dtype=torch.float32)
        zt = (vnet.encode(real_t) - ZM) / ZS
        acts = [1, 1] + SCRIPT
        a1h = F.one_hot(torch.tensor([acts]), 3).float()
        pred_seq, _ = mnet(zt[None, :-1], a1h[:, :len(zt) - 1])   # teacher-forced
        znext = zt[None, :-1] + pred_seq                          # predicted next codes
        painted = vnet.decode(znext[0] * ZS + ZM).numpy()         # painted frame t+1

    # frame indices: painted[i] is the model's guess of real[i+1]
    # show steps 2.. (skip warm-up)
    offs = 2
    pairs = [(real[offs + 1 + k], painted[offs + k], SCRIPT[k]) for k in range(N)]

    # ---------------- deck strip
    show = list(range(0, N, 3))
    glyph = {0: "left", 1: "stay", 2: "right"}
    fig, axes = plt.subplots(2, len(show), figsize=(13, 4.0))
    for i, t in enumerate(show):
        r, p, a = pairs[t]
        axes[0, i].imshow(np.clip(r, 0, 1)); axes[0, i].axis("off")
        axes[0, i].set_title(f"step {t+1} · key {glyph[a]}", fontsize=8, color=MUTED)
        axes[1, i].imshow(np.clip(p, 0, 1)); axes[1, i].axis("off")
    fig.text(0.075, 0.68, "what really\nhappened", fontsize=11, color=TEAL,
             ha="right", va="center", fontweight="bold")
    fig.text(0.075, 0.27, "what the model\npainted — before\nit happened", fontsize=11,
             color=CLAY, ha="right", va="center", fontweight="bold")
    fig.suptitle("At every step, the network paints the NEXT frame — before the game shows it",
                 y=0.99, fontsize=13, fontweight="bold")
    plt.subplots_adjust(left=0.10, top=0.85, bottom=0.03, wspace=0.06, hspace=0.16)
    plt.savefig("plot_pred_next.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("plot_pred_next.png written")

    # ---------------- animated opener
    S = 11
    PW, PH = 32 * S, 32 * S
    W, HH = PW * 2 + 72, PH + 168
    PAPER_RGB, INK_RGB = (251, 249, 241), (40, 38, 34)
    TEAL_RGB, CLAY_RGB = (46, 143, 143), (201, 100, 66)
    try:
        f_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 26)
        f_lab = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 19)
        f_small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia.ttf", 16)
        f_key = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 24)
    except OSError:
        f_title = f_lab = f_small = f_key = ImageFont.load_default()

    def up(fr):
        return Image.fromarray((np.clip(fr, 0, 1) * 255).astype(np.uint8)).resize(
            (PW, PH), Image.NEAREST)

    def render(t):
        r, p, a = pairs[t]
        img = Image.new("RGB", (W, HH), PAPER_RGB)
        d = ImageDraw.Draw(img)
        d.text((24, 12), "The network paints the next frame — before it happens",
               font=f_title, fill=INK_RGB)
        x1, x2, y0 = 24, PW + 48, 56
        img.paste(up(r), (x1, y0)); img.paste(up(p), (x2, y0))
        d.rectangle([x1 - 1, y0 - 1, x1 + PW, y0 + PH], outline=INK_RGB, width=2)
        d.rectangle([x2 - 1, y0 - 1, x2 + PW, y0 + PH], outline=INK_RGB, width=2)
        d.text((x1, y0 + PH + 8), "the real game", font=f_lab, fill=TEAL_RGB)
        d.text((x2, y0 + PH + 8), "the network's painting", font=f_lab, fill=CLAY_RGB)
        d.text((x2, y0 + PH + 32), "predicted from the PREVIOUS frames + the key", font=f_small, fill=INK_RGB)
        ky = y0 + PH + 58
        cx = W // 2 - 80
        for i, (g, code) in enumerate([("<", 0), ("o", 1), (">", 2)]):
            kx = cx + i * 54
            pressed = (a == code)
            d.rounded_rectangle([kx, ky, kx + 44, ky + 44], radius=8,
                                fill=(TEAL_RGB if pressed else PAPER_RGB),
                                outline=INK_RGB, width=3 if pressed else 2)
            tw = d.textlength(g, font=f_key)
            d.text((kx + 22 - tw / 2, ky + 8), g, font=f_key,
                   fill=(PAPER_RGB if pressed else INK_RGB))
        d.text((cx - 4, ky + 48), f"the player's key · step {t+1:02d}", font=f_small, fill=INK_RGB)
        return img

    imgs = [render(t) for t in range(N)]
    imgs += [imgs[-1]] * 3
    imgs[0].save("dream_play.gif", save_all=True, append_images=imgs[1:],
                 duration=300, loop=0)
    print(f"dream_play.gif written  ({len(imgs)} frames)")
    for i, im in enumerate(imgs):
        im.save(f"/tmp/openf_{i:03d}.png")
    r = subprocess.run(
        ["ffmpeg", "-y", "-framerate", "3", "-i", "/tmp/openf_%03d.png",
         "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "dream_play.mp4"], capture_output=True)
    print("dream_play.mp4 written" if r.returncode == 0 else "GIF only")


if __name__ == "__main__":
    main()
