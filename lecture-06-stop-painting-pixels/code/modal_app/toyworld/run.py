"""Train every toy architecture and dump raw arrays for the motivation deck.

Outputs to /results/toy/:
  fork_models.npz     energy grids over ball position per architecture,
                      gen_det's blurry prediction, the two true futures,
                      jea collapse trace, sample context/target frames
  camera.npz          true vs inferred (dx, dy) for CameraJEPA + samples
  audio.npz           energy over continuation pitch per architecture,
                      gen_det's blurred spectrogram, sample spectrograms
  summary.json        the headline numbers
"""
import json
import os
import numpy as np
import torch

from . import world
from .models import GenDet, AE, JEA, JEPA, CameraJEPA

OUT = "/results/toy"
STEPS = 4000
BATCH = 128


def train(model, batch_fn, steps=STEPS, lr=1e-3, device="cuda", trace=None):
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    for step in range(steps):
        x, y, _ = batch_fn(BATCH, rng)
        x, y = x.to(device), y.to(device)
        loss = model.loss(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if hasattr(model, "ema_update"):
            model.ema_update()
        if trace is not None and step % 50 == 0 and hasattr(model, "feat_std"):
            trace.append((step, model.feat_std(y)))
    model.eval()
    return model


@torch.no_grad()
def energy_grid(model, ctx, target_at, xs, ys, device):
    """Mean energy over a fixed context batch for targets rendered on a grid."""
    grid = np.zeros((len(ys), len(xs)), np.float32)
    for i, py in enumerate(ys):
        targets = torch.from_numpy(np.stack(
            [target_at((px, py)) for px in xs])).to(device)   # (nx,1,S,S)
        for j in range(len(xs)):
            y = targets[j:j + 1].expand(ctx.size(0), -1, -1, -1)
            grid[i, j] = float(model.energy(ctx, y).mean())
    return grid


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(7)
    summary = {}

    # ---------------------------------------------------------------- fork
    ctx_fix, _, _ = world.fork_batch(64, np.random.default_rng(123))
    # reference context without jitter for clean landscapes
    c0 = np.stack([world.render_ball(*world.fork_positions(0, t))
                   for t in (0, 1, 2)])
    ctx_ref = torch.from_numpy(c0)[None].expand(64, -1, -1, -1).to(device)

    xs = np.linspace(2, 46, 23)
    ys = np.linspace(2, 46, 23)
    grids, extras = {}, {}

    jea_trace = []
    models = {
        "gen_det": train(GenDet(3), world.fork_batch, device=device),
        "ae": train(AE(3), world.anyball_batch, device=device),
        "jea": train(JEA(3), world.fork_batch, device=device,
                     trace=jea_trace),
        "jepa_z": train(JEPA(3, K=4), world.fork_batch, steps=8000, device=device),
        "jepa_noz": train(JEPA(3, K=1), world.fork_batch, steps=8000, device=device),
    }
    for name, m in models.items():
        grids[name] = energy_grid(m, ctx_ref, world.fork_target_at,
                                  xs, ys, device)

    # gen_det's blurry mean prediction + the two true futures
    pred = models["gen_det"].predict(ctx_ref[:1]).cpu().numpy()[0, 0]
    modes = world.fork_mode_positions()
    true0 = world.fork_target_at(modes[0])[0]
    true1 = world.fork_target_at(modes[1])[0]

    # does jepa's discrete z separate the two branches?
    xb, yb, br = world.fork_batch(512, np.random.default_rng(9))
    _, zhat = models["jepa_z"].energy(xb.to(device), yb.to(device),
                                     return_z=True)
    zhat = zhat.cpu().numpy()
    br = br.numpy()
    # purity: best mapping z->branch
    purity = 0.0
    for k in np.unique(zhat):
        sel = zhat == k
        if sel.sum():
            purity += max((br[sel] == 0).sum(), (br[sel] == 1).sum())
    purity /= len(br)
    summary["fork_z_purity"] = float(purity)
    summary["fork_jea_final_feat_std"] = float(jea_trace[-1][1])

    np.savez_compressed(
        os.path.join(OUT, "fork_models.npz"),
        xs=xs, ys=ys, ctx=c0, pred_blur=pred, true0=true0, true1=true1,
        mode_positions=np.array(modes), jea_trace=np.array(jea_trace),
        **{f"grid_{k}": v for k, v in grids.items()})

    # --------------------------------------------------------------- camera
    cam = train(CameraJEPA(1), world.camera_batch, steps=6000, device=device)
    xc, yc, dxy = world.camera_batch(512, np.random.default_rng(11))
    zin = cam.infer_z(xc.to(device), yc.to(device)).cpu().numpy()
    err = np.abs(zin - dxy.numpy()).mean()
    summary["camera_mae_px"] = float(err)
    np.savez_compressed(
        os.path.join(OUT, "camera.npz"),
        true=dxy.numpy(), inferred=zin,
        scene=world.render_scene(),
        shifted_example=world.render_scene(4.0, -3.0))

    # ---------------------------------------------------------------- audio
    a_models = {
        "gen_det": train(GenDet(1), world.audio_batch, device=device),
        "jepa_z": train(JEPA(1, K=4), world.audio_batch, steps=8000, device=device),
        "jepa_noz": train(JEPA(1, K=1), world.audio_batch, steps=8000, device=device),
    }
    actx, _, _ = world.audio_batch(64, np.random.default_rng(5))
    actx = actx.to(device)
    rows = np.arange(2, 40, 1.0)
    curves = {}
    for name, m in a_models.items():
        c = []
        for r in rows:
            y = torch.from_numpy(world.audio_target_at(r))[None] \
                .expand(actx.size(0), -1, -1, -1).to(device)
            c.append(float(m.energy(actx, y).mean()))
        curves[name] = np.array(c)
    apred = a_models["gen_det"].predict(actx[:1]).cpu().numpy()[0, 0]
    np.savez_compressed(
        os.path.join(OUT, "audio.npz"),
        rows=rows, ctx=actx[0, 0].cpu().numpy(),
        true_c=world.render_tones("A", "C"), true_g=world.render_tones("A", "G"),
        pred_blur=apred, note_rows=np.array([world.NOTE_BANDS["C"],
                                             world.NOTE_BANDS["G"]]),
        **{f"curve_{k}": v for k, v in curves.items()})

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary
