"""Re-run ONLY the camera experiment (soft-min fix) and update summary.json."""
import json
import os
import numpy as np
import torch

from . import world
from .models import CameraJEPA
from .run import train, OUT


def main(seed: int = 0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(seed)
    cam = train(CameraJEPA(1), world.camera_batch, steps=4000, device=device)
    xc, yc, dxy = world.camera_batch(512, np.random.default_rng(11))
    zin = cam.infer_z(xc.to(device), yc.to(device)).cpu().numpy()
    err = float(np.abs(zin - dxy.numpy()).mean())
    np.savez_compressed(
        os.path.join(OUT, f"camera_seed{seed}.npz"),
        true=dxy.numpy(), inferred=zin,
        scene=world.render_scene(),
        shifted_example=world.render_scene(4.0, -3.0))
    p = os.path.join(OUT, "summary.json")
    summary = json.load(open(p)) if os.path.exists(p) else {}
    summary[f"camera_mae_px_seed{seed}"] = err
    with open(p, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary
