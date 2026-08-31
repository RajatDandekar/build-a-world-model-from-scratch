"""A tiny controllable world for making LeCun's energy-landscape arguments
runnable. 48x48 grayscale.

Three generators:
  fork    — a ball moves right; at the fork it goes up-right or down-right
            with equal probability. Context = 3 frames before the fork
            (stacked as channels), target = the frame 4 steps after.
            The future is bimodal BY CONSTRUCTION.
  camera  — a fixed scene of blobs; the target is the same scene shifted by a
            random (dx, dy). The shift is the ONLY thing the context cannot
            know: a ready-made latent variable (LeCun's camera example).
  audio   — 48x48 "spectrograms": time on x, log-freq bands on y. Context =
            first half playing one note; the continuation is note C or G
            with equal probability. Same bimodality, different modality.
"""
import math
import numpy as np
import torch

S = 48


def _blob(canvas, cx, cy, r=3.2, amp=1.0):
    y, x = np.mgrid[0:S, 0:S]
    canvas += amp * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r)))
    return canvas


def render_ball(cx, cy):
    return np.clip(_blob(np.zeros((S, S), np.float32), cx, cy), 0, 1)


# ---------------------------------------------------------------- fork world
FORK_X = 24.0
V = 4.0
START = (8.0, 24.0)


def fork_positions(branch, t, ox=0.0, oy=0.0):
    """Ball position at integer time t (t=0,1,2 context; fork at t=4).
    (ox, oy) jitters the whole trajectory so contexts vary a little."""
    x = START[0] + V * t
    if x <= FORK_X:
        return x + ox, START[1] + oy
    d = x - FORK_X
    dy = d * math.tan(math.radians(35.0))
    y = START[1] - dy if branch == 0 else START[1] + dy
    return x + ox, y + oy


def fork_sample(rng, jitter=True):
    branch = int(rng.integers(2))
    ox = float(rng.uniform(-1.5, 1.5)) if jitter else 0.0
    oy = float(rng.uniform(-3.0, 3.0)) if jitter else 0.0
    ctx = np.stack([render_ball(*fork_positions(branch, t, ox, oy))
                    for t in (0, 1, 2)])                 # (3,S,S)
    tgt = render_ball(*fork_positions(branch, 8, ox, oy))  # bimodal frame
    return ctx, tgt[None], branch


def fork_target_at(pos):
    return render_ball(*pos)[None]


def fork_mode_positions():
    return [fork_positions(0, 8), fork_positions(1, 8)]


def fork_batch(n, rng):
    ctx, tgt, br = zip(*[fork_sample(rng) for _ in range(n)])
    return (torch.from_numpy(np.stack(ctx)), torch.from_numpy(np.stack(tgt)),
            torch.tensor(br))


# -------------------------------------------------------------- camera world
SCENE_BLOBS = [(10, 10, 1.6), (30, 16, 2.2), (18, 30, 1.4), (38, 36, 1.8),
               (26, 40, 1.5), (40, 22, 1.3), (14, 22, 1.2), (34, 8, 1.5)]


def render_scene(dx=0.0, dy=0.0):
    c = np.zeros((S, S), np.float32)
    for (cx, cy, r) in SCENE_BLOBS:
        c = _blob(c, cx + dx, cy + dy, r)
    return np.clip(c, 0, 1)


def camera_batch(n, rng, max_shift=6):
    ctx = np.stack([render_scene()[None]] * n)
    dxy = rng.uniform(-max_shift, max_shift, size=(n, 2)).astype(np.float32)
    tgt = np.stack([render_scene(dx, dy)[None] for dx, dy in dxy])
    return (torch.from_numpy(ctx), torch.from_numpy(tgt),
            torch.from_numpy(dxy))


# --------------------------------------------------------------- audio world
NOTE_BANDS = {"A": 30, "C": 18, "G": 8}   # y-rows of the frequency bands


def render_tones(first="A", second=None, band_h=3):
    c = np.zeros((S, S), np.float32)
    c[NOTE_BANDS[first]:NOTE_BANDS[first] + band_h, 0:S // 2] = 1.0
    if second is not None:
        r = int(round(NOTE_BANDS[second])) if isinstance(second, str) \
            else int(round(second))
        c[r:r + band_h, S // 2:S] = 1.0
    return np.clip(c, 0, 1)


def audio_sample(rng):
    branch = int(rng.integers(2))
    note = "C" if branch == 0 else "G"
    full = render_tones("A", note)
    ctx = full.copy()
    ctx[:, S // 2:] = 0.0                  # mask the continuation
    return ctx[None], full[None], branch


def audio_batch(n, rng):
    ctx, tgt, br = zip(*[audio_sample(rng) for _ in range(n)])
    return (torch.from_numpy(np.stack(ctx)), torch.from_numpy(np.stack(tgt)),
            torch.tensor(br))


def audio_target_at(band_row):
    return render_tones("A", float(band_row))[None]


def anyball_batch(n, rng):
    """Frames with the ball anywhere — the AE's view of 'the world'."""
    ctx = np.zeros((n, 3, S, S), np.float32)          # unused by the AE
    pos = rng.uniform(4, 44, size=(n, 2))
    tgt = np.stack([render_ball(px, py)[None] for px, py in pos])
    return torch.from_numpy(ctx), torch.from_numpy(tgt), torch.zeros(n)
