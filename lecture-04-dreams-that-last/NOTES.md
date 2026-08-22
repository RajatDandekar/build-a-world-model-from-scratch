# Build notes — Dreams That Last

The honest log. Everything below happened in the order it is written, and the
numbers are the ones we actually measured. If you are reproducing this, the
dead ends are the useful part.

---

## The premise we had to validate first

Lecture 3 built a world model on a toy game and it failed the long-horizon test:
one-step prediction was near-perfect, but once the model ate its own predictions
the imagined ball dissolved within a handful of steps. We had spent an afternoon
trying to fix that with **training** tricks — a smoothness penalty, closed-loop
practice, a decoded-frame loss, delta prediction, sticky actions. Each helped a
little. None fixed it.

That afternoon is why this lecture is structured the way it is. **It was a design
problem wearing a training problem's clothes.** So before writing a single slide
we trained an RSSM on real robot data to check that the design change actually
buys what the paper claims.

**Rule we followed: no slide gets written until the result behind it exists.**

## The data

`lerobot/svla_so101_pickplace` — 50 tele-operated SO-101 pick-and-place episodes,
11,939 frames at 30 fps, Apache-2.0. Each row has a camera frame, six true joint
angles (`observation.state`), and six joint commands (`action`).

This dataset is pedagogically ideal for one reason: **it makes the Lecture-1
abstraction concrete.** The state is six numbers you can read; the observation is
12,288 pixels pointed at those six numbers plus a cube. Students see the
distinction instead of being told it.

Decoded to 64×64. Episode lengths 183 / 239 / 306 (min / mean / max). Last five
episodes held out and never trained on.

## The model

7.67M parameters, Dreamer-style RSSM:

- conv encoder (4 layers) → 1,024-number embedding
- **h**: 256 numbers, `GRUCell`, never sampled
- **s**: 32 numbers, diagonal Gaussian, sampled every step from a distribution
  computed from `h`
- conv decoder reading `[h, s]`
- a **joint-angle head** reading the six joints out of `[h, s]` — not needed for
  prediction, added purely to make the latent legible to a human

Loss = pixel MSE + 10 × joint MSE + KL(posterior ‖ prior), with free bits 1.0 and
KL balancing 0.8 / 0.2. 21,000 steps on one A10G, about 50 minutes.

## What we measured

**The dream holds.** Five held-out episodes, 5 frames of context, then 60 steps
open-loop driven only by the recorded actions:

| | @1 | @30 | @60 |
|---|---|---|---|
| pixel MSE | ~0.001 | ~0.004 | ~0.001–0.011 |
| joint MSE | ~0.002 | ~0.004 | ~0.002–0.02 |

It does not compound. On one episode the error at step 60 is *lower* than at
step 30 — the dream re-converges rather than running away.

**The ablation** — all three designs, same 45 training episodes, same 16,000
steps, same seed, matched parameter counts:

| design | parameters | pixel @60 | joint @60 |
|---|---|---|---|
| deterministic only | 7,718,313 | 0.0106 | 0.409 |
| stochastic only | 7,543,881 | 0.0155 | 2.598 |
| **RSSM (both)** | 7,665,993 | **0.0050** | **0.009** |

45× better than deterministic-only, 288× better than stochastic-only.

The stochastic-only arm is the most instructive: its joint error **rose** during
training (0.26 → 0.39) while its KL cost climbed past 90 — a model paying more
and more to re-transmit facts it should simply have kept. That is the leaky-state
argument turning into a measurement.

Note the pixel column separates the three far less than the joint column does.
Pixels are dominated by the static table and background; the *robot* is where the
designs differ. **Choose the metric that measures the thing you care about.**

**The probe.** Rather than assert that `h` "carries information", we fit a plain
linear readout from `h` (256 numbers) to the six true joint angles over 25
episodes: **R² = 0.996–0.999 on every joint**, shuffled-label control ≈ 0, and the
readout keeps tracking the real robot 40 steps after the camera is switched off.
"The memory carries the arm" is literal.

## What we could not show

We expected the posterior's uncertainty to **spike at contact** — the grasp is
where the future genuinely forks, and that would have been the perfect
illustration of why the stochastic path exists. Averaged over the state's
dimensions, **it does not.**

What *is* there: at t = 0, with no history at all, uncertainty is at its maximum
(σ ≈ 0.54) and **one frame collapses it** (≈ 0.31). Belief-narrowing, measured on
a real robot.

Two candidate explanations, both testable and neither yet tested: the forks may
live in a few dimensions that the mean washes out, or 50 episodes of a reliably
successful grasp may simply contain little for the model to be unsure about. We
put this on a slide instead of dropping the plot.

The sampled-futures figure in the lecture raises the sampling temperature so the
spread the model *does* carry is visible at all — the honest spread is about 0.05
in normalised gripper units. That is stated on the slide.

## Things that cost us time

**Modal `--detach` only keeps the last triggered function alive.** An entrypoint
chaining `prep → train → evaluate` dies when the client process is killed. We
lost a training run to this. Fix: wrap the whole job in one function that calls
the stages with `.local()`, launch with `nohup modal run --detach`, checkpoint
every 1,000 steps and resume if a checkpoint exists.

**Do not pin `torch` in an image that installs `lerobot`.** A pinned
`torch==2.6.0` broke torchcodec's shared library
(`undefined symbol: _ZN3c1013MessageLogger6stream…`) and the dataset would not
decode. Just `pip_install("lerobot", "matplotlib")` and let it resolve.

**Metric traps, twice.** First: pixel error nearly hides the difference between
the three designs (see above). Second, from the Lecture-3 work that led here: we
scored the imagined ball by total bright pixels, which rewarded a model that had
smeared the ball across the frame — energy was preserved while the ball was gone.
Both times the fix was to measure the *thing*, not a proxy for it.

**One text-match edit swallowed five slides** during a deck restructure, because
two slides had near-identical section labels. Caught only because the exported
page count dropped from 62 to 60. The deck build now runs a gate that checks page
count, blank pages, text overflow, and part-label consistency on every export.

## Reproducing

```bash
pip install modal && modal setup
modal run --detach code/modal_rssm_pilot.py        # prep + train + evaluate (~50 min, A10G)
modal run --detach code/modal_ablation.py::run_all # the three-design comparison (~2 h)
modal run --detach code/modal_deep_figs.py::make   # probe + one-real-training-step figures
```

Or open the notebook — it downloads a 12-episode subset and the trained
checkpoint and runs everything except training on a free Colab CPU.

## Reading

- Hafner et al., *Learning Latent Dynamics for Planning from Pixels* (PlaNet),
  [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) — read Sections 2–3 for the
  RSSM. The CEM planner is a separate topic; skip it on a first pass. Latent
  overshooting is scaffolding that Dreamer later dropped — understand the
  motivation, skip the equations.
- Ha & Schmidhuber, *World Models*, [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
  — the Lecture-3 design this one replaces.
