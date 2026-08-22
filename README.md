<p align="center">
  <img src="assets/hero.png" alt="Build a World Model from Scratch — pixels → encoder → code → memory → prediction → imagined frame" width="100%">
</p>

<h1 align="center">Build a World Model from Scratch</h1>

<p align="center">
  <b>A lecture series by <a href="https://www.vizuara.ai">Vizuara AI</a></b> — slides, runnable code, and Colab notebooks.<br>
  Everything is built from first principles, every number on every slide is real output from the code in this repo.
</p>

<p align="center">
  <a href="https://www.youtube.com/@vizuara"><img src="https://img.shields.io/badge/YouTube-Vizuara-red?logo=youtube" alt="YouTube"></a>
  <a href="https://colab.research.google.com/github/RajatDandekar/build-a-world-model-from-scratch/blob/main/lecture-03-your-first-world-model/pong_worldmodel.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
  <img src="https://img.shields.io/badge/dependencies-numpy%20·%20torch%20·%20matplotlib-blue" alt="Dependencies">
  <img src="https://img.shields.io/badge/GPU-not%20required-success" alt="No GPU required">
</p>

---

## What is this?

A world model is a neural network that learns **how a world works** — well enough to predict what
happens next, and eventually well enough that an agent can plan, imagine, and train *inside* the
model instead of the real world. World models sit behind some of the most exciting results in
modern AI: Dreamer agents that learn in imagination, video models that simulate reality, robots
that rehearse before they act.

This series builds that entire idea up **from scratch** — small worlds, small networks, complete
code, honest engineering. No magic, no hand-waving, no "trust us, it works." When something broke
while we built it (and plenty did), the failure and the fix are part of the lecture.

## The lectures

Each lecture folder holds its slides, its runnable code, and a README with the full write-up.

| # | Lecture | Materials | What you'll learn |
|---|---------|-----------|-------------------|
| 0 | **Series Introduction** | [slides](lecture-00-series-introduction/lecture-00.pdf) | Why world models, the Renderer/Simulator/Planner map of the field, and where this series goes |
| 1 | **What Is a World Model, Really?** | [slides](lecture-01-what-is-a-world-model/lecture-01.pdf) · [code](lecture-01-what-is-a-world-model/code) | The agent–environment loop, why **state ≠ observation**, 80 years of the idea, and the taxonomy |
| 2 | **The World Modeler's Toolkit** | [slides](lecture-02-the-world-modelers-toolkit/lecture-02.pdf) | The four tools every world model stands on: latent spaces, reward over time, value, actor-critic |
| 3 | **Your First World Model** | [**write-up**](lecture-03-your-first-world-model) · [slides](lecture-03-your-first-world-model/lecture-03.pdf) · [notebook](lecture-03-your-first-world-model/pong_worldmodel.ipynb) | Build a complete world model on MiniPong: encoder + memory + prediction, then run it as the game |
| 4 | **Dreams That Last — the RSSM** | [**write-up**](lecture-04-dreams-that-last) · [slides](lecture-04-dreams-that-last/lecture-04.pdf) · [notebook](lecture-04-dreams-that-last/rssm_so101.ipynb) | Build an RSSM on **real SO-101 robot data**: track a belief, carry a memory and a doubt, dream 60 steps with the camera off |

More on the way — next, an agent that learns to *act* entirely inside the dream.

---

## Latest result — Lecture 4

Lecture 3 built a world model whose dream dissolved after a handful of steps. Lecture 4 fixes it,
on real robot data. Given five frames of context and then **nothing but the joint commands**, the
model imagines two full seconds of an SO-101 pick-and-place it has never seen:

<p align="center">
  <img src="lecture-04-dreams-that-last/assets/fig_joint_curves.png" alt="Dreamed joint angles tracking the real robot for 60 open-loop steps" width="90%">
</p>

The idea is the **RSSM**: one state with two halves — a deterministic memory `h` that is never
sampled (facts live here) and a stochastic `s` sampled every step (doubt lives here). We built all
three designs to check the claim, at matched parameter counts:

| design | pixel error @60 | **joint error @60** |
|---|---|---|
| deterministic only | 0.0106 | 0.409 |
| stochastic only | 0.0155 | 2.598 |
| **both — the RSSM** | **0.0050** | **0.009** |

**45× better than deterministic-only, 288× better than stochastic-only.** And a linear probe of the
memory recovers **99.9% of every joint angle** — the state literally carries the arm.
[Full write-up →](lecture-04-dreams-that-last)

---

## Running things locally

Everything needs only `numpy`, `torch`, and `matplotlib`:

```bash
# Lecture 1 — partial observability made into a number (~1 second)
python lecture-01-what-is-a-world-model/code/world_model_lecture1.py

# Lecture 3 — the full MiniPong world model (~8 minutes on a laptop CPU)
cd lecture-03-your-first-world-model/code && python pong_worldmodel.py
```

Every notebook also runs on a free Colab with no setup — the badges in each lecture's write-up
open them directly.

---

## About

Created by **[Rajat Dandekar](https://www.youtube.com/@vizuara)** (Vizuara AI). The series is
taught on the [Vizuara YouTube channel](https://www.youtube.com/@vizuara) — slides here are the
exact decks used in the recordings, and the hand-drawn "whiteboard notebook" figures throughout
are part of the series' visual language.

If this helped you understand world models, a ⭐ on the repo helps others find it.

*Reference: D. Ha & J. Schmidhuber, ["World Models"](https://arxiv.org/abs/1803.10122), 2018.*
