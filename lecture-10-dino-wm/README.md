# Lecture 10 — DINO-WM: Planning in Feature Space

> **Watch:** [Lecture 10 on YouTube](https://youtu.be/5l0jG_aIncs) · [full playlist](https://www.youtube.com/playlist?list=PLahzD_BXtne4)
> **Paper:** Zhou, Pan, LeCun, Pinto — [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://arxiv.org/abs/2411.04983)

DINO-WM keeps the classic world-model recipe — *(observation, action) → next observation* — but never
predicts a pixel. A **frozen DINOv2** encoder turns each camera frame into 196 patch features, a small
**transformer predictor** learns how those features move when the robot acts, and at test time
**model-predictive control (CEM)** imagines hundreds of action sequences and keeps the one whose
imagined final frame is closest to a goal image. Only the predictor is trained.

## What you'll build

| Notebook | What it does | Open |
|---|---|---|
| [`DINO_World_Model_From_Scratch.ipynb`](DINO_World_Model_From_Scratch.ipynb) | Every DINO-WM component from scratch on a small **Ball Arena** world: frozen DINOv2 encoder, action/proprio encoders, block-causal transformer predictor, optional decoder, and a CEM planner that reaches a goal image | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RajatDandekar/build-a-world-model-from-scratch/blob/main/lecture-10-dino-wm/DINO_World_Model_From_Scratch.ipynb) |
| [`DINO_WM_PushT.ipynb`](DINO_WM_PushT.ipynb) | The same model on **real robot data**: 206 expert Push-T episodes from LeRobot, pre-encoded with DINOv2, next-frame prediction, autoregressive rollouts, and **zero-shot CEM planning** from a held-out start frame to a goal frame | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RajatDandekar/build-a-world-model-from-scratch/blob/main/lecture-10-dino-wm/DINO_WM_PushT.ipynb) |

Both notebooks run top-to-bottom on a free Colab **T4** (runtimes are in the table at the bottom).

## The ideas in the lecture

- **Why feature space?** A 256×256 image is 196,608 numbers; DINOv2 summarises it as a 14×14 grid of
  semantic features (object identity, layout, pose, affordances). Predicting those is far cheaper and
  ignores lighting and texture.
- **Tokens per step:** 256 patch tokens + action tokens + proprio tokens = 264 tokens per timestep; with a
  3-frame history that is 792 tokens through the predictor.
- **Block-causal attention:** tokens inside a timestep see each other; a timestep only sees the past.
- **Planning, not a policy:** the world model is trained once; actions are chosen at test time by CEM
  rollouts in imagination. That is why it needs only ~50 trajectories — and why inference is slow.
- **Limitations:** you must supply a goal image, there is no language conditioning, and errors compound in
  long autoregressive rollouts.

## Next

[Lecture 11 — DIAMOND](../lecture-11-diamond) swaps feature prediction for diffusion in pixel space.
