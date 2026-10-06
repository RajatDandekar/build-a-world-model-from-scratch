# Lecture 15 — World Action Models: DreamZero (and our SO-101 fine-tune)

> **Watch:** [Lecture 15 on YouTube](https://youtu.be/IGn2b9RmUQs) · [full playlist](https://www.youtube.com/playlist?list=PLahzD_BXtne4)
> **Paper:** NVIDIA (2026) — [World Action Models are Zero-shot Policies (DreamZero)](https://arxiv.org/abs/2602.15922) · [code](https://github.com/dreamzero0/dreamzero)

**Paradigm 3:** one network that **jointly** predicts the future video *and* the actions that cause it — no
MPC, no separately trained RL agent, no inverse-dynamics model.

## The ideas in the lecture

- **Backbone:** Wan2.1-I2V-14B, a web-scale video diffusion transformer. Visual context (VAE), language
  (text encoder) and robot state (state encoder) go in; a **video head** and an **action head** come out.
- **Joint flow matching:** for chunk *k* and a shared denoising time *t*, both the noisy video latent and the
  noisy action are linear interpolations between noise (t = 0) and data (t = 1). The network predicts the
  velocity `(z₁ − z₀, a₁ − a₀)`, conditioned on the instruction, proprioception and the **clean previous
  chunks**. The loss is MSE between predicted and target velocity — not between actions directly.
- **Autoregressive chunks + KV cache:** earlier chunks are cached, and predicted frames are replaced by real
  observations in the cache, which stops errors compounding and gives real-time control at ~7 Hz.

## Project: fine-tune DreamZero on the SO-101 arm

We adapted DreamZero to the low-cost SO-101 arm and released the result as **DreamZero-SO101**:

- **Weights:** [Vizuara/dreamzero-so101-lora](https://huggingface.co/Vizuara/dreamzero-so101-lora) (LoRA, ~217 MB,
  with the loss log and training curve) · intermediate checkpoints and the 1K-step proof-of-concept run in
  [Vizuara/dreamzero-so101-checkpoints](https://huggingface.co/Vizuara/dreamzero-so101-checkpoints)
- **Project site & interactive gallery:** [vizuara-ai-lab.github.io/dreamzero-so101](https://vizuara-ai-lab.github.io/dreamzero-so101/) ·
  [demo](https://vizuara-ai-lab.github.io/dreamzero-so101/demo.html) — paper, imagined rollouts on held-out
  episodes, evaluation results
- **Base code:** [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero) (Apache 2.0) +
  [Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)

The recipe, as walked through in the lecture:

1. **Find SO-101 data** — community LeRobot datasets on Hugging Face (the run used 715 episodes, mostly
   `whosricky/so101-megamix-v1`).
2. **Convert LeRobot → GEAR format** (DreamZero's data layout) and register SO-101 as a new embodiment
   (6-DoF action: 5 joints + gripper, 3 cameras, 30 fps).
3. **Proof-of-concept run** — 2× H100, LoRA rank 4, 1,000 steps: total loss 0.42 → 0.068. If your loss falls
   like this, the pipeline is wired correctly.
4. **Full training** — LoRA converges around 70K steps; a full fine-tune is the quality option.
5. **Evaluate & infer** — give one camera frame + an instruction; the model imagines the future video and the
   joint commands in one pass.

Seven things broke before training worked — worth knowing before you start: a cuDNN SDPA backend crash
(wrap attention in the flash/efficient/math kernel context), a PyTorch 2.11 LR-scheduler change, DataLoader
OOM (`dataloader_num_workers` 10 → 4), the LoRA rank living under `action_head_cfg` rather than top level, a
missing `annotation.task` mapping in `modality.json`, a `blinker` distutils conflict
(`pip install --ignore-installed blinker`), and torch / huggingface-hub / deepspeed version drift.

**Make it yours:** the run used 715 episodes; with ~10,000 episodes of your own task the model should improve
substantially — collecting good SO-101 data is the highest-leverage contribution.
