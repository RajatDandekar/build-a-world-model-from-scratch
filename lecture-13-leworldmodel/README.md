# Lecture 13 — LeWorldModel: A JEPA World Model on One GPU

> **Watch:** [Lecture 13 on YouTube](https://youtu.be/dTq1nNGbv28) · [full playlist](https://www.youtube.com/playlist?list=PLahzD_BXtne4)
> **Paper:** Maes, Le Lidec, Scieur, LeCun, Balestriero (2026) — [LeWorldModel: Stable End-to-End JEPA from Pixels](https://arxiv.org/abs/2603.19312) ·
> [code](https://github.com/lucas-maes/le-wm) · [project page](https://le-wm.github.io/) · [checkpoints & data](https://huggingface.co/collections/quentinll/lewm)

LeWorldModel (LeWM) is the opposite bet to DINO-WM (Lecture 10). Instead of a heavy *frozen* foundation encoder
(86M parameters, ~200 tokens per frame, ~47 s per CEM planning step), it trains a **ViT-Tiny encoder from scratch**
that emits **one CLS token per frame**, plus a 6-layer predictor — **~15M parameters in total**, trainable on one
GPU in a few hours and ~48× faster at planning.

## The ideas in the lecture

- **End-to-end JEPA:** encoder and predictor are trained together on `MSE(pred_emb, next_emb)` — no frozen
  backbone, no EMA target encoder, no stop-gradient.
- **SIGReg** prevents collapse instead: project the embeddings onto 1,024 random directions, test each 1-D
  projection for normality, and penalise deviation from a Gaussian. Collapsed embeddings can't be Gaussian.
  Total loss = `pred_loss + λ · sigreg_loss` with **one** hyperparameter (λ ≈ 0.09).
- **Actions enter through AdaLN** in the predictor; planning is the same CEM loop as DINO-WM, toward a goal image.
- **Takeaways:** foundation encoders are not mandatory; tiny models can plan; one good loss beats six fragile
  ones. It is still task-specific (train one per task) and still needs action labels.

## Project: LeWorldModel on the SO-101 arm

Three ways to build with LeWM — pick one. All three start from the same verified pipeline below.

| Option | Goal | Deliverable |
|---|---|---|
| **A. Teach your robot to imagine** | Train LeWM on SO-101 pick-and-place demos and plan with CEM toward a goal photo — no policy network | Robot picks a cube from imagination alone (≥3/10), plus ablations over λ, number of demos, and CEM samples |
| **B. The surprise detector** | Use prediction error `‖Enc(xₜ₊₁) − Pred(sₜ, aₜ)‖` as a live anomaly signal | ROC curve separating successful vs failed episodes; auto-label new data by surprise |
| **C. One model, four tasks** | Train one LeWM on several SO-101 datasets; switch tasks by changing only the goal image | 4×4 task confusion matrix, data-scaling curve, leave-one-task-out transfer |

### Step 1 — install LeWM and replicate Push-T

```bash
git clone https://github.com/lucas-maes/le-wm.git && cd le-wm
uv venv --python=3.10 && source .venv/bin/activate
uv pip install "stable-worldmodel[train,env]"
export STABLEWM_HOME=~/.stable_worldmodel        # datasets live in $STABLEWM_HOME/datasets/
python train.py data=pusht                        # official dataset: quentinll/lewm-pusht
python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

### Step 2 — convert SO-101 data (LeRobot → LeWM HDF5)

[`project/convert_lerobot_to_h5.py`](project/convert_lerobot_to_h5.py) writes exactly the schema of the official
Push-T file (`pixels` uint8 224×224, `action`, `proprio`, `state`, `ep_len`, `ep_offset`, `episode_idx`, `step_idx`):

```bash
pip install lerobot h5py
python project/convert_lerobot_to_h5.py --repo-id lerobot/svla_so101_pickplace \
    --out $STABLEWM_HOME/datasets/so101_pickplace_train.h5
# Option C: pass several --repo-id values to pool tasks into one file
```

Note: the SO-101 action is **6-dimensional** (5 joints + gripper), not 7.

### Step 3 — train on SO-101

```bash
cp project/so101.yaml le-wm/config/train/data/so101.yaml
cd le-wm && python train.py data=so101 trainer.max_epochs=200 loader.batch_size=64
```

Watch two curves: `pred_loss` should fall, and `sigreg_loss` should stay **above zero** — if it collapses to 0
while `pred_loss` stalls, the encoder has collapsed; raise `loss.sigreg.weight`.

**What a healthy start looks like** (we ran Steps 2–3 exactly as written on `lerobot/svla_so101_pickplace`:
50 episodes / 11,939 frames, one A10G, batch 64, ~45 s per epoch):

| | sanity check | epoch 1 | epoch 2 |
|---|---|---|---|
| validation `pred_loss` | 0.600 | 0.184 | **0.054** |
| validation `sigreg_loss` | 45.5 | 21.7 | **4.34** (falling, not collapsing to 0) |

Continue to ~200 epochs for a model you can plan with.

### Step 4 — plan and deploy

At test time: encode the last 3 camera frames and a goal photo, run CEM (300 samples × 30 iterations × horizon 5)
in latent space, execute the first action chunk (with `frameskip=5`, each planned action is 5 raw 6-D actions),
grab a new frame, re-plan. Preprocess camera frames exactly like training (224×224, ImageNet normalisation) and
un-normalise actions with the training mean/std before sending them to the arm.

### Compute

| Phase | Hardware | Time |
|---|---|---|
| Training | 1× A100 / A10G | a few hours |
| Inference + CEM | laptop GPU or Apple Silicon | < 1 s per planning step |

**SO-101 datasets to start from:** `lerobot/svla_so101_pickplace` (50 episodes), `whosricky/so101-megamix-v1`
(400), `youliangtan/so101-table-cleanup` (80) — or record your own with `lerobot-record`.

## Next

[Lecture 14 — Video World Models](../lecture-14-video-world-models): learning from video that has no actions at all.
