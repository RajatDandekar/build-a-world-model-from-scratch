# Lecture 6 — Stop Painting Pixels (I-JEPA)

The experiment suite behind World-Models Lecture 6: what the joint-embedding
predictive architecture is, why it beats its two parent families, and a
replication of the I-JEPA paper's ImageNet linear-probe study
(Assran et al., arXiv:2301.08243).

## The four experiments

| id | question | how |
|---|---|---|
| E1 | do the two parent families really fail the way the paper says? | 5 objectives × 1 identical ViT-S/8 on STL-10 (100k unlabeled): naive joint embedding, MAE, MAE-with-JEPA-masks, I-JEPA, I-JEPA-without-EMA/stop-grad. Same data, steps, seed. Linear probe + kNN + collapse metrics. |
| E2 | does the paper's ImageNet claim replicate? | frozen official checkpoints (facebook/ijepa_vith14_1k, facebook/vit-mae-huge, random ViT-H, supervised ViT-B) → linear probe on ImageNet-100 at 100% / 10% / 1% labels. |
| E3 | does the pretrain→probe pipeline work end-to-end in our hands? | I-JEPA ViT-S/16 pretrained from scratch on ImageNet-100, 150 epochs, one H100. |
| E4 | what do the masks actually look like? | the real multi-block collator run on real images (reproduces paper Fig. 4). |

## Layout

```
modal_app/
  app.py            Modal entrypoints (prep / train / probe / viz / smoke)
  ijepa/
    vit.py          ViT encoder (subset-capable), predictor, pixel decoder
    masks.py        multi-block mask collator (paper App. A) + MAE random mask
    models.py       the five objectives over one encoder
    train.py        unified trainer, JSONL logs, collapse metrics
    probe.py        frozen-feature linear probe, kNN, low-shot
    pretrained.py   official-checkpoint probes (HF transformers)
    viz.py          mask-example and MAE-reconstruction dumps
tests/test_core.py  27 CPU checks (masking, shapes, gradient routing, EMA)
analysis/
  make_figures.py   every deck figure, regenerated from raw logs
  verify_deck.py    numbers gate: slide values vs figs/headlines.json
  paper_numbers.md  verified paper tables (ar5iv, 2026-08-29)
figs_concept/       Style-A concept diagrams (needs a live GEMINI_API_KEY)
deck_assets/svg/    hand-built architecture diagrams used by the deck
sync_results.sh     pull logs/probes/viz from the Modal volume
```

The deck lives at
`/Users/raj/Downloads/Vizuara/World Model Series/lecture-06-ijepa/`
(assemble.py builds slides.md from parts/ + these assets).

## Running

```bash
alias modal="MODAL_PROFILE=rajatworkspace /Users/raj/.venv/bin/modal"
cd modal_app
modal run app.py::prep --dataset stl10        # stage data (once)
modal run app.py::prep --dataset in100
modal run app.py::smoke                       # tiny end-to-end check first
modal run --detach app.py::launch_e1          # E1: five STL-10 runs
modal run --detach app.py::launch_e2          # E2: official-checkpoint probes
modal run --detach app.py::train_one --arch ijepa --dataset in100 --epochs 150
modal run app.py::probe_runs                  # after E1 finishes
modal run app.py::viz_remote --mae-run stl10_mae_s0
cd .. && ./sync_results.sh && python3 analysis/make_figures.py
```

## Fidelity notes (code vs paper)

- masking: 4 targets scale (0.15,0.2) ar (0.75,1.5); context (0.85,1.0)
  minus targets — exactly the paper.
- loss: smooth-L1 (the official CODE), not L2 (the paper's text).
- target representations are `F.layer_norm`-ed before the loss —
  undocumented in the paper, present in the official code, kept here.
- EMA momentum 0.996 → 1.0 linear; AdamW; cosine lr.
- deltas at our scale: ViT-S, batch 256, lr 5e-4, constant wd 0.05,
  100–150 epochs (paper: ViT-H, batch 2048, wd 0.04→0.4, 300 epochs).

## Discipline

Every number on a slide is recomputed from raw logs by
`analysis/make_figures.py` (→ `figs/headlines.json`) and gated by
`analysis/verify_deck.py`. Runs are compared only at the largest step both
reached. The probe always includes a random-init baseline.
