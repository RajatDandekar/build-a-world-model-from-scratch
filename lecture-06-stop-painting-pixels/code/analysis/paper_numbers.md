# I-JEPA paper numbers (verified from ar5iv 2301.08243 on 2026-08-29)

## Table 1 — IN-1k linear evaluation
| Method | Arch | Epochs | Top-1 |
|---|---|---|---|
| data2vec | ViT-L/16 | 1600 | 77.3 |
| MAE | ViT-B/16 | 1600 | 68.0 |
| MAE | ViT-L/16 | 1600 | 76.0 |
| MAE | ViT-H/14 | 1600 | 77.2 |
| CAE | ViT-L/16 | 1600 | 78.1 |
| I-JEPA | ViT-B/16 | 600 | 72.9 |
| I-JEPA | ViT-L/16 | 600 | 77.5 |
| I-JEPA | ViT-H/14 | 300 | 79.3 |
| I-JEPA | ViT-H/14 @448 | 300 | 81.1 |
| DINO | ViT-B/8 | 300 | 80.1 |
| iBOT | ViT-L/16 | 250 | 81.0 |

## Table 2 — IN-1k 1% semi-supervised
MAE ViT-H/14 71.5 (1600 ep) · I-JEPA ViT-H/14 73.3 (300 ep) ·
I-JEPA ViT-H/14@448 77.3 · data2vec ViT-L 73.3 · MSN ViT-B/4 75.7

## Masking (paper App. A, matches our collator)
4 target blocks, scale (0.15, 0.2), aspect (0.75, 1.5), may overlap each other;
1 context block, scale (0.85, 1.0), unit aspect; targets removed from context.

## Compute claims
ViT-H/14 on IN-1k: 16×A100, <72 h; ~10× less compute than MAE ViT-H/14;
converges in ~5× fewer iterations than pixel-reconstruction methods.

## Pretraining recipe (paper)
AdamW, batch 2048, lr 1e-4→1e-3 warmup 15 ep then cosine→1e-6,
wd 0.04→0.4 linear, EMA 0.996→1.0 linear, loss = average L2 on patch reps.
(Official CODE uses smooth-L1 and F.layer_norm on target reps — code-vs-paper
divergence, worth one honest slide.)

## Deltas in OUR recipe (state on the methods slide)
ViT-S (STL-10: /8 @96px; IN-100: /16 @224px), batch 256, lr 5e-4 cosine,
wd 0.05 constant, 100–150 epochs, smooth-L1 + target layer-norm (code-faithful).
