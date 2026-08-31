# Build log — Lecture 6

- Five objectives, one ViT-S/8, STL-10, same seed/steps: jea (naive joint embedding), mae,
  genmask (MAE with I-JEPA's block masks — the control that isolates masking), ijepa,
  ijepa_nostop (guard removed). H100s via Modal; ~40k steps each.
- The two collapse results are exact: jea ends at cosine loss −1.0 with feature std 4×10⁻⁶;
  the guard-free I-JEPA at 1.0×10⁻⁶ (it is NOT zero — we print tiny values in scientific
  notation after last lecture's rounding lesson). Both probe BELOW an untrained encoder.
- Honest finding kept on the slides: at this scale MAE beats I-JEPA on a linear probe
  (75.3 vs 72.1); I-JEPA leads on kNN. The latent-target advantage is a scale claim — the
  official-checkpoint replication settles it (88.3/86.6 linear, 77.1/58.4 at 1% labels,
  84.5/30.7 kNN).
- From-scratch ImageNet-100 pretraining: ViT-S/16, 150 epochs, 2.67 h on one H100 → 59.1%
  linear vs 18.8% random-init. First attempt was silently cancelled when the local Modal client
  died — long runs must use `.spawn()`, not `.remote()`.
- Fidelity to the official code, disclosed on a slide: smooth-L1 (not the paper's L2) and an
  undocumented `F.layer_norm` on teacher outputs.
- Discipline carried from the previous Lecture 6: every slide number recomputed from raw logs
  (`analysis/make_figures.py` → headlines.json) and gated (`analysis/verify_deck.py`); runs
  compared only at the largest step both reached; a random-encoder baseline in every probe;
  render every page and look at it (the layout gate caught overflow and a blank page the eye
  had skipped).
- The nearest-neighbour figure is a straight cosine top-4 over the official checkpoints'
  frozen features on the ImageNet-100 validation set; query classes chosen with a fixed seed;
  indices in `results/neighbors_spec.json`.
