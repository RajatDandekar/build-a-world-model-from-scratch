# Build log — Lecture 6a

- The deck was rebuilt once, hard. v1 carried a synthetic "fork world" (a ball with a bimodal
  future) plus a camera-shift latent-variable experiment and a part on latent capacity. The
  practice review cut all of it: one story, ending at the measured contours on real encoders.
  The toy code survives in `../lecture-06-stop-painting-pixels/code/modal_app/toyworld/`.
- The camera experiment (infer z = camera shift by energy minimization) was run six ways across
  three seeds and never reliably beat a guess-zero baseline. Two theory-faithful failure modes:
  a trained target encoder drifts toward shift-invariant features (deleting exactly what z must
  carry), and frozen random features decorrelate within a pixel on detailed scenes. Kept out of
  the deck; kept in the record.
- The energy symbol E is introduced only AFTER a concrete quiz slide (real masked context, three
  candidates, measured scores 0.15 / 0.50 / 0.50). Nobody meets a symbol before its meaning.
- Every architecture gets three slides: the diagram, its real trained output, its own measured
  energy map. The 5-panel contour figure is the destination of the whole lecture.
- Figures: whiteboard-style concept diagrams are generated (gemini image model) and verified
  label-by-label; results figures are matplotlib from raw npz. Numbers gated against
  `results/headlines/lecture-06a-headlines.json`.
