# Lecture 5 — A Vector or a Vocabulary? (IRIS) · build log

Honest record of what was built, what was measured, what broke, and what we
chose not to claim. Same discipline as Lecture 4: **validate the result before
writing a single slide.**

---

## The shape of the lecture

**110 slides. Five parts**, two of them on text and three on pixels.

| Part | Question | Evidence |
|---|---|---|
| 1 | How does a model read a sequence? | 20 Newsgroups + a synthetic long-range task |
| 2 | What can a latent variable be? | the same corpus, two representations |
| 3 | How do you build a vocabulary for pictures? | VQ-VAE, IRIS |
| 4 | **What does the transformer actually do?** | one real forward pass, traced layer by layer |
| 5 | Which one is better? | CoinRun, both models, matched |

Parts 1 and 2 run on a laptop CPU in under two minutes (`pilot/text_demos.py`).
Parts 4 and 5 run on one A10G (`pilot/modal_coinrun.py`).

---

## Data

`gen_data()` — 600 CoinRun episodes under a random policy.
**256,151 frames**, episode length min/mean/max **72 / 427 / 512**, 64×64 RGB.
Last 40 episodes held out; neither model ever trained on them.

---

## The two models

|  | Model A · RSSM | Model B · IRIS-mini |
|---|---|---|
| state | 32 continuous (Gaussian) | 16 tokens from a 512-word codebook |
| dynamics | GRU, h = 256 | causal transformer, d=256, 6 layers, 8 heads |
| context | one vector | 8 steps × 17 positions |
| params | 5,364,035 | 6,882,499 (1,842,115 tokenizer + 5,040,384 transformer) |

Matched on training wall-clock; the parameter counts land within 28% of each
other and are printed on the slide rather than smoothed over.

---

## Two real bugs, found and fixed before any slide was written

**1 · The GPT was never trained to predict across frames.**
The token stream is `[t0…t15, action]` per step. The original loss kept every
position with `pos % 17 < 16`, which
  - trained position 15 to predict the *action placeholder* (a constant 0), and
  - excluded position 16 (the action token) — the only position whose output
    predicts token 0 of the **next** frame.

So the model learned to complete a frame from its own first token, and nothing
about how one frame follows another. Cross-entropy looked healthy (0.25) partly
because predicting a constant is free. Fixed to `pos % 17 != 15` and retrained
from scratch. **This one would have silently produced a world model that cannot
roll forward.**

**2 · The dream sampler was not autoregressive.**
The rollout called the transformer 16 times per frame but never appended the
sampled token before drawing the next one — so all 16 tokens were i.i.d. draws
from the *same* distribution. Rewritten to build the sequence incrementally
(`gpt.build(...)` for the history, then append `tok_emb(sampled) + pos`), which
is what IRIS actually does.

A third, smaller one: the context-need experiment scored *every* position while
hiding old frames, which mixes "can you finish this frame" with "can you predict
the next one". Now it scores only the newest frame's 16 tokens, and hides the
old actions as well as the old frames.

---

## Figures that were rebuilt because they were not honest enough

- **`fig_t_attention`** was an *untrained* attention layer — a near-uniform
  matrix presented as "actual attention weights". Replaced with the attention
  model actually trained on the long-range task, reading out where the question
  token looks. It puts **25% of its attention on a name 27 words back** and
  answers correctly 100% of the time. That is a real result, and it is the
  better figure.
- **`fig_t_rnn_summary`** was an untrained GRU (i.e. noise). Now a GRU language
  model trained on the corpus for 1,500 steps.
- **`fig_t_continuous`**'s interpolation panel decoded each step to the nearest
  *article*, which snapped between two articles — the opposite of the point.
  Replaced with topic-similarity curves along the path, which genuinely slide.

---

## Measured results

**Part 1 — the long-range task** (8 names, chance = 0.125), same data, same
budget, same size for both models:

| gap (words) | 4 | 12 | 24 | 48 | 96 | 160 |
|---|---|---|---|---|---|---|
| RNN | 1.00 | 1.00 | **0.11** | 0.14 | 0.13 | 0.12 |
| attention | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

A cliff, not a slope. Perfect at 12 words, at chance by 24.

**Part 5 — CoinRun** (all from `pilot/out/figs/stats.json`):

| | RSSM | IRIS-mini |
|---|---|---|
| parameters | 5,364,035 | 6,882,499 |
| imagined frames / sec (A10G, batch 1) | ~5,600 | ~27 |
| dream edge sharpness (real = 0.0678) | 0.0513 | 0.0674 ± 0.0015 |

- **Codebook perplexity 472 of 512, all 512 codes alive.** No collapse. The
  EMA + dead-code revival did their job.
- **Held-out reconstruction MSE 0.00081** from 16 token IDs per frame.
- **Sharpness went the way Part 2 predicts.** The RSSM lands well *below* the
  real game (0.051 vs 0.068 — too smooth, which is what averaging futures looks
  like). IRIS lands essentially *level* with it (0.067). IRIS/RSSM = 1.31×.
  **This number is a sampled quantity**, so one rollout is not a measurement:
  it is averaged over 5 seeded rollouts (0.0661 / 0.0666 / 0.0704 / 0.0670 /
  0.0668, sd 0.0015) and the slide prints the spread. An earlier single-sample
  run read 0.0741 and would have supported the *opposite* claim ("IRIS is
  sharper than reality"); the slide text is now derived from the numbers by
  `assemble.py` rather than written by hand, so it cannot drift from them again.
- **Context curve: 1.57 → 0.099 → 0.019, then flat.** The second frame is worth
  a factor of ~16 (that is where motion lives); past three frames, more history
  buys nothing on CoinRun. Honest and slightly deflating — a random policy on
  procedurally generated levels does not create long-range dependencies.

Every number on a Part 5 slide is substituted from `stats.json` by
`assemble.py`, and the build **fails** if a placeholder cannot be filled — so no
slide can ship with an invented figure on it.

### A claim the data killed

The efficiency slide was originally going to say the gap *widens with context
length*. It does not, at our scale: throughput was **28.2 / 28.3 / 28.2 fps** at
contexts of 2, 4 and 8 frames — flat. At 136 positions the sequence is far too
short for attention's quadratic term to matter; the 206× gap is entirely the
**sixteen sequential token decodes** per imagined frame. The slide now says
that, and separates the asymptotic argument (true) from our measurement (also
true, and different).

---

## What we deliberately did **not** claim

- **That IRIS "wins".** It flips two independent choices at once (continuous →
  discrete, recurrent → attention), so a single head-to-head cannot attribute
  the difference to either. The slide says so.
- **That the throughput gap is the final word.** Our transformer recomputes the
  whole window per token; a KV cache would narrow it. The caption says so.
- **That sharper means better.** A token model can paint a crisp frame of
  something that never happens. That distinction is the closing slide.

---

## The export gate, and what it now catches

`gate.py` reads the rendered PDF, not the source. Five checks:

1. **Render artefacts** — `[object Object]`, `undefined`, an unfilled `⟦…⟧`.
2. **Blank pages** — a duplicate `---` separator swallowing a slide.
3. **Text-over-text** — two rendered *lines* whose boxes intersect.
4. **Part-label drift** — an eyebrow saying "Part 3" inside Part 2.
5. **Silent truncation** — `.slidev-layout` scrolls, so overflowing text is
   *dropped* from the PDF rather than spilling past the page edge. The gate
   re-splits `slides.md` and checks each slide's last words actually rendered.

Two of these were written because the deck failed them:

- **Check 3 caught a caption overprinting the running footer on 22 of 61
  slides.** `.fullfig .cap` sat at `bottom: 1.1rem`, the footer at `.85rem`.
  The first version of the check compared text *blocks* and passed cleanly —
  PyMuPDF **merges two colliding runs into one block**, so a block-level check
  is structurally blind to exactly the bug it is looking for. Comparing lines
  reproduced all 22.
- Check 5 exists because a scrolled-away paragraph looks identical to a
  paragraph that was never written.

A check that has never fired is not evidence of anything, so both were verified
against a known-bad input before being trusted.

**The gate cannot see inside a PNG.** Everything a matplotlib figure does wrong
— a label struck through by its own curve, a legend printed over data, an axis
line glued to a spine, 7pt type — is invisible to it. Three separate visual
passes over the rendered pages found 24, then 12, then the remainder. That is
the only way these were caught, and it is worth budgeting for.

## Rerunning

```bash
# Parts 1 & 2 — laptop, ~2 min, no GPU, no download
python3 pilot/text_demos.py

# Part 4 — one A10G
modal run --detach pilot/modal_coinrun.py::gen_data
modal run --detach pilot/modal_coinrun.py::train_rssm --steps 12000
modal run --detach pilot/modal_coinrun.py::train_iris --tok-steps 20000 --gpt-steps 60000
modal run --detach pilot/modal_coinrun.py::compare

# the deck
python3 gen_figures.py            # the hand-drawn Style A scenes
python3 assemble.py               # fills every ⟦PLACEHOLDER⟧ from stats.json
npx slidev export --output export/lecture-05.pdf --timeout 300000 --wait 900
python3 gate.py export/lecture-05.pdf
```

---

## The opener and the class demo

**`public/img/opener.mp4`** — 23 s, rendered by `modal_coinrun.py::opener`. Real
CoinRun, then the same scene imagined by both models with the camera off, then
the 512-word vocabulary. Every imagined frame is real model output.

**`simulator/index.html`** — a live demo. Both models are primed on the same
eight real frames; the presenter then picks LEFT / RIGHT / JUMP and watches each
model imagine. **3,279 genuine rollouts, depth 7**, pre-computed on a GPU by
`modal_coinrun.py::simulator` and packed into one sprite sheet + one JSON, so it
runs from disk with no server and no network.

### What building the demo taught us

The first version felt broken: pressing a different action barely changed the
next frame. It is not broken — it is a **resolution limit**, and it is
measurable:

| steps after the choice | 1 | 2 | 3 | 6 |
|---|---|---|---|---|
| of 16 tokens, how many differ across the 3 actions | 2.0 | 7.0 | 12.7 | **14.3** |
| pixel difference between the 3 RSSM futures | .004 | .009 | .039 | .040 |
| pixel difference between the 3 IRIS futures | .004 | .012 | .045 | .050 |

Each token covers a 16×16 patch; one frame of player movement is a few pixels,
so it often does not change which word wins. **The continuous model shows the
identical delay**, so this is a fact about the world and the resolution, not
about discreteness — which is exactly why the demo shows both. Consistent with
the transformer's own behaviour: only **0.1%–5.5%** of its attention lands on
action tokens.

The simulator therefore shows all three branches side by side with a live count
of how many tokens differ. The divergence is the lesson, so it is on screen.

## Part 4 — inside the transformer

Added after review: the deck described IRIS as "tokens + a transformer" and
never opened the stack. 26 slides, every figure traced from one real forward
pass (`modal_coinrun.py::internals`).

The measured results that carry the part:

- **The residual stream.** At the sampling position, vector length grows
  16.2 → 52.8 while cosine similarity to its own input embedding falls
  **1.00 → 0.20**. The "context vector" is not a special mechanism — it is this
  stream, at the last position, after the last layer.
- **A division of labour nobody designed.** Share of attention landing inside
  the newest frame, layer 1→6: **25% · 52% · 64% · 61% · 99% · 74%**. Early
  layers gather across history; late layers settle onto the frame being
  finished. Emerged from the loss.
- **Logit lens.** Decoding the intermediate vector with the final head gives
  221 → 221 → 221 → 221 → 54 → 54 → **76**. The network changes its mind late
  rather than refining early, and confidence does not rise monotonically.
- **The output is genuinely multimodal**: 79% / 13% / 7% on three different real
  patches — Part 2's argument, visible in the actual output layer.

One point worth teaching explicitly: **the codebook vector never enters the
transformer.** The tokenizer emits an *integer*; the transformer looks that
integer up in its own separate embedding table. Two different objects for the
same word, joined by an int — which is precisely why the tokenizer can be frozen
and the dynamics model trained on top.

## A gate check that was dead on arrival

The part-label consistency check never fired. Eyebrows render **letter-spaced**
(`P A R T 4 O F 5`), so a regex for `Part 4 of 5` matched nothing, on every
page, silently — while the check sat in the file looking reassuring. Found only
by noticing the gate had stopped *printing* which page each part opened on.

That is the second time in this build the same failure mode bit: a check whose
output looks like a pass when it is really a no-op. Both are now verified
against known-bad input before being trusted. The gate prints the part map on
every run precisely so a dead check is visible.

## The figure bug that mattered most

Every "here is what one codebook word looks like" tile — in `fig_vocabulary`,
`fig_x_embedding`, `fig_x_logitlens` and `fig_x_logits` — was a **flat grey
wash**. Four figures whose entire job is to show pictures were showing nothing,
and the first review pass described the vocabulary as "wallpaper swatches"
without either of us realising that was the bug rather than the aesthetic.

The cause was mine and it was conceptual, not a typo:

```python
codes = tok.vq.emb.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, GRID, GRID)
patch = tok.decode_q(codes[i])          # one word, replicated 4x4
```

Handing the decoder a **constant latent field** is asking it to paint an image
in which every region is the same thing. A flat wash is the correct answer. The
output was genuinely the model's — it just meant nothing.

A word's real meaning is the set of image patches that get assigned to it. So
the gallery is now built by tokenising held-out frames and **averaging the
actual 16×16 patches** for each code:

```
vocabulary gallery: 491/512 words seen in 34,560 real patches
per-tile std over the 512 tiles: mean 49.5 (min 3.6, max 94.6)
tiles still essentially flat: 2 of 512
```

**The lesson:** "the figure came out of the real model" is not the same as "the
figure shows something". Both were true of the old tiles. Only the second one
matters, and it is not something a mechanical gate can check — it took a human
looking at the pixels and saying *these are all identical grey squares*.

---

## Review round 3 — Rajat's pass over the rendered deck

Twenty-odd points, several of them real errors rather than gaps. **90 → 110
slides.** What changed and why:

### One factual error

Slide 29 claimed *"we already fixed this in Lecture 3 — Ha & Schmidhuber's
answer was the mixture density network."* **We never taught the MDN.** Grepping
both the re-recorded Lecture 3 (MiniPong) and Lecture 4 returns nothing; the
MiniPong model used a single Gaussian. Rewritten to introduce the mixture idea
fresh, credit the World Models paper for it, and say plainly in the text that we
did not build one.

### Attention was compared before it was introduced

The long-range experiment raced "RNN vs attention" and plotted the result
*eleven slides before* attention was defined. Two new slides now state the idea
(keep everything, query/key/value, distance is free) with a figure, immediately
before the race. The full mechanics still live in Part 4 — but the comparison is
no longer against an undefined thing.

### Ideas that had no picture

New hand-drawn figures for: the running summary (a newspaper feeding one fixed
box), the long-range task itself, how accuracy is defined (8 names, chance
0.125, 400 held-out passages), attention in one picture, parallel training, the
two VQ loss terms as a pipeline, IRIS end-to-end, the whole transformer stack at
a glance, training-vs-dreaming, and the context probe.

Nine of the ten passed review first time. `fig_context_probe` needed a regen —
my prompt said "on every row, a label…" and the model duly repeated the same
annotation four times.

### A missing section

IRIS had **two slides** before the deck dived into transformer internals. Added
a proper block-diagram section at the end of Part 3: what the three components
are, what goes in and what comes out of each, why the tokenizer is frozen before
the dynamics model is trained, and an explicit note that IRIS's actor-critic is
out of scope because this series is still on the *simulation* half of the
taxonomy.

### Wording that was actually wrong

- *"the vector grows, length 16.2 → 52.8"* — reads as if the dimension changes.
  It does not: always 256 numbers. What grows is the **magnitude**. Fixed on the
  slide and in the figure's axis label.
- *"cost grows with the context window"* next to a rollout where our window is
  **fixed at 8 frames / 136 positions** and never grows. Now says which is which.
- The mask figure's dotted line was never explained. It marks the action token.

### Questions that deserved a slide each

- **"One forward pass gives 136 predictions? I thought we predicted one at a
  time."** Both true: training scores all 136 in one pass because the answers are
  already known and the mask keeps it honest; imagining needs 16 sequential
  passes per frame. Now a slide plus a figure, and it sets up the throughput
  result in Part 5.
- **"How is the share of attention apparent from the figure?"** It wasn't — the
  number was only in the caption. Each panel now shades the newest frame and
  prints its share inside the panel.
- **"How is the vocabulary calculated? Does the model learn it during
  training?"** Separated the two things: the model learns 512 *vectors*; the
  pictures are 34,560 held-out patches averaged per word. Includes the flat-grey
  mistake, because it is instructive.
- **"How did you measure how many past frames it uses?"** The blanking
  experiment now has its own figure.
- **"What is imagined throughput?"** Defined before the chart: one GRU step
  versus sixteen sequential transformer passes, batch 1, same GPU.

### And a new measured result

Pushing on "steps 2 and 3 need more depth" turned up something worth having.
The positional table is initialised to **zeros** — no sine waves, no imposed
structure. After training, the strongest off-diagonal similarity across all
offsets beyond ten sits at **exactly 17**: one frame of 16 picture words plus
one action. The model recovered the rhythm of its own input from next-token
prediction alone. That is now a figure (`fig_x_pos`) and a slide.
