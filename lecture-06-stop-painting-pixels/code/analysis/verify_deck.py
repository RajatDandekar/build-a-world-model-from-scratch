#!/usr/bin/env python3
"""Numbers gate: every experimental number quoted in slides.md must match
figs/headlines.json (recomputed from raw logs by make_figures.py).

Convention in slides.md: experimental numbers are wrapped in an HTML comment
anchor the first time they appear on each slide, e.g.
    <!--N probe_ijepa 74.2 -->
meaning "the value 74.2 on this slide is headlines[probe_ijepa]". The gate
recomputes and fails if the quoted value differs by more than the tolerance
(0.05 for percentages/floats). Slides with no anchors are ignored, so paper
numbers (cited, not measured) don't need anchors.

Usage: python3 verify_deck.py path/to/slides.md
"""
import json
import os
import re
import sys

slides = sys.argv[1] if len(sys.argv) > 1 else "slides.md"
head = (sys.argv[2] if len(sys.argv) > 2 else
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "figs", "headlines.json"))
H = json.load(open(head))

text = open(slides).read()
anchors = re.findall(r"<!--N\s+(\S+)\s+([-\d.]+)\s*-->", text)
if not anchors:
    sys.exit("verify_deck: no <!--N key value--> anchors found — "
             "either the deck quotes no experimental numbers (unlikely) "
             "or the anchors were forgotten. Failing safe.")

bad = []
for key, val in anchors:
    if key not in H:
        bad.append(f"  unknown key: {key}")
        continue
    want = float(H[key])
    got = float(val)
    if abs(want - got) > 0.05:
        bad.append(f"  {key}: slide says {got}, logs say {want:.4g}")

if bad:
    print("verify_deck: STALE NUMBERS\n" + "\n".join(bad))
    sys.exit(1)
print(f"verify_deck: all {len(anchors)} anchored numbers match headlines.json")
