#!/usr/bin/env python3
"""Convert rssm_so101.py (percent format) -> ../rssm_so101.ipynb for Colab."""
import json, re

src = open("rssm_so101.py").read()

cells, cur, kind = [], [], None

def flush():
    global cur, kind
    if kind is None or not cur:
        cur = []
        return
    text = "\n".join(cur).strip("\n")
    if not text:
        cur[:] = []
        return
    if kind == "markdown":
        lines = [re.sub(r"^# ?", "", l) for l in text.split("\n")]
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]})
    else:
        lines = text.split("\n")
        cells.append({"cell_type": "code", "metadata": {}, "outputs": [],
                      "execution_count": None,
                      "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]})
    cur[:] = []

for line in src.split("\n"):
    if line.startswith("# %% [markdown]"):
        flush(); kind = "markdown"
    elif line.startswith("# %%"):
        flush(); kind = "code"
    else:
        cur.append(line)
flush()

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("../rssm_so101.ipynb", "w"), indent=1)
print(f"wrote rssm_so101.ipynb  ({len(cells)} cells)")
