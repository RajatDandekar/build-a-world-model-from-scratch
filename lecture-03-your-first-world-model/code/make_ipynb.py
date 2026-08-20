#!/usr/bin/env python3
"""Convert pong_worldmodel.py (percent format) -> pong_worldmodel.ipynb.

The notebook must be self-contained on Colab, so the MiniPong environment
(pong_env.py) is inlined as the first code cell.
"""
import json, re

env_src = open("pong_env.py").read()
src = open("pong_worldmodel.py").read()
# drop the local import; the env class is inlined instead
src = src.replace("from pong_env import MiniPong\n", "")

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

# inline the environment right after the title cell (cell 0 is the intro markdown)
env_lines = env_src.strip("\n").split("\n")
env_cell = {"cell_type": "code", "metadata": {}, "outputs": [],
            "execution_count": None,
            "source": [l + "\n" for l in env_lines[:-1]] + [env_lines[-1]]}
cells.insert(1, env_cell)

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("pong_worldmodel.ipynb", "w"), indent=1)
print(f"wrote pong_worldmodel.ipynb  ({len(cells)} cells)")
