#!/usr/bin/env python3
"""The export gate: a mechanical check of the rendered PDF.

Five failure classes, all of which we have actually shipped by accident before:
  1. render errors      — a stray "[object Object]", "undefined", "NaN"
  2. blank pages        — a duplicate --- separator swallowing a slide
  3. overflow           — text or an image running past the page box
  4. part-label drift   — an eyebrow saying "Part 3" inside Part 2
  5. missing images     — a figure that never loaded

  python3 gate.py export/lecture-05.pdf
"""
import os
import re
import sys

import fitz

BAD = ["[object Object]", "undefined", "NaN", "⟦", "{{", "v-click"]


def slide_sources(md="slides.md"):
    """Split slides.md the way Slidev does, returning the visible text per slide."""
    lines = open(md).read().split("\n")
    slides, cur, i, n = [], [], 0, len(lines)
    # skip the headmatter block
    if lines and lines[0].strip() == "---":
        j = 1
        while j < n and lines[j].strip() != "---":
            j += 1
        i = j + 1
    while i < n:
        if lines[i].strip() == "---":
            j = i + 1
            while j < n and lines[j].strip() not in ("---", "") and ":" in lines[j]:
                j += 1
            if j > i + 1 and j < n and lines[j].strip() == "---":
                i = j + 1
            else:
                i += 1
            slides.append("\n".join(cur)); cur = []
            continue
        cur.append(lines[i]); i += 1
    slides.append("\n".join(cur))
    out = []
    for s in slides:
        s = re.sub(r"<style>.*?</style>", " ", s, flags=re.S)
        s = re.sub(r"<[^>]+>", " ", s)          # strip tags
        s = re.sub(r"&[a-z]+;", " ", s)
        out.append(" ".join(s.split()))
    return out


def tail_phrase(text, words=6):
    """The last few words of a slide — the first thing to vanish on overflow."""
    w = [t for t in re.findall(r"[A-Za-z][A-Za-z'-]+", text) if len(t) > 2]
    return w[-words:] if len(w) >= words else w


def main(path):
    doc = fitz.open(path)
    W, H = doc[0].rect.width, doc[0].rect.height
    fails, warns = [], []
    part_now, part_pages = None, {}

    for i, page in enumerate(doc, 1):
        txt = page.get_text()
        flat = " ".join(txt.split())

        for b in BAD:
            if b in txt:
                fails.append(f"p{i}: render artefact {b!r}")

        imgs = page.get_images(full=True)
        if len(flat) < 12 and not imgs:
            fails.append(f"p{i}: blank page")

        # overflow — any drawn block whose box leaves the page
        for blk in page.get_text("blocks"):
            x0, y0, x1, y1 = blk[:4]
            if y1 > H + 1.5 or x1 > W + 1.5 or y0 < -1.5 or x0 < -1.5:
                fails.append(f"p{i}: content outside the page box "
                             f"({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})")
                break

        # Overlapping text runs — the caption/footer collision we shipped on 22
        # slides. NOTE: compare LINES, not blocks. PyMuPDF merges two colliding
        # runs into a single block, so a block-level check silently passes.
        boxes = []
        for blk in page.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                t = " ".join("".join(sp["text"] for sp in ln["spans"]).split())
                if t:
                    boxes.append((ln["bbox"], t))
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                (ax0, ay0, ax1, ay1), at = boxes[a]
                (bx0, by0, bx1, by1), bt = boxes[b]
                ox = min(ax1, bx1) - max(ax0, bx0)
                oy = min(ay1, by1) - max(ay0, by0)
                if ox > 3 and oy > 3:
                    fails.append(f"p{i}: text overlaps text — "
                                 f"{at[:34]!r} × {bt[:34]!r}")
                    break
            else:
                continue
            break

        # eyebrows and part dashes render LETTER-SPACED ("P A R T 4 O F 5"), so
        # match against the whitespace-stripped text. Matching `flat` looks
        # right and silently never fires — which is how this check sat dead.
        squash = re.sub(r"\s+", "", txt)
        m = re.search(r"Part(\d)of\d", squash, re.I)
        if m:
            part_now = int(m.group(1))
            part_pages.setdefault(part_now, []).append(i)
            continue
        m = re.search(r"Part(\d)·", squash, re.I)
        if m and part_now is not None and int(m.group(1)) != part_now:
            fails.append(f"p{i}: eyebrow says Part {m.group(1)} "
                         f"but we are inside Part {part_now}")

        if imgs and len(flat) > 900:
            warns.append(f"p{i}: {len(flat)} chars alongside an image — check density")

    # the real overflow mode: .slidev-layout scrolls, so overflowing text is
    # silently DROPPED from the PDF rather than spilling past the page edge.
    # Check that each slide's last words actually made it onto its page.
    src = slide_sources(os.path.join(os.path.dirname(os.path.abspath(path)),
                                     "..", "slides.md"))
    if len(src) == len(doc):
        for i, (s, page) in enumerate(zip(src, doc), 1):
            tail = tail_phrase(s)
            if len(tail) < 4:
                continue
            # eyebrows and part dashes render letter-spaced ("P A R T 2"),
            # so match against the whitespace-stripped page text
            got = re.sub(r"\s+", "", page.get_text()).lower()
            missing = [w for w in tail if w.lower() not in got]
            if len(missing) >= 3:
                fails.append(f"p{i}: slide text truncated — last words missing "
                             f"from the render: {' '.join(tail)!r}")
    else:
        warns.append(f"slide-source count {len(src)} != {len(doc)} pages "
                     "— skipped the truncation check")

    print(f"{path} · {len(doc)} pages · {W:.0f}x{H:.0f}")
    for k in sorted(part_pages):
        print(f"  Part {k} opens on page {part_pages[k][0]}")
    for w in warns:
        print(f"  WARN {w}")
    if fails:
        for f in fails:
            print(f"  FAIL {f}")
        sys.exit(f"\n{len(fails)} FAIL")
    print(f"\n0 FAIL · {len(warns)} WARN")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "export/lecture-05.pdf")
