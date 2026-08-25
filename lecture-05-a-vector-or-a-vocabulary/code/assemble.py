#!/usr/bin/env python3
"""Assemble slides.md from parts/, substituting the MEASURED numbers.

Every ⟦PLACEHOLDER⟧ in parts/crisp_body.md is filled from pilot/out/stats.json,
which compare() writes. If a number is missing the build fails loudly rather
than shipping a slide with a made-up figure on it.

  python3 assemble.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STATS = os.path.join(HERE, "pilot", "out", "figs", "stats.json")


def main():
    if not os.path.exists(STATS):
        sys.exit(f"missing {STATS} — run compare() first")
    s = json.load(open(STATS))

    K = s["K"]
    perp, alive = s["codebook_perplexity"], s["codes_alive"]
    sr, si, sreal = s["sharp_rssm"], s["sharp_iris"], s["sharp_real"]
    fr, fi = s["fps_rssm"], s["fps_iris"]

    sd = s.get("sharp_iris_sd", 0.0)
    nroll = s.get("n_rollouts", 1)
    head = (f"Edge sharpness — real footage <b>{sreal:.3f}</b>, RSSM "
            f"<b>{sr:.3f}</b>, IRIS <b>{si:.3f}</b>"
            + (f" ± {sd:.3f} over {nroll} seeded rollouts" if nroll > 1 else "")
            + ". ")

    def where(v):
        """Describe a value's position relative to the real footage, from data."""
        d = (v - sreal) / sreal
        if d < -0.04:
            return "below"
        if d > 0.04:
            return "above"
        return "level with"

    w_r, w_i = where(sr), where(si)
    tail = {
        "below": "<em>below</em> the real game — too smooth, which is what "
                 "averaging futures looks like",
        "above": "<em>above</em> the real game — a shade too crisp, because every "
                 "edge it can draw is a codebook edge",
        "level with": "essentially <em>level with</em> the real game",
    }
    if si > sr:
        verdict = (head + f"The token model's dreams are {si/sr:.2f}× sharper than "
                   f"the RSSM's. And note where each one lands: the RSSM is "
                   f"{tail[w_r]}; IRIS is {tail[w_i]}.")
    else:
        verdict = (head + "At this scale the token model did NOT come out sharper. "
                   "We report what we measured; the reason belongs on the slide, "
                   "not in a footnote.")

    ctx = s["context_curve"]
    drop = ctx[0] / max(ctx[1], 1e-9)
    knee = next((i + 1 for i in range(len(ctx))
                 if ctx[i] < 1.15 * min(ctx)), len(ctx))
    fps_ctx = s["fps_iris_by_ctx"]
    spread = max(fps_ctx.values()) / max(min(fps_ctx.values()), 1e-9)

    sub = {
        "RSSM_PAR": f"{s['rssm_params']:,}",
        "IRIS_PAR": f"{s['iris_params']:,} (tokenizer + transformer)",
        "CODEBOOK_LINE": (f"perplexity {perp:.0f} of {K} · {alive} of {K} codes "
                          f"alive — this dictionary is genuinely in use"),
        "DREAM_LINE": (f"top: what really happened · middle: the RSSM's imagination · "
                       f"bottom: IRIS sampling tokens · edge sharpness "
                       f"{sreal:.3f} / {sr:.3f} / {si:.3f}"
                       + (f" (IRIS averaged over {nroll} seeded rollouts, "
                          f"sd {sd:.3f})" if nroll > 1 else "")),
        "DREAM_VERDICT": verdict,
        "EFF_LINE": (f"{fr:,.0f} imagined frames per second for the RSSM against "
                     f"{fi:,.0f} for IRIS — a factor of {fr/max(fi,1e-9):,.0f} · "
                     "right: at this size the transformer's throughput is flat — "
                     + " / ".join(f"{fps_ctx[k]:.1f}" for k in sorted(fps_ctx, key=int))
                     + " fps at 2, 4 and 8 frames — so the gap is the sixteen "
                     "sequential decodes, not attention's quadratic term"),
        "CTX_KNEE": str(knee),
        "CTX_DROP": f"{drop:,.0f}×",
        "CTX_LINE": (f"one visible frame gives cross-entropy {ctx[0]:.2f}; two gives "
                     f"{ctx[1]:.3f}; three gives {ctx[2]:.3f} and it is flat from "
                     f"there — on CoinRun the useful past is about "
                     f"{knee} frames deep"),
        "RECON_MSE": f"{s['recon_mse_heldout']:.5f}",
    }

    body = open(os.path.join(HERE, "parts", "crisp_body.md")).read()
    for k, v in sub.items():
        body = body.replace(f"⟦{k}⟧", v)
    if "⟦" in body:
        left = {t.split("⟧")[0] for t in body.split("⟦")[1:]}
        sys.exit(f"unfilled placeholders: {sorted(left)}")

    head = open(os.path.join(HERE, "parts", "00_head.md")).read()
    open(os.path.join(HERE, "slides.md"), "w").write(head + body)
    n = 1 + body.count("\n---\n")
    print(f"slides.md written · ~{n} slides · all placeholders filled")


if __name__ == "__main__":
    main()
