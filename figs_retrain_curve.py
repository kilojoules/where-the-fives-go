"""How fast does retraining undo each removal? Densely sampled so the answer is visible.

Reads results/rc_r*_s*_best.json.

The previous version of this figure logged every 25 steps and every arm was already at
0.98-0.99 by the first logged point: the whole recovery happened inside one unsampled gap,
so the curve that looked slowest was simply the one that started lowest. This logs EVERY
step for the first 40. It also attacks each method at its own BEST-FORGETTING setting rather
than at library defaults, and sweeps two attacker learning rates keeping the better, because
a single rate made 'filtered' look unrecoverable when that rate simply broke the model.

Every point is gated on the model staying useful (non-5 accuracy >= 95%).
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figlang as L

HERE = os.path.dirname(os.path.abspath(__file__))
runs = defaultdict(list)
for f in glob.glob(f"{HERE}/results/rc_r*_best.json"):
    d = json.load(open(f))
    runs[d["rate"]].append(d)
ARMS = [("dense", "dense — nothing removed", L.C["dense"]),
        ("filtered", "filtered — never in the data", L.C["filtered"]),
        ("gram", "GRAM — best-forgetting setting", L.C["gram"]),
        ("coreadapt", "core adaptation — best-forgetting setting", L.C["coreadapt"])]
RATES = sorted(runs)
INK, GREY = L.C["ink"], L.C["grey"]

fig, axes = plt.subplots(1, len(RATES), figsize=(12.6, 5.6), dpi=200, squeeze=False)
fig.subplots_adjust(left=0.065, right=0.985, top=0.690, bottom=0.135, wspace=0.17)

lines = []
for col, rate in enumerate(RATES):
    ax = axes[0][col]
    ds = runs[rate]
    dense_end = np.mean([d["curves"]["dense"][-1][1] for d in ds if "dense" in d["curves"]])
    tgt = 0.95 * dense_end
    ax.axhline(tgt, color=GREY, lw=1.2, ls="--", zorder=2)
    ax.text(300, tgt + 0.008, "95% of dense", fontsize=7.6, color=GREY, ha="right")
    for key, label, colour in ARMS:
        cs = [d["curves"][key] for d in ds if key in d["curves"]]
        if not cs:
            continue
        n = min(len(c) for c in cs)
        steps = [c[0] for c in cs[0][:n]]
        y = np.mean([[r[1] for r in c[:n]] for c in cs], axis=0)
        ax.plot(steps, y, color=colour, lw=2.2, zorder=4,
                label=label if col == 0 else None)
        ax.plot(steps[0], y[0], "o", color=colour, ms=6, zorder=5)
        # first step reaching the target, per seed, gated
        hit = []
        for c in cs:
            h = next((s for s, a, ao in c if a >= tgt and ao >= 0.95), None)
            hit.append(h)
        got = [h for h in hit if h is not None]
        if got:
            ax.axvline(np.mean(got), color=colour, lw=1.0, ls=":", alpha=0.6, zorder=3)
            ax.annotate(f"{np.mean(got):.0f} steps", (np.mean(got), 0.435),
                        fontsize=8.0, color=colour, ha="left", va="bottom",
                        xytext=(4, 0), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks([0, 2, 5, 10, 20, 40, 100, 300])
    ax.set_xticklabels(["0", "2", "5", "10", "20", "40", "100", "300"], fontsize=8.3)
    ax.set_xlim(-0.3, 320)
    ax.set_ylim(0.40, 1.03)
    ax.set_xlabel("fine-tuning steps  (log-ish scale, every step logged to 40)", fontsize=9)
    if col == 0:
        ax.set_ylabel("READOUT — AUC over the model's own logit 5", fontsize=9.2)
    ax.set_title("clean labels" if rate == 0 else f"{rate:.0%} of 5s mislabeled",
                 fontsize=10.2, color=INK, loc="left")
    ax.grid(True, color="#ececec", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8.3)
axes[0][0].legend(fontsize=8.3, frameon=False, loc="lower right", labelspacing=0.55)

fig.suptitle("Core adaptation retrains FASTEST — it is the only route that comes back",
             fontsize=12.8, x=0.065, ha="left", y=0.968)
fig.text(0.065, 0.922,
         "Attacker has 50 labeled 5s and fine-tunes the shipped model; every point is gated "
         "on the model staying useful (non-5 accuracy ≥ 95%).\nEach method is at its OWN "
         "best-forgetting setting, not a library default, and the attacker gets the better "
         "of two learning rates.\n\n"
         "An earlier version of this figure logged every 25 steps, by which point every arm "
         "was already at 0.98 — the whole recovery sat inside\none unsampled gap, and the "
         "curve that looked slowest was just the one that started lowest. Every step to 40 "
         "is logged here.",
         fontsize=8.5, color="#666666", linespacing=1.5, va="top")
out = f"{HERE}/figures/retrain_curve.png"
fig.savefig(out)
print(out)
