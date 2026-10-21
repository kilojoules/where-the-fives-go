"""The finding, on one axis: mislabel some 5s and three routes forget harder — one re-learns.

Reads results/absorb3_seed{0..7}.json.

Every route here decides what to remove by reading the LABEL (train.py:108, train.py:120,
core_adapt5.py:166), so a true 5 wearing another digit's label is invisible to all of them
identically. What differs is the gradient each then applies to the example it missed:

  filtering, GRAM   ordinary cross-entropy toward a WRONG digit -> logit 5 is pushed DOWN
  core adaptation   the retain-OFF term pins the SHIPPED core to the TEACHER's output
                    (core_adapt5.py:246), and the teacher is the donor, which still reads the
                    image as a 5 whatever its label says -> logit 5 is pushed UP

So core adaptation's retention objective re-teaches the capability on exactly the examples its
removal objective failed to identify. Every arm runs at its DEFAULT hyperparameters.

Bars are the paired change per seed, so each seed carries its own baseline; dots are the eight
seeds. Scored on raw logit 5 -- softmax P(5) rises for every removal route here and would hide
the whole effect (INSTRUMENT.md).
"""
import glob
import json
import os
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = sorted(int(f.split("seed")[-1].split(".")[0])
               for f in glob.glob(f"{HERE}/results/absorb3_seed*.json"))
D = {s: json.load(open(f"{HERE}/results/absorb3_seed{s}.json")) for s in SEEDS}
R0, R1 = "0.0", "0.3"

ROWS = [  # key, label, colour  — worst-to-best so the flip lands at the bottom
    ("gram_main",   "GRAM\nrouted away during training", "#1baf7a"),
    ("filter_no5",  "filtered\n5s never in the data",    "#eb6834"),
    ("dense_all",   "dense\nnothing removed at all",     "#2a78d6"),
    ("core_adapt5", "core adaptation\nmoved out of a trained core", "#6250d6"),
]
INK, GREY = "#333333", "#9a9a92"


def sign_p(d):
    d = [x for x in d if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    m = max(k, n - k)
    return n, k, min(1.0, 2 * sum(comb(n, i) for i in range(m, n + 1)) / 2 ** n)


fig, ax = plt.subplots(figsize=(12.6, 6.4), dpi=200)
fig.subplots_adjust(left=0.205, right=0.805, top=0.655, bottom=0.205)

deltas = {}
for key, _, _ in ROWS:
    deltas[key] = [D[s]["arms"][key][R1]["auc_clean_rawlogit"] -
                   D[s]["arms"][key][R0]["auc_clean_rawlogit"] for s in SEEDS]
lo = min(min(v) for v in deltas.values())
hi = max(max(v) for v in deltas.values())
pad = 0.06 * (hi - lo)

tx = ax.get_yaxis_transform()        # x in axes fraction, y in data coords
for i, (key, label, colour) in enumerate(ROWS):
    d = deltas[key]
    m = float(np.mean(d))
    ax.barh(i, m, height=0.50, color=colour, alpha=0.88, zorder=3)
    ax.scatter(d, [i] * len(d), s=26, color="#ffffff", edgecolors=colour, linewidths=1.1,
               zorder=5, alpha=0.95)
    n, k, p = sign_p(d)
    ax.text(1.035, i + 0.12, f"{m:+.3f}", transform=tx, va="center", ha="left",
            fontsize=11.2, color=colour, fontweight="bold")
    ax.text(1.035, i - 0.20, f"{k}/{n} seeds up    p={p:.3f}", transform=tx, va="center",
            ha="left", fontsize=7.9, color=GREY)

ax.axvline(0, color="#555555", lw=1.3, zorder=4)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([r[1] for r in ROWS], fontsize=9.4, linespacing=1.45)
ax.set_xlim(lo - pad, hi + pad)
ax.set_ylim(-0.74, len(ROWS) - 0.30)
ax.set_xlabel("change in how well the SHIPPED model ranks held-out 5s\n"
              "AUC over logit 5:  clean labels  →  30% of 5s mislabeled", fontsize=9.3)
ax.grid(True, axis="x", color="#ececec", lw=0.6, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=8.6, length=0)

ax.text(lo - pad / 2, -0.66, "←  FORGETS 5s HARDER", fontsize=9.0, color=GREY, ha="left")
ax.text(hi + pad / 2, -0.66, "RE-LEARNS 5s  →", fontsize=9.0, color="#6250d6", ha="right",
        fontweight="bold")

fig.suptitle("Mislabel some 5s, and three routes forget harder — one re-learns them",
             fontsize=13.2, x=0.045, ha="left", y=0.965)
fig.text(0.045, 0.895,
         "Every route decides what to remove by reading the LABEL, so a true 5 wearing another "
         "digit's label is invisible to all four — the same 1,626 of 5,421\ntraining 5s slip "
         "past each one. What differs is the gradient each then applies to it. Filtering and "
         "GRAM train on it with the wrong digit's label,\nso the model learns ‘this is "
         "not a 5’. Core adaptation never looks at the label: it tells the shipped core to "
         "match its TEACHER on that image, and\nthe teacher is the original model, which still "
         "sees a 5 (core_adapt5.py:246) — its retention objective re-teaches what its removal "
         "objective missed.",
         fontsize=8.7, color="#666666", linespacing=1.55, va="top")
fig.text(0.045, 0.055,
         "All four arms at their DEFAULT hyperparameters — GRAM p_as=0.5, p_cr=0.5; core "
         "adaptation retain_off_weight=2.0, core_lr=1e-5, 60 cycles.\n"
         f"{len(SEEDS)} seeds, paired within seed so each carries its own baseline; dots are "
         "seeds. Scored on raw logit 5 — softmax P(5) rises for every\nremoval route here and "
         "would hide the effect entirely (INSTRUMENT.md).",
         fontsize=7.9, color=GREY, linespacing=1.5, va="top")
out = f"{HERE}/figures/the_flip.png"
fig.savefig(out)
print(out)
