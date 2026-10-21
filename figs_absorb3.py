"""Mislabeled 5s: the two statistics move in opposite directions.

Reads results/absorb3_seed{0..7}.json (written by absorb3.py).

Every route decides what to remove by reading the LABEL -- filtering drops label-5 examples
(train.py:108), GRAM routes label-5 examples into module 5 (train.py:120), core adaptation
splits target from retain on whether the label is 5 (core_adapt5.py:166). So a true 5 wearing
another digit's label is invisible to all three, identically, and absorption VOLUME is not
what separates them. What separates them is the gradient each mechanism then applies to it.

Drawn as a paired slope chart because the pairing is the evidence: each thin line is one seed
carried from clean labels to 30% of its 5s mislabeled, same seed and same noise realization at
both ends, so a seed's own baseline cannot be confused for an effect. Seed spread here is
+-0.13 on some cells, larger than several of the effects, and only the paired view survives it.

Panel A is softmax P(5), what every earlier table in this repo reports. Panel B is raw logit 5.
Both pass a capability gate on a clean dose (results/gate5_fine.json) -- and on the filtered
arm they disagree in SIGN. See INSTRUMENT.md.
"""
import json
import os
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [s for s in range(8) if os.path.exists(f"{HERE}/results/absorb3_seed{s}.json")]
D = {s: json.load(open(f"{HERE}/results/absorb3_seed{s}.json")) for s in SEEDS}
R0, R1 = "0.0", "0.3"

ROWS = [  # key, label, colour
    ("dense_all",   "dense\nnothing removed",  "#2a78d6"),
    ("filter_no5",  "filtered\nnever in data", "#eb6834"),
    ("gram_main",   "GRAM\nrouted away",       "#1baf7a"),
    ("core_adapt5", "core adaptation\nmoved out", "#6250d6"),
]
GREY, INK, FAINT = "#9a9a92", "#333333", "#c9ccd1"


def sign_p(d):
    d = [x for x in d if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    m = max(k, n - k)
    return n, k, min(1.0, 2 * sum(comb(n, i) for i in range(m, n + 1)) / 2 ** n)


fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), dpi=200)
fig.subplots_adjust(left=0.062, right=0.988, top=0.735, bottom=0.145, wspace=0.165)

for ax, stat, title in (
        (axes[0], "auc_clean", "softmax P(5) — what this repo has been reporting"),
        (axes[1], "auc_clean_rawlogit", "raw logit 5 — the same weights, scored alone")):
    for i, (key, label, colour) in enumerate(ROWS):
        a = [D[s]["arms"][key][R0][stat] for s in SEEDS]
        b = [D[s]["arms"][key][R1][stat] for s in SEEDS]
        x0, x1 = i - 0.17, i + 0.17
        for u, v in zip(a, b):                      # one line per seed
            ax.plot([x0, x1], [u, v], color=colour, lw=0.85, alpha=0.42, zorder=2)
        ax.plot([x0, x1], [np.mean(a), np.mean(b)], color=colour, lw=2.8, zorder=4,
                solid_capstyle="round")
        ax.plot([x0, x1], [np.mean(a), np.mean(b)], "o", color=colour, ms=5.5, zorder=5)
        n, k, p = sign_p([v - u for u, v in zip(a, b)])
        d = np.mean(b) - np.mean(a)
        ax.annotate(f"{d:+.3f}\n{k}/{n}  p={p:.3f}", (x1 + 0.055, np.mean(b)),
                    fontsize=7.6, color=colour, va="center", linespacing=1.35)

    ax.axhline(0.5, color=GREY, lw=1.1, ls="--", zorder=1)
    ax.text(-0.42, 0.512, "chance", fontsize=7.2, color=GREY, va="bottom")
    ax.set_xticks(range(len(ROWS)))
    ax.set_xticklabels([r[1] for r in ROWS], fontsize=8.1, linespacing=1.4)
    ax.set_xlim(-0.45, len(ROWS) - 0.30)
    ax.set_ylim(0.16, 1.045)
    ax.set_title(title, fontsize=9.6, color=INK, loc="left")
    ax.grid(True, axis="y", color="#ececec", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8.1, length=0)

axes[0].set_ylabel("AUC, held-out 5s, shipped readout", fontsize=9)
# Each route's pair of dots is left=clean, right=30% mislabeled; say so once.
axes[0].text(-0.42, 0.205, "left dot: clean labels      right dot: 30% of 5s mislabeled",
             fontsize=7.6, color=GREY)

n5 = D[SEEDS[0]]["arms"]["dense_all"][R1]["n_relabeled"]
fig.suptitle("Mislabeled 5s: every route absorbs the same ones, and only one route re-learns them",
             fontsize=12.4, x=0.062, ha="left", y=0.952)
fig.text(0.062, 0.845,
         f"All three removal routes read the label, so the same {n5} of 5,421 training 5s slip "
         "past all of them. A wrong label then pushes logit 5 DOWN under\nfiltering and GRAM, "
         "but core adaptation pins its shipped core to a teacher that still knows 5s "
         "(core_adapt5.py:246) — and a wrong label does not fool\nthe teacher. Paired on seed: "
         "core adaptation rises against each other route 8/8 seeds, p=0.008. Note the filtered "
         "arm disagrees in SIGN between panels.",
         fontsize=8.5, color="#666666", linespacing=1.5)
out = f"{HERE}/figures/absorb3.png"
fig.savefig(out)
print(out, f"({len(SEEDS)} seeds)")
