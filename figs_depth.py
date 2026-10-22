"""Where in the network does the 5 stop being decodable?

A linear probe is trained on the frozen representation at each depth, under each config's own
forward mask, and scored by AUC on held-out 5s.

Read the raw-pixel line before anything else: pixels ALONE separate 5s at 0.956, so a high
score in the trunk describes MNIST, not the model. The informative part of this plot is where
each arm sits relative to that line, and three of them fall below it.

Two readouts are shown at the output because they disagree, and only one is trustworthy:

  o  probe refit on the logits -- a 10-way cross-entropy head read at column 5. It is not
     optimising the ranking we score, so it lands BELOW raw for gram_OFF and ABOVE it for
     coreadapt_OFF. It cannot be right in both directions. Shown hollow.
  *  the shipped readout's own P(5) -- what an attacker reading the output actually gets.

Reads results/instruments3.json and results/auc_check.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{HERE}/results/instruments3.json"))
A = json.load(open(f"{HERE}/results/auc_check.json"))
P, STAGES = D["probe"], D["probe_stages"]

SERIES = [
    ("dense",         "dense — nothing removed",         "#2a78d6", "-",  ("dense", "(0, 0)")),
    ("gram_ON",       "GRAM, module reattached",         "#1baf7a", "--", ("gram", "(1, 1)")),
    ("coreadapt_ON",  "core-adapted, module reattached", "#6250d6", "--", ("coreadapt", "(1, 0)")),
    ("gram_OFF",      "GRAM-OFF — routed away (ships)",  "#1baf7a", "-",  ("gram", "(0, 1)")),
    ("coreadapt_OFF", "core-adapted (ships)",            "#6250d6", "-",  ("coreadapt", "(0, 0)")),
    ("filtered",      "filtered — never in the data",    "#eb6834", "-",  ("filtered", "(0, 0)")),
]
# dense and the two module-reattached configs all sit at 0.998-1.000 and cannot be told apart
# on this axis, so they get one shared label rather than three overlapping ones.
CEILING = {"dense", "gram_ON", "coreadapt_ON"}
NICE = {"input": "raw\npixels", "embed": "embed", "block1": "block 1",
        "block2": "block 2", "logits": "logits"}
GREY, INK = "#9a9a92", "#333333"

fig, ax = plt.subplots(figsize=(11.2, 5.6), dpi=200)
fig.subplots_adjust(left=0.085, right=0.655, top=0.775, bottom=0.135)

x = list(range(len(STAGES)))
for key, label, col, ls, (araw, amask) in SERIES:
    y = [P[key][s] for s in STAGES]
    ax.plot(x, y, color=col, ls=ls, lw=2.3, marker="o", ms=5,
            markerfacecolor="white" if ls == "-" else col, markeredgewidth=1.6, zorder=4)
    ax.plot([x[-1]], [y[-1]], marker="o", ms=9, markerfacecolor="none",
            markeredgecolor=col, markeredgewidth=1.4, zorder=5)
    raw = A[araw][amask]
    ax.plot([x[-1] + 0.42], [raw], marker="*", ms=14, color=col, zorder=6)
    ax.plot([x[-1], x[-1] + 0.42], [y[-1], raw], color=col, lw=0.8, ls=":", alpha=.6, zorder=3)
    if key not in CEILING:
        ax.text(x[-1] + 0.60, raw, f"{label}  ({raw:.3f})", fontsize=8.4, color=col,
                va="center", ha="left")

ax.axhline(P["dense"]["input"], color=GREY, lw=1, ls=":", zorder=2)
ax.text(0.03, P["dense"]["input"] - 0.012, "raw pixels already give 0.956 — every arm starts here",
        fontsize=7.6, color=GREY, va="top")
ax.text(x[-1] + 0.60, 0.9996, "dense, and BOTH modules reattached\nall ≥ 0.998 — nothing was lost",
        fontsize=8.4, color="#2a78d6", va="center", ha="left", linespacing=1.4)

ax.set_xticks(x + [x[-1] + 0.42])
ax.set_xticklabels([NICE[s] for s in STAGES] + ["shipped\nreadout"], fontsize=8.4)
ax.set_xlim(-0.25, x[-1] + 0.78)
ax.set_ylim(0.655, 1.025)   # nothing falls below 0.70; chance (0.5) is far off-scale
ax.set_xlabel("depth — where the probe reads", fontsize=9)
ax.set_ylabel("AUC for “is this a 5?”   (0.5 = chance, off-scale below)", fontsize=9)
ax.grid(True, color="#ececec", lw=0.6, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=8.2)

from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], color=INK, marker="o", ms=9, markerfacecolor="none", ls="none",
           label="probe refit on logits (unreliable — see note)"),
    Line2D([], [], color=INK, marker="*", ms=13, ls="none",
           label="shipped readout’s own P(5) — trust this one"),
], loc="lower left", fontsize=8, frameon=False, handletextpad=0.4,
          bbox_to_anchor=(0.012, 0.012))

fig.suptitle("Where the 5 stops being decodable", fontsize=12.8, x=0.085, ha="left", y=0.955)
fig.text(0.085, 0.845,
         "Every arm carries the 5 almost perfectly through the trunk. They separate only at the "
         "very end — which is the custody law:\nthese models did not forget the 5, they differ in "
         "whether the readout will use it. Seed 0; one checkpoint per arm.",
         fontsize=8.7, color="#666666", linespacing=1.5)
out = f"{HERE}/figures/depth_auc.png"
fig.savefig(out)
print(out)
