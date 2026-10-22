"""Does the network ADD or DESTROY 5-information, relative to what the pixels already give?

Absolute probe AUC is close to meaningless on MNIST: raw pixels alone separate 5s from
non-5s at 0.956, so a high number in the trunk describes the dataset, not the model. The
informative quantity is the DISTANCE FROM THAT FLOOR.

  above 0   the network has built representation the pixels did not hand it
  below 0   the network has actively discarded information it was given for free

Everything is scored under each config's own forward mask. Two readouts are shown at the
output because they disagree, and only one is trustworthy:

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
FLOOR = P["dense"]["input"]          # raw pixels, identical for every arm
TRUNK = [s for s in STAGES if s != "input"]

SERIES = [
    ("dense",         "dense — nothing removed",         "#2a78d6", "-",  ("dense", "(0, 0)")),
    ("gram_ON",       "GRAM, module reattached",         "#1baf7a", "--", ("gram", "(1, 1)")),
    ("coreadapt_ON",  "core-adapted, module reattached", "#6250d6", "--", ("coreadapt", "(1, 0)")),
    ("gram_OFF",      "GRAM-OFF — routed away (ships)",  "#1baf7a", "-",  ("gram", "(0, 1)")),
    ("coreadapt_OFF", "core-adapted (ships)",            "#6250d6", "-",  ("coreadapt", "(0, 0)")),
    ("filtered",      "filtered — never in the data",    "#eb6834", "-",  ("filtered", "(0, 0)")),
]
CEILING = {"dense", "gram_ON", "coreadapt_ON"}
NICE = {"embed": "embed", "block1": "block 1", "block2": "block 2", "logits": "logits"}
GREY, INK = "#9a9a92", "#333333"

fig, ax = plt.subplots(figsize=(11.4, 5.9), dpi=200)
fig.subplots_adjust(left=0.085, right=0.655, top=0.755, bottom=0.125)

x = list(range(len(TRUNK)))
xs = x[-1] + 0.45
ax.axhspan(-0.35, 0, color="#fbeae2", zorder=0)
ax.axhline(0, color="#c2451b", lw=1.3, zorder=2)
ax.text(-0.18, 0.005, "the raw-pixel floor (AUC 0.956)", fontsize=7.8,
        color="#c2451b", va="bottom")
ax.text(-0.14, -0.148, "everything in this band scores WORSE than the bare pixels:\n"
        "the network has destroyed information it was handed for free",
        fontsize=8.0, color="#c2451b", va="center", style="italic", linespacing=1.45)

for key, label, col, ls, (araw, amask) in SERIES:
    y = [P[key][s] - FLOOR for s in TRUNK]
    ax.plot(x, y, color=col, ls=ls, lw=2.3, marker="o", ms=5,
            markerfacecolor="white" if ls == "-" else col, markeredgewidth=1.6, zorder=4)
    ax.plot([x[-1]], [y[-1]], marker="o", ms=9, markerfacecolor="none",
            markeredgecolor=col, markeredgewidth=1.4, zorder=5)
    raw = A[araw][amask] - FLOOR
    ax.plot([xs], [raw], marker="*", ms=15, color=col, zorder=6)
    ax.plot([x[-1], xs], [y[-1], raw], color=col, lw=0.8, ls=":", alpha=.6, zorder=3)
    if key not in CEILING:
        ax.text(xs + 0.16, raw, f"{label}\n{raw:+.3f}", fontsize=8.4, color=col,
                va="center", ha="left", linespacing=1.35)
ax.text(xs + 0.16, A["dense"]["(0, 0)"] - FLOOR,
        "dense, and BOTH modules reattached\nall +0.042 — nothing was lost",
        fontsize=8.4, color="#2a78d6", va="center", ha="left", linespacing=1.35)

ax.set_xticks(x + [xs])
ax.set_xticklabels([NICE[s] for s in TRUNK] + ["shipped\nreadout"], fontsize=8.4)
ax.set_xlim(-0.22, xs + 0.30)
ax.set_ylim(-0.283, 0.072)   # gram_OFF reaches -0.256 at the logits probe
ax.set_xlabel("depth — where the probe reads", fontsize=9)
ax.set_ylabel("AUC for “is this a 5?”, minus the raw-pixel floor", fontsize=9)
ax.grid(True, axis="y", color="#ececec", lw=0.6, zorder=1)
ax.set_axisbelow(False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=8.2)

from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], color=INK, marker="o", ms=9, markerfacecolor="none", ls="none",
           label="probe refit on logits — disagrees with the readout, do not trust"),
    Line2D([], [], color=INK, marker="*", ms=13, ls="none",
           label="the shipped readout’s own P(5) — trust this one"),
], loc="lower left", fontsize=8, frameon=False, handletextpad=0.4, bbox_to_anchor=(0.012, 0.008))

fig.suptitle("Which removal routes destroy the 5, and which only hide it",
             fontsize=12.8, x=0.085, ha="left", y=0.955)
fig.text(0.085, 0.825,
         "On MNIST the raw pixels already separate 5s at AUC 0.956, so absolute probe scores "
         "describe the dataset rather than the model.\nWhat matters is the distance from that "
         "floor. Filtering and GRAM routing push the representation BELOW it; core adaptation "
         "does not.\nSeed 0; one checkpoint per arm.",
         fontsize=8.7, color="#666666", linespacing=1.5)
out = f"{HERE}/figures/depth_auc_floor.png"
fig.savefig(out)
print(out)
