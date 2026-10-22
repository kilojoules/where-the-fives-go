"""How far down does the 5 survive, and does a probe see more than the model emits?

Reads results/instruments3.json (seed 0, clean labels).

Five probe depths, then a sixth column that is NOT a probe: the model's own output, scored as
AUC over its raw logit 5. That column is what the deployed model actually emits; the five to
its left are what a linear readout can recover from inside it. The gap between them is the
whole elicitation question -- an attacker who can fit a readout gets the probe column, a user
who only sees outputs gets the last one.

The pixel floor matters more than any single number here. Raw pixels give AUC 0.956 before any
model touches them, because MNIST 5s are close to linearly separable. A probe scoring 0.96 in
the trunk is therefore describing the dataset. Only the shaded region below that line says
anything about removal, which is why the DECAY across depth carries the signal and the
absolute height does not.

Caveat on the 'probe @ logits' column: it comes from layer_probe.probe_fit, a 10-way CE head
read at column 5. INSTRUMENT.md shows that head is unreliable -- here it lands BELOW the
model's own output for GRAM (0.701 vs 0.851) and far ABOVE it for core adaptation (0.943 vs
0.737), which cannot both be a faithful reading. The trunk columns are sound; that one is
superseded by the binary probe in scale_study.py.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{HERE}/results/instruments3.json"))
STAGES = ["input", "embed", "block1", "block2", "logits"]
COLS = STAGES + ["model's own\noutput"]
# viridis 0.3-0.9: perceptually uniform, so these six are separable by luminance alone and
# need no line-style channel. One smooth solid line throughout.
import matplotlib.cm as _cm, matplotlib.colors as _mc
_KEYS = ["dense", "gram_OFF", "gram_ON", "coreadapt_OFF", "coreadapt_ON", "filtered"]
_LABS = ["dense — nothing removed",
         "GRAM — module detached (ships)", "GRAM — module reattached",
         "core adapt — detached (ships)", "core adapt — reattached",
         "filtered — never in the data"]
_COLS = [_mc.to_hex(_cm.viridis(v)) for v in
         [0.3 + 0.6 * i / (len(_KEYS) - 1) for i in range(len(_KEYS))]]
_MKS = ["o", "s", "v", "^", "P", "D"]   # one shape per series
ROWS = [(k, l, c, "-", m) for k, l, c, m in zip(_KEYS, _LABS, _COLS, _MKS)]
INK, GREY = "#333333", "#9a9a92"
FLOOR = D["probe"]["dense"]["input"]

fig, ax = plt.subplots(figsize=(11.6, 6.6), dpi=200)
fig.subplots_adjust(left=0.075, right=0.985, top=0.735, bottom=0.125)

ax.axhspan(0.50, FLOOR, color="#fafaf8", zorder=0)
ax.axhline(FLOOR, color=GREY, lw=1.3, ls="--", zorder=2)
ax.text(0.05, FLOOR + 0.006,
        f"raw-pixel floor {FLOOR:.3f} — above this line a probe is reading MNIST, not the model",
        fontsize=8.0, color=GREY, va="bottom")
# the last column is a different kind of measurement; separate it visually
ax.axvline(4.5, color="#cfcfc9", lw=1.1, ls="-", zorder=1)
ax.text(5.0, 0.515, "what the model\nactually emits", fontsize=8.0, color=GREY,
        ha="center", va="bottom", linespacing=1.4)

x = list(range(len(STAGES)))
for key, label, colour, ls, mk in ROWS:
    y = [D["probe"][key][s] for s in STAGES]
    ax.plot(x, y, color=colour, lw=2.2, ls=ls, marker=mk, ms=6.5, label=label, zorder=4,
            markerfacecolor="white", markeredgewidth=1.5)
    nat = D["probe"][key]["native_auroc"]
    ax.plot([4, 5], [y[-1], nat], color=colour, lw=1.1, ls="-", alpha=0.45, zorder=3)
    ax.plot(5, nat, marker=mk, ms=9.5, color=colour, zorder=6,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.annotate(f"{nat:.3f}", (5, nat), xytext=(11, -3), textcoords="offset points",
                fontsize=8.2, color=colour)

ax.set_xticks(list(range(6)))
ax.set_xticklabels(COLS, fontsize=8.8)
ax.set_xlim(-0.15, 5.65)
ax.set_ylim(0.52, 1.015)
ax.set_xlabel("where the 5-vs-rest score is read from", fontsize=9.2)
ax.set_ylabel("AUC for 5-vs-rest", fontsize=9.2)
ax.grid(True, color="#ececec", lw=0.6, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=8.4)
ax.legend(fontsize=8.3, frameon=False, loc="lower left", labelspacing=0.55)

fig.suptitle("How far down does the 5 survive — and does a probe see more than the model emits?",
             fontsize=12.4, x=0.075, ha="left", y=0.972)
fig.text(0.075, 0.922,
         "Circles are linear probes fitted at that depth. Diamonds are the model's OWN output, "
         "scored as AUC over its raw logit 5 — not a probe.\nThe gap between the last circle "
         "and the diamond is the elicitation question: an attacker who can fit a readout gets "
         "the circle, a user\nwho only sees outputs gets the diamond.\n\n"
         "Read the DECAY, not the height. Filtering decays hardest (0.981→0.780), GRAM next "
         "(0.996→0.701), core adaptation barely at all\n(0.997→0.943) — detaching GRAM's module "
         "costs 8 points of block2, detaching core adaptation's costs 0.3, because its core "
         "never gave\nthe capability up.",
         fontsize=8.5, color="#666666", linespacing=1.5, va="top")
out = f"{HERE}/figures/probe_depth.png"
fig.savefig(out)
print(out)
for k, lab, _c, _ls, _mk in ROWS:
    print(f"  {k:>14}: probe@logits {D['probe'][k]['logits']:.3f}   own output "
          f"{D['probe'][k]['native_auroc']:.3f}   (gap {D['probe'][k]['logits'] - D['probe'][k]['native_auroc']:+.3f})")
