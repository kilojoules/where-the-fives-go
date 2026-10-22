"""Can you steer the 5 back into a model that had it removed? No.

Reads results/instruments3.json (seed 0, clean labels).

Take the difference-of-means direction between 5s and non-5s in the residual stream, add
alpha * that direction, and rescore. Every arm is also run against a MATCHED-NORM RANDOM
direction. That control is what makes the panel interpretable: the learned curves move a lot,
and without something of the same norm to compare against, movement alone looks like evidence
that a 5-direction was found.

The result is a flat null. The best alpha for every arm is 0 -- the untouched shipped model --
and past alpha=0.5 the learned direction destroys the readout FASTER than random noise of the
same size. Whatever difference-of-means finds here, it is not a lever that puts the capability
back.

elicit5.py mandates this control for the story track; steer.py's MNIST path lacked one, which
is why it was added before any of these numbers were read.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{HERE}/results/instruments3.json"))
# viridis 0.3-0.9 separates these by luminance, so one smooth solid line is enough.
# The matched-norm control stays dotted: that is a different MEASUREMENT, not a different arm.
import figlang as _L
SHOW = [("dense", "dense — nothing removed", _L.VIRIDIS["dense"], "-", "o"),
        ("gram_OFF", "GRAM — module detached (ships)", _L.VIRIDIS["gram"], "-", "s"),
        ("coreadapt_OFF", "core adapt — detached (ships)", _L.VIRIDIS["coreadapt"], "-", "^"),
        ("filtered", "filtered — never in the data", _L.VIRIDIS["filtered"], "-", "D")]
INK, GREY = "#333333", "#9a9a92"
alphas = sorted(D["steer"]["dense"]["self_dm"], key=float)
xa = [float(a) for a in alphas]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.8, 5.9), dpi=200,
                             gridspec_kw={"width_ratios": [1.25, 1]})
fig.subplots_adjust(left=0.065, right=0.985, top=0.715, bottom=0.135, wspace=0.21)

for key, label, colour, ls, mk in SHOW:
    sd = [D["steer"][key]["self_dm"][a]["auc5"] for a in alphas]
    rd = [D["steer"][key]["random"][a]["auc5"] for a in alphas]
    ax.plot(xa, sd, color=colour, lw=2.3, ls=ls, marker=mk, ms=6.5, zorder=4, label=label,
            markerfacecolor="white", markeredgewidth=1.5)
    ax.plot(xa, rd, color=colour, lw=1.2, ls=(0, (1, 2)), marker=mk, ms=4.0, alpha=0.55,
            zorder=3)
    ax.plot(xa[0], sd[0], marker="*", ms=16, color=colour, zorder=6,
            markeredgecolor="white", markeredgewidth=0.9)

ax.axhline(0.5, color=GREY, lw=1.1, ls="--", zorder=1)
ax.text(7.8, 0.515, "chance", fontsize=7.8, color=GREY, ha="right")
ax.set_xscale("symlog", linthresh=0.5)
ax.set_xticks(xa); ax.set_xticklabels([f"{a:g}" for a in xa], fontsize=8.6)
ax.set_xlabel("steering strength  α      (α = 0 is the shipped model, untouched)", fontsize=9.2)
ax.set_ylabel("AUC over P(5) on held-out 5s", fontsize=9.2)
ax.set_ylim(0.10, 1.05)
ax.set_title("steering only makes it worse — for every arm, at every strength",
             fontsize=10.2, color=INK, loc="left")
ax.legend(fontsize=9.0, frameon=False, loc="lower left", labelspacing=0.85,
          handlelength=3.2, handletextpad=0.9, borderpad=0.9)

# right: how much does the learned direction buy over its own matched-norm control?
w = 0.19
for i, (key, label, colour, _ls, _mk) in enumerate(SHOW):
    d = [D["steer"][key]["self_dm"][a]["auc5"] - D["steer"][key]["random"][a]["auc5"]
         for a in alphas]
    bx.bar([j + (i - 1.5) * w for j in range(len(alphas))], d, width=w, color=colour,
           zorder=3, label=label)
bx.axhline(0, color="#555555", lw=1.1, zorder=4)
bx.set_xticks(range(len(alphas))); bx.set_xticklabels([f"{a:g}" for a in xa], fontsize=8.6)
bx.set_xlabel("steering strength  α", fontsize=9.2)
bx.set_ylabel("learned direction − matched-norm random", fontsize=9.2)
bx.set_title("the learned direction never beats random", fontsize=10.2, color=INK, loc="left")
bx.text(0.04, 0.95, "above 0 would mean the learned\ndirection recovers more than noise",
        transform=bx.transAxes, fontsize=7.8, color=GREY, va="top", linespacing=1.4)

for a_ in (ax, bx):
    a_.grid(True, color="#ececec", lw=0.6, zorder=0)
    a_.set_axisbelow(True)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
    a_.tick_params(labelsize=8.4)

fig.suptitle("Can you steer the capability back? No.", fontsize=12.6, x=0.065,
             ha="left", y=0.968)
fig.text(0.065, 0.920,
         "Add α × (difference-of-means direction between 5s and non-5s) to the residual "
         "stream and rescore. Solid = that learned direction;\ndotted = a RANDOM direction of "
         "the same norm, the control that makes the comparison mean anything.\n"
         "★ marks the best α for each arm: it is 0 everywhere, so nothing beats leaving the "
         "model alone. Past α=0.5 the learned direction\ndestroys the readout faster than "
         "noise of the same size does — the right panel is that difference, and it is "
         "negative throughout.",
         fontsize=8.5, color="#666666", linespacing=1.5, va="top")
out = f"{HERE}/figures/steering.png"
fig.savefig(out)
print(out)
