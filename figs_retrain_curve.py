"""How fast does retraining undo each removal? Densely sampled, legible in black and white.

Reads results/rc_r*_s*_best.json.

Identity is carried by LINE STYLE and MARKER first and colour second, so the figure survives
greyscale printing and colour blindness. The legend sits outside the axes and the
steps-to-recover numbers live in their own panel rather than on top of the curves, because an
earlier version put both over the lines and they were unreadable.

An earlier version also logged only every 25 steps, by which point every arm was already at
0.98: the whole recovery sat inside one unsampled gap, so the curve that looked slowest was
simply the one that started lowest. Every step to 40 is logged here.
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import figlang as L

HERE = os.path.dirname(os.path.abspath(__file__))
runs = defaultdict(list)
for f in glob.glob(f"{HERE}/results/rc_r*_best.json"):
    d = json.load(open(f))
    runs[d["rate"]].append(d)
ARMS = ["dense", "gram", "coreadapt", "filtered"]
RATES = sorted(runs)
INK, GREY = L.C["ink"], L.C["grey"]
TICKS = [0, 2, 5, 10, 20, 40, 100, 300, 1000]

fig, axes = plt.subplots(1, len(RATES), figsize=(12.4, 6.8), dpi=200, squeeze=False)
fig.subplots_adjust(left=0.070, right=0.985, top=0.780, bottom=0.255, wspace=0.14)

summary = {}
for col, rate in enumerate(RATES):
    ax = axes[0][col]
    ds = runs[rate]
    dense_end = np.mean([d["curves"]["dense"][-1][1] for d in ds if "dense" in d["curves"]])
    tgt = 0.95 * dense_end
    ax.axhline(tgt, color="#555555", lw=1.1, ls=(0, (2, 2)), zorder=2)
    ax.text(1120, tgt + 0.009, "95% of dense", fontsize=8.0, color="#555555", ha="right")

    for key in ARMS:
        cs = [d["curves"][key] for d in ds if key in d["curves"]]
        if not cs:
            continue
        n = min(len(c) for c in cs)
        steps = [c[0] for c in cs[0][:n]]
        y = np.mean([[r[1] for r in c[:n]] for c in cs], axis=0)
        st = L.STYLE[key]
        # markers only on a subset of points, so dense early sampling stays readable
        mk_at = [i for i, s_ in enumerate(steps) if s_ in TICKS]
        ax.plot(steps, y, color=st["color"], ls=st["ls"], lw=st["lw"], zorder=4,
                marker=st["marker"], ms=st["ms"], markevery=mk_at,
                markerfacecolor="white", markeredgewidth=1.5)
        hit = [next((s_ for s_, a_, ao in c if a_ >= tgt and ao >= 0.95), None) for c in cs]
        got = [h for h in hit if h is not None]
        summary[(rate, key)] = (np.mean(got) if got else None, len(got), len(hit))

    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks(TICKS)
    ax.set_xticklabels([str(t) for t in TICKS], fontsize=8.6)
    ax.set_xlim(-0.3, 1150)
    ax.set_ylim(0.38, 1.04)
    ax.set_xlabel("fine-tuning steps", fontsize=9.4)
    if col == 0:
        ax.set_ylabel("READOUT — AUC over the model's own logit 5", fontsize=9.4)
    ax.set_title("clean labels" if rate == 0 else f"{rate:.0%} of 5s mislabeled",
                 fontsize=10.6, color=INK, loc="left")
    ax.grid(True, color="#ececec", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8.6)

# legend OUTSIDE the axes, one row, so it cannot sit on a curve
handles = [Line2D([], [], color=L.STYLE[k]["color"], ls=L.STYLE[k]["ls"],
                  lw=L.STYLE[k]["lw"], marker=L.STYLE[k]["marker"], ms=L.STYLE[k]["ms"],
                  markerfacecolor="white", markeredgewidth=1.5, label=L.LABEL[k])
           for k in ARMS]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.098), ncol=4,
           frameon=False, fontsize=10.0, handlelength=3.8, columnspacing=3.4,
           handletextpad=0.9, borderpad=1.0)

# steps-to-recover in its own strip, not on top of the lines
def cell(rate, key):
    v = summary.get((rate, key))
    if v is None or v[0] is None:
        return "never"
    s = f"{v[0]:.0f}"
    return s if v[1] == v[2] else f"{s}  ({v[2]-v[1]}/{v[2]} never)"


txt = "steps to reach 95% of dense:     " + "     ".join(
    f"{'clean' if r == 0 else f'{r:.0%} noise'} — " +
    ", ".join(f"{L.LABEL[k].split(' —')[0]} {cell(r, k)}"
              for k in ARMS if k != "dense")
    for r in RATES)
fig.text(0.5, 0.030, txt, fontsize=9.0, color=INK, ha="center")

fig.suptitle("Core adaptation retrains fastest — the only route that comes back",
             fontsize=13.0, x=0.070, ha="left", y=0.970)
fig.text(0.070, 0.930,
         "Attacker has 50 labeled 5s and fine-tunes the shipped model. Every point is gated "
         "on the model staying useful (non-5 accuracy ≥ 95%).\nEach method sits at its OWN "
         "best-forgetting setting, not a library default, and the attacker gets the better "
         "of two learning rates.\nEvery step is logged to 40, coarsening out to a 1000-step "
         "budget. An earlier version sampled every 25 steps and stopped at 300: the whole\n"
         "recovery fell inside one unsampled gap, and \u201cnever recovers\u201d described "
         "the budget rather than the method.",
         fontsize=8.7, color="#666666", linespacing=1.55, va="top")
out = f"{HERE}/figures/retrain_curve.png"
fig.savefig(out)
print(out)
