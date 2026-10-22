"""Pareto front: how little can you leave in the core, and still have a working module?

Reads results/pcrgrid_s*_r*.json (13 seeds, 25 p_as x p_cr cells, 3 noise rates).

The two objectives pull against each other through the same channel. p_as is the only route
that teaches the core "5-pixels -> 5", which is what makes the module's job learnable -- and
it is also the leak. So every point here is a choice about how much of the capability you are
willing to leave recoverable in what you ship.

  x  READOUT ABOVE CHANCE = |AUC(logit 5) - 0.5| on the shipped core.   Lower is better.
     Not raw AUC: a below-chance readout is still information with the sign flipped, so a cell at
     0.28 is anti-ranked, not clean. See INSTRUMENT.md.
  y  CAPABILITY = module-ON recall on 5s.                      Higher is better.

A point is only on the front if it is RELIABLE -- at least 80% of its 13 seeds actually
learned the digit. Hollow markers failed that bar and are drawn faint: their low information
means the model never learned 5, not that removal worked, which is the single most repeated
error in this campaign. The front is computed over filled markers only.
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
cells = defaultdict(list)
for f in glob.glob(f"{HERE}/results/pcrgrid_s*_r*.json"):
    for c in json.load(open(f))["cells"].values():
        cells[(c["rate"], c["p_as"], c["p_cr"])].append(c)
RATES = sorted({k[0] for k in cells})
NSEED = max(len(v) for v in cells.values())
RELIABLE = 0.8

STYLE = {0.0: ("o", "#2a78d6", "clean labels"),
         0.1: ("s", "#eda100", "10% of 5s mislabeled"),
         0.3: ("^", "#c0392b", "30% of 5s mislabeled")}
INK, GREY = "#333333", "#9a9a92"


def front(pts):
    """Non-dominated set: minimise info, maximise capability."""
    out = []
    for p in pts:
        if not any(q["info"] <= p["info"] and q["cap"] >= p["cap"]
                   and (q["info"] < p["info"] or q["cap"] > p["cap"]) for q in pts):
            out.append(p)
    return sorted(out, key=lambda d: d["info"])


fig, ax = plt.subplots(figsize=(11.6, 7.6), dpi=200)
fig.subplots_adjust(left=0.078, right=0.985, top=0.690, bottom=0.095)

summary, allpts = [], []
for rate in RATES:
    mk, colour, label = STYLE[rate]
    pts = []
    for (r, pa, pc), v in cells.items():
        if r != rate:
            continue
        lk = [x["leak_raw"] for x in v]
        cap = [x["capability"] for x in v]
        pts.append(dict(info=abs(np.mean(lk) - 0.5), cap=float(np.mean(cap)),
                        use=float(np.mean([c > RELIABLE for c in cap])), pa=pa, pc=pc))
    ok = [p for p in pts if p["use"] >= RELIABLE]
    F = front(ok)

    # Only the non-dominated set is drawn. Dominated settings are strictly worse on both
    # objectives, so they are choices nobody should make; the counts stay in the caption.
    ax.plot([p["info"] for p in F], [p["cap"] for p in F], color=colour, lw=1.9,
            alpha=0.80, zorder=3, drawstyle="steps-post")
    ax.scatter([p["info"] for p in F], [p["cap"] for p in F], marker=mk, s=86,
               color=colour, alpha=0.92, zorder=5, label=label,
               edgecolors="white", linewidths=0.8)
    for p in F:
        p["rate"], p["mk"], p["colour"] = rate, mk, colour
        allpts.append(p)
    summary.append(f"{rate:.0%}: {len(F)} of 25 settings on the front "
                   f"({len(pts) - len(ok)} unreliable, {len(ok) - len(F)} dominated)")

# label every front point; nudge vertically where two sit on top of each other
allpts.sort(key=lambda p: (p["info"], p["cap"]))
for i, p in enumerate(allpts):
    near = [q for q in allpts[:i] if abs(q["info"] - p["info"]) < 0.018
            and abs(q["cap"] - p["cap"]) < 0.025]
    ax.annotate(f"{p['pa']:g} / {p['pc']:g}", (p["info"], p["cap"]),
                xytext=(0, 11 + 11 * len(near)), textcoords="offset points",
                fontsize=7.4, color=p["colour"], ha="center")

ax.text(0.015, 0.955, "labels are  p_as / p_cr", transform=ax.transAxes,
        fontsize=8.0, color=GREY)

ax.set_xlabel(L.READOUT_FOLDED,
              fontsize=9.2)
ax.set_ylabel(L.CAPABILITY_DIR, fontsize=9.2)
xs = [p["info"] for p in allpts]; ys = [p["cap"] for p in allpts]
ax.set_xlim(min(xs) - 0.035, max(xs) + 0.035)
ax.set_ylim(min(ys) - 0.045, max(ys) + 0.055)
ax.grid(True, color="#ececec", lw=0.6, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=8.4)

ax.legend(fontsize=8.6, frameon=False, loc="lower right", handletextpad=0.5,
          labelspacing=0.7)

fig.suptitle("What you can actually buy: readout left above chance, against capability kept",
             fontsize=12.6, x=0.078, ha="left", y=0.980)
fig.text(0.078, 0.945,
         "Each point is one (p_as, p_cr) setting averaged over 13 seeds, and only the best trade-offs are drawn — a setting appears if nothing\nbeat it on BOTH axes at once. The step line is the front.\nThe two objectives pull against each other through one channel — p_as is the only route that teaches the core ‘5-pixels → 5’,\nwhich is what makes the module's job learnable, and it is also the leak.\nSettings where fewer than 80% of seeds learned the digit are dropped before the front is computed: their low readout means\nthe model never learned 5, not that removal worked, and that confusion is the most repeated error in this campaign.",
         fontsize=8.4, color="#666666", linespacing=1.5, va="top")
fig.text(0.078, 0.790, "\n".join(summary), fontsize=8.4, color=INK,
         linespacing=1.5, va="top")
out = f"{HERE}/figures/pareto.png"
fig.savefig(out)
print(out)
for s in summary:
    print(" ", s)
