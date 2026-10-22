"""Core adaptation's operating front, the analogue of figures/pareto.png for GRAM.

Reads results/cag_s*_r*_w*_lr*.json (one file per cell, written by ca_grid.py).

Two knobs, mapped onto GRAM's pair:

  retain_off_weight  weight on kl(s_off_r, tr_), which pins the SHIPPED core to the donor
                     teacher on retain data (core_adapt5.py:246). retain_target.py showed
                     changing this term changes the leak in 5/5 seeds, so it IS the leak
                     path -- the structural analogue of p_as.
  core_lr            how far the core can move per step: removal, and damage.

The reliability filter differs from the GRAM front and that difference is the point. There the
question was whether the module ever learned 5. Here the capability is COPIED from a donor
that already has it, so what breaks instead is RETENTION of the other nine digits -- and it
does break: at retain_off_weight=0 the pinning term is gone and retention falls to 0.18-0.55.
Those cells post the lowest leak in the whole grid while having lost up to 45% of the other
digits, which is the same trap as a GRAM cell that never learned a 5.

  x  INFORMATION = |AUC(logit 5) - 0.5| on the shipped core (module detached).  Lower better.
  y  CAPABILITY  = module-ON recall on 5s.                                      Higher better.

Only non-dominated, retention-safe settings are drawn.
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RET_BAR = 0.95

byseed = defaultdict(dict)
for f in glob.glob(f"{HERE}/results/cag_s*.json"):
    try:
        D = json.load(open(f))
    except Exception:
        continue
    for c in D.get("cells", {}).values():
        byseed[D["seed"]][(c["rate"], c["retain_off_weight"], c["core_lr"])] = c
# only seeds with the full 30-cell grid, so every point averages the same seeds
NEED = len({k for s in byseed for k in byseed[s]})
SEEDS = sorted(s for s in byseed if len(byseed[s]) >= 30)
keys = sorted({k for s in SEEDS for k in byseed[s]})
RATES = sorted({k[0] for k in keys})

STYLE = {0.0: ("o", "#2a78d6", "clean labels"),
         0.3: ("^", "#c0392b", "30% of 5s mislabeled")}
INK, GREY = "#333333", "#9a9a92"


def front(pts):
    out = []
    for p in pts:
        if not any(q["info"] <= p["info"] and q["cap"] >= p["cap"]
                   and (q["info"] < p["info"] or q["cap"] > p["cap"]) for q in pts):
            out.append(p)
    return sorted(out, key=lambda d: d["info"])


fig, ax = plt.subplots(figsize=(11.6, 7.2), dpi=200)
fig.subplots_adjust(left=0.080, right=0.985, top=0.700, bottom=0.098)

summary, allpts = [], []
for rate in RATES:
    mk, colour, label = STYLE[rate]
    pts, dropped = [], 0
    for (r, w, lr) in keys:
        if r != rate:
            continue
        v = [byseed[s][(r, w, lr)] for s in SEEDS if (r, w, lr) in byseed[s]]
        if not v:
            continue
        ret = float(np.mean([x["acc_others"] for x in v]))
        if ret < RET_BAR:          # lost the other nine -- not a removal result
            dropped += 1
            continue
        lk = [x["leak_raw"] for x in v]
        pts.append(dict(info=abs(np.mean(lk) - 0.5), cap=float(np.mean([x["capability"] for x in v])),
                        ret=ret, w=w, lr=lr))
    F = front(pts)
    ax.plot([p["info"] for p in F], [p["cap"] for p in F], color=colour, lw=1.9,
            alpha=0.80, zorder=3, drawstyle="steps-post")
    ax.scatter([p["info"] for p in F], [p["cap"] for p in F], marker=mk, s=88,
               color=colour, alpha=0.92, zorder=5, label=label,
               edgecolors="white", linewidths=0.8)
    for p in F:
        p["colour"] = colour
        allpts.append(p)
    summary.append(f"{'clean' if rate == 0 else f'{rate:.0%} mislabeled'}: "
                   f"{len(F)} of {len(pts) + dropped} settings on the front "
                   f"({dropped} dropped for retention < {RET_BAR:.2f}), "
                   f"best information {min(p['info'] for p in F):.3f}")

allpts.sort(key=lambda p: (p["info"], p["cap"]))
for i, p in enumerate(allpts):
    near = [q for q in allpts[:i] if abs(q["info"] - p["info"]) < 0.02
            and abs(q["cap"] - p["cap"]) < 0.03]
    ax.annotate(f"w={p['w']:g}, lr={p['lr']:g}", (p["info"], p["cap"]),
                xytext=(0, 11 + 11 * len(near)), textcoords="offset points",
                fontsize=7.4, color=p["colour"], ha="center")

ax.set_xlabel("INFORMATION left in the shipped core   |AUC(logit 5) − 0.5|   ←  better",
              fontsize=9.2)
ax.set_ylabel("CAPABILITY   module-ON recall on 5s   better  →", fontsize=9.2)
ax.grid(True, color="#ececec", lw=0.6, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=8.4)
ax.legend(fontsize=8.8, frameon=False, loc="lower right", handletextpad=0.5, labelspacing=0.7)

fig.suptitle("Core adaptation: no setting escapes the teacher leak",
             fontsize=12.6, x=0.080, ha="left", y=0.975)
fig.text(0.080, 0.938,
         "Each point is one (retain_off_weight, core_lr) setting averaged over "
         f"{len(SEEDS)} seeds; only the best trade-offs are drawn \u2014 a setting appears if nothing beat it on BOTH axes.\n"
         "retain_off_weight is the term that pins the shipped core to the donor teacher "
         "(core_adapt5.py:246) — the leak path itself.\n"
         "Settings whose other-nine retention fell below 0.95 are dropped first: at "
         "retain_off_weight=0 the pinning term is gone and retention\n"
         "collapses to 0.18–0.55, which posts the lowest leak in the grid from a model that "
         "lost up to 45% of its other digits.",
         fontsize=8.4, color="#666666", linespacing=1.5, va="top")
fig.text(0.080, 0.800, "\n".join("      " + s for s in summary),
         fontsize=8.4, color=INK, linespacing=1.55, va="top")
out = f"{HERE}/figures/ca_pareto.png"
fig.savefig(out)
print(out, f"({len(SEEDS)} seeds: {SEEDS})")
for s in summary:
    print("  " + s)
