"""Probe on the logits against the model's own readout. Every point is labelled.

Only the best trade-offs are drawn: a setting appears if no other setting beat it on BOTH
axes at once. Labelling only a few of those (as an earlier version did) reads as though the
unlabelled points have no hyperparameters, so every drawn point carries its values.

Both axes read the output layer and differ only in what may look at it: the readout is
logit 5 alone, the probe is a linear combination of all ten. Lower is better on both.
The probe can never score lower than the readout, because "use logit 5" is one of the
combinations available to it -- so every point lies above the diagonal, and that distance
is capability hiding in the other nine logits.
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
GREEN, PURPLE, INK, GREY = "#1baf7a", "#6250d6", "#333333", "#9a9a92"

cells = defaultdict(list)
for f in glob.glob(f"{HERE}/results/pvr_*.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if "probe_logits" not in d:
        continue
    cells[(d["method"], d["rate"], d["a"], d["b"])].append(d)

SERIES = [("gram", 0.0, GREEN, "o", "none", "GRAM, clean labels"),
          ("gram", 0.3, GREEN, "o", GREEN, "GRAM, 30% mislabeled"),
          ("coreadapt", 0.0, PURPLE, "s", "none", "core adaptation, clean labels"),
          ("coreadapt", 0.3, PURPLE, "s", PURPLE, "core adaptation, 30% mislabeled")]

fig, ax = plt.subplots(figsize=(10.6, 7.8), dpi=200)
fig.subplots_adjust(left=0.105, right=0.975, top=0.770, bottom=0.095)

def best_tradeoffs(pts):
    """Keep a setting only if nothing else beat it on BOTH axes at once."""
    return [p for p in pts if not any(
        q["x"] <= p["x"] and q["y"] <= p["y"] and (q["x"] < p["x"] or q["y"] < p["y"])
        for q in pts)]


NAMES = {"gram": ("p_as", "p_cr"), "coreadapt": ("weight", "lr")}
pts_all, drawn = [], []
for meth, rate, colour, mk, fill, label in SERIES:
    pts = []
    for (m, r, a, b), v in cells.items():
        if m != meth or r != rate:
            continue
        ok = (np.mean([x["capability"] > 0.8 for x in v]) >= 0.8) if meth == "gram" \
             else (np.mean([x["acc_others"] for x in v]) >= 0.95)
        if not ok:
            continue
        pts.append(dict(x=float(np.mean([q["probe_logits"] for q in v])),
                        y=float(np.mean([q["readout_auc"] for q in v])), a=a, b=b, m=meth))
    if not pts:
        continue
    pts_all += [(p["x"], p["y"]) for p in pts]
    F = best_tradeoffs(pts)
    ax.scatter([p["x"] for p in F], [p["y"] for p in F], marker=mk, s=105,
               facecolors=fill, edgecolors=colour, linewidths=2.0, alpha=0.95,
               zorder=5, label=f"{label}  ({len(F)} of {len(pts)} settings)")
    for p in F:
        p["colour"] = colour
        drawn.append(p)

# every drawn point gets its hyperparameters; nudge apart where they collide
drawn.sort(key=lambda p: (p["x"], p["y"]))
for i, p in enumerate(drawn):
    n = sum(1 for q in drawn[:i]
            if abs(q["x"] - p["x"]) < 0.012 and abs(q["y"] - p["y"]) < 0.05)
    k1, k2 = NAMES[p["m"]]
    ax.annotate(f"{k1}={p['a']:g}, {k2}={p['b']:g}", (p["x"], p["y"]),
                xytext=(9, 6 + 13 * n), textcoords="offset points", fontsize=8.0,
                color=p["colour"], ha="left",
                arrowprops=dict(arrowstyle="-", color=p["colour"], lw=0.6, alpha=0.5))

FLOOR = min(p[0] for p in pts_all)
ax.plot([0.40, 1.01], [0.40, 1.01], color=GREY, lw=1.2, ls=":", zorder=1)
ax.text(0.999, 0.985, "probe = readout", fontsize=8.6, color=GREY, ha="right", va="top")

ax.set_xlim(0.885, 1.008)
ax.set_ylim(0.44, 1.01)
ax.set_xlabel(L.PROBE, fontsize=10.5)
ax.set_ylabel(L.READOUT, fontsize=10.5)
ax.grid(True, color="#ececec", lw=0.6, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=9)
ax.legend(fontsize=9.2, frameon=False, loc="lower right", labelspacing=0.7,
          handletextpad=0.7)

fig.suptitle("Removal shows up in the output, not in the logits",
             fontsize=14, x=0.105, ha="left", y=0.962)
fig.text(0.105, 0.918,
         "Each point is one hyperparameter setting, labelled with its values. Lower is better "
         "on both axes.\nOnly the best trade-offs are drawn — a setting appears if no other "
         f"setting beat it on BOTH axes at once.\nAcross all {len(pts_all)} settings the "
         f"readout falls as low as {min(p[1] for p in pts_all):.2f}, but the probe never "
         f"drops below {FLOOR:.2f}.",
         fontsize=10, color="#555555", linespacing=1.6, va="top")
out = f"{HERE}/figures/probe_vs_readout.png"
fig.savefig(out)
print(out, f"| {len(pts_all)} settings | probe floor {FLOOR:.3f} | readout min {min(p[1] for p in pts_all):.3f}")
