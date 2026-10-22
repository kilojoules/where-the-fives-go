"""How many labeled 5s does it take to undo the removal?

Reads results/elicit_three.json: the shipped model is fine-tuned on k labeled 5s plus 4k
non-5s, and rescored as it trains. Scored by AUC over P(5), never recall -- recall at a fixed
argmax is zero for any model whose logit 5 sits below its best competitor, which is exactly
what suppression arranges, so it reads 0.00 right up until it doesn't.

Every point is gated on the model STAYING USEFUL: non-5 accuracy >= 95%. An attack that wins
by wrecking the model is not an attack.

LEFT   the attack as it runs, at k=50.
RIGHT  the end state against attacker budget. The dashed line is dense -- a model that never
       removed anything -- which is the ceiling an attacker is trying to reach.
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figlang as L

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(f"{HERE}/results/elicit_three.json"))
SHIPS = [("dense_all", "dense — nothing removed", "#2a78d6"),
         ("filter_no5", "filtered — never in the data", "#eb6834"),
         ("gram_ablate5", "GRAM — module detached", "#1baf7a"),
         ("core_adapt5", "core adaptation — moved out", "#6250d6")]
INK, GREY = "#333333", "#9a9a92"
KS = sorted({r["k"] for r in rows})

best = defaultdict(lambda: -1.0)
before, traj = {}, {}
for r in rows:
    best[(r["ship"], r["k"])] = max(best[(r["ship"], r["k"])], r.get("auc5_best_useful", -1))
    before[r["ship"]] = r.get("auc5_before")
    if r["k"] == 50 and r.get("auc5_best_useful", -1) >= best[(r["ship"], 50)]:
        traj[r["ship"]] = r["auc5_traj"]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.8, 6.6), dpi=200,
                             gridspec_kw={"width_ratios": [1.15, 1]})
fig.subplots_adjust(left=0.065, right=0.985, top=0.640, bottom=0.120, wspace=0.21)

# ---- left: the attack running, k=50
for key, label, colour in SHIPS:
    y = traj.get(key)
    if not y:
        continue
    x = np.linspace(0, 300, len(y))
    ax.plot(x, y, color=colour, lw=2.3, zorder=4, label=label)
    ax.plot(0, y[0], "o", color=colour, ms=7, zorder=5)
ax.axhline(before["dense_all"], color="#2a78d6", lw=1.1, ls="--", alpha=0.6, zorder=2)
ax.text(298, before["dense_all"] - 0.012, "dense, untouched", fontsize=7.6,
        color="#2a78d6", ha="right", va="top")
ax.set_xlabel("fine-tuning step  (k = 50 labeled 5s)", fontsize=9.2)
ax.set_ylabel("AUC over P(5) on held-out 5s", fontsize=9.2)
ax.set_title("50 examples is enough", fontsize=10.4, color=INK, loc="left")
ax.set_ylim(0.70, 1.02)
ax.legend(fontsize=8.3, frameon=False, loc="lower right", labelspacing=0.5)

# ---- right: end state vs attacker budget
xs = range(len(KS))
for key, label, colour in SHIPS:
    y = [best[(key, k)] for k in KS]
    bx.plot(xs, y, color=colour, lw=2.3, marker="o", ms=6.5, zorder=4)
    bx.plot(-0.55, before[key], marker="D", ms=8, color=colour, zorder=5,
            markeredgecolor="white", markeredgewidth=0.8)
    bx.plot([-0.55, 0], [before[key], y[0]], color=colour, lw=1.0, ls=":", alpha=0.5, zorder=3)
for k_i, k in enumerate(KS):
    gaps = [best[("dense_all", k)] - best[(s, k)] for s, _, _ in SHIPS[1:]]
    bx.annotate(f"max gap\n{max(gaps):+.3f}", (k_i, 0.735), fontsize=7.8, color=GREY,
                ha="center", va="bottom", linespacing=1.35)
bx.set_xticks([-0.55] + list(xs))
bx.set_xticklabels(["as\nshipped"] + [str(k) for k in KS], fontsize=8.6)
bx.set_xlim(-0.95, len(KS) - 0.55)
bx.set_ylim(0.70, 1.02)
bx.set_xlabel("labeled 5s the attacker has", fontsize=9.2)
bx.set_title("every route converges on the ceiling", fontsize=10.4, color=INK, loc="left")

for a_ in (ax, bx):
    a_.grid(True, color="#ececec", lw=0.6, zorder=0)
    a_.set_axisbelow(True)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
    a_.tick_params(labelsize=8.4)

fig.suptitle("How many labeled 5s does it take to undo the removal?",
             fontsize=12.8, x=0.065, ha="left", y=0.975)
fig.text(0.065, 0.935,
         "The shipped model is fine-tuned on k labeled 5s plus 4k non-5s. Every point is "
         "gated on the model staying useful — non-5 accuracy ≥ 95% —\nso an attack that "
         "wins by wrecking the model does not count. Scored by AUC, never recall: recall is "
         "0.00 for any model whose logit 5 sits\nbelow its best competitor, which is exactly "
         "what suppression arranges.\n\n"
         "At k=50 all three removal routes are within 0.007 AUC of a model that never "
         "removed anything. Core adaptation is the only one that resists\nk=10, holding at "
         "its shipped 0.752 while filtering and GRAM jump past 0.98.",
         fontsize=8.5, color="#666666", linespacing=1.5, va="top")
out = f"{HERE}/figures/retrain.png"
fig.savefig(out)
print(out)
for key, label, _ in SHIPS:
    print(f"  {key:>13}: shipped {before[key]:.4f} -> " +
          "  ".join(f"k={k}: {best[(key,k)]:.4f}" for k in KS))
