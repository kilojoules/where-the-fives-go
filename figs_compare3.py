"""Three ways to not recognize 5s, scored the same way.

Reads results/compare3.json (compare3.py) and results/core_adapt5_c12.json (core_adapt5.py).

Left: where each route lands. Right: how core adaptation got there, and that it has not
finished -- the curve is still falling at the last cycle.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(f"{HERE}/results/compare3.json"))["arms"]
T = json.load(open(f"{HERE}/results/core_adapt5_c12.json"))["history"]

ROWS = [
    ("dense_all",   "knows 5\n(nothing removed)",        "#2a78d6"),
    ("filter_no5",  "never in the data\n(filtered)",     "#eb6834"),
    ("gram_main",   "routed away\nduring training",      "#1baf7a"),
    ("core_adapt5", "moved out after\ntraining (12 cyc)", "#6250d6"),
]
GREY, INK = "#9a9a92", "#333333"

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=200,
                             gridspec_kw={"width_ratios": [1, 1.15]})
fig.subplots_adjust(left=0.115, right=0.985, top=0.745, bottom=0.145, wspace=0.30)

# ---- left: recall on 5, in each arm's shipped configuration
y = list(range(len(ROWS)))[::-1]
for yi, (k, lab, col) in zip(y, ROWS):
    v = C[k]["recall5"]
    ax.barh(yi, v, color=col, height=.6, zorder=3)
    ax.text(v + .012 if v > 0.02 else .012, yi, f"{v:.4f}", va="center", fontsize=9,
            color=INK if v > 0.02 else GREY)
ax.set_yticks(y)
ax.set_yticklabels([r[1] for r in ROWS], fontsize=8.8, linespacing=1.4)
ax.set_xlim(0, 1.06)
ax.set_xlabel("recall on the 892 held-out 5s, as shipped", fontsize=9)
ax.set_title("filtering and routing reach zero; moving it out afterwards does not",
             fontsize=9.8, color=INK, loc="left")

# ---- right: the core-adaptation trajectory against the two floors
x = list(range(len(T)))
off5 = [h["ablate5"]["recall5"] for h in T]
bx.axhline(0, color="#eb6834", lw=1.6, ls="--", zorder=2)
bx.text(len(T) - 0.2, 0.028, "filtered and GRAM-routed both sit here", fontsize=7.8,
        color="#eb6834", ha="right", va="bottom")
bx.axhline(C["dense_all"]["recall5"], color="#2a78d6", lw=1.6, ls=":", zorder=2)
bx.text(5.6, C["dense_all"]["recall5"] - 0.035, "a model that still knows 5", fontsize=7.8,
        color="#2a78d6", va="top")
bx.plot(x, off5, color="#6250d6", lw=2.6, marker="o", ms=4.6, zorder=4)
bx.annotate("still falling at the last cycle:\nthis has not finished",
            (x[-1], off5[-1]), xytext=(x[-1] - 4.3, 0.74), fontsize=8.2, color="#6250d6",
            va="center", linespacing=1.45,
            arrowprops=dict(arrowstyle="-", color="#6250d6", lw=0.9))
bx.set_xticks(x)
bx.set_xticklabels(["donor"] + [str(i) for i in x[1:]], fontsize=8)
bx.set_xlabel("core-adaptation cycle", fontsize=9)
bx.set_ylabel("recall on 5, module removed", fontsize=9)
bx.set_ylim(-0.03, 1.04)
bx.set_title("how the third route gets there", fontsize=9.8, color=INK, loc="left")

for a_ in (ax, bx):
    a_.grid(True, axis="x" if a_ is ax else "both", color="#ececec", lw=0.6, zorder=0)
    a_.set_axisbelow(True)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
    a_.tick_params(labelsize=8.2)

fig.suptitle("Three ways to not recognize 5s", fontsize=12.6, x=0.115, ha="left", y=0.945)
fig.text(0.115, 0.845,
         "Every arm keeps the other nine digits intact (0.977-0.982), so none of this was bought "
         "by damaging the model.\nRecall of 0 is a floor any model reaches by never emitting the "
         "label — it is not evidence that the 5 is gone.",
         fontsize=8.6, color="#666666", linespacing=1.5)
out = f"{HERE}/figures/three_ways.png"
fig.savefig(out)
print(out)
