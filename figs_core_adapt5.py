"""Figures for core adaptation on MNIST: does the 5 leave the core, and what does it cost?

Reads results/core_adapt5_c12.json (written by core_adapt5.py).

Panel A is the whole claim in three lines. Panel B is the control: if the method were simply
damaging the model, every digit would fall, not one.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{HERE}/results/core_adapt5_c12.json"))
H = D["history"]
TARGET = str(D["target_digit"])
x = list(range(len(H)))

OFF5 = [h["ablate5"]["recall5"] for h in H]
OFF9 = [h["ablate5"]["recall_others"] for h in H]
ON5 = [h["full"]["recall5"] for h in H]

C_OFF5, C_OFF9, C_ON5, GREY, INK = "#eb6834", "#1baf7a", "#6250d6", "#9a9a92", "#333333"

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 4.9), dpi=200,
                             gridspec_kw={"width_ratios": [1.25, 1]})
fig.subplots_adjust(left=0.065, right=0.985, top=0.775, bottom=0.185, wspace=0.26)

# ---------------- panel A: the three lines
ax.plot(x, ON5, color=C_ON5, lw=2.2, marker="o", ms=4, label="5s, module ON (capability kept)")
ax.plot(x, OFF9, color=C_OFF9, lw=2.2, marker="o", ms=4, label="other nine digits, module OFF")
ax.plot(x, OFF5, color=C_OFF5, lw=2.6, marker="o", ms=4.6, label="5s, module OFF (what ships)")
ax.axvline(0, color=GREY, lw=1, ls=":", zorder=1)
ax.text(0.15, 0.40, "donor:\nknows 5", fontsize=7.4, color=GREY, va="center", linespacing=1.4)
ax.annotate(f"{OFF5[0]:.3f} → {OFF5[-1]:.3f}", (x[-1], OFF5[-1]), xytext=(x[-1] - 3.4, 0.175),
            fontsize=8.2, color=C_OFF5, va="center",
            arrowprops=dict(arrowstyle="-", color=C_OFF5, lw=0.9))
ax.set_xticks(x)
ax.set_xticklabels(["donor"] + [str(i) for i in x[1:]], fontsize=8)
ax.set_xlabel("core-adaptation cycle", fontsize=9)
ax.set_ylabel("recall on the MNIST test set", fontsize=9)
ax.set_ylim(0, 1.04)
ax.set_title("the 5 leaves the core, and only the 5", fontsize=10, color=INK, loc="left")
ax.legend(fontsize=8.2, frameon=False, loc="lower left", bbox_to_anchor=(0.012, 0.085))

# ---------------- panel B: per-digit, donor vs final, module OFF
d0 = H[0]["ablate5"]["recall_per_digit"]
dN = H[-1]["ablate5"]["recall_per_digit"]
digits = [str(i) for i in range(10)]
w = 0.38
bx.bar([i - w / 2 for i in range(10)], [d0[d] for d in digits], width=w,
       color="#c9ccd1", label="donor", zorder=3)
bx.bar([i + w / 2 for i in range(10)], [dN[d] for d in digits], width=w,
       color=[C_OFF5 if d == TARGET else "#2a78d6" for d in digits],
       label=f"after {len(H) - 1} cycles", zorder=3)
bx.annotate(f"−{100 * (d0[TARGET] - dN[TARGET]):.0f} pts", (int(TARGET) + w / 2, dN[TARGET]),
            xytext=(int(TARGET) + 1.35, 0.60), fontsize=8.6, color=C_OFF5, va="center",
            arrowprops=dict(arrowstyle="-", color=C_OFF5, lw=0.9))
bx.set_xticks(range(10))
bx.set_xticklabels(digits, fontsize=8.5)
bx.set_xlabel("digit", fontsize=9)
bx.set_ylabel("recall with the module removed", fontsize=9)
bx.set_ylim(0, 1.04)
bx.set_title("every other digit is untouched", fontsize=10, color=INK, loc="left")
# The bars leave no clear space inside the axes, so the legend goes underneath.
from matplotlib.patches import Patch
bx.legend(handles=[Patch(color="#c9ccd1", label="donor"),
                   Patch(color="#2a78d6", label=f"after {len(H) - 1} cycles"),
                   Patch(color=C_OFF5, label="the target digit")],
          ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.145),
          frameon=False, fontsize=8.4, handlelength=1.4, columnspacing=1.8)

for a_ in (ax, bx):
    a_.grid(True, color="#ececec", lw=0.6, zorder=0)
    a_.set_axisbelow(True)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
    a_.tick_params(labelsize=8.2)

fig.suptitle("Moving the 5 out of a model that already learned it",
             fontsize=12.4, x=0.065, ha="left", y=0.955)
fig.text(0.065, 0.862,
         "Hillary Sanders' core adaptation (AIAO-121 \"No-MLP C5\"), ported to MNIST. The donor "
         "is a trained classifier; each cycle teaches an\nauxiliary module to carry the 5 and "
         "pins the core to the teacher on the other nine with the module OFF.",
         fontsize=8.6, color="#666666", linespacing=1.5)
out = f"{HERE}/figures/core_adapt5.png"
fig.savefig(out)
print(out)
