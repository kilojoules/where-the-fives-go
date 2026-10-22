"""Does core adaptation's leak come from the teacher? The causal test.

Reads results/rtarget_s*.json (5 seeds x 3 rates x 3 variants).

core_adapt5.py's retain-OFF term pins the SHIPPED core to the donor teacher on retain data
(core_adapt5.py:246). A mislabeled 5 sits in that retain set, because the split reads the
label -- and the teacher reads the image correctly whatever the label says. The proposal was
that this one term is why core adaptation RE-LEARNS 5s under mislabeling while every other
removal route loses them.

Three variants change only that term, on the same donor, seed and noise realisation:
  teacher        KL to the donor over all ten logits -- the recipe as written
  teacher_mask5  KL over the nine NON-target logits only -- the term's actual job
  label          cross-entropy on the (noisy) retain labels -- what the other routes do

Thin lines are seeds. Read the middle and right panels together with the left one: a variant
that lowers the leak by destroying the capability or the retained digits has not fixed
anything.
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
bys = defaultdict(dict)
for f in glob.glob(f"{HERE}/results/rtarget_s*.json"):
    D = json.load(open(f))
    for c in D["cells"].values():
        bys[D["seed"]][(c["rate"], c["variant"])] = c
SEEDS = sorted(bys)
RATES = [0.0, 0.1, 0.3]
VARS = [("teacher", "teacher, all ten logits\n(the recipe)", "#6250d6"),
        ("teacher_mask5", "teacher, logit 5 masked out", "#1baf7a"),
        ("label", "noisy labels\n(what the other routes do)", "#eb6834")]
GREY, INK = "#9a9a92", "#333333"

fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.9), dpi=200)
fig.subplots_adjust(left=0.058, right=0.988, top=0.705, bottom=0.165, wspace=0.235)

PANELS = [("leak_raw", "LEAK — AUC over logit 5, shipped core", (0.15, 1.04), True),
          ("capability", "CAPABILITY — recall on 5s, module attached", (0.0, 1.04), False),
          ("acc_others", "RETENTION — accuracy on the other nine", (0.0, 1.04), False)]

for ax, (key, title, ylim, chance) in zip(axes, PANELS):
    for vk, vlabel, colour in VARS:
        for s in SEEDS:
            ys = [bys[s][(r, vk)][key] for r in RATES if (r, vk) in bys[s]]
            ax.plot(RATES[:len(ys)], ys, color=colour, lw=0.8, alpha=0.33, zorder=2)
        m = [np.mean([bys[s][(r, vk)][key] for s in SEEDS]) for r in RATES]
        ax.plot(RATES, m, color=colour, lw=2.6, marker="o", ms=5.5, zorder=4, label=vlabel)
    if chance:
        ax.axhline(0.5, color=GREY, lw=1.1, ls="--", zorder=1)
        ax.text(0.002, 0.512, "chance", fontsize=7.4, color=GREY, va="bottom")
    ax.set_xticks(RATES)
    ax.set_xticklabels(["clean", "10%", "30%"], fontsize=8.4)
    ax.set_xlabel("fraction of true 5s given another digit's label", fontsize=8.6)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.016, 0.316)
    ax.set_title(title, fontsize=9.4, color=INK, loc="left")
    ax.grid(True, color="#ececec", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8.1)

axes[0].legend(fontsize=7.8, frameon=False, loc="lower right", labelspacing=0.9,
               handlelength=1.6)
# the one seed where masking the target logit destroyed the retained digits
bad = [(s, c) for s in SEEDS for (r, v), c in bys[s].items()
       if v == "teacher_mask5" and c["acc_others"] < 0.9]
if bad:
    s, c = bad[0]
    axes[2].annotate(f"seed {s}: {c['acc_others']:.2f}\nmasking broke this run",
                     (0.0, c["acc_others"]), xytext=(0.075, 0.33), fontsize=7.8,
                     color="#1baf7a", va="center",
                     arrowprops=dict(arrowstyle="->", color="#1baf7a", lw=1.0))

fig.suptitle("Is the teacher the reason core adaptation re-learns what it absorbed?",
             fontsize=12.6, x=0.058, ha="left", y=0.950)
fig.text(0.058, 0.805,
         "Partly. Changing the retain-OFF term lowers the leak in 5 of 5 seeds (mask5 −0.215, "
         "label −0.258 against teacher at 30%, p=0.063 — the floor at n=5).\nBut masking logit "
         "5 does NOT remove the rise with noise (0.52 → 0.73), so the teacher's target logit "
         "is not the whole path. And neither alternative is\nfree: both roughly halve the "
         "capability by 30% mislabeling, and masking destroyed the retained digits outright in "
         "one seed. 5 seeds.",
         fontsize=8.5, color="#666666", linespacing=1.5)
out = f"{HERE}/figures/rtarget.png"
fig.savefig(out)
print(out)
