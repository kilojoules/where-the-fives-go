"""Summary: label noise and the three removal routes, 8 seeds.

Panel A -- the comparison. All three routes decide what to remove by reading the LABEL
(train.py:108, train.py:120, core_adapt5.py:166), so a true 5 wearing another digit's label
is invisible to all three identically. What differs is the gradient each then applies to it.
Paired on seed, because seed spread reaches +-0.16 and swamps several of these effects.

Panel B -- the GRAM knob. p_as (train.py:188) is the probability the core receives gradient on
a routed-5 sub-batch: the only channel that teaches the core "5-pixels -> 5". It buys
capability and costs leak. False negatives are never routed, so they reach the core at every
p_as and shift the whole frontier.

Reads results/absorb3_seed{0..7}.json and results/pas_absorb_seed{0..7}.json.
Both axes are scored on RAW LOGIT 5, not softmax P(5) -- see INSTRUMENT.md, where the two
disagree in sign on the filtered arm.
"""
import json
import os
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [s for s in range(8) if os.path.exists(f"{HERE}/results/absorb3_seed{s}.json")]
A = {s: json.load(open(f"{HERE}/results/absorb3_seed{s}.json")) for s in SEEDS}
P = {s: json.load(open(f"{HERE}/results/pas_absorb_seed{s}.json")) for s in SEEDS}
R0, R1 = "0.0", "0.3"

ROUTES = [
    ("dense_all",   "dense\nnothing removed",     "#2a78d6"),
    ("filter_no5",  "filtered\nnever in data",    "#eb6834"),
    ("gram_main",   "GRAM\nrouted away",          "#1baf7a"),
    ("core_adapt5", "core adaptation\nmoved out", "#6250d6"),
]
PAS = [(0.0, "#cdd3f0"), (0.25, "#9aa3e0"), (0.5, "#6250d6"), (1.0, "#2e2191")]
GREY, INK = "#9a9a92", "#333333"


def sign_p(d):
    d = [x for x in d if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    m = max(k, n - k)
    return n, k, min(1.0, 2 * sum(comb(n, i) for i in range(m, n + 1)) / 2 ** n)


fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.0, 5.5), dpi=200,
                             gridspec_kw={"width_ratios": [1.05, 1]})
fig.subplots_adjust(left=0.058, right=0.986, top=0.725, bottom=0.145, wspace=0.225)

# ---------------- Panel A: the four routes, paired
for i, (key, label, colour) in enumerate(ROUTES):
    a = [A[s]["arms"][key][R0]["auc_clean_rawlogit"] for s in SEEDS]
    b = [A[s]["arms"][key][R1]["auc_clean_rawlogit"] for s in SEEDS]
    x0, x1 = i - 0.16, i + 0.16
    for u, v in zip(a, b):
        ax.plot([x0, x1], [u, v], color=colour, lw=0.8, alpha=0.38, zorder=2)
    ax.plot([x0, x1], [np.mean(a), np.mean(b)], color=colour, lw=2.9, zorder=4,
            solid_capstyle="round")
    ax.plot([x0, x1], [np.mean(a), np.mean(b)], "o", color=colour, ms=5.5, zorder=5)
    d = np.mean(b) - np.mean(a)
    ax.annotate(f"{d:+.3f}", (x1 + 0.05, np.mean(b)), fontsize=8.4, color=colour,
                va="center", fontweight="bold")

ax.axhline(0.5, color=GREY, lw=1.1, ls="--", zorder=1)
ax.text(-0.42, 0.512, "chance", fontsize=7.4, color=GREY, va="bottom")
ax.set_xticks(range(len(ROUTES)))
ax.set_xticklabels([r[1] for r in ROUTES], fontsize=8.2, linespacing=1.4)
ax.set_xlim(-0.45, len(ROUTES) - 0.52)
ax.set_ylim(0.24, 1.04)
ax.set_ylabel("AUC over raw logit 5, held-out 5s, shipped readout", fontsize=8.9)
ax.set_title("Only one route re-learns what it absorbed", fontsize=10.2, color=INK, loc="left")
ax.text(-0.42, 0.275, "left dot: clean labels      right dot: 30% of 5s mislabeled\n"
                      "core adaptation rises against each other route 8/8 seeds, p=0.008",
        fontsize=7.6, color=GREY, linespacing=1.5)

# ---------------- Panel B: the p_as frontier
for pas, colour in PAS:
    xs, ys, xe, ye = [], [], [], []
    for r in (R0, R1):
        leak = [P[s]["cells"][f"r{float(r)}_pas{pas}"]["auc_ship_rawlogit"] for s in SEEDS]
        cap = [P[s]["cells"][f"r{float(r)}_pas{pas}"]["module_on_recall5"] for s in SEEDS]
        xs.append(np.mean(leak)); ys.append(np.mean(cap))
        xe.append(np.std(leak)); ye.append(np.std(cap))
    bx.annotate("", xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.9,
                                shrinkA=5.5, shrinkB=6.5, alpha=0.85), zorder=3)
    for j, mk in enumerate(("o", "s")):
        bx.errorbar(xs[j], ys[j], xerr=xe[j], yerr=ye[j], fmt=mk, color=colour, ms=7.5,
                    elinewidth=1.0, capsize=2.5, alpha=0.95, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.8)
    bx.annotate(f"p_as={pas:g}", (xs[0], ys[0]), xytext=(-8, -15),
                textcoords="offset points", fontsize=8.0, color=colour, ha="right")

bx.set_xlabel("LEAK  —  AUC over raw logit 5 with module 5 ablated  →  worse removal",
              fontsize=8.6)
bx.set_ylabel("CAPABILITY  —  recall on 5s with the module attached", fontsize=8.9)
bx.set_title("GRAM's p_as buys capability and costs leak", fontsize=10.2, color=INK, loc="left")
bx.set_xlim(0.38, 1.03)
bx.set_ylim(-0.06, 1.10)
bx.plot([], [], "o", color=GREY, ms=7, label="clean labels")
bx.plot([], [], "s", color=GREY, ms=7, label="30% of 5s mislabeled")
bx.legend(fontsize=8.0, frameon=False, loc="lower right", handletextpad=0.4)

for a_ in (ax, bx):
    a_.grid(True, color="#ececec", lw=0.6, zorder=0)
    a_.set_axisbelow(True)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
    a_.tick_params(labelsize=8.1)
ax.tick_params(axis="x", length=0)

fig.suptitle("Mislabeled 5s: what each removal route does with the examples it cannot see",
             fontsize=12.6, x=0.058, ha="left", y=0.950)
fig.text(0.058, 0.838,
         "Every route reads the label, so the same 1,626 of 5,421 training 5s slip past all of "
         "them. Filtering and GRAM then apply a wrong-label gradient and\ntheir logit 5 falls; "
         "core adaptation pins its shipped core to a teacher that still knows 5s "
         "(core_adapt5.py:246), which a wrong label does not fool, and\nits logit 5 rises. "
         "8 seeds, paired within seed. Scored on raw logit 5: softmax P(5) moves the opposite "
         "way on the filtered arm — see INSTRUMENT.md.",
         fontsize=8.5, color="#666666", linespacing=1.5)
fig.text(0.058, 0.035,
         "At 30% mislabeling every p_as has ablated RECALL of exactly 0.0000, including "
         "p_as=1.0 which leaks at 0.83 — recall is a floor any model reaches by never emitting "
         "the label, which is why neither axis uses it.",
         fontsize=7.6, color=GREY)
out = f"{HERE}/figures/absorb_summary.png"
fig.savefig(out)
print(out, f"({len(SEEDS)} seeds)")
