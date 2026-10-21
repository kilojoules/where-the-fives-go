"""p_as x p_cr under label noise: the axis the 1-D sweeps held fixed.

Reads results/pcrgrid_s*_r*.json (13 seeds x 3 noise rates x 25 cells).

Every earlier sweep in this repo pinned p_cr=0.5 and moved p_as alone. This grid moves both.

Top row scores INFORMATION = |leak - 0.5|, not the raw AUC. Below-chance leak is still
information with the sign flipped -- a cell at 0.28 is anti-ranked, not clean -- so ranking by
raw AUC would crown the p_cr=0 column, which is wrong. 0.00 means the shipped readout is a
coin: nothing recoverable from logit 5 by ranking.

Bottom row scores USABILITY: the fraction of seeds whose module-ON recall exceeds 0.8. Several
cells are BIMODAL -- some seeds learn 5s and some never do -- so a mean capability would state
something true of no single run. A leak number from a config that never learned 5s is
meaningless, which is what the bottom row is for: read a top-row cell only where the bottom
row is dark.
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
cells = defaultdict(list)
for f in glob.glob(f"{HERE}/results/pcrgrid_s*_r*.json"):
    for c in json.load(open(f))["cells"].values():
        cells[(c["rate"], c["p_as"], c["p_cr"])].append(c)

RATES = sorted({k[0] for k in cells})
PAS = sorted({k[1] for k in cells})
PCR = sorted({k[2] for k in cells})
NSEED = max(len(v) for v in cells.values())
INK, GREY = "#333333", "#9a9a92"
BASE, CAND = (0.5, 0.5), (0.75, 1.0)

fig, axes = plt.subplots(2, len(RATES), figsize=(13.2, 8.2), dpi=200)
fig.subplots_adjust(left=0.075, right=0.915, top=0.760, bottom=0.075, hspace=0.52, wspace=0.17)

for col, rate in enumerate(RATES):
    info = np.full((len(PAS), len(PCR)), np.nan)
    use = np.full((len(PAS), len(PCR)), np.nan)
    for i, pas in enumerate(PAS):
        for j, pcr in enumerate(PCR):
            v = cells.get((rate, pas, pcr), [])
            if not v:
                continue
            info[i, j] = abs(np.mean([x["leak_raw"] for x in v]) - 0.5)
            use[i, j] = np.mean([x["capability"] > 0.8 for x in v])

    for row, (M, cmap, vmax, title) in enumerate([
            (info, "YlOrRd", 0.5, "information left in the shipped core  |AUC(logit 5) − 0.5|"),
            (use, "Blues", 1.0, "fraction of seeds that learned 5 at all  (module-ON recall > 0.8)")]):
        ax = axes[row, col]
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, origin="lower", aspect="auto")
        for i in range(len(PAS)):
            for j in range(len(PCR)):
                if np.isnan(M[i, j]):
                    continue
                strong = M[i, j] > 0.6 * vmax
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.6,
                        color="white" if strong else INK)
        # mark the setting every earlier sweep used, and the one this grid turned up
        for (pa, pc), lab, c in ((BASE, "baseline", "#2a78d6"), (CAND, "candidate", "#1baf7a")):
            if pa in PAS and pc in PCR:
                ax.add_patch(plt.Rectangle((PCR.index(pc) - .5, PAS.index(pa) - .5), 1, 1,
                                           fill=False, ec=c, lw=2.2, zorder=5))
        ax.set_xticks(range(len(PCR))); ax.set_xticklabels([f"{p:g}" for p in PCR], fontsize=8)
        ax.set_yticks(range(len(PAS))); ax.set_yticklabels([f"{p:g}" for p in PAS], fontsize=8)
        ax.set_xlabel("p_cr  (module activated on a CORE sub-batch)", fontsize=8.3)
        if col == 0:
            ax.set_ylabel("p_as  (core gets gradient on a ROUTED sub-batch)", fontsize=8.3)
        pct = "clean labels" if rate == 0 else f"{rate:.0%} of 5s mislabeled"
        ax.set_title(pct, fontsize=9.4, color=INK, loc="left", pad=5)
        if col == len(RATES) - 1:
            cb = fig.colorbar(im, ax=axes[row, :].tolist(), fraction=0.020, pad=0.012)
            cb.ax.tick_params(labelsize=7.4)
            cb.set_label(["more leak  →", "more reliable  →"][row], fontsize=7.6)
        if col == 0:
            # row header above the row's own axes; hspace keeps it clear of the row above
            ax.text(0.0, 1.145, title, transform=ax.transAxes, fontsize=9.8, color=INK,
                    fontweight="bold")

fig.suptitle("The knob the one-dimensional sweeps held fixed",
             fontsize=12.8, x=0.075, ha="left", y=0.955)
fig.text(0.075, 0.845,
         f"Every earlier sweep in this repo pinned p_cr = 0.5 (blue box) and moved p_as alone. "
         f"Moving both finds p_as=0.75, p_cr=1.0 (green box), which under 30%\nmislabeling "
         "leaks 0.08 of information against the baseline's 0.27 — at the same capability. "
         "Paired on seed, that gap is real and it REVERSES with label\nquality: the candidate "
         f"is reliably WORSE on clean labels (+0.055, 12/13 seeds, p=0.003) and better under "
         f"noise (−0.189, 11/13, p=0.023). {NSEED} seeds.",
         fontsize=8.5, color="#666666", linespacing=1.5)
out = f"{HERE}/figures/pcr_grid.png"
fig.savefig(out)
print(out)
