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

# --- hyperparameter selection, honestly: choose on one half of the seeds, report on the
# other. With 25 cells the best-looking one is biased low by construction, so a cell picked
# and quoted on the same seeds is a winner's curse, not a result.
byseed = defaultdict(dict)
for f in glob.glob(f"{HERE}/results/pcrgrid_s*_r*.json"):
    D = json.load(open(f))
    for c in D["cells"].values():
        byseed[D["seed"]][(c["rate"], c["p_as"], c["p_cr"])] = c
ALLSEEDS = sorted(byseed)
SEL, EVAL = ALLSEEDS[:len(ALLSEEDS) // 2], ALLSEEDS[len(ALLSEEDS) // 2:]


def agg(ss, key):
    v = [byseed[s][key] for s in ss if key in byseed[s]]
    if not v:
        return None
    lk = [x["leak_raw"] for x in v]
    return dict(info=abs(np.mean(lk) - 0.5), leak=np.mean(lk), sd=np.std(lk),
                use=np.mean([x["capability"] > 0.8 for x in v]),
                cap=np.mean([x["capability"] for x in v]),
                ao=np.mean([x["acc_others"] for x in v]))


RATES = sorted({k[0] for k in cells})
PAS = sorted({k[1] for k in cells})
PCR = sorted({k[2] for k in cells})
NSEED = max(len(v) for v in cells.values())
INK, GREY = "#333333", "#9a9a92"
BASE = (0.5, 0.5)

# RULE: least information, subject to every SELECTION seed learning 5 and the other nine
# digits held >= 0.95. Applied to a selection half only; the number quoted is from the other
# half. One arbitrary split is itself noisy, so we repeat over 200 random half-splits and mark
# the MODAL pick, reporting how often it wins. A pick that only survives one split is not a
# hyperparameter choice, it is a coin landing.
import random as _rnd
from collections import Counter as _Counter
_rng = _rnd.Random(0)
PICK, HELD, FREQ = {}, {}, {}
_PAS = sorted({k[1] for k in cells}); _PCR = sorted({k[2] for k in cells})


def _pick(ss, rate):
    c = []
    for pa in _PAS:
        for pc in _PCR:
            a = agg(ss, (rate, pa, pc))
            if a and a["use"] >= 1.0 and a["ao"] >= 0.95:
                c.append((a["info"], pa, pc))
    return sorted(c)[0][1:] if c else None


for _r in RATES:
    cnt, held = _Counter(), defaultdict(list)
    for _ in range(200):
        sh = ALLSEEDS[:]; _rng.shuffle(sh)
        sel, ev = sh[:len(sh) // 2 + 1], sh[len(sh) // 2 + 1:]
        p = _pick(sel, _r)
        if p:
            cnt[p] += 1
            h = agg(ev, (_r,) + p)
            if h:
                held[p].append(h["info"])
    if cnt:
        p, n = cnt.most_common(1)[0]
        PICK[_r], FREQ[_r] = p, n / sum(cnt.values())
        HELD[_r] = dict(info=float(np.mean(held[p])))

fig, axes = plt.subplots(2, len(RATES), figsize=(13.2, 9.8), dpi=200)
fig.subplots_adjust(left=0.075, right=0.915, top=0.715, bottom=0.065, hspace=0.52, wspace=0.17)

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
        if BASE[0] in PAS and BASE[1] in PCR:
            ax.add_patch(plt.Rectangle((PCR.index(BASE[1]) - .5, PAS.index(BASE[0]) - .5), 1, 1,
                                       fill=False, ec="#2a78d6", lw=2.0, ls=(0, (3, 2)), zorder=5))
        if rate in PICK:
            _pa, _pc = PICK[rate]
            ax.add_patch(plt.Rectangle((PCR.index(_pc) - .5, PAS.index(_pa) - .5), 1, 1,
                                       fill=False, ec="#111111", lw=2.8, zorder=6))
            ax.plot(PCR.index(_pc) + 0.33, PAS.index(_pa) + 0.33, marker="*", ms=9,
                    color="#111111", zorder=7)
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

hl = "\n".join(
    f"  {r:.0%} mislabeled  →  p_as={PICK[r][0]:<4g} p_cr={PICK[r][1]:<4g}   "
    f"held-out information {HELD[r]['info']:.3f} "
    f"(baseline {agg(ALLSEEDS, (r, 0.5, 0.5))['info']:.3f})   "
    f"modal pick in {FREQ[r]:.0%} of splits" for r in RATES if r in PICK)
fig.suptitle("Choosing p_as and p_cr for each noise level",
             fontsize=12.8, x=0.075, ha="left", y=0.972)
fig.text(0.075, 0.935,
         "★ is the setting chosen at each noise level. With 25 cells the best-looking one is "
         "biased low by construction, so a cell chosen and quoted on the same seeds is a "
         "winner's curse.\nEach pick is made on half the seeds and scored on the other half, "
         "over 200 random half-splits; ★ is the modal pick, scored on held-out seeds. Rule: "
         "least information,\nsubject to every selection seed learning 5 and the other nine "
         "digits above 0.95. Dashed blue is p_cr=0.5, which every earlier 1-D sweep here "
         "pinned.",
         fontsize=8.4, color="#666666", linespacing=1.5, va="top")
fig.text(0.075, 0.878, hl, fontsize=8.4, color=INK, linespacing=1.55, va="top",
         family="DejaVu Sans Mono")
fig.text(0.075, 0.818,
         "At CLEAN labels tuning buys nothing — 0.311 against the baseline's 0.325 — so the "
         f"whole gain is a noise effect. p_as is the stable axis (0.5 clean, 0.75 under noise, "
         f"in\n75–100% of splits); p_cr is the wobbly one. {NSEED} seeds total.",
         fontsize=8.4, color="#666666", linespacing=1.5, va="top")
out = f"{HERE}/figures/pcr_grid.png"
fig.savefig(out)
print(out)
