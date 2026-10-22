"""Core adaptation's two knobs as a grid — the counterpart to figures/pcr_grid.png.

Reads results/cag_s*.json (5 retain_off_weight x 3 core_lr x 2 noise levels x 5 seeds).

Same layout as the GRAM grid so the two can be read side by side, with one deliberate
difference in the bottom row. For GRAM the question was whether the module ever learned 5 --
a cell with a low readout because the digit was never acquired is not a removal result. Here
the capability is COPIED from a donor that already has it, so that failure cannot happen;
what breaks instead is RETENTION of the other nine digits. At retain_off_weight=0 the pinning
term is gone and retention collapses, and those cells post the lowest readout in the grid
from models that lost up to 45% of the remaining digits.

Hatched cells on the top row are the ones whose retention fell below 0.95: their low readout
means the model was damaged, not that removal worked.
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
RET_BAR = 0.95

byseed = defaultdict(dict)
for f in glob.glob(f"{HERE}/results/cag_s*.json"):
    try:
        D = json.load(open(f))
    except Exception:
        continue
    for c in D.get("cells", {}).values():
        byseed[D["seed"]][(c["rate"], c["retain_off_weight"], c["core_lr"])] = c
SEEDS = sorted(s for s in byseed if len(byseed[s]) >= 30)
keys = sorted({k for s in SEEDS for k in byseed[s]})
RATES = sorted({k[0] for k in keys})
W = sorted({k[1] for k in keys})
LR = sorted({k[2] for k in keys})
INK, GREY = L.C["ink"], L.C["grey"]
BASE = (2.0, 1e-5)        # the recipe's default, for the same dashed box pcr_grid draws

# Selection protocol copied from figs_pcr_grid so the two figures mean the same thing:
# lowest readout above chance subject to retention >= 0.95, chosen on half the seeds and
# scored on the other half, over 200 random half-splits; the star is the modal pick.
import random as _rnd
from collections import Counter as _Counter


def _agg(ss, key):
    v = [byseed[s][key] for s in ss if key in byseed[s]]
    if not v:
        return None
    lk = [x["leak_raw"] for x in v]
    return dict(info=abs(np.mean(lk) - 0.5),
                ret=float(np.mean([x["acc_others"] for x in v])))


_rng = _rnd.Random(0)
PICK, HELD, FREQ = {}, {}, {}
for _r in RATES:
    cnt, held = _Counter(), defaultdict(list)
    for _ in range(200):
        sh = SEEDS[:]
        _rng.shuffle(sh)
        sel, ev = sh[:len(sh) // 2 + 1], sh[len(sh) // 2 + 1:]
        cands = []
        for _w in W:
            for _l in LR:
                a = _agg(sel, (_r, _w, _l))
                if a and a["ret"] >= RET_BAR:
                    cands.append((a["info"], _w, _l))
        if not cands:
            continue
        _, _w, _l = sorted(cands)[0]
        cnt[(_w, _l)] += 1
        h = _agg(ev, (_r, _w, _l))
        if h:
            held[(_w, _l)].append(h["info"])
    if cnt:
        p, n = cnt.most_common(1)[0]
        PICK[_r], FREQ[_r] = p, n / sum(cnt.values())
        HELD[_r] = float(np.mean(held[p])) if held[p] else float("nan")

fig, axes = plt.subplots(2, len(RATES), figsize=(9.6, 9.4), dpi=200, squeeze=False)
fig.subplots_adjust(left=0.105, right=0.885, top=0.635, bottom=0.062,
                    hspace=0.52, wspace=0.16)

for col, rate in enumerate(RATES):
    read = np.full((len(W), len(LR)), np.nan)
    ret = np.full((len(W), len(LR)), np.nan)
    for i, w in enumerate(W):
        for j, lr in enumerate(LR):
            v = [byseed[s][(rate, w, lr)] for s in SEEDS if (rate, w, lr) in byseed[s]]
            if not v:
                continue
            read[i, j] = abs(np.mean([x["leak_raw"] for x in v]) - 0.5)
            ret[i, j] = np.mean([x["acc_others"] for x in v])

    for row, (M, cmap, vmax, title) in enumerate([
            (read, "YlOrRd", 0.5, "READOUT above chance  |AUC(logit 5) − 0.50|"),
            (ret, "Blues", 1.0, "RETENTION — accuracy on the other nine digits")]):
        ax = axes[row][col]
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, origin="lower", aspect="auto")
        for i in range(len(W)):
            for j in range(len(LR)):
                if np.isnan(M[i, j]):
                    continue
                dead = row == 0 and ret[i, j] < RET_BAR
                if dead:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=True,
                                               facecolor="white", alpha=0.62, lw=0, zorder=3))
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               hatch="////", ec="#b0b0a8", lw=0.0, zorder=4))
                strong = M[i, j] > 0.6 * vmax
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.6,
                        zorder=5,
                        color=("#8a8a82" if dead else ("white" if strong else INK)))
        if BASE[0] in W and BASE[1] in LR:
            ax.add_patch(plt.Rectangle((LR.index(BASE[1]) - .5, W.index(BASE[0]) - .5), 1, 1,
                                       fill=False, ec="#2a78d6", lw=2.0, ls=(0, (3, 2)),
                                       zorder=5))
        if rate in PICK:
            _w, _l = PICK[rate]
            ax.add_patch(plt.Rectangle((LR.index(_l) - .5, W.index(_w) - .5), 1, 1,
                                       fill=False, ec="#111111", lw=2.8, zorder=6))
            ax.plot(LR.index(_l) + 0.33, W.index(_w) + 0.33, marker="*", ms=9,
                    color="#111111", zorder=7)
        ax.set_xticks(range(len(LR)))
        ax.set_xticklabels([f"{l:g}" for l in LR], fontsize=8.4)
        ax.set_yticks(range(len(W)))
        ax.set_yticklabels([f"{w:g}" for w in W], fontsize=8.4)
        ax.set_xlabel("core_lr  (how far the core may move)", fontsize=8.5)
        if col == 0:
            ax.set_ylabel("retain_off_weight\n(pins the shipped core to the teacher)",
                          fontsize=8.5, linespacing=1.4)
            ax.text(0.0, 1.145, title, transform=ax.transAxes, fontsize=9.8, color=INK,
                    fontweight="bold")
        ax.set_title("clean labels" if rate == 0 else f"{rate:.0%} of 5s mislabeled",
                     fontsize=9.4, color=INK, loc="left", pad=5)
        if col == len(RATES) - 1:
            cb = fig.colorbar(im, ax=[a[col] for a in [axes[row]]], fraction=0.045, pad=0.03)
            cb.ax.tick_params(labelsize=7.4)
            cb.set_label(["more left behind  →", "more intact  →"][row], fontsize=7.6)

hl = "\n".join(
    f"  {r:.0%} mislabeled → weight={PICK[r][0]:g}, lr={PICK[r][1]:g}   "
    f"held-out readout above chance {HELD[r]:.3f}   modal pick in {FREQ[r]:.0%} of splits"
    for r in RATES if r in PICK)
fig.suptitle("Core adaptation's two knobs", fontsize=13.2, x=0.105, ha="left", y=0.972)
fig.text(0.105, 0.938,
         "The counterpart to figures/pcr_grid.png, same layout so the two read side by side.\n"
         "retain_off_weight is the term pinning the shipped core to the donor teacher "
         "(core_adapt5.py:246) — the leak path itself.\n\n"
         "The bottom row differs from the GRAM grid on purpose. There the question was "
         "whether the module ever learned 5; here the\ncapability is copied from a donor "
         "that already has it, so what breaks instead is RETENTION of the other nine digits.\n"
         "HATCHED cells had retention below 0.95: their low readout means the model was "
         f"damaged, not that removal worked.\n"
         "★ is the setting chosen at each noise level, picked on half the seeds and scored "
         "on the other half over 200 random half-splits\n(same protocol as pcr_grid); "
         f"dashed blue is the recipe default. {len(SEEDS)} seeds.",
         fontsize=8.6, color="#666666", linespacing=1.5, va="top")
fig.text(0.105, 0.712, hl, fontsize=8.4, color=INK, linespacing=1.5, va="top",
         family="DejaVu Sans Mono")
out = f"{HERE}/figures/ca_grid.png"
fig.savefig(out)
print(out, f"({len(SEEDS)} seeds, {len(W)}x{len(LR)} grid, rates {RATES})")
