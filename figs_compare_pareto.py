"""One front per method per noise level: GRAM against core adaptation.

Both methods are scored identically -- readout left above chance in the SHIPPED model, against the
capability it still has when the module is attached:

  x  READOUT ABOVE CHANCE = |AUC(raw logit 5) - 0.5| on held-out 5s, shipped configuration.
                   0 means the shipped readout orders 5s no better than a coin.
                   Not softmax P(5): the two disagree in sign under label noise (INSTRUMENT.md).
  y  CAPABILITY  = recall on 5s with the module attached.

What differs is each method's own knobs and its own failure mode, so each gets the filter that
matches what actually breaks for it:

  GRAM             p_as x p_cr, 25 settings. Drops settings where fewer than 80% of seeds
                   learned 5 at all -- their low readout means the model never acquired the digit.
  core adaptation  retain_off_weight x core_lr, 15 settings. The capability is COPIED from a
                   donor that already has it, so instead retention breaks: drops settings whose
                   other-nine accuracy fell below 0.95.

Only the best trade-offs are drawn: a setting appears if nothing beat it on BOTH axes. A dominated one is worse on both axes, so nobody should
pick it, and plotting it only adds ink.
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
GRAM_C, CA_C, INK, GREY = "#1baf7a", "#6250d6", "#333333", "#9a9a92"


def front(pts):
    out = [p for p in pts if not any(
        q["info"] <= p["info"] and q["cap"] >= p["cap"]
        and (q["info"] < p["info"] or q["cap"] > p["cap"]) for q in pts)]
    return sorted(out, key=lambda d: d["info"])


# ---- GRAM: p_as x p_cr
g = defaultdict(list)
for f in glob.glob(f"{HERE}/results/pcrgrid_s*_r*.json"):
    for c in json.load(open(f))["cells"].values():
        g[(c["rate"], c["p_as"], c["p_cr"])].append(c)
GRAM = {}
for rate in (0.0, 0.3):
    pts = []
    for (r, a, b), v in g.items():
        if r != rate:
            continue
        cap = [x["capability"] for x in v]
        if np.mean([c > 0.8 for c in cap]) < 0.8:      # never learned 5 -> not a removal result
            continue
        lk = [x["leak_raw"] for x in v]
        pts.append(dict(info=abs(np.mean(lk) - 0.5), cap=float(np.mean(cap)),
                        lab=f"p_as={a:g}/p_cr={b:g}"))
    GRAM[rate] = front(pts)

# ---- core adaptation: retain_off_weight x core_lr
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
CA = {}
for rate in (0.0, 0.3):
    pts = []
    for (r, w, lr) in keys:
        if r != rate:
            continue
        v = [byseed[s][(r, w, lr)] for s in SEEDS if (r, w, lr) in byseed[s]]
        if not v or np.mean([x["acc_others"] for x in v]) < 0.95:   # lost the other nine
            continue
        lk = [x["leak_raw"] for x in v]
        pts.append(dict(info=abs(np.mean(lk) - 0.5),
                        cap=float(np.mean([x["capability"] for x in v])),
                        lab=f"w={w:g}/lr={lr:g}"))
    CA[rate] = front(pts)

fig, ax = plt.subplots(figsize=(11.4, 8.8), dpi=200)
fig.subplots_adjust(left=0.080, right=0.980, top=0.660, bottom=0.085)

_ALL_LABELS = []   # every drawn point gets its hyperparameters
SERIES = [
    (GRAM[0.0], GRAM_C, "o", "none", "--", "GRAM — clean labels"),
    (GRAM[0.3], GRAM_C, "o", GRAM_C, "-",  "GRAM — 30% mislabeled"),
    (CA[0.0],   CA_C,   "s", "none", "--", "core adaptation — clean labels"),
    (CA[0.3],   CA_C,   "s", CA_C,   "-",  "core adaptation — 30% mislabeled"),
]
for F, colour, mk, fill, ls, label in SERIES:
    if not F:
        continue
    ax.plot([p["info"] for p in F], [p["cap"] for p in F], color=colour, lw=2.0, ls=ls,
            alpha=0.85, zorder=3, drawstyle="steps-post")
    ax.scatter([p["info"] for p in F], [p["cap"] for p in F], marker=mk, s=74,
               facecolors=fill, edgecolors=colour, linewidths=1.8, zorder=5, label=label)
    # Label EVERY drawn point. Labelling only the knee reads as though the other points
    # have no hyperparameters, which confused a reader of an earlier version.
    for j, p in enumerate(F):
        _ALL_LABELS.append((p["info"], p["cap"], p["lab"], colour))

# The single most important thing to see: the two methods move in OPPOSITE directions when
# labels go wrong. Draw that movement rather than leaving it to be inferred from line style.
for Fc, Fn, colour, tag in ((GRAM[0.0], GRAM[0.3], GRAM_C, "GRAM"),
                            (CA[0.0], CA[0.3], CA_C, "core adaptation")):
    x0, x1 = Fc[0]["info"], Fn[0]["info"]
    y = 0.995 if tag == "GRAM" else 0.975
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=3.0, alpha=0.5,
                                shrinkA=2, shrinkB=2))
    better = "better" if x1 < x0 else "WORSE"
    ax.text((x0 + x1) / 2, y + 0.004, f"{tag}: {better} under noise",
            fontsize=8.6, color=colour, ha="center", va="bottom", fontweight="bold")

# the reference that makes the x-axis mean something
ax.axvline(0.4996 - 0.5 + 0.4996, color=GREY, lw=0)   # no-op, keeps layout stable
ax.axvspan(0.47, 0.52, color="#f3f3f1", zorder=0)
ax.text(0.495, 0.748, "a model that\nremoved nothing", fontsize=7.6, color=GREY,
        ha="center", va="bottom", linespacing=1.4)

# place every label, nudging vertically where points nearly coincide
_ALL_LABELS.sort()
for i, (xx, yy, lab, col) in enumerate(_ALL_LABELS):
    n = sum(1 for (qx, qy, _, _) in _ALL_LABELS[:i]
            if abs(qx - xx) < 0.045 and abs(qy - yy) < 0.030)
    ax.annotate(lab, (xx, yy), xytext=(((-1) ** n) * 6 * n, 10 + 15 * n),
                textcoords="offset points",
                fontsize=7.3, color=col, ha="center",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.6, alpha=0.45))

ax.set_xlabel(L.READOUT_FOLDED, fontsize=9.4)
ax.set_ylabel(L.CAPABILITY_DIR,
              fontsize=9.4)
ax.set_xlim(0.015, 0.545)
ax.set_ylim(0.735, 1.025)
ax.grid(True, color="#ececec", lw=0.6, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=8.4)
ax.legend(fontsize=8.6, frameon=False, loc="lower center", handletextpad=0.7, labelspacing=0.75)

gc, gn = GRAM[0.0][0]["info"], GRAM[0.3][0]["info"]
cc, cn = CA[0.0][0]["info"], CA[0.3][0]["info"]
fig.suptitle("Label noise moves the two methods in opposite directions",
             fontsize=12.8, x=0.080, ha="left", y=0.978)
fig.text(0.080, 0.948,
         "Dashed/hollow = clean labels.   Solid/filled = 30% of 5s wearing another digit's label.\n"
         "Every point is one hyperparameter setting, labelled with its values. Only the best trade-offs are shown —\n"
         "a setting appears if nothing beat it on BOTH axes at once.\n"
         "Each method is filtered by what actually breaks for it: GRAM by whether the module ever learned 5,\n"
         "core adaptation by whether the other nine digits survived.",
         fontsize=8.5, color="#666666", linespacing=1.5, va="top")
fig.text(0.080, 0.833,
         f"GRAM gets BETTER under noise:  {gc:.3f} → {gn:.3f} readout above chance at its best "
         f"setting.    Core adaptation gets WORSE:  {cc:.3f} → {cn:.3f}.\n"
         "A mislabeled 5 reaches GRAM as a wrong LABEL, which pushes logit 5 down. It reaches "
         "core adaptation as the TEACHER's\nopinion, and the teacher reads the image "
         "correctly whatever the label says.",
         fontsize=8.8, color=INK, linespacing=1.55, va="top")
# Both axes here are the model's OWN behaviour. A linear probe fitted on the ten logits
# recovers far more than logit 5 alone, for every usable setting of either method -- so this
# figure says which method EMITS less, not which leaves less behind.
fig.text(0.080, 0.737,
         "Caveat: both axes are what the model EMITS. A linear probe on its ten logits "
         "recovers 0.93\u20130.99 for every usable setting of either\nmethod, against readouts "
         "as low as 0.48 \u2014 GRAM's knee here reads 0.549 by readout and 0.927 by probe. "
         "Read this as which method emits\nless, not which leaves less behind; see "
         "figures/probe_vs_readout.png.",
         fontsize=8.0, color="#9a6a2a", linespacing=1.5, va="top")

out = f"{HERE}/figures/compare_pareto.png"
fig.savefig(out)
print(out)
print(f"GRAM  clean knee {gc:.3f} | noisy knee {gn:.3f}   ({len(GRAM[0.0])}, {len(GRAM[0.3])} on front)")
print(f"CA    clean knee {cc:.3f} | noisy knee {cn:.3f}   ({len(CA[0.0])}, {len(CA[0.3])} on front)")
