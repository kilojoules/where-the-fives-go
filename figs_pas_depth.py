"""5-decodability by depth, across the p_as sweep.

p_as is the probability the CORE receives gradient on a routed-5 batch. The question this
answers: does turning it down keep the 5 out of the REPRESENTATION, or only out of the
readout? The curves separate at embed and block 1, so it is the representation -- p_as is a
real lever on what the trunk encodes, not just on what the model will say.

Two of the five settings are degenerate and are drawn dashed: at p_as 0 and 0.1 the system
never learned the capability at all (module-ON recall 0.017 and 0.053), so those curves show
a network that cannot do the task, not a tighter version of GRAM.

Reads results/pas_depth.json and results/pas_sweep.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{HERE}/results/pas_depth.json"))
S = json.load(open(f"{HERE}/results/pas_sweep.json"))["rows"]
R, STAGES = D["rows"], D["stages"]
XS = STAGES + ["shipped"]

# p_as is a magnitude, so the palette is one hue light -> dark, not categorical colours.
RAMP = {"0.0": "#a3ddc0", "0.1": "#63c79b", "0.25": "#2fb57f", "0.5": "#17815a", "1.0": "#0a4f36"}
DEGENERATE = {"0.0", "0.1"}      # module-ON recall 0.017 / 0.053 -- never learned the task
NICE = {"input": "raw\npixels", "embed": "embed", "block1": "block 1", "block2": "block 2",
        "logits": "logits\n(probe)", "shipped": "shipped\nreadout"}
GREY, INK = "#9a9a92", "#333333"

fig, ax = plt.subplots(figsize=(11.6, 6.0), dpi=200)
fig.subplots_adjust(left=0.078, right=0.695, top=0.760, bottom=0.135)

x = list(range(len(XS)))
ax.axhspan(0.30, 0.5, color="#f6f0f6", zorder=0)
ax.axhline(0.5, color="#8a5a9a", lw=1.2, ls="--", zorder=2)
ax.text(0.04, 0.515, "chance = 0.5", fontsize=7.8, color="#8a5a9a", va="bottom")
ax.text(0.04, 0.478, "below chance: the representation ANTI-ranks 5s", fontsize=7.8,
        color="#8a5a9a", va="top", style="italic")
ax.axhline(R["0.0"]["input"], color=GREY, lw=1, ls=":", zorder=2)
ax.text(2.25, R["0.0"]["input"] + 0.009, "raw-pixel floor 0.956", fontsize=7.6,
        color=GREY, va="bottom")

ends = sorted(((R[pas][XS[-1]], pas) for pas in RAMP), reverse=True)
label_y, GAP = {}, 0.038
for i, (v, pas) in enumerate(ends):
    label_y[pas] = v if i == 0 else min(v, label_y[ends[i - 1][1]] - GAP)

for pas in ["0.0", "0.1", "0.25", "0.5", "1.0"]:
    y = [R[pas][s] for s in XS]
    deg = pas in DEGENERATE
    ax.plot(x, y, color=RAMP[pas], lw=2.6, ls="--" if deg else "-",
            marker="o", ms=5.4, markerfacecolor="white" if deg else RAMP[pas],
            markeredgewidth=1.7, zorder=4)
    rec = S[pas]["ablated(0,1)"]["recall5"]
    tail = "  ✗ never learned the task" if deg else (
        f"  ✗ ablation stopped working (recall {rec:.2f})" if rec > 0.01 else "  ✓ works")
    ly = label_y[pas]
    if abs(ly - y[-1]) > 0.004:
        ax.plot([x[-1] + 0.02, x[-1] + 0.10], [y[-1], ly], color=RAMP[pas], lw=0.8,
                clip_on=False, zorder=3)
    ax.text(x[-1] + 0.12, ly, f"p_as = {pas}{tail}", fontsize=8.5, color=RAMP[pas],
            va="center", ha="left")

ax.annotate("p_as = 0 drives block 2 to 0.376 —\nfar below chance", (3, R["0.0"]["block2"]),
            xytext=(1.25, 0.405), fontsize=8.2, color="#8a5a9a", va="center", linespacing=1.4,
            arrowprops=dict(arrowstyle="-", color="#8a5a9a", lw=0.9))

ax.set_xticks(x)
ax.set_xticklabels([NICE[s] for s in XS], fontsize=8.4)
ax.set_xlim(-0.18, x[-1] + 0.14)
ax.set_ylim(0.32, 1.035)
ax.set_xlabel("depth — where the probe reads", fontsize=9)
ax.set_ylabel("AUC for “is this a 5?”, module 5 detached", fontsize=9)
ax.grid(True, color="#ececec", lw=0.6, zorder=1)
ax.set_axisbelow(False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=8.2)

fig.suptitle("Turning p_as down keeps the 5 out of the representation, not just the readout",
             fontsize=12.4, x=0.078, ha="left", y=0.957)
fig.text(0.078, 0.815,
         "p_as is how often the CORE gets gradient on a routed 5. The curves separate at embed "
         "and block 1, so it governs what the trunk\nencodes. But no setting is both safe and "
         "useful: the two that strip the trunk never learn the task, and the one that keeps the\n"
         "trunk clean enough to matter has stopped suppressing. Seed 0; one run per setting; "
         "p_as is our MNIST knob, not real GRAM’s.",
         fontsize=8.6, color="#666666", linespacing=1.5)
out = f"{HERE}/figures/pas_depth.png"
fig.savefig(out)
print(out)
