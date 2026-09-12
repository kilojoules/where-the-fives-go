"""Depth-profile probes as contours over labeler FP/FN operating points.

Same stage-wise probing as layer_probe.py, but the lines are labeler operating
points from the Part IV grid: the FN axis (labeler recall r, misses to the
all-active pool) and the FP axis (precision p, junk routed into the module
with true targets). Custody-law prediction: the contours collapse — latent
5-information in the ablated core is insensitive to labeler quality even
though full-profile FP behavior varies wildly.

Outputs figures/layer_probe_labeler.png and results/contour_probe.json.
"""

import json

import numpy as np
import torch

from layer_probe import mnist_profile_one, SEEDS

STAGES = ["input", "embed", "block1", "block2", "logits"]

FN_AXIS = [("r = 1.00 (no misses)", "ck_gram_main"),
           ("r = 0.30", "rtck_r0.3_p1.0"),
           ("r = 0.03", "ck_gram_f0.03")]
FP_AXIS = [("p = 1.0 (clean)", "ck_gram_main"),
           ("p = 0.9", "rtck_r1.0_p0.9"),
           ("p = 0.7", "rtck_r1.0_p0.7"),
           ("p = 0.5 (coin flip)", "rtck_r1.0_p0.5")]


def main():
    import multiprocessing as mp
    names = sorted({n for _, n in FN_AXIS + FP_AXIS})
    jobs = [(n, s) for n in names for s in SEEDS]
    res = {n: {**{st: [] for st in STAGES}, "native_auroc": []} for n in names}
    with mp.get_context("spawn").Pool(3) as pool:
        for name, s, out in pool.imap_unordered(mnist_profile_one, jobs):
            for st in STAGES:
                res[name][st].append(out[st])
            res[name]["native_auroc"].append(out["native_auroc"])
    json.dump(res, open("results/contour_probe.json", "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.arange(len(STAGES))
    labels = ["raw\ninput", "embed", "block 1", "block 2", "logits"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, axis, cmap_name, title in [
            (axes[0], FN_AXIS, "viridis", "FN axis: labeler recall r (misses → all-active pool)"),
            (axes[1], FP_AXIS, "magma", "FP axis: labeler precision p (junk routed in)")]:
        cmap = plt.get_cmap(cmap_name)
        for i, (lab, name) in enumerate(axis):
            c = cmap(0.2 + 0.6 * i / max(1, len(axis) - 1))
            m = [np.mean(res[name][st]) for st in STAGES]
            sd = [np.std(res[name][st]) for st in STAGES]
            ax.errorbar(xs, m, yerr=sd, fmt="o-", color=c, capsize=3, label=lab)
            ax.scatter([xs[-1] + 0.35], [np.mean(res[name]["native_auroc"])],
                       marker="*", s=160, color=c, zorder=5, edgecolors="k", linewidths=0.5)
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.set_xticks(list(xs) + [xs[-1] + 0.35])
        ax.set_xticklabels(labels + ["native\nreadout"])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("probe AUROC: 5 vs rest (ablated core)")
    axes[0].set_ylim(0.35, 1.03)
    plt.tight_layout()
    plt.savefig("figures/layer_probe_labeler.png", dpi=150)
    print("CONTOUR PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
