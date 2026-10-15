"""Grokking reliability on modular arithmetic: does routing help the model grok x?

Held-out multiplication accuracy after 5000 epochs under the identical recipe,
10 seeds per arm: GRAM (t2_gram seeds 0-2 from t2_suite.py, t2x_gram seeds 3-9
from t2_seeds.py), scored with its module attached, and dense (t2_dense /
t2x_dense), scored on its core. A seed counts as grokked at >= 80%.

Outputs figures/grokking_reliability.png from results/.
"""

import json
from math import comb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GROKKED = 0.8
ARMS = [("gram", "GRAM, module attached", "#1f8a70"), ("dense", "dense", "#8a8a8a")]


def final_acc(arm):
    """Held-out x accuracy for seeds 0-9: GRAM with its module attached (full), dense on its core."""
    prof = "full" if arm == "gram" else "core"
    return np.array([json.load(open(f"results/{'t2' if s < 3 else 't2x'}_{arm}_seed{s}.json"))[prof]["mult_acc"]
                     for s in range(10)])


def fisher(a, b, n=10):
    """Exact Fisher test of a/n vs b/n grokked: (one-sided p for the first arm grokking more often, two-sided p)."""
    k = a + b
    pr = {x: comb(n, x) * comb(n, k - x) / comb(2 * n, k) for x in range(max(0, k - n), min(k, n) + 1)}
    return sum(p for x, p in pr.items() if x >= a), min(1.0, sum(p for p in pr.values() if p <= pr[a] * (1 + 1e-9)))


def main():
    acc = {arm: final_acc(arm) for arm, _, _ in ARMS}
    k = {arm: int((v >= GROKKED).sum()) for arm, v in acc.items()}
    one, two = fisher(k["gram"], k["dense"])
    for arm, v in acc.items():
        print(f"{arm:5s} " + " ".join(f"{x:.3f}" for x in v) + f"  grokked {k[arm]}/10")
    print(f"Fisher: one-sided p = {one:.3f} for GRAM grokking more often, two-sided p = {two:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    offs = np.linspace(-0.2, 0.2, 10)
    for i, (arm, label, c) in enumerate(ARMS):
        v = acc[arm]
        ax.scatter(i + offs[:3], v[:3], s=60, color=c, edgecolors="k", linewidths=0.5, zorder=3)
        ax.scatter(i + offs[3:], v[3:], s=60, facecolors="white", edgecolors=c, linewidths=1.6, zorder=3)
        ax.annotate(f"{k[arm]}/10 grokked", (i, 1.06), ha="center", fontsize=10, color=c, fontweight="bold")
    ax.axhline(GROKKED, color="k", lw=0.8, ls="--")
    ax.annotate("grokked: ≥ 80%", (1.45, GROKKED + 0.015), ha="right", fontsize=8)
    ax.axhline(1 / 53, color="k", lw=0.6, ls=":")
    ax.annotate("chance", (1.45, 1 / 53 + 0.015), ha="right", fontsize=8)
    ax.scatter([], [], s=60, color="#bbbbbb", edgecolors="k", linewidths=0.5, label="seeds 0–2 (t2_suite.py)")
    ax.scatter([], [], s=60, facecolors="white", edgecolors="#555555", linewidths=1.6,
               label="seeds 3–9 (t2_seeds.py)")
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label for _, label, _ in ARMS])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.03, 1.13)
    ax.set_ylabel("held-out × accuracy after 5000 epochs")
    ax.set_title("Routing neither helps nor hurts grokking: 10 seeds per arm, identical recipe", fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    fig.text(0.01, 0.01, f"One-sided Fisher p = {one:.2f} for GRAM grokking more often; two-sided p = {two:.2f}.",
             fontsize=8, color="#52514e")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig("figures/grokking_reliability.png", dpi=150)
    print("wrote figures/grokking_reliability.png", flush=True)


if __name__ == "__main__":
    main()
