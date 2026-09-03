"""Analysis of the labeler operating-point grid: absorption vs (recall r, precision p)."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEEDS = [0, 1, 2]
RECALLS = [0.03, 0.3, 1.0]
PRECISIONS = [1.0, 0.9, 0.7, 0.5]


def cell(name, profile, fn):
    vals = []
    for s in SEEDS:
        js = json.load(open(f"results/{name}_seed{s}.json"))
        vals.append(fn(js[profile]))
    return np.mean(vals), np.std(vals)


def recall_of(d):
    return lambda m: m["recall"][str(d)]


def fp5(m):
    c = np.array(m["confusion"])
    return (c[:, 5].sum() - c[5, 5]) / (c.sum() - c[5].sum())


def acc_excl(m, digits={5, 7}):
    c = np.array(m["confusion"])
    keep = [i for i in range(10) if i not in digits]
    return sum(c[i, i] for i in keep) / c[keep].sum()


def main():
    lines = []
    say = lambda t="": (print(t), lines.append(t))
    say("# Labeler operating point vs absorption\n")
    say("Routed set for module-5 built by a simulated 5-domain labeler with recall r "
        "(missed 5s -> all-active pool) and precision p (junk non-5 images routed in, "
        "true class targets kept).\n")

    grids = {}
    for metric, prof, fn, label in [
        ("absorb", "core_only", recall_of(5), "core-only recall on 5s (absorption: lower = stronger)"),
        ("ablate", "ablate5", recall_of(5), "ablate-5 recall on 5s (deployable removal)"),
        ("full", "full", recall_of(5), "full-profile recall on 5s (capability delivery)"),
        ("fp", "full", fp5, "full-profile FP rate into 5"),
        ("collat", "ablate5", acc_excl, "ablate-5 accuracy on digits excl {5,7} (collateral)"),
    ]:
        g = np.zeros((len(RECALLS), len(PRECISIONS)))
        gs = np.zeros_like(g)
        say(f"\n## {label}\n")
        say("| r \\ p | " + " | ".join(str(p) for p in PRECISIONS) + " |")
        say("|---" * (len(PRECISIONS) + 1) + "|")
        for i, r in enumerate(RECALLS):
            row = []
            for j, p in enumerate(PRECISIONS):
                m, s = cell(f"rt_r{r}_p{p}", prof, fn)
                g[i, j], gs[i, j] = m, s
                row.append(f"{100*m:.2f}±{100*s:.2f}")
            say(f"| {r} | " + " | ".join(row) + " |")
        grids[metric] = (g, gs)

    say("\n## Concentrated FPs from digit 3 (r=1.0)\n")
    say("| p | source | ablate5 recall-3 | core-only recall-3 | full recall-3 | ablate5 recall-5 |")
    say("|---|---|---|---|---|---|")
    base_a3 = cell("rt_r1.0_p1.0", "ablate5", recall_of(3))
    base_c3 = cell("rt_r1.0_p1.0", "core_only", recall_of(3))
    base_f3 = cell("rt_r1.0_p1.0", "full", recall_of(3))
    base_a5 = cell("rt_r1.0_p1.0", "ablate5", recall_of(5))
    say(f"| 1.0 | (none) | {100*base_a3[0]:.2f}±{100*base_a3[1]:.2f} | {100*base_c3[0]:.2f}±{100*base_c3[1]:.2f} | "
        f"{100*base_f3[0]:.2f}±{100*base_f3[1]:.2f} | {100*base_a5[0]:.2f}±{100*base_a5[1]:.2f} |")
    for p in [0.9, 0.7]:
        for tag, src in [(f"rt_r1.0_p{p}", "uniform"), (f"rt_r1.0_p{p}_from3", "all 3s")]:
            a3 = cell(tag, "ablate5", recall_of(3))
            c3 = cell(tag, "core_only", recall_of(3))
            f3 = cell(tag, "full", recall_of(3))
            a5 = cell(tag, "ablate5", recall_of(5))
            say(f"| {p} | {src} | {100*a3[0]:.2f}±{100*a3[1]:.2f} | {100*c3[0]:.2f}±{100*c3[1]:.2f} | "
                f"{100*f3[0]:.2f}±{100*f3[1]:.2f} | {100*a5[0]:.2f}±{100*a5[1]:.2f} |")

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for ax, key, title, cmap in [
            (axes[0], "absorb", "Absorption: core-only recall on 5s", "viridis"),
            (axes[1], "full", "Delivery: full-profile recall on 5s", "viridis"),
            (axes[2], "fp", "Cost: full-profile FP rate into 5", "magma"),
            (axes[3], "collat", "Collateral: ablate-5 acc on other digits", "viridis")]:
        g, gs = grids[key]
        im = ax.imshow(g * 100, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(PRECISIONS))); ax.set_xticklabels(PRECISIONS)
        ax.set_yticks(range(len(RECALLS))); ax.set_yticklabels(RECALLS)
        ax.set_xlabel("labeler precision p"); ax.set_ylabel("labeler recall r")
        ax.set_title(title, fontsize=10)
        for i in range(len(RECALLS)):
            for j in range(len(PRECISIONS)):
                ax.text(j, i, f"{100*g[i,j]:.1f}\n±{100*gs[i,j]:.1f}", ha="center", va="center",
                        fontsize=8, color="white" if g[i, j] < 0.7 * g.max() else "black")
        plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig("figures/routing_grid.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ps = [1.0, 0.9, 0.7]
    for tag_fn, label, style in [
            (lambda p: f"rt_r1.0_p{p}" if p == 1.0 else f"rt_r1.0_p{p}_from3", "FPs all from 3s", "o-"),
            (lambda p: f"rt_r1.0_p{p}", "FPs uniform", "s--")]:
        m = [cell(tag_fn(p), "ablate5", recall_of(3))[0] for p in ps]
        s = [cell(tag_fn(p), "ablate5", recall_of(3))[1] for p in ps]
        ax.errorbar([1 - p for p in ps], np.array(m) * 100, yerr=np.array(s) * 100,
                    fmt=style, capsize=3, label=f"recall on 3s, ablate-5 ({label})")
    ax.set_xlabel("labeler FP contamination (1 − p)")
    ax.set_ylabel("clean-test recall on digit 3 (%)")
    ax.set_title("Collateral absorption of the FP-source class")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("figures/routing_collateral.png", dpi=150); plt.close()

    with open("results/summary_routing.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote results/summary_routing.md, figures/routing_grid.png, figures/routing_collateral.png")


if __name__ == "__main__":
    main()
