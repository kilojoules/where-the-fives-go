"""Analysis of elicitation attacks and the continuous-ablation sweep."""

import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SHIPS = [
    ("dense_all", "dense, all data (capability present)", "#888888"),
    ("filter_no5", "data-filtered (never saw 5s)", "#4c72b0"),
    ("gram_ablate5", "GRAM, module-5 ablated", "#1f8a70"),
    ("gramF03_ablate5", "GRAM @3% labels, module-5 ablated", "#b3423a"),
]
KS = [10, 50, 200]


def main():
    res = json.load(open("results/elicitation.json"))
    lines = []
    say = lambda t="": (print(t), lines.append(t))

    say("# Elicitation: recovering the removed 5 by finetuning\n")
    say("Best recall on 5s the attacker reaches while the model stays useful "
        "(acc on non-5 digits >= 95%); best over 2 learning rates; mean±sd over 3 seeds.\n")
    say("| shipped model | k=10 | k=50 | k=200 |")
    say("|---|---|---|---|")
    best = defaultdict(dict)  # ship -> k -> list over seeds of best-over-lr
    for r in res:
        key = (r["ship"], r["k"], r["seed"])
        cur = best[r["ship"]].setdefault(r["k"], {})
        cur[r["seed"]] = max(cur.get(r["seed"], 0.0), r["best_recall5_useful"])
    for ship, label, _ in SHIPS:
        cells = []
        for k in KS:
            vals = list(best[ship][k].values())
            cells.append(f"{100*np.mean(vals):.1f}±{100*np.std(vals):.1f}")
        say(f"| {label} | " + " | ".join(cells) + " |")

    # recovery curves: per ship/k, per seed pick lr maximizing useful-best, average trajs
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, k in zip(axes, KS):
        for ship, label, color in SHIPS:
            trajs = []
            for seed in [0, 1, 2]:
                cand = [r for r in res if r["ship"] == ship and r["k"] == k and r["seed"] == seed]
                r = max(cand, key=lambda r: r["best_recall5_useful"])
                trajs.append([t[1] for t in r["traj"]])
            steps = [t[0] for t in cand[0]["traj"]]
            m, s = np.mean(trajs, axis=0), np.std(trajs, axis=0)
            ax.plot(steps, m, color=color, label=label)
            ax.fill_between(steps, m - s, m + s, color=color, alpha=0.15)
        ax.set_title(f"k = {k} labeled 5s")
        ax.set_xlabel("finetune steps"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("test recall on 5s")
    axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("figures/elicitation.png", dpi=150); plt.close()

    # alpha sweep
    al = json.load(open("results/alpha_sweep.json"))
    alphas = sorted({a["alpha"] for a in al})
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for key, label, color, scale in [("recall5", "recall on 5s", "#1f8a70", 1),
                                     ("fp5", "FP rate into 5 (x10)", "#b3423a", 10)]:
        m = [np.mean([r[key] for r in al if r["alpha"] == a]) * scale for a in alphas]
        s = [np.std([r[key] for r in al if r["alpha"] == a]) * scale for a in alphas]
        ax.errorbar(alphas, m, yerr=s, fmt="o-", color=color, capsize=3, label=label)
    ax.set_xlabel("module-5 output scale α (module-7 = 1)")
    ax.set_ylabel("rate")
    ax.set_title("Continuous ablation of module-5")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("figures/alpha_sweep.png", dpi=150); plt.close()

    say("\nAlpha sweep (mean over seeds):")
    say("| α | recall 5 | FP->5 |")
    say("|---|---|---|")
    for a in alphas:
        r5 = np.mean([r["recall5"] for r in al if r["alpha"] == a])
        f5 = np.mean([r["fp5"] for r in al if r["alpha"] == a])
        say(f"| {a} | {100*r5:.2f} | {100*f5:.2f} |")

    with open("results/summary_elicit.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote results/summary_elicit.md, figures/elicitation.png, figures/alpha_sweep.png")


if __name__ == "__main__":
    main()
