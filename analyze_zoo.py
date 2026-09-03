"""Probe-predicts-elicitation analysis (review corrections encoded).

Per protocol review:
- PRIMARY outcome: rec_any = best target over trajectory, max over lrs
  (raw best_useful is a degenerate threshold race — never used).
- SECONDARY: rec_rel = best target at eval points where usefulness stayed
  within 0.05 of the ship's own step-0 usefulness (drop-adjusted).
- Behavior matching uses recorded per-ship baseline_target (<= 0.05), never
  config labels. Partial rank correlation controls baseline_target.
- Budget check: modmath trajectories flagged if still rising at final eval.
- Below-chance probe flags per family.
"""

import glob
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHANCE = {"mnist": 0.10, "fashion": 0.05, "modmath": 1 / 53, "lm": 0.11}
SMALL_K = {"mnist": 10, "fashion": 100, "modmath": 50, "lm": 50}


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x, y, z):
    """Rank-based partial correlation of x,y given z."""
    def ranks(v):
        return np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    rx, ry, rz = ranks(x), ranks(y), ranks(z)
    def resid(a, b):
        if b.std() == 0:
            return a - a.mean()
        beta = np.cov(a, b)[0, 1] / np.var(b)
        return a - beta * b
    ex, ey = resid(rx, rz), resid(ry, rz)
    if ex.std() == 0 or ey.std() == 0:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def load_ships():
    ships = {}
    for f in glob.glob("results/zoo/probe_*.json"):
        js = json.load(open(f))
        ships[(js["family"], js["config"], js["seed"])] = {
            "probe": js["probe_target"], "baseline": js["baseline_target"],
            "auroc": js.get("probe_auroc"),
        }
    for f in glob.glob("results/zoo/*_s*_k*_lr*.json"):
        js = json.load(open(f))
        key = (js["family"], js["config"], js["seed"])
        if key not in ships:
            continue
        s = ships[key]
        k = js["k"]
        base_useful = js["traj"][0]["useful"]
        rec_rel = max((t["target"] for t in js["traj"]
                       if t["useful"] >= base_useful - 0.05), default=0.0)
        rising = (len(js["traj"]) >= 3 and
                  js["traj"][-1]["target"] > js["traj"][-2]["target"] + 0.02)
        d = s.setdefault(f"k{k}", {"rec_any": 0.0, "rec_rel": 0.0, "rising": False})
        d["rec_any"] = max(d["rec_any"], js["best_any"])
        d["rec_rel"] = max(d["rec_rel"], rec_rel)
        d["rising"] = d["rising"] or rising
    return ships


def main():
    ships = load_ships()
    lines = []
    say = lambda t="": (print(t), lines.append(t))
    say("# Does the probe predict elicitation?\n")

    fams = sorted({f for f, _, _ in ships})
    fig, axes = plt.subplots(1, len(fams) + 1, figsize=(4.2 * (len(fams) + 1), 4.0))
    matched_all = []

    for ax, fam in zip(axes, fams):
        rows = [(c, s, v) for (f, c, s), v in ships.items() if f == fam and f"k{SMALL_K[fam]}" in v]
        probe = [v["probe"] for _, _, v in rows]
        base = [v["baseline"] for _, _, v in rows]
        rec = [v[f"k{SMALL_K[fam]}"]["rec_any"] for _, _, v in rows]
        rec_rel = [v[f"k{SMALL_K[fam]}"]["rec_rel"] for _, _, v in rows]
        rising = sum(v[f"k{SMALL_K[fam]}"]["rising"] for _, _, v in rows)
        say(f"\n## {fam}  (n={len(rows)} ships, k={SMALL_K[fam]})")
        say(f"Spearman(probe, recovery) = {spearman(probe, rec):.3f} | "
            f"partial given baseline = {partial_spearman(probe, rec, base):.3f} | "
            f"drop-adjusted recovery corr = {spearman(probe, rec_rel):.3f}")
        say(f"still-rising trajectories at final eval: {rising}/{len(rows)}")
        below = [(c, s, v["probe"]) for c, s, v in rows if v["probe"] < CHANCE[fam]]
        say(f"below-chance probes: {below if below else 'none'}")

        m = [(c, s, v) for c, s, v in rows if v["baseline"] <= 0.05]
        if len(m) >= 4:
            mp_, mr = [v["probe"] for _, _, v in m], [v[f"k{SMALL_K[fam]}"]["rec_any"] for _, _, v in m]
            say(f"behavior-matched subset (baseline<=0.05, n={len(m)}): "
                f"Spearman = {spearman(mp_, mr):.3f}")
            dense_ref = np.mean([v["baseline"] for c, s, v in rows if "dense" in c]) or 1.0
            for c, s, v in m:
                matched_all.append((fam, c, s, v["probe"] / max(dense_ref, 1e-9),
                                    v[f"k{SMALL_K[fam]}"]["rec_any"] / max(dense_ref, 1e-9)))
        say("| config | seed | baseline | probe | rec_any | rec_rel |")
        say("|---|---|---|---|---|---|")
        for c, s, v in sorted(rows):
            d = v[f"k{SMALL_K[fam]}"]
            say(f"| {c} | {s} | {v['baseline']:.3f} | {v['probe']:.3f} | "
                f"{d['rec_any']:.3f} | {d['rec_rel']:.3f} |")

        colors = ["#b3423a" if v["baseline"] <= 0.05 else "#8899aa" for _, _, v in rows]
        ax.scatter(probe, rec, c=colors, s=28, alpha=0.85)
        ax.set_xlabel("probe accuracy (ship mask)")
        ax.set_ylabel(f"attack recovery (k={SMALL_K[fam]})")
        ax.set_title(f"{fam}  ρ={spearman(probe, rec):.2f}", fontsize=10)
        ax.grid(alpha=0.3)

    ax = axes[-1]
    if matched_all:
        xs = [p for _, _, _, p, _ in matched_all]
        ys = [r for _, _, _, _, r in matched_all]
        fam_colors = {"mnist": "#1f8a70", "fashion": "#b3423a", "modmath": "#4c72b0", "lm": "#c78a2d"}
        for fam in fams:
            fx = [p for f, _, _, p, _ in matched_all if f == fam]
            fy = [r for f, _, _, _, r in matched_all if f == fam]
            if fx:
                ax.scatter(fx, fy, color=fam_colors.get(fam, "k"), s=30, label=fam, alpha=0.85)
        ax.set_title(f"behavior-matched, all families  ρ={spearman(xs, ys):.2f}", fontsize=10)
        ax.set_xlabel("probe / family ceiling")
        ax.set_ylabel("recovery / family ceiling")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        say(f"\n## Pooled behavior-matched (n={len(matched_all)}): "
            f"Spearman = {spearman(xs, ys):.3f}")
    plt.tight_layout(); plt.savefig("figures/zoo_correlation.png", dpi=150); plt.close()

    with open("results/summary_zoo.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote results/summary_zoo.md, figures/zoo_correlation.png")


if __name__ == "__main__":
    main()
