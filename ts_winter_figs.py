"""Figures for the real-TinyStories winter sweep (ts_winter_pas.py), from results/tsw_eval_*.json.

  python ts_winter_figs.py   writes figures/tsw_story.png

Three panels across p_as, pooled over seeds, with 95% Wilson intervals:
  behavior   share of 2048 sampled stories per seed that use a winter word
             (no steering), GRAM with the module deleted and attached;
  prompted   share of continuations of real winter openings (341 per seed,
             no steering) that use a winter word again;
  steering   share of 512 steered stories per seed that use a winter word,
             at each vector's pre-registered cell.
Dense references are drawn as bands (their interval) or lines, the filter as
a dotted line.
"""

import glob
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PAS = [0.5, 0.9, 0.95, 0.98, 1.0]
TEAL, VIOLET, BRICK, ORANGE, SLATE, GREY = "#1f8a70", "#7b5ea7", "#b3423a", "#d08a2e", "#33557b", "#7d8994"


def wilson(k, n, z=1.96):
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    m = (p + z * z / (2 * n)) / (1 + z * z / n)
    return p, m - c, m + c


def load():
    rows = {}
    for f in sorted(glob.glob("results/tsw_eval_*.json")):
        if "smoke" in f:
            continue
        r = json.load(open(f))
        rows.setdefault(r["arm"] + (f" {r['p_as']:g}" if r["arm"] == "gram" else ""), []).append(r)
    return rows


def rate(parts):
    """Pooled (rate, lo, hi) over a list of sample-stat dicts."""
    return wilson(sum(p["winter"] for p in parts), sum(p["n"] for p in parts))


def best_random(rs):
    return rate([max((c["constant"] for k, c in r["cells"].items() if k.startswith("random")),
                     key=lambda c: c["winter"]) for r in rs])


def line(ax, xs, vals, color, label, ls="-", marker="o"):
    v = np.array(vals)
    ax.errorbar(xs, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=color, lw=2.2, ls=ls,
                marker=marker, ms=6, capsize=3, label=label)


def band(ax, v, color, label, xr, below=False):
    ax.axhspan(v[1], v[2], color=color, alpha=0.15, lw=0)
    ax.axhline(v[0], color=color, ls="--", lw=1.2)
    ax.annotate(label, (xr, v[1] if below else v[2]), xytext=(0, -3 if below else 3), textcoords="offset points",
                ha="right", va="top" if below else "bottom", fontsize=8.5, color=color)


def style(ax, title, ylabel, ymax):
    ax.set_xticks(range(len(PAS)))
    ax.set_xticklabels([f"{p:g}" for p in PAS])
    ax.set_xlim(-0.4, len(PAS) - 0.6)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0 if ymax > 0.1 else 1))
    ax.set_xlabel("GRAM p_as (not to scale)", fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10.5, loc="left")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def main():
    rows = load()
    g = [rows[f"gram {p:g}"] for p in PAS]
    dense, filt = rows["dense"], rows["filter"]
    xs = np.arange(len(PAS))
    xr = len(PAS) - 0.6
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    a = axes[0]
    band(a, rate([r["base"]["neutral"] for r in dense]), SLATE, "trained without GRAM", xr, below=True)
    line(a, xs, [rate([r["full"]["neutral"] for r in rs]) for rs in g], VIOLET, "GRAM, module attached")
    line(a, xs, [rate([r["base"]["neutral"] for r in rs]) for rs in g], TEAL, "GRAM, module deleted (shipped)")
    f = rate([r["base"]["neutral"] for r in filt])
    a.axhline(f[0], color=GREY, ls=":", lw=1.4)
    a.annotate(f"filtered: {f[0]:.2%}", (xr, f[0]), xytext=(0, 3), textcoords="offset points", ha="right",
               fontsize=8.5, color=GREY)
    style(a, "The behavior: unprompted stories with a winter word", "stories with a winter word", 0.036)
    a.legend(loc="upper left", frameon=False, fontsize=8.5)

    b = axes[1]
    band(b, rate([r["base"]["prompted"] for r in dense]), SLATE, "trained without GRAM", xr, below=True)
    line(b, xs, [rate([r["full"]["prompted"] for r in rs]) for rs in g], VIOLET, "GRAM, module attached")
    line(b, xs, [rate([r["base"]["prompted"] for r in rs]) for rs in g], TEAL, "GRAM, module deleted")
    f = rate([r["base"]["prompted"] for r in filt])
    b.axhline(f[0], color=GREY, ls=":", lw=1.4, clip_on=False, zorder=5)
    b.annotate(f"filtered: {sum(r['base']['prompted']['winter'] for r in filt)} of "
               f"{sum(r['base']['prompted']['n'] for r in filt)}", (xr, f[0]), xytext=(0, 4),
               textcoords="offset points", ha="right", fontsize=8.5, color=GREY)
    style(b, "Continuing a real winter opening (no steering)", "continuations with a winter word", 1.0)
    b.legend(loc="lower right", frameon=False, fontsize=8.5, bbox_to_anchor=(1, 0.06))

    c = axes[2]
    band(c, rate([r["cells"]["token_snow"]["constant"] for r in dense]), SLATE,
         "trained without GRAM + push", xr)
    do = rate([r["cells"]["own_dm"]["constant"] for r in dense])
    c.axhline(do[0], color=ORANGE, ls=(0, (1.5, 2.5)), lw=1.4)
    c.annotate("trained without GRAM + own vector", (xr, do[0]), xytext=(0, -3), textcoords="offset points",
               ha="right", va="top", fontsize=8.5, color=ORANGE)
    line(c, xs, [rate([r["cells"]["token_snow"]["constant"] for r in rs]) for rs in g], BRICK,
         "deleted + push toward “snow”")
    line(c, xs, [rate([r["cells"]["own_dm"]["constant"] for r in rs]) for rs in g], ORANGE,
         "deleted + own winter vector", ls="--", marker="s")
    line(c, xs, [best_random(rs) for rs in g], "#b9c0c6", "deleted + best of 8 random", marker="^")
    fp = rate([r["cells"]["token_snow"]["constant"] for r in filt])
    fo = rate([r["cells"]["own_dm"]["constant"] for r in filt])
    c.axhline(max(fp[0], fo[0]), color=GREY, ls=":", lw=1.4,
              label=f"filtered + push ({fp[0]:.2%}) or own vector ({fo[0]:.2%})")
    style(c, "Steering the shipped model", "steered stories with a winter word", 0.75)
    c.legend(loc="upper left", frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig("figures/tsw_story.png", dpi=160)
    print("wrote figures/tsw_story.png")


if __name__ == "__main__":
    main()
