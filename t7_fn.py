"""Track 7, winter: how do labeler FALSE NEGATIVES change steerability?

GRAM's steering resistance on winter (Track 7: 0.000 under every vector within
core loss <= 0.55, vs 0.16 -> 0.72 on dense) assumed perfect labels. Real labelers miss examples.
A missed winter story is treated as retained data: under GRAM it is routed
through the core path (so the core learns winter from it directly); under
filtering it stays in the training set. Same miss rate, same missed windows,
both arms.

Sweep fn_rate in {0, .01, .03, .1, .3, .5} x 3 seeds, GRAM and filter (fn=0
reuses the Track 7 checkpoints; dense is the reference). Per shipped model:
behaviour (P(winter | "one day in"), binding A/B), the key-held-out probe,
and the t7_eval steering sweep (caa / window_mean / random, up and down,
sites all/L2/L3/L4), compared at matched collateral.

Run:  python t7_fn.py          (trains missing checkpoints, writes results/t7_fn.json)
      python t7_fn.py report | figonly
"""

import json

import numpy as np

try:
    import torch
    import track7_concepts as T
    import t7_eval as E
except ImportError:
    T = None

CONCEPT = "winter"
FNS = [0.0, 0.01, 0.03, 0.1, 0.3, 0.5]
SEEDS = [0, 1, 2]
ARMS = ["gram", "filter"]
MASK = (0,)


def tag(arm, fn, seed):
    fs = f"_fn{fn:g}" if fn > 0 else ""
    return f"t7_{CONCEPT}_{arm}{fs}_seed{seed}"


def eval_one(job):
    arm, fn, seed = job
    torch.set_num_threads(2)
    core, target, _, _, xvt, yvt = T.val_sets(CONCEPT)
    xvc, yvc = T.windows(core[1_100_000:], 150, 99)
    xc, _ = T.windows(core[:1_100_000], 256, 8)
    xt, _ = T.windows(target[:180_000], 256, 7)
    m = T.LM()
    m.load_state_dict(torch.load(f"results/{tag(arm, fn, seed)}.pt"))
    m.eval()
    pre = f"{arm}|{fn:g}"
    out = {f"{pre}|behaviour": T.behaviour(m, CONCEPT, MASK, xvc, yvc, xvt, yvt),
           f"{pre}|probe": E.probe(m, CONCEPT, seed)}
    E.steer_sweep(m, CONCEPT, seed, xc, xt, xvc, yvc, out, pre)
    print(f"eval {pre} seed{seed} done", flush=True)
    return out


def main():
    import multiprocessing as mp
    import os
    cfgs = [T.Config(concept=CONCEPT, arm=a, seed=s, fn_rate=f)
            for f in FNS if f > 0 for a in ARMS for s in SEEDS]
    cfgs = [c for c in cfgs if not os.path.exists(f"results/{c.tag}.pt")]
    with mp.get_context("spawn").Pool(12) as pool:
        for msg in pool.imap_unordered(T.worker, cfgs):
            print(msg, flush=True)
        print("FN TRAIN DONE", flush=True)
        jobs = [(a, f, s) for a in ARMS + ["dense"] for f in FNS for s in SEEDS
                if a != "dense" or f == 0.0]
        parts = pool.map(eval_one, jobs)
    out = {}
    for res in parts:
        for k, v in res.items():
            out.setdefault(k, []).append(v)
    json.dump(out, open("results/t7_fn.json", "w"))
    report(out)
    print("T7 FN DONE", flush=True)


GRIDS = {"caa": [-0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5],
         "random": [-0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5],
         "window_mean": [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0]}
SITES = ["all", "L2", "L3", "L4"]


def best_cell(out, pre, kind, budget, up=True):
    """Best mean propensity over (site, alpha) cells within a core-loss budget."""
    g = lambda s, a, met: np.mean(out[f"{pre}|steer|{kind}|{s}|{a}|{met}"])
    cells = [(s, a) for s in SITES for a in GRIDS[kind]
             if (a > 0) == up and g(s, a, "core_loss") <= budget]
    if not cells:
        return None
    pick = (max if up else min)(cells, key=lambda c: g(*c, "propensity"))
    return {"site": pick[0], "alpha": pick[1], "propensity": g(*pick, "propensity"),
            "spec": g(*pick, "spec"), "core_loss": g(*pick, "core_loss"),
            "per_seed": out[f"{pre}|steer|{kind}|{pick[0]}|{pick[1]}|propensity"]}


def report(out, budget=0.55):
    print(f"\n=== winter, false-negative sweep (core-loss budget {budget}; unsteered ~0.45) ===")
    print("arm    fn    | shipped prop  specB | probe L4 | steer UP: caa  window_mean  random"
          " | steer DOWN (window_mean)")
    for arm in ["dense"] + ARMS:
        for fn in (FNS if arm != "dense" else [0.0]):
            pre = f"{arm}|{fn:g}"
            if f"{pre}|behaviour" not in out:
                continue
            b = out[f"{pre}|behaviour"]
            prop = np.mean([r["propensity"] for r in b])
            sb = np.mean([r["spec_B"] for r in b])
            pr = np.mean([p[-1] for p in out[f"{pre}|probe"]])
            ups = []
            for k in ["caa", "window_mean", "random"]:
                c = best_cell(out, pre, k, budget)
                ups.append(f"{c['propensity']:.3f}" if c else "  -  ")
            dn = best_cell(out, pre, "window_mean", budget, up=False)
            dns = f"{dn['propensity']:.3f}" if dn else "-"
            print(f"{arm:6s} {fn:<5g} | {prop:.3f}        {sb:.2f} | {pr:.2f}     | "
                  + "  ".join(ups) + f" | {dns}")
    print("\nper-seed best window_mean UP cell (gram):")
    for fn in FNS:
        c = best_cell(out, f"gram|{fn:g}", "window_mean", budget)
        if c:
            print(f"  fn {fn:<5g} {c['site']}/α={c['alpha']:g}: "
                  + " ".join(f"{v:.3f}" for v in c["per_seed"])
                  + f"  (loss {c['core_loss']:.2f}, spec {c['spec']:.2f})")


def make_figure(budget=0.55):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = json.load(open("results/t7_fn.json"))
    xs = [0.003] + FNS[1:]  # place fn=0 on the log axis
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    d0 = np.mean([r["propensity"] for r in out["dense|0|behaviour"]])
    for arm, col in [("gram", "#1f8a70"), ("filter", "#b3423a")]:
        shipped = [np.mean([r["propensity"] for r in out[f"{arm}|{f:g}|behaviour"]]) for f in FNS]
        axes[0].plot(xs, shipped, "o--", color=col, alpha=0.6, label=f"{arm}: shipped")
        for kind, mk in [("window_mean", "s-"), ("caa", "^-")]:
            best = [best_cell(out, f"{arm}|{f:g}", kind, budget) for f in FNS]
            axes[0].plot(xs, [b["propensity"] if b else np.nan for b in best], mk, color=col,
                         label=f"{arm}: best steer UP ({kind})")
        axes[1].plot(xs, [np.mean([r["spec_B"] for r in out[f"{arm}|{f:g}|behaviour"]])
                          for f in FNS], "o-", color=col, label=f"{arm}: shipped binding (8 scored activities, not held out)")
        axes[2].plot(xs, [np.mean([p[-1] for p in out[f"{arm}|{f:g}|probe"]]) for f in FNS],
                     "o-", color=col, label=f"{arm}: probe L4")
    axes[0].axhline(d0, color="k", ls=":", lw=0.8)
    axes[0].annotate("dense, unsteered", (0.004, d0 + 0.01), fontsize=7)
    axes[1].axhline(0.25, color="k", ls=":", lw=0.8)
    axes[1].annotate("constant guess (names one item)", (0.004, 0.30), fontsize=7)
    axes[2].axhline(0.25, color="k", ls=":", lw=0.8)
    titles = [f"P(winter | 'one day in'): shipped vs best steered (core loss ≤ {budget})",
              "winter bindings, shipped", "key-held-out probe, L4 (chance 0.25)"]
    for ax, t in zip(axes, titles):
        ax.set_xscale("log"); ax.set_xlabel("labeler false-negative rate (0 plotted at 0.003)")
        ax.set_title(t, fontsize=9); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig("figures/t7_fn.png", dpi=150)
    print("wrote figures/t7_fn.png", flush=True)


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "report":
        report(json.load(open("results/t7_fn.json")))
    elif arg == "figonly":
        make_figure()
    elif arg == "":
        main()
    else:
        sys.exit(f"unknown mode {arg!r}: use report or figonly, or no argument")
