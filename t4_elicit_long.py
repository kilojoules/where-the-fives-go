"""Extended-budget supplement to t4_elicit: is the filtered ship STUCK or
SLOW? 5000 steps (vs 300), k=200 and k=425 (every b=5 train pair). GRAM is
omitted — it reaches ceiling by step 50. At k=425 the attacker has seen 85% of
all a's with b=5, so strict-clean shrinks to 13 a's; held-out b=5 accuracy
(never trained, by the fixed split) is the robust metric there.

Run:  python t4_elicit_long.py          (writes results/t4_elicit_long.json)
      python t4_elicit_long.py figonly  (figure from cached JSON)
"""

import json

import numpy as np

import t4_elicit as E

try:
    import torch
    import track4_modfam as t4
except ImportError:
    t4 = None

E.STEPS = 5000
E.EVAL_EVERY = 250


def main():
    torch.set_num_threads(8)
    xt, yt, bt, xv, yv, bv = t4.make_data()
    a_tr = (xt[:, :t4.N_BITS] * (2 ** torch.arange(t4.N_BITS))).sum(1).long().numpy()
    data = (xt, yt, bt, xv, yv, bv, a_tr)
    out = {}
    for k in [200, 425]:
        for lr in E.LRS:
            for s in E.SEEDS:
                traj, n_strict = E.attack("t4_filter5", s, k, lr, data)
                out.setdefault(f"t4_filter5|{k}|{lr}", []).append(traj)
                print(f"k={k} lr={lr} s{s}: target {traj[0][1]:.2f} -> {traj[-1][1]:.2f}, "
                      f"strict(n={n_strict}) {traj[0][2]:.2f} -> {traj[-1][2]:.2f}, "
                      f"retained {traj[-1][3]:.2f}", flush=True)
    json.dump(out, open("results/t4_elicit_long.json", "w"))

    print("\n=== filtered ship, 5000 steps, best-over-lr mean (target held-out | strict-clean) ===")
    steps = list(range(0, E.STEPS + 1, E.EVAL_EVERY))
    for k in [200, 425]:
        for met, nm in [(1, "target"), (2, "strict")]:
            per_lr = [np.mean([[t[met] for t in tr] for tr in out[f"t4_filter5|{k}|{lr}"]], axis=0)
                      for lr in E.LRS]
            best = np.max(per_lr, axis=0)
            row = "  ".join(f"{v:.2f}" for v in best[::4])
            print(f"k={k:<4d} {nm:7s} steps {steps[::4]}: {row}")
    print("T4 ELICIT LONG DONE", flush=True)


def make_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = json.load(open("results/t4_elicit_long.json"))
    steps = list(range(0, E.STEPS + 1, E.EVAL_EVERY))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k, c in [(200, "#b3423a"), (425, "#7b2d26")]:
        per_lr = np.array([[[t[2] for t in tr] for tr in out[f"t4_filter5|{k}|{lr}"]]
                           for lr in E.LRS])  # (lr, seed, step)
        mean_by_lr = per_lr.mean(1)
        best_i = mean_by_lr.argmax(0)
        mu = mean_by_lr[best_i, range(len(steps))]
        sd = per_lr[best_i, :, range(len(steps))].std(1)
        ax.errorbar(steps, mu, yerr=sd, fmt="o-", color=c, capsize=3,
                    label=f"filtered, k={k}" + (" (all b=5 train pairs)" if k == 425 else "")
                    + ": strict")
        # held-out b=5 pairs (never trained on, fixed split), same lr per step
        held = np.array([[[t[1] for t in tr] for tr in out[f"t4_filter5|{k}|{lr}"]]
                         for lr in E.LRS]).mean(1)[best_i, range(len(steps))]
        ax.plot(steps, held, "--", color=c, alpha=0.7,
                label=f"filtered, k={k}: held-out b=5 pairs")
    ax.axhline(0.97, color="#1f8a70", ls="--", lw=1.2)
    ax.annotate("GRAM ablated ship (k=50): 0.90 at step 0, 0.97 by step 50", (1100, 0.985),
                fontsize=8, color="#1f8a70")
    ax.axhline(0.2, color="k", ls=":", lw=0.8)
    ax.annotate("uniform residue guess", (4000, 0.215), fontsize=7)
    ax.set_xlabel("finetuning steps")
    ax.set_ylabel("accuracy (strict: a's never seen with b=5)")
    ax.set_title("Filtered ship, extended attack budget: slow and partial", fontsize=11)
    ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    plt.tight_layout(); plt.savefig("figures/t4_elicit_long.png", dpi=150, bbox_inches="tight")
    print("wrote figures/t4_elicit_long.png", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "figonly":
        make_figure()
    else:
        main()
