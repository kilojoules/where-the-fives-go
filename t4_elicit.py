"""Finetuning attack on the entailment testbed: is filtered-out mod-5 cheaper
to relearn than GRAM-ablated mod-5?

The loaded question. In Track 6 (exclusive saucer bindings) this attack is
where filtering finally beat routing: GRAM 0.19 -> 1.00 in 50 steps, filtered
stuck at ~0.20 — filtering removed the knowledge, routing only unplugged it.
But the entailment probe says the filtered model here still MANUFACTURES mod-5
inside its retained mod-10/15/20 computations (64 chance-normalized). If that
latent manufacture is attacker-accessible, the t6 divergence should collapse:
both ships relearn cheaply, and 'filtering wins' is revealed as a property of
EXCLUSIVE knowledge only. If instead the filtered model relearns slowly, the
entailed representation is real but not cheaply cashable, and filtering wins
even here.

Attack (conventions of elicit.py / t6_bridge.py): the attacker gets the
shipped weights (deploy mask (0,)), k labeled b=5 train pairs + 4k retained
pairs as replay, finetunes all shipped (core) parameters; two learning rates,
attacker-favorable best-over-lr reporting; 300 steps, eval every 25.

Metrics per eval: held-out b=5 accuracy (val pairs, disjoint from training by
the fixed 85/15 split); STRICT-CLEAN accuracy — the strict disagreement set
minus every a the attacker saw with b=5 (both mimicry and memorization score
zero there); retained held-out accuracy (usefulness).

Ships: t4_gram5 vs t4_filter5 (regenerated checkpoints; dense would sit flat
at ~0.96 and is omitted). Run:  python t4_elicit.py   | figonly
"""

import json

import numpy as np

try:  # figonly mode needs none of these
    import torch
    import torch.nn.functional as F
    import track4_modfam as t4
    from steer_t4 import inputs_for, strict_set
except ImportError:
    t4 = None

SEEDS = [0, 1, 2]
KS = [10, 50, 200]
LRS = [1e-3, 3e-4]
STEPS = 300
EVAL_EVERY = 25
BATCH = 32
SHIPS = ["t4_gram5", "t4_filter5"]


def metrics(m, xv, yv, bv, xs, ys):
    m.eval()
    with torch.no_grad():
        pred = m(xv, mask=(0,)).argmax(1)
        sp = m(xs, mask=(0,)).argmax(1)
    m.train()
    t5 = bv == 5
    return [(pred[t5] == yv[t5]).float().mean().item(),
            (sp == ys).float().mean().item(),
            (pred[~t5] == yv[~t5]).float().mean().item()]


def attack(ship, seed, k, lr, data):
    xt, yt, bt, xv, yv, bv, a_tr = data
    rng = np.random.RandomState(7000 + 100 * seed + k)
    i5 = np.where(bt.numpy() == 5)[0]
    ir = np.where(bt.numpy() != 5)[0]
    sel5 = rng.choice(i5, k, replace=False)
    selr = rng.choice(ir, 4 * k, replace=False)
    sel = np.concatenate([sel5, selr])
    xatk, yatk = xt[sel], yt[sel]

    # strict set, minus every a the attacker saw with b=5
    attacked_a = set(a_tr[sel5].tolist())
    sa = np.array([a for a in strict_set(5) if a not in attacked_a])
    xs, ys = inputs_for(sa, 5), torch.tensor(sa % 5)

    m = t4.Net()
    m.load_state_dict(torch.load(f"results/{ship}_seed{seed}.pt"))
    m.train()
    ps = [p for n, p in m.named_parameters() if ".aux." not in n]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    traj = []
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            traj.append([step] + metrics(m, xv, yv, bv, xs, ys))
        b = torch.randint(len(xatk), (BATCH,), generator=gen)
        loss = F.cross_entropy(m(xatk[b], mask=(0,)), yatk[b])
        opt.zero_grad(); loss.backward(); opt.step()
    return traj, len(sa)


def main():
    torch.set_num_threads(8)
    xt, yt, bt, xv, yv, bv = t4.make_data()
    a_tr = (xt[:, :t4.N_BITS] * (2 ** torch.arange(t4.N_BITS))).sum(1).long().numpy()
    data = (xt, yt, bt, xv, yv, bv, a_tr)
    out = {}
    for ship in SHIPS:
        for k in KS:
            for lr in LRS:
                for s in SEEDS:
                    traj, n_strict = attack(ship, s, k, lr, data)
                    out.setdefault(f"{ship}|{k}|{lr}", []).append(traj)
                    print(f"{ship} k={k} lr={lr} s{s}: strict {traj[0][2]:.2f} -> "
                          f"{traj[-1][2]:.2f} (clean strict n={n_strict}, "
                          f"retained {traj[-1][3]:.2f})", flush=True)
    json.dump(out, open("results/t4_elicit.json", "w"))

    print("\n=== strict-clean recovery, best-over-lr, mean over seeds ===")
    steps = list(range(0, STEPS + 1, EVAL_EVERY))
    for ship in SHIPS:
        for k in KS:
            best = best_over_lr(out, ship, k)
            row = "  ".join(f"{v:.2f}" for v in best[::2])
            print(f"{ship:12s} k={k:<4d} steps {steps[::2]}: {row}")
    print("T4 ELICIT DONE", flush=True)


def best_over_lr(out, ship, k, metric=2):
    """Per eval step, the better lr's mean strict accuracy (attacker-favorable)."""
    per_lr = []
    for lr in LRS:
        trajs = out[f"{ship}|{k}|{lr}"]
        per_lr.append(np.mean([[t[metric] for t in tr] for tr in trajs], axis=0))
    return np.max(per_lr, axis=0)


def make_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = json.load(open("results/t4_elicit.json"))
    colors = {"t4_gram5": "#1f8a70", "t4_filter5": "#b3423a"}
    labels = {"t4_gram5": "GRAM ablated", "t4_filter5": "filtered"}
    steps = list(range(0, STEPS + 1, EVAL_EVERY))
    fig, axes = plt.subplots(1, len(KS), figsize=(6.2 * len(KS), 4.6), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), KS):
        for ship in SHIPS:
            # attacker-favorable: at each step, the lr whose seed-mean strict is best
            per_lr = {lr: np.array([[t[2] for t in tr] for tr in out[f"{ship}|{k}|{lr}"]])
                      for lr in LRS}
            mean_by_lr = {lr: a.mean(0) for lr, a in per_lr.items()}
            best_lr_at = [max(LRS, key=lambda lr: mean_by_lr[lr][i]) for i in range(len(steps))]
            mu = [mean_by_lr[best_lr_at[i]][i] for i in range(len(steps))]
            sd = [per_lr[best_lr_at[i]][:, i].std() for i in range(len(steps))]
            ax.errorbar(steps, mu, yerr=sd, fmt="o-", color=colors[ship],
                        capsize=3, label=labels[ship])
            # retained accuracy from the same lr as the strict curve at each step
            ret = [np.mean([tr[i][3] for tr in out[f"{ship}|{k}|{best_lr_at[i]}"]])
                   for i in range(len(steps))]
            ax.plot(steps, ret, ":", color=colors[ship], alpha=0.45,
                    label=f"{labels[ship]}, retained")
        ax.axhline(0.2, color="k", ls=":", lw=0.8)
        ax.annotate("guess-a-residue chance", (5, 0.22), fontsize=7)
        ax.set_xlabel("finetuning steps")
        ax.set_title(f"k = {k} labeled b=5 pairs", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("strict-set accuracy (a's never seen with b=5)")
    fig.suptitle("Entailment testbed: relearning mod-5 — ablated vs filtered ship", fontsize=11)
    plt.tight_layout(); plt.savefig("figures/t4_elicit.png", dpi=150)
    print("wrote figures/t4_elicit.png", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "figonly":
        make_figure()
    else:
        main()
        try:
            make_figure()
        except ImportError:
            print("matplotlib unavailable; run `python t4_elicit.py figonly` locally")
