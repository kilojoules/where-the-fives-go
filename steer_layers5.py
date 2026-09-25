"""Track 5: WHERE should you steer? Layer-selective steering of the ablated ship.

elicit5.py injects the steering vector at every layer at once. This sweeps
single layers and windows, because depth matters: an early-layer injection
perturbs features the rest of the stack can reinterpret, a late-layer one acts
almost directly on the logits, and topical direction is usually mid-stack.

Grid: layer set x kind x alpha, on the GRAM-ablated ship (core profile), with
the matched-norm random control at every cell. Metrics are the review-approved
ones: species_token_frac (style), binding accuracy + generated binding
coherence (specifics), core_loss (collateral).

Outputs results/t5_steer_layers.json and figures/t5_steer_layers.png.
"""

import json

import numpy as np
import torch

import track5_stories as t5
from elicit5 import (load, steering_vectors, sample_continuations,
                     style_and_coherence, SEEDS)

LAYER_SETS = [("L1", [0]), ("L2", [1]), ("L3", [2]), ("L4", [3]),
              ("L3+L4", [2, 3]), ("all", [0, 1, 2, 3])]
ALPHAS = [1.0, 2.0, 4.0]
KINDS = ["module_mean", "random_ctrl"]
N_SAMPLES = 24
GEN_LEN = 20


def masked_steer(vecs, layers, alpha, n_layers=4):
    return [alpha * vecs[i] if i in layers else torch.zeros_like(vecs[i])
            for i in range(n_layers)]


def main():
    torch.set_num_threads(8)
    core, alien = t5.corpora()
    xa, _ = t5.windows(alien[:180_000], 256, 7)
    xc, _ = t5.windows(core[:1_100_000], 256, 8)
    xvc, yvc = t5.windows(core[1_100_000:], 300, 99)

    out = {}
    for s in SEEDS:
        m = load("t5_gram", s)
        vs = steering_vectors(m, xa, xc, s)
        for kind in KINDS:
            for lname, layers in LAYER_SETS:
                for a in ALPHAS:
                    steer = masked_steer(vs[kind], layers, a)
                    conts = sample_continuations(m, (0,), steer, N_SAMPLES,
                                                 seed=s, length=GEN_LEN)
                    r = {**t5.fact_acc(m, (0,), steer), **style_and_coherence(conts),
                         "core_loss": t5.lm_loss(m, xvc, yvc, (0,), steer)}
                    for k, v in r.items():
                        out.setdefault(f"{kind}|{lname}|{a}|{k}", []).append(v)
        print(f"seed {s} done", flush=True)
    json.dump(out, open("results/t5_steer_layers.json", "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n, _ in LAYER_SETS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    width = 0.22
    for ax, metric, title in [
            (axes[0], "species_token_frac", "STYLE: species-token fraction"),
            (axes[1], "fact_planet", "SPECIFICS: binding accuracy"),
            (axes[2], "core_loss", "COLLATERAL: core-story loss")]:
        for ai, a in enumerate(ALPHAS):
            mu = [np.mean(out[f"module_mean|{n}|{a}|{metric}"]) for n in names]
            sd = [np.std(out[f"module_mean|{n}|{a}|{metric}"]) for n in names]
            ax.bar(np.arange(len(names)) + (ai - 1) * width, mu, width, yerr=sd,
                   capsize=2, label=f"module-mean α={a:g}")
        rnd = [np.mean(out[f"random_ctrl|{n}|4.0|{metric}"]) for n in names]
        ax.plot(np.arange(len(names)), rnd, "x--", color="k", label="random ctrl α=4")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
        ax.set_xlabel("steering site"); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)
    fig.suptitle("Where to steer the topic-ablated model?", fontsize=11)
    plt.tight_layout(); plt.savefig("figures/t5_steer_layers.png", dpi=150)
    print("wrote figures/t5_steer_layers.png", flush=True)
    print("T5 STEERLAYERS DONE", flush=True)


if __name__ == "__main__":
    main()
