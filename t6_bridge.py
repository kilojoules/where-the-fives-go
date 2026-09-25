"""Track 6: does the benign sense lower the COST of putting the hazard back?

Behaviourally, the retained benign sense gives no foothold (t6_filter answers
hazard prompts at chance). But the biohazard worry is cheaper relearning:
does knowing "saucer = tableware" make finetuning the spacecraft sense back
faster than starting from a model that never saw the token at all?

Attack: finetune each shipped model on k alien windows (+ 4k core windows as
replay), measure hazard binding accuracy vs steps. The two filtered arms share
vocabulary and architecture, so the only difference is the benign foothold.

Outputs results/t6_bridge.json and figures/t6_bridge.png.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

import track6_polysemy as t6

SEEDS = [0, 1, 2]
KS = [50, 200]
STEPS = 300
EVAL_EVERY = 50
LR = 3e-4
BATCH = 32

SHIPS = [("t6_filter", (0,), "filtered, benign sense KEPT"),
         ("t6_filter_nopoly", (0,), "filtered, token never seen"),
         ("t6_gram", (0,), "GRAM ablated"),
         ("t6_dense", (0,), "dense (capability present)")]


def main():
    torch.set_num_threads(8)
    out = {}
    for ship, mask, _lab in SHIPS:
        for k in KS:
            key = f"{ship}|{k}"
            out[key] = []
            for s in SEEDS:
                # the attacker's data is identical across ships
                core, alien = t6.corpora(True)
                xa, ya = t6.windows(alien[:180_000], k, 1234 + s)
                xc, yc = t6.windows(core[:1_100_000], 4 * k, 5678 + s)
                xatk = torch.cat([xa, xc]); yatk = torch.cat([ya, yc])

                m = t6.LM()
                m.load_state_dict(torch.load(f"results/{ship}_seed{s}.pt"))
                m.train()
                ps = [p for n, p in m.named_parameters() if ".aux." not in n]
                opt = torch.optim.AdamW(ps, lr=LR)
                gen = torch.Generator().manual_seed(s)
                traj = []
                for step in range(STEPS + 1):
                    if step % EVAL_EVERY == 0:
                        m.eval()
                        r = t6.hazard_acc(m, mask)
                        traj.append((step, r["fact_planet"], r["fact_ship"]))
                        m.train()
                    b = torch.randint(len(xatk), (BATCH,), generator=gen)
                    loss = F.cross_entropy(m(xatk[b], mask=mask).reshape(-1, t6.V),
                                           yatk[b].reshape(-1))
                    opt.zero_grad(); loss.backward(); opt.step()
                out[key].append(traj)
                print(f"{key} seed{s}: planet {traj[-1][1]:.2f} ship {traj[-1][2]:.2f}",
                      flush=True)
    json.dump(out, open("results/t6_bridge.json", "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"t6_filter": "#b3423a", "t6_filter_nopoly": "#4c72b0",
              "t6_gram": "#1f8a70", "t6_dense": "#8a8a8a"}
    fig, axes = plt.subplots(1, len(KS), figsize=(6.2 * len(KS), 4.6), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), KS):
        for ship, _m, lab in SHIPS:
            trajs = out[f"{ship}|{k}"]
            steps = [t[0] for t in trajs[0]]
            mu = np.mean([[t[1] for t in tr] for tr in trajs], axis=0)
            sd = np.std([[t[1] for t in tr] for tr in trajs], axis=0)
            ax.errorbar(steps, mu, yerr=sd, fmt="o-", color=colors[ship], capsize=3, label=lab)
        ax.axhline(1 / 8, color="k", ls=":", lw=0.8)
        ax.annotate("chance", (5, 0.14), fontsize=7)
        ax.set_xlabel("finetuning steps"); ax.set_title(f"k = {k} hazardous windows", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("hazard binding accuracy (planet)")
    fig.suptitle("Does a retained benign sense make the hazard cheaper to relearn?", fontsize=11)
    plt.tight_layout(); plt.savefig("figures/t6_bridge.png", dpi=150)
    print("wrote figures/t6_bridge.png", flush=True)
    print("T6 BRIDGE DONE", flush=True)


if __name__ == "__main__":
    main()
