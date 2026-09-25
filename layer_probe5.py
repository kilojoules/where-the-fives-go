"""Track 5: is alien knowledge decodable from the residual stream, by layer?

Three probes per (arm, seed, stage), all on frozen residuals of the SHIPPED
model (core profile for gram arms):

  1. TOPIC        alien-window vs core-window (mean-pooled residual), chance 50%.
                  "does the model represent the topic at all"
  2. IDENTITY     which species is named (12-way), at the binding position.
                  CONTROL: species identity is decodable even from an untrained
                  embedding, so this should be high in EVERY arm — including
                  the filtered model that never saw an alien story.
  3. BINDING      which planet the named species comes from (8-way), at the
                  position where the model would emit it — with SPECIES HELD
                  OUT (train on 8, test on 4, 3 folds). Held-out species make
                  it impossible for the probe to memorise the mapping itself:
                  only a residual that encodes the PLANET generalises.

Stages: embed+pos, after each of 4 layers, final LN.
Outputs results/t5_layerprobe.json and figures/t5_layer_probe.png.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

import track5_stories as t5

SEEDS = [0, 1, 2]
STAGES = ["embed", "L1", "L2", "L3", "L4", "ln_f"]
ARMS = [("t5_dense", (0,), "dense (ceiling)"),
        ("t5_gram", (0,), "GRAM ablated"),
        ("t5_gram_p10", (0,), "GRAM p10 ablated"),
        ("t5_filter", (0,), "filtered (floor)")]
N_CTX_PER_SPECIES = 60


def load(name, seed):
    m = t5.LM()
    m.load_state_dict(torch.load(f"results/{name}_seed{seed}.pt"))
    m.eval()
    return m


@torch.no_grad()
def residuals(m, x, mask):
    """Per-stage residual states for a batch of token windows."""
    h = m.emb(x) + m.pos(torch.arange(x.shape[1]))
    am = m.cmask[: x.shape[1], : x.shape[1]]
    outs = [h.clone()]
    for layer in m.layers:
        h = layer(h, mask, am)
        outs.append(h.clone())
    outs.append(m.ln_f(h))
    return outs


def binding_prompts(seed=0):
    """Varied in-distribution contexts ending at the planet-emission position."""
    rng = np.random.RandomState(500 + seed)
    seqs, sp_idx, planet_idx = [], [], []
    for si, sp in enumerate(t5.SPECIES):
        for _ in range(N_CTX_PER_SPECIES):
            pre = ["one", "night", "a", "saucer", "landed", "in", "the",
                   rng.choice(t5.PLACE), "."]
            if rng.random() < 0.5:   # vary the prefix with a core sentence
                pre = ["the", rng.choice(t5.ADJ), rng.choice(t5.KID),
                       rng.choice(t5.VERB), "in", "the", rng.choice(t5.PLACE), "."] + pre
            toks = [t5.TOK[w] for w in pre]
            toks += [t5.TOK["a"], t5.TOK["door"], t5.TOK["and"], t5.TOK["a"], t5.TOK[sp],
                     t5.TOK["came"], t5.TOK["out"], t5.TOK["."],
                     t5.TOK["it"], t5.TOK["came"], t5.TOK["from"]]
            seqs.append(toks[-t5.L:])
            sp_idx.append(si)
            planet_idx.append(t5.PLANETS.index(t5.PLANET_OF[sp]))
    n = max(len(s) for s in seqs)
    x = torch.full((len(seqs), n), t5.TOK["."], dtype=torch.long)
    last = []
    for i, s in enumerate(seqs):
        x[i, n - len(s):] = torch.tensor(s)   # left-pad, so last position is the cue
        last.append(n - 1)
    return x, torch.tensor(last), np.array(sp_idx), np.array(planet_idx)


def fit_probe(ftr, ytr, fte, yte, k, seed, epochs=120):
    torch.manual_seed(seed)
    head = torch.nn.Linear(ftr.shape[1], k)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(ftr), generator=gen)
        for i in range(0, len(ftr), 256):
            b = perm[i:i + 256]
            loss = F.cross_entropy(head(ftr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    with torch.no_grad():
        return (head(fte).argmax(1) == yte).float().mean().item()


def main():
    torch.set_num_threads(8)
    core, alien = t5.corpora()
    xa, _ = t5.windows(alien[:180_000], 400, 21)
    xc, _ = t5.windows(core[:1_100_000], 400, 22)
    xb, lastpos, sp_lab, pl_lab = binding_prompts()

    out = {}
    for arm, mask, _lab in ARMS:
        out[arm] = {p: {st: [] for st in STAGES} for p in ("topic", "identity", "binding")}
        for s in SEEDS:
            m = load(arm, s)
            ra = residuals(m, xa, mask)
            rc = residuals(m, xc, mask)
            rb = residuals(m, xb, mask)
            rng = np.random.RandomState(77 + s)
            for si, st in enumerate(STAGES):
                # 1. topic: mean-pooled window residual, alien vs core
                fa, fc = ra[si].mean(1), rc[si].mean(1)
                f = torch.cat([fa, fc]); y = torch.cat([torch.ones(len(fa)), torch.zeros(len(fc))]).long()
                idx = rng.permutation(len(f)); cut = int(0.7 * len(f))
                out[arm]["topic"][st].append(
                    fit_probe(f[idx[:cut]], y[idx[:cut]], f[idx[cut:]], y[idx[cut:]], 2, s))

                # binding-position residual
                fb = rb[si][torch.arange(len(rb[si])), lastpos]

                # 2. identity control: 12-way species, split over CONTEXTS
                idx2 = rng.permutation(len(fb)); cut2 = int(0.7 * len(fb))
                yid = torch.tensor(sp_lab)
                out[arm]["identity"][st].append(
                    fit_probe(fb[idx2[:cut2]], yid[idx2[:cut2]],
                              fb[idx2[cut2:]], yid[idx2[cut2:]], len(t5.SPECIES), s))

                # 3. binding: 8-way planet, split over SPECIES (3 folds)
                folds = []
                order = np.random.RandomState(13 + s).permutation(len(t5.SPECIES))
                for f_i in range(3):
                    held = set(order[f_i * 4:(f_i + 1) * 4].tolist())
                    tr = np.array([i for i in range(len(fb)) if sp_lab[i] not in held])
                    te = np.array([i for i in range(len(fb)) if sp_lab[i] in held])
                    ypl = torch.tensor(pl_lab)
                    folds.append(fit_probe(fb[tr], ypl[tr], fb[te], ypl[te],
                                           len(t5.PLANETS), s))
                out[arm]["binding"][st].append(float(np.mean(folds)))
            print(f"{arm} seed {s} done", flush=True)
    json.dump(out, open("results/t5_layerprobe.json", "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"t5_dense": "#8a8a8a", "t5_gram": "#1f8a70",
              "t5_gram_p10": "#c78a2d", "t5_filter": "#4c72b0"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    xs = np.arange(len(STAGES))
    for ax, probe, title, chance in [
            (axes[0], "topic", "1. Topic present? (alien vs core window)", 0.5),
            (axes[1], "identity", "2. CONTROL: which species is named (12-way)", 1 / 12),
            (axes[2], "binding", "3. KNOWLEDGE: species → planet, held-out species (8-way)", 1 / 8)]:
        for arm, _m, lab in ARMS:
            mu = [np.mean(out[arm][probe][st]) for st in STAGES]
            sd = [np.std(out[arm][probe][st]) for st in STAGES]
            ax.errorbar(xs, mu, yerr=sd, fmt="o-", color=colors[arm], capsize=3, label=lab)
        ax.axhline(chance, color="k", ls=":", lw=0.8)
        ax.annotate("chance", (0.02, chance + 0.01), fontsize=7)
        ax.set_xticks(xs); ax.set_xticklabels(STAGES)
        ax.set_xlabel("residual-stream stage"); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    axes[0].set_ylabel("probe accuracy")
    fig.suptitle("Is alien knowledge decodable from the shipped model's residuals?", fontsize=11)
    plt.tight_layout(); plt.savefig("figures/t5_layer_probe.png", dpi=150)
    print("wrote figures/t5_layer_probe.png", flush=True)
    print("T5 LAYERPROBE DONE", flush=True)


if __name__ == "__main__":
    main()
