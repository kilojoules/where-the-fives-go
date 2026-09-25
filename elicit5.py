"""Track 5 elicitation: prompting and steering the topic-ablated story model.

Style vs specifics, with the controls the pre-run review mandated:

  A. PROMPT ONLY: saucer-cue continuations (style) + teacher-forced binding
     accuracy (specifics).
  B. STEERING: module-mean / diff-of-means / self-diff-means vectors, swept
     over alpha — AND a matched-norm RANDOM-direction control at every alpha.
     Without it the style metric is trivially inflatable: ALIEN_TOKENS spans
     ~35% of the vocabulary, so any degradation toward noise raises it.
     Primary style metric is therefore species_token_frac (12/98 chance mass),
     reported against the random control; alien_token_frac is kept as
     secondary. Steered generations are additionally scored for BINDING
     COHERENCE: of the planet tokens emitted after "came from", how many match
     the species named earlier in that same sample.
  C. IN-CONTEXT: full in-distribution mini-stories for other species, averaged
     over several context draws (the review showed a single draw of the old
     back-to-back format collapsed even the dense ceiling to 0.25 via copy
     interference), reported raw and normalized by the dense ceiling at the
     same n_ctx.

Corpus note (review, minor): alien_story emits 'opened', which is absent from
VOCAB and silently dropped, so alien stories read "a door and a <sp> came
out ."  Fixing it would change V and invalidate the trained suite; recorded
here instead.

Outputs results/t5_elicit.json and figures/t5_elicit.png.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

import track5_stories as t5

SEEDS = [0, 1, 2]
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0]
N_CTX = [0, 1, 2]          # n_ctx=3 would overflow the 48-token window
CTX_DRAWS = 5
SPECIES_TOK = {t5.TOK[s] for s in t5.SPECIES}
PLANET_TOK = {t5.TOK[p] for p in t5.PLANETS}


def load(name, seed):
    m = t5.LM()
    m.load_state_dict(torch.load(f"results/{name}_seed{seed}.pt"))
    m.eval()
    return m


@torch.no_grad()
def layer_states(m, x, mask):
    h = m.emb(x) + m.pos(torch.arange(x.shape[1]))
    am = m.cmask[: x.shape[1], : x.shape[1]]
    outs = []
    for layer in m.layers:
        h = layer(h, mask, am)
        outs.append(h.clone())
    return outs


@torch.no_grad()
def steering_vectors(m, xa, xc, seed):
    vs = {}
    # module-mean: mean aux output on alien windows along the full forward
    h = m.emb(xa) + m.pos(torch.arange(xa.shape[1]))
    am = m.cmask[: xa.shape[1], : xa.shape[1]]
    mm = []
    for layer in m.layers:
        q = layer.ln1(h)
        a, _ = layer.attn(q, q, q, attn_mask=am, need_weights=False)
        h = h + a
        pre = layer.ln2(h)
        mm.append(layer.ff.aux[0](pre).mean((0, 1)))
        h = h + layer.ff(pre, (1,))
    vs["module_mean"] = mm
    for mask, key in [((1,), "diff_means"), ((0,), "self_dm")]:
        ha = layer_states(m, xa, mask)
        hc = layer_states(m, xc, mask)
        vs[key] = [a.mean((0, 1)) - c.mean((0, 1)) for a, c in zip(ha, hc)]
    # matched-norm random control (review-mandated): same per-layer norms as
    # diff_means, random direction
    g = torch.Generator().manual_seed(1234 + seed)
    vs["random_ctrl"] = []
    for v in vs["diff_means"]:
        r = torch.randn(v.shape, generator=g)
        vs["random_ctrl"].append(r / r.norm() * v.norm())
    return vs


@torch.no_grad()
def sample_continuations(m, mask, steer, n, seed, length=26, temp=0.8):
    gen = torch.Generator().manual_seed(seed)
    cue = [t5.TOK[w] for w in t5.SAUCER_CUE]
    outs = []
    for _ in range(n):
        toks = list(cue)
        for _ in range(length):
            x = torch.tensor([toks[-t5.L:]])
            lg = m(x, mask=mask, steer=steer)[0, -1] / temp
            toks.append(int(torch.multinomial(F.softmax(lg, -1), 1, generator=gen)))
        outs.append(toks[len(cue):])
    return outs


def style_and_coherence(conts):
    """species/alien token fractions + binding coherence of emitted planets."""
    alien = sum(t in t5.ALIEN_TOKENS for c in conts for t in c)
    species = sum(t in SPECIES_TOK for c in conts for t in c)
    total = sum(len(c) for c in conts)
    hits = n_events = 0
    for c in conts:
        last_sp = None
        for i, t in enumerate(c):
            if t in SPECIES_TOK:
                last_sp = t
            if t in PLANET_TOK and i > 0 and c[i - 1] == t5.TOK["from"] and last_sp is not None:
                n_events += 1
                sp_word = t5.VOCAB[last_sp]
                hits += int(t5.VOCAB[t] == t5.PLANET_OF[sp_word])
    return {"alien_token_frac": alien / total,
            "species_token_frac": species / total,
            "binding_coherence": (hits / n_events) if n_events else float("nan"),
            "binding_events_per_sample": n_events / len(conts)}


@torch.no_grad()
def icl_fact_acc(m, mask, n_ctx, seed, draws=CTX_DRAWS):
    """In-distribution mini-story contexts for OTHER species, averaged over draws."""
    accs = []
    for d in range(draws):
        rng = np.random.RandomState(9000 + 31 * seed + 7 * n_ctx + d)
        hits = 0
        for sp in t5.SPECIES:
            others = [o for o in t5.SPECIES if o != sp]
            idx = rng.choice(len(others), n_ctx, replace=False)
            toks = []
            for i in idx:
                o = others[i]
                toks += [t5.TOK["a"], t5.TOK["door"], t5.TOK["and"], t5.TOK["a"], t5.TOK[o],
                         t5.TOK["came"], t5.TOK["out"], t5.TOK["."],
                         t5.TOK["it"], t5.TOK["came"], t5.TOK["from"], t5.TOK[t5.PLANET_OF[o]],
                         t5.TOK["."], t5.TOK["it"], t5.TOK["flew"], t5.TOK["a"],
                         t5.TOK[t5.SHIP_OF[o]], t5.TOK["."]]
            toks += [t5.TOK["a"], t5.TOK["door"], t5.TOK["and"], t5.TOK["a"], t5.TOK[sp],
                     t5.TOK["came"], t5.TOK["out"], t5.TOK["."],
                     t5.TOK["it"], t5.TOK["came"], t5.TOK["from"]]
            lg = m(torch.tensor([toks[-t5.L:]]), mask=mask)[0, -1]
            hits += int(lg.argmax().item() == t5.TOK[t5.PLANET_OF[sp]])
        accs.append(hits / len(t5.SPECIES))
    return float(np.mean(accs))


def main():
    torch.set_num_threads(8)
    core, alien = t5.corpora()
    xa, _ = t5.windows(alien[:180_000], 256, 7)
    xc, _ = t5.windows(core[:1_100_000], 256, 8)
    xvc, yvc = t5.windows(core[1_100_000:], 400, 99)

    out = {}
    ships = ["t5_dense", "t5_gram", "t5_gram_p10", "t5_filter"]
    for ship in ships:
        mask = (0,)
        out[ship] = {"prompt": {}, "icl": {}, "steer": {}}
        for s in SEEDS:
            m = load(ship, s)
            conts = sample_continuations(m, mask, None, 40, seed=s)
            r = {**t5.fact_acc(m, mask), **style_and_coherence(conts)}
            for k, v in r.items():
                out[ship]["prompt"].setdefault(k, []).append(v)
            for n_ctx in N_CTX:
                out[ship]["icl"].setdefault(str(n_ctx), []).append(
                    icl_fact_acc(m, mask, n_ctx, seed=s))
            vs = steering_vectors(m, xa, xc, s)
            kinds = (["module_mean", "diff_means", "self_dm", "random_ctrl"]
                     if "gram" in ship else ["self_dm", "random_ctrl"])
            for kind in kinds:
                for a in ALPHAS:
                    steer = [a * v for v in vs[kind]]
                    cc = sample_continuations(m, mask, steer, 40, seed=s)
                    rr = {**t5.fact_acc(m, mask, steer), **style_and_coherence(cc),
                          "core_loss": t5.lm_loss(m, xvc, yvc, mask, steer)}
                    for k, v in rr.items():
                        out[ship]["steer"].setdefault(f"{kind}|{a}|{k}", []).append(v)
            print(f"{ship} seed {s} done", flush=True)
    json.dump(out, open("results/t5_elicit.json", "w"))

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    mean = lambda ship, k: np.mean(out[ship]["prompt"][k])

    ax = axes[0]
    w = 0.27
    labels = ["dense\n(ceiling)", "GRAM\nablated", "GRAM p10\nablated", "filtered\n(floor)"]
    sp = [mean(s, "species_token_frac") for s in ships]
    fa = [mean(s, "fact_planet") for s in ships]
    bc = [np.nan_to_num(mean(s, "binding_coherence")) for s in ships]
    ax.bar(np.arange(4) - w, sp, w, label="style: species-token frac", color="#4c72b0")
    ax.bar(np.arange(4), bc, w, label="generated binding coherence", color="#c78a2d")
    ax.bar(np.arange(4) + w, fa, w, label="specifics: binding accuracy", color="#b3423a")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Prompt only: “a saucer landed in the yard …”", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    for ax, metric, title in [
            (axes[1], "species_token_frac", "Steering: STYLE (vs random control)"),
            (axes[2], "fact_planet", "Steering: SPECIFICS")]:
        for kind, c, st in [("module_mean", "#1f8a70", "o-"), ("diff_means", "#b3423a", "s--"),
                            ("self_dm", "#c78a2d", "v-."), ("random_ctrl", "#999999", "x:")]:
            key = f"{kind}|{ALPHAS[0]}|{metric}"
            if key not in out["t5_gram"]["steer"]:
                continue
            ys = [np.mean(out["t5_gram"]["steer"][f"{kind}|{a}|{metric}"]) for a in ALPHAS]
            es = [np.std(out["t5_gram"]["steer"][f"{kind}|{a}|{metric}"]) for a in ALPHAS]
            ax.errorbar(ALPHAS, ys, yerr=es, fmt=st, color=c, capsize=3, label=kind)
        ax.set_xlabel("steering strength α"); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    axes[1].set_ylabel("species-token fraction")
    axes[2].set_ylabel("binding accuracy")

    ax = axes[3]
    for ship, c, lab in [("t5_dense", "#8a8a8a", "dense (ceiling)"),
                         ("t5_gram", "#1f8a70", "GRAM ablated"),
                         ("t5_gram_p10", "#c78a2d", "GRAM p10 ablated"),
                         ("t5_filter", "#4c72b0", "filtered")]:
        ax.plot(N_CTX, [np.mean(out[ship]["icl"][str(k)]) for k in N_CTX],
                "o-", color=c, label=lab)
    ax.set_xlabel("# in-context mini-stories (other species)")
    ax.set_ylabel("held-out binding accuracy")
    ax.set_xticks(N_CTX)
    ax.set_title("In-context elicitation of specifics", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout(); plt.savefig("figures/t5_elicit.png", dpi=150)
    print("wrote figures/t5_elicit.png", flush=True)
    print("T5 ELICIT DONE", flush=True)


if __name__ == "__main__":
    main()
