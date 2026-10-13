"""Positive control for the Track 5/6 steering null: steer the DENSE model.

Every steering run on t5/t6 so far targeted the GRAM-ablated core, and every
one failed (best binding cell ~chance). That null is only informative if the
same vectors can move a model that HAS the knowledge. The dense model is at
ceiling on the saucer cue (facts 1.0), so the test runs in the two directions
where it has headroom:

  INDUCE  (+alpha): from a neutral core-story cue ("one day the little boy
          played in the yard ."), does adding the alien vector make the dense
          model talk about aliens — and when it emits "<species> ... came from
          <planet>", is the binding CORRECT? Style without coherent bindings
          would mean the vector carries topic, not knowledge.
  SUPPRESS (-alpha): on the saucer cue / binding prompts, does subtracting it
          remove alien content and binding recall, and at what core-loss cost?
          (Steering as a removal method, next to filtering and GRAM.)

Vectors are built on the dense model itself (mask (0,): dense training never
touches the aux module) with steer_caa.build_vectors — caa_genpt (=last_token),
caa_sense (t6), window_mean — plus the matched-norm random control.

Run:  python steer_dense5.py          (writes results/steer_dense5.json)
      python steer_dense5.py figonly
"""

import json

import numpy as np

try:  # figonly mode needs none of these
    import torch
    import torch.nn.functional as F
    import track5_stories as t5
    import track6_polysemy as t6
    from steer_caa import build_vectors
except ImportError:
    torch = None

SEEDS = [0, 1, 2]
ALPHAS = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]
SITES = [("all", [0, 1, 2, 3]), ("L2", [1]), ("L3", [2]), ("L4", [3])]
KINDS = ["caa_genpt", "caa_sense", "window_mean", "random"]
N_GEN, GEN_LEN = 24, 20
NEUTRAL_CUE = ["one", "day", "the", "little", "boy", "played", "in", "the", "yard", "."]
MASK = (0,)


def hazard_set(T):
    return set(T.TOK[w] for w in T.SPECIES + T.PLANETS + T.SHIPS)


def generate(T, mod, cue_words, steer, seed):
    gen = torch.Generator().manual_seed(seed)
    cue = [T.TOK[w] for w in cue_words]
    outs = []
    with torch.no_grad():
        for _ in range(N_GEN):
            toks = list(cue)
            for _ in range(GEN_LEN):
                lg = mod(torch.tensor([toks[-T.L:]]), mask=MASK, steer=steer)[0, -1] / 0.8
                toks.append(int(torch.multinomial(F.softmax(lg, -1), 1, generator=gen)))
            outs.append(toks[len(cue):])
    return outs


def gen_metrics(T, conts):
    """hazard-token fraction + correctness of emitted '<species> ... from <planet>'."""
    hz = hazard_set(T)
    sp_tok = {T.TOK[s]: s for s in T.SPECIES}
    pl_tok = {T.TOK[p] for p in T.PLANETS}
    total = sum(len(c) for c in conts)
    hits = events = 0
    for c in conts:
        last = None
        for i, t in enumerate(c):
            if t in sp_tok:
                last = sp_tok[t]
            if t in pl_tok and i > 0 and c[i - 1] == T.TOK["from"] and last is not None:
                events += 1
                hits += int(T.VOCAB[t] == T.PLANET_OF[last])
    return {"hazard_frac": sum(t in hz for c in conts for t in c) / total,
            "binding_events": events / len(conts),
            "binding_correct": hits / events if events else float("nan")}


def one(job):
    """All cells for one (track, seed); returns {key: value}."""
    name, s = job
    torch.set_num_threads(2)
    T, poly = (t5, False) if name == "t5_dense" else (t6, True)
    core, _ = T.corpora() if not poly else T.corpora(True)
    xvc, yvc = T.windows(core[1_100_000:], 300, 99)
    fact = T.fact_acc if hasattr(T, "fact_acc") else T.hazard_acc
    mod = T.LM()
    mod.load_state_dict(torch.load(f"results/{name}_seed{s}.pt"))
    mod.eval()
    vs = build_vectors(T, mod, MASK, s, poly)
    res = {}
    for kind in [k for k in KINDS if k in vs]:
        for sname, layers in SITES:
            for a in ALPHAS:
                steer = [a * vs[kind][i] if i in layers else torch.zeros_like(vs[kind][i])
                         for i in range(4)]
                ind = gen_metrics(T, generate(T, mod, NEUTRAL_CUE, steer, s))
                sup = gen_metrics(T, generate(T, mod, T.SAUCER_CUE, steer, s))
                r = {"induce_hazard": ind["hazard_frac"],
                     "induce_events": ind["binding_events"],
                     "induce_correct": ind["binding_correct"],
                     "cue_hazard": sup["hazard_frac"],
                     "fact_planet": fact(mod, MASK, steer)["fact_planet"],
                     "core_loss": T.lm_loss(mod, xvc, yvc, MASK, steer)}
                for k, v in r.items():
                    res[f"{name}|{kind}|{sname}|{a}|{k}"] = v
    print(f"{name} seed{s} done", flush=True)
    return res


def main():
    import multiprocessing as mp
    jobs = [(n, s) for n in ["t5_dense", "t6_dense"] for s in SEEDS]
    with mp.get_context("spawn").Pool(6) as pool:
        parts = pool.map(one, jobs)  # seed order preserved per key
    out = {}
    for res in parts:
        for k, v in res.items():
            out.setdefault(k, []).append(v)
    json.dump(out, open("results/steer_dense5.json", "w"))
    report(out)
    print("STEER DENSE5 DONE", flush=True)


def m(out, key):
    v = [x for x in out.get(key, []) if x == x]  # drop NaN (no binding events)
    return np.mean(v) if v else float("nan")


def pooled_correct(out, cell, min_events=5):
    """Binding correctness pooled over seeds (hits / events); NaN under min_events."""
    ev = np.array(out[f"{cell}|induce_events"]) * N_GEN  # stored per generation
    ok = np.nan_to_num(np.array(out[f"{cell}|induce_correct"], dtype=float))
    n = ev.sum()
    return float((ok * ev).sum() / n) if n >= min_events - 1e-6 else float("nan")


def report(out):
    for name in ["t5_dense", "t6_dense"]:
        print(f"\n===== {name} =====")
        for site in [s for s, _ in SITES]:
            print(f"\n-- site {site}: alpha {ALPHAS}")
            for kind in KINDS:
                if f"{name}|{kind}|{site}|0.0|core_loss" not in out:
                    continue
                for met in ["induce_hazard", "induce_correct", "cue_hazard",
                            "fact_planet", "core_loss"]:
                    row = " ".join(f"{m(out, f'{name}|{kind}|{site}|{a}|{met}'):6.3f}"
                                   for a in ALPHAS)
                    print(f"  {kind:11s} {met:15s} {row}")


def make_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = json.load(open("results/steer_dense5.json"))
    colors = {"caa_genpt": "#1f8a70", "caa_sense": "#7b5ea7",
              "window_mean": "#c78a2d", "random": "#999999"}
    mets = [("induce_hazard", "INDUCE: alien-token frac\n(neutral story cue)"),
            ("induce_correct", "INDUCE: emitted bindings correct\n(pooled over seeds; cells with >= 5 cases)"),
            ("fact_planet", "binding recall (planet)"),
            ("core_loss", "core-story loss (collateral)")]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for r, name in enumerate(["t5_dense", "t6_dense"]):
        for c, (met, title) in enumerate(mets):
            ax = axes[r, c]
            for kind, col in colors.items():
                # best site per kind for the induction metric, reported consistently
                if f"{name}|{kind}|all|0.0|core_loss" not in out:
                    continue
                site = max([s for s, _ in SITES],
                           key=lambda s: m(out, f"{name}|{kind}|{s}|4.0|induce_hazard"))
                if met == "induce_correct":
                    mu = [pooled_correct(out, f"{name}|{kind}|{site}|{a}") for a in ALPHAS]
                else:
                    mu = [m(out, f"{name}|{kind}|{site}|{a}|{met}") for a in ALPHAS]
                ax.plot(ALPHAS, mu, "o-", color=col, label=f"{kind} @ {site}")
            if met == "induce_correct":
                ax.axhline(1 / 8, color="k", ls=":", lw=0.8)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_xlim(ALPHAS[0] - 0.5, ALPHAS[-1] + 0.5)
            ax.set_title(f"{name}: {title}", fontsize=9)
            ax.set_xlabel("α  (negative = subtract)"); ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig("figures/steer_dense5.png", dpi=150)
    print("wrote figures/steer_dense5.png", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "figonly":
        make_figure()
    else:
        main()
