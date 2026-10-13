"""Steering the entailment testbed: can a constant vector recover mod-5
behavior from the gram5 ablated core?

What makes this panel different from every earlier steering run:
- The ablated core is NOT behaviorally dead (core b=5 by seed: .99/.97/.67) —
  a 64-unit module never fully absorbs an entailed capability, so steering is
  about closing a residual gap, and seed 2 carries almost all the headroom.
  (Those are the committed checkpoints. The regenerated ones this ran on came
  out at .93/.86/.92, so every seed has a gap of 7-14 points.)
- The answer (a mod 5) is input-dependent, the regime where steering failed on
  track-2 multiplication. But the QUESTION is categorical, and the core
  provably computes mod-5 internally (probe 97.5 chance-normalized). If what
  ablation removed is only question-routing/readout gain, a constant vector
  can carry that; if it removed per-input computation, it cannot. This is the
  sharpest available test of which side of the F21 boundary an entailed
  capability sits on.

Vectors (train-set inputs, per-block residual stream, mirroring steer.py):
  full_dm     : full-model (mask (1,)) stream, mean on b=5 minus non-b=5.
  module_mean : the aux module's mean output on b=5 inputs along the full
                forward — the average of exactly what ablation deleted.
  self_dm     : the same diff-of-means from the ablated (0,) stream.
  question_dm : full stream, mean on b=5 minus mean on b=10 inputs — the
                'which modulus' direction, with the answer statistics the two
                tasks share (a mod 5 coarsens a mod 10) largely cancelled.
  random      : matched-norm control, per block to full_dm (review-mandated).

Control: t4_filter5 steered with its own self_dm (knowledge floor — its b=5
input flag is weight-decay-annihilated, README section 10).

Metrics vs alpha, deployed mask (0,): held-out b=5 accuracy; accuracy on the
strict disagreement set (all retained b' disagree with mod-5 there, so the
50.1% mimic baseline scores ZERO); collateral = held-out non-5 accuracy.

Run:  python steer_t4.py          (writes results/steer_t4.json)
      python steer_t4.py figonly  (figure from cached JSON)
"""

import json

import numpy as np

try:  # figonly mode needs neither torch nor the testbed module
    import torch
    import track4_modfam as t4
except ImportError:
    torch = t4 = None

SEEDS = [0, 1, 2]
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
GRAM_KINDS = ["full_dm", "module_mean", "self_dm", "question_dm", "random"]


def strict_set(tgt=5):
    A = np.arange(t4.A_MAX + 1)
    mask = np.ones(len(A), dtype=bool)
    for bp in [b for b in t4.BS if b != tgt]:
        mask &= (A % bp) != (A % tgt)
    return A[mask]


def inputs_for(a_vals, b):
    xs = np.zeros((len(a_vals), t4.N_IN), dtype=np.float32)
    for i in range(t4.N_BITS):
        xs[:, i] = (np.asarray(a_vals) >> i) & 1
    xs[:, t4.N_BITS + t4.BS.index(b)] = 1
    return torch.tensor(xs)


def stream_means(m, x, sel, mask):
    """Per-block residual-stream mean over sel minus over ~sel."""
    vecs = []
    with torch.no_grad():
        h = torch.relu(m.embed(x))
        for b in m.blocks:
            h = h + b(h, mask)
            vecs.append(h[sel].mean(0) - h[~sel].mean(0))
    return vecs


def build_vectors(m, xt, bt, seed):
    i5 = bt == 5
    vs = {"full_dm": stream_means(m, xt, i5, (1,)),
          "self_dm": stream_means(m, xt, i5, (0,))}
    # module-mean: aux output on b=5 inputs along the FULL forward
    with torch.no_grad():
        h = torch.relu(m.embed(xt))
        mm = []
        for b in m.blocks:
            mm.append(b.aux[0](h)[i5].mean(0))
            h = h + b(h, (1,))
        vs["module_mean"] = mm
        # question_dm: b=5 stream mean minus b=10 stream mean (full forward)
        i10 = bt == 10
        h = torch.relu(m.embed(xt))
        qd = []
        for b in m.blocks:
            h = h + b(h, (1,))
            qd.append(h[i5].mean(0) - h[i10].mean(0))
        vs["question_dm"] = qd
    g = torch.Generator().manual_seed(1000 + seed)
    vs["random"] = []
    for v in vs["full_dm"]:
        r = torch.randn(v.shape, generator=g)
        vs["random"].append(r / r.norm() * v.norm())
    return vs


def steered_eval(m, vecs, alpha, xv, yv, bv, xs_strict, ys_strict, mask=(0,)):
    def fwd(x):
        h = torch.relu(m.embed(x))
        for bi, b in enumerate(m.blocks):
            h = h + b(h, mask) + alpha * vecs[bi]
        return m.head(h).argmax(1)

    with torch.no_grad():
        pred = fwd(xv)
        t5v = bv == 5
        return {"target_val": (pred[t5v] == yv[t5v]).float().mean().item(),
                "collateral": (pred[~t5v] == yv[~t5v]).float().mean().item(),
                "strict": (fwd(xs_strict) == ys_strict).float().mean().item()}


def main():
    torch.set_num_threads(8)
    xt, yt, bt, xv, yv, bv = t4.make_data()
    sa = strict_set(5)
    xs_strict = inputs_for(sa, 5)
    ys_strict = torch.tensor(sa % 5)
    print(f"strict set: {len(sa)} a's", flush=True)

    out = {}
    for s in SEEDS:
        m = t4.Net()
        m.load_state_dict(torch.load(f"results/t4_gram5_seed{s}.pt"))
        m.eval()
        vs = build_vectors(m, xt, bt, s)
        for kind in GRAM_KINDS:
            for a in ALPHAS:
                r = steered_eval(m, vs[kind], a, xv, yv, bv, xs_strict, ys_strict)
                for k, v in r.items():
                    out.setdefault(f"gram:{kind}|{a}|{k}", []).append(v)
        mf = t4.Net()
        mf.load_state_dict(torch.load(f"results/t4_filter5_seed{s}.pt"))
        mf.eval()
        fvecs = stream_means(mf, xt, bt == 5, (0,))
        for a in ALPHAS:
            r = steered_eval(mf, fvecs, a, xv, yv, bv, xs_strict, ys_strict)
            for k, v in r.items():
                out.setdefault(f"filter:self_dm|{a}|{k}", []).append(v)
        print(f"seed {s} done", flush=True)

    json.dump(out, open("results/steer_t4.json", "w"))

    print("\n=== strict-set accuracy (mimic baseline scores 0 here; guess-a-residue chance 0.2) ===")
    arms = [f"gram:{k}" for k in GRAM_KINDS] + ["filter:self_dm"]
    hdr = "  ".join(f"a={a:<4g}" for a in ALPHAS)
    print(f"{'arm':18s} {hdr}")
    for arm in arms:
        row = "  ".join(f"{np.mean(out[f'{arm}|{a}|strict']):.3f}" for a in ALPHAS)
        print(f"{arm:18s} {row}")
    print("\n=== held-out b=5 accuracy ===")
    for arm in arms:
        row = "  ".join(f"{np.mean(out[f'{arm}|{a}|target_val']):.3f}" for a in ALPHAS)
        print(f"{arm:18s} {row}")
    print("\n=== per-seed strict at alpha 0 vs best alpha (gram arms) ===")
    for arm in [f"gram:{k}" for k in GRAM_KINDS]:
        base = out[f"{arm}|0.0|strict"]
        best_a = max(ALPHAS, key=lambda a: np.mean(out[f"{arm}|{a}|strict"]))
        best = out[f"{arm}|{best_a}|strict"]
        coll = out[f"{arm}|{best_a}|collateral"]
        print(f"{arm:18s} base {['%.2f' % b for b in base]} -> "
              f"a={best_a:g}: {['%.2f' % b for b in best]} (collateral {np.mean(coll):.3f})")
    print("\nSTEER_T4 DONE", flush=True)


def make_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = json.load(open("results/steer_t4.json"))
    styles = {"gram:module_mean": ("#1f8a70", "o-", "module-mean (the deleted signal)"),
              "gram:full_dm": ("#b3423a", "s--", "diff-of-means, full model"),
              "gram:question_dm": ("#7b5ea7", "D-", "question vector (b5 − b10, full)"),
              "gram:self_dm": ("#c78a2d", "v-.", "self-steering (ablated core)"),
              "gram:random": ("#999999", ".:", "matched-norm random"),
              "filter:self_dm": ("#4c72b0", "x:", "filtered control, self-steering")}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    panels = [("strict", "strict disagreement set accuracy",
               "entailment: steering the ablated core toward mod-5 (strict set)"),
              ("target_val", "held-out b=5 accuracy", "held-out target accuracy"),
              ("collateral", "held-out non-5 accuracy", "collateral")]
    for ax, (met, yl, ti) in zip(axes, panels):
        for key, (c, st, lab) in styles.items():
            mu = [np.mean(out[f"{key}|{a}|{met}"]) for a in ALPHAS]
            sd = [np.std(out[f"{key}|{a}|{met}"]) for a in ALPHAS]
            ax.errorbar(ALPHAS, mu, yerr=sd, fmt=st, color=c, capsize=3, label=lab)
        if met == "strict":
            ax.axhline(0.2, color="k", ls=":", lw=0.7)
            ax.annotate("guess-a-residue chance", (ALPHAS[-1], 0.2),
                        fontsize=6, ha="right", va="bottom")
        ax.set_xlabel("steering strength α"); ax.set_ylabel(yl)
        ax.set_title(ti, fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig("figures/steer_t4.png", dpi=150)
    print("wrote figures/steer_t4.png", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "figonly":
        make_figure()
    else:
        main()
        try:
            make_figure()
        except ImportError:
            print("matplotlib unavailable; run `python steer_t4.py figonly` locally")
