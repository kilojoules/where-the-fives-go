"""Activation steering against the ablated model: can a single fixed vector
re-elicit the removed digit?

Vectors (all computed on the train set, applied unconditionally at test time
to the ablated ship, mask (0,1) — module-7 present as deployed):
  full_dm    : difference of means (5s vs non-5s) of the FULL model's residual
               stream after each block. Same trunk weights, same basis.
  module_mean: the mean OUTPUT of module-5 on 5s at each block — the average
               of exactly what ablation deleted, re-injected as a constant.
  self_dm    : difference of means from the ABLATED core's own stream — pure
               latent-knowledge steering, no access to the full model needed.

Steering: h <- h + alpha * v_b after each block b. Metrics vs alpha: recall
on 5s (elicitation) and accuracy on non-5s (collateral). Controls:
filter_no5 with its own self_dm (knowledge floor), dense (no steering needed).
Also a modmath panel: steer the ablated core toward x with full-model diff-
of-means over op domains.

Outputs figures/steering.png, results/steering.json.
"""

import json

import numpy as np
import torch

SEEDS = [0, 1, 2]
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def mnist_vectors(m, x, y):
    """Per-block steering vectors of each type, computed on the train set."""
    i5 = y == 5
    vs = {"full_dm": [], "module_mean": [], "self_dm": []}
    with torch.no_grad():
        # full-model stream (mask (1,1)) and ablated stream (mask (0,1))
        for mask, key in [((1, 1), "full_dm"), ((0, 1), "self_dm")]:
            h = m.embed(x)
            for b in m.blocks:
                h = h + b(h, mask)
                vs[key].append(h[i5].mean(0) - h[~i5].mean(0))
        # module-5 mean output per block along the FULL forward
        h = m.embed(x)
        for b in m.blocks:
            vs["module_mean"].append(b.aux[0](h)[i5].mean(0))
            h = h + b(h, (1, 1))
    return vs


@torch.no_grad()
def mnist_steered_eval(m, xte, yte, vecs, alpha, mask=(0, 1), bs=2048):
    preds = []
    for i in range(0, len(xte), bs):
        h = m.embed(xte[i:i + bs])
        for bi, b in enumerate(m.blocks):
            h = h + b(h, mask) + alpha * vecs[bi]
        preds.append(m.head(h).argmax(1))
    pred = torch.cat(preds)
    i5 = yte == 5
    return {"recall5": (pred[i5] == 5).float().mean().item(),
            "acc_excl5": (pred[~i5] == yte[~i5]).float().mean().item()}


def run_mnist():
    from model import GramNet
    from train import load_mnist
    xtr, ytr, xte, yte = load_mnist(torch.device("cpu"))
    out = {}
    for vec_kind in ["full_dm", "module_mean", "self_dm"]:
        out[f"gram:{vec_kind}"] = {a: {"recall5": [], "acc_excl5": []} for a in ALPHAS}
    out["filter:self_dm"] = {a: {"recall5": [], "acc_excl5": []} for a in ALPHAS}

    for s in SEEDS:
        m = GramNet(); m.load_state_dict(torch.load(f"results/ck_gram_main_seed{s}.pt")); m.eval()
        vs = mnist_vectors(m, xtr, ytr)
        for vec_kind in ["full_dm", "module_mean", "self_dm"]:
            for a in ALPHAS:
                r = mnist_steered_eval(m, xte, yte, vs[vec_kind], a)
                for k in r:
                    out[f"gram:{vec_kind}"][a][k].append(r[k])
        mf = GramNet(); mf.load_state_dict(torch.load(f"results/ck_filter_no5_seed{s}.pt")); mf.eval()
        with torch.no_grad():
            i5 = ytr == 5
            fvecs = []
            h = mf.embed(xtr)
            for b in mf.blocks:
                h = h + b(h, (0, 0))
                fvecs.append(h[i5].mean(0) - h[~i5].mean(0))
        for a in ALPHAS:
            r = mnist_steered_eval(mf, xte, yte, fvecs, a, mask=(0, 0))
            for k in r:
                out["filter:self_dm"][a][k].append(r[k])
        print(f"mnist seed {s} done", flush=True)
    return out


def run_modmath():
    import track2_modmath as t2
    xt, yt, opt_, xv, yv, opv = t2.make_data()
    mult_t, mult_v = opt_ == 1, opv == 1
    out = {"gram:full_dm": {a: {"mult": [], "add": []} for a in ALPHAS},
           "gram:self_dm": {a: {"mult": [], "add": []} for a in ALPHAS}}
    for s in SEEDS:
        m = t2.Net(); m.load_state_dict(torch.load(f"results/t2_gram_seed{s}.pt")); m.eval()
        vs = {}
        with torch.no_grad():
            for mask, key in [((1,), "full_dm"), ((0,), "self_dm")]:
                vecs = []
                h = torch.relu(m.embed(xt))
                for b in m.blocks:
                    h = h + b(h, mask)
                    vecs.append(h[mult_t].mean(0) - h[~mult_t].mean(0))
                vs[key] = vecs
        for key in ["full_dm", "self_dm"]:
            for a in ALPHAS:
                with torch.no_grad():
                    h = torch.relu(m.embed(xv))
                    for bi, b in enumerate(m.blocks):
                        h = h + b(h, (0,)) + a * vs[key][bi]
                    pred = m.head(h).argmax(1)
                out[f"gram:{key}"][a]["mult"].append(
                    (pred[mult_v] == yv[mult_v]).float().mean().item())
                out[f"gram:{key}"][a]["add"].append(
                    (pred[~mult_v] == yv[~mult_v]).float().mean().item())
        print(f"modmath seed {s} done", flush=True)
    return out


def make_figure(mn, mm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    styles = {"gram:module_mean": ("#1f8a70", "o-", "module-mean vector (the deleted signal)"),
              "gram:full_dm": ("#b3423a", "s--", "diff-of-means from full model"),
              "gram:self_dm": ("#c78a2d", "v-.", "self-steering (ablated core only)"),
              "filter:self_dm": ("#4c72b0", "x:", "filtered model, self-steering")}

    ax = axes[0]
    for key, (c, st, lab) in styles.items():
        r5 = [np.mean(mn[key][a]["recall5"]) for a in ALPHAS]
        ax.errorbar(ALPHAS, r5, yerr=[np.std(mn[key][a]["recall5"]) for a in ALPHAS],
                    fmt=st, color=c, capsize=3, label=lab)
    ax.set_xlabel("steering strength α"); ax.set_ylabel("recall on 5s (ablated ship)")
    ax.set_title("MNIST: elicitation vs steering strength")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1]
    for key, (c, st, lab) in styles.items():
        r5 = [np.mean(mn[key][a]["recall5"]) for a in ALPHAS]
        ax_ = [np.mean(mn[key][a]["acc_excl5"]) for a in ALPHAS]
        ax.plot(ax_, r5, st, color=c, label=lab)
        for a, x_, y_ in zip(ALPHAS, ax_, r5):
            if a in (1.0, 4.0):
                ax.annotate(f"α={a:g}", (x_, y_), fontsize=6, alpha=0.7)
    ax.set_xlabel("accuracy on non-5 digits (collateral)")
    ax.set_ylabel("recall on 5s")
    ax.set_title("MNIST: the elicitation-collateral frontier")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[2]
    for key in ["gram:full_dm", "gram:self_dm"]:
        c_, st_, lab_ = styles[key]
        mu = [np.mean(mm[key][a]["mult"]) for a in ALPHAS]
        ad = [np.mean(mm[key][a]["add"]) for a in ALPHAS]
        ax.errorbar(ALPHAS, mu, yerr=[np.std(mm[key][a]["mult"]) for a in ALPHAS],
                    fmt=st_, color=c_, capsize=3, label=f"× recall, {lab_}")
        ax.plot(ALPHAS, ad, st_, color=c_, alpha=0.35, label=f"+ accuracy, {lab_}")
    ax.axhline(1 / 53, color="k", ls=":", lw=0.7)
    ax.set_xlabel("steering strength α"); ax.set_ylabel("held-out accuracy")
    ax.set_title("mod-arithmetic: steering the ablated core toward ×")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    plt.tight_layout(); plt.savefig("figures/steering.png", dpi=150)
    print("wrote figures/steering.png", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "figonly":
        d = json.load(open("results/steering.json"))
        def fk(dd):
            return {k: {float(a): v for a, v in vv.items()} for k, vv in dd.items()}
        make_figure(fk(d["mnist"]), fk(d["modmath"]))
        raise SystemExit
    torch.set_num_threads(8)
    mn = run_mnist()
    mm = run_modmath()
    json.dump({"mnist": mn, "modmath": mm},
              open("results/steering.json", "w"))
    make_figure(mn, mm)
    print("STEER DONE", flush=True)
