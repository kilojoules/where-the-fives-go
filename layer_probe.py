"""Depth profile of decodable information vs expressed performance.

For each ship (core-only mask), train a converged linear probe on the frozen
representation at every depth stage — raw input, embed, block 1, block 2
(penultimate), and the native logits — and compare with the NATIVE readout's
own performance. The custody law predicts: probe stays high through the trunk
(MNIST) or rises through it (modmath, where the input is chance-level), and
the collapse happens at exactly one place — the shipped readout.

Outputs figures/layer_probe.png and results/layer_probe.json.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

SEEDS = [0, 1, 2]
EPOCHS = 100


def probe_fit(ftr, ytr, n_classes, seed, lr=1e-2):
    torch.manual_seed(seed)
    head = torch.nn.Linear(ftr.shape[1], n_classes)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(ftr), generator=gen)
        for i in range(0, len(ftr), 512):
            b = perm[i:i + 512]
            loss = F.cross_entropy(head(ftr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return head


def auroc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def mnist_stages(model, x, mask=(0, 0)):
    """Representations at each depth under the core-only forward."""
    outs = {}
    with torch.no_grad():
        flat = x.flatten(1)
        outs["input"] = flat
        h = model.embed(x)
        outs["embed"] = h.clone()
        for i, b in enumerate(model.blocks):
            h = h + b(h, mask)
            outs[f"block{i+1}"] = h.clone()
        outs["logits"] = model.head(h)
    return outs


MNIST_STAGES = ["input", "embed", "block1", "block2", "logits"]


def mnist_profile_one(job):
    """One (name, seed) depth profile, cached. The input stage is model-
    independent and shared via its own cache entry."""
    import os
    name, s = job
    cache = f"results/lp_cache/mnist_{name}_s{s}.json"
    if os.path.exists(cache):
        return name, s, json.load(open(cache))
    torch.set_num_threads(3)
    from model import GramNet
    from train import load_mnist
    xtr, ytr, xte, yte = load_mnist(torch.device("cpu"))
    i5te = (yte == 5).numpy().astype(int)
    m = GramNet(); m.load_state_dict(torch.load(f"results/{name}_seed{s}.pt")); m.eval()
    ftr_all, fte_all = {}, {}
    for src_, dst in [(xtr, ftr_all), (xte, fte_all)]:
        accum = {st: [] for st in MNIST_STAGES}
        for i in range(0, len(src_), 2048):
            outs = mnist_stages(m, src_[i:i + 2048])
            for st in MNIST_STAGES:
                accum[st].append(outs[st])
        for st in MNIST_STAGES:
            dst[st] = torch.cat(accum[st])
    out = {}
    for st in MNIST_STAGES:
        if st == "input":
            icache = f"results/lp_cache/mnist_INPUT_s{s}.json"
            if os.path.exists(icache):
                out[st] = json.load(open(icache))["auroc"]
                continue
        head = probe_fit(ftr_all[st], ytr, 10, s)
        with torch.no_grad():
            lg = head(fte_all[st])
        out[st] = float(auroc(lg[:, 5].numpy(), i5te))
        if st == "input":
            json.dump({"auroc": out[st]}, open(f"results/lp_cache/mnist_INPUT_s{s}.json", "w"))
    native = fte_all["logits"]
    out["native_auroc"] = float(auroc(native[:, 5].numpy(), i5te))
    out["native_recall"] = float((native.argmax(1)[yte == 5] == 5).float().mean())
    json.dump(out, open(cache, "w"))
    print(f"mnist {name} s{s}: " + " ".join(f"{st}={out[st]:.3f}" for st in MNIST_STAGES) +
          f" native_auroc={out['native_auroc']:.3f}", flush=True)
    return name, s, out


def run_mnist(names=("ck_gram_main", "ck_filter_no5", "ck_dense_all")):
    import multiprocessing as mp
    stages = MNIST_STAGES
    jobs = [(n, s) for n in names for s in SEEDS]
    res = {n: {**{st: [] for st in stages}, "native_auroc": [], "native_recall": []} for n in names}
    with mp.get_context("spawn").Pool(3) as pool:
        for name, s, out in pool.imap_unordered(mnist_profile_one, jobs):
            for st in stages:
                res[name][st].append(out[st])
            res[name]["native_auroc"].append(out["native_auroc"])
            res[name]["native_recall"].append(out.get("native_recall", 0.0))
    return res, stages


def modmath_stages(model, x, mask=(0,)):
    outs = {}
    with torch.no_grad():
        outs["input"] = x
        h = torch.relu(model.embed(x))
        outs["embed"] = h.clone()
        for i, b in enumerate(model.blocks):
            h = h + b(h, mask)
            outs[f"block{i+1}"] = h.clone()
        outs["logits"] = model.head(h)
    return outs


def run_modmath():
    import track2_modmath as t2
    xt, yt, opt_, xv, yv, opv = t2.make_data()
    mult_v = (opv == 1)
    stages = ["input", "embed", "block1", "block2", "logits"]
    res = {}
    for name in ["t2_gram", "t2_filter", "t2_dense"]:
        res[name] = {st: [] for st in stages}
        res[name]["native_mult"] = []
        for s in SEEDS:
            m = t2.Net(); m.load_state_dict(torch.load(f"results/{name}_seed{s}.pt")); m.eval()
            otr, ote = modmath_stages(m, xt), modmath_stages(m, xv)
            for st in stages:
                head = probe_fit(otr[st], yt, 53, s)
                with torch.no_grad():
                    pred = head(ote[st]).argmax(1)
                res[name][st].append(float((pred[mult_v] == yv[mult_v]).float().mean()))
            with torch.no_grad():
                pred = ote["logits"].argmax(1)
            res[name]["native_mult"].append(float((pred[mult_v] == yv[mult_v]).float().mean()))
            print(f"modmath {name} s{s}: " +
                  " ".join(f"{st}={res[name][st][-1]:.3f}" for st in stages) +
                  f" native={res[name]['native_mult'][-1]:.3f}", flush=True)
    return res, stages


def make_figure(mn, mm, stages):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.arange(len(stages))
    labels = ["raw\ninput", "embed", "block 1", "block 2", "logits"]
    styles = {
        "gram": ("#1f8a70", "o-", "GRAM, module ablated"),
        "filter": ("#4c72b0", "s--", "data-filtered"),
        "dense": ("#8a8a8a", "d:", "dense (capability present)"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    for key, name in [("gram", "ck_gram_main"), ("filter", "ck_filter_no5"),
                      ("dense", "ck_dense_all")]:
        c, st_, lab = styles[key]
        mvals = [np.mean(mn[name][st]) for st in stages]
        svals = [np.std(mn[name][st]) for st in stages]
        ax.errorbar(xs, mvals, yerr=svals, fmt=st_, color=c, capsize=3, label=lab)
        ax.scatter([xs[-1] + 0.35], [np.mean(mn[name]["native_auroc"])],
                   marker="*", s=180, color=c, zorder=5, edgecolors="k", linewidths=0.5)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.annotate("chance", (0.05, 0.51), fontsize=8)
    ax.annotate("★ = native readout\n(no retraining)", (3.05, 0.56), fontsize=8)
    ax.set_xticks(list(xs) + [xs[-1] + 0.35])
    ax.set_xticklabels(labels + ["native\nreadout"])
    ax.set_ylabel("probe AUROC: 5 vs rest")
    ax.set_title("MNIST — information is everywhere; the cliff is the readout")
    ax.set_ylim(0.35, 1.03); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    for key, name in [("gram", "t2_gram"), ("filter", "t2_filter"), ("dense", "t2_dense")]:
        c, st_, lab = styles[key]
        mvals = [np.mean(mm[name][st]) for st in stages]
        svals = [np.std(mm[name][st]) for st in stages]
        ax.errorbar(xs, mvals, yerr=svals, fmt=st_, color=c, capsize=3, label=lab)
        ax.scatter([xs[-1] + 0.35], [np.mean(mm[name]["native_mult"])],
                   marker="*", s=180, color=c, zorder=5, edgecolors="k", linewidths=0.5)
    ax.axhline(1 / 53, color="k", lw=0.6, ls=":")
    ax.annotate("chance", (0.05, 0.035), fontsize=8)
    ax.set_xticks(list(xs) + [xs[-1] + 0.35])
    ax.set_xticklabels(labels + ["native\nreadout"])
    ax.set_ylabel("probe accuracy: multiplication (held-out pairs)")
    ax.set_title("mod-arithmetic — the core computes ×, the readout won't say it")
    ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig("figures/layer_probe.png", dpi=150)
    print("wrote figures/layer_probe.png", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(8)
    mn, stages = run_mnist()
    mm, _ = run_modmath()
    json.dump({"mnist": mn, "modmath": mm}, open("results/layer_probe.json", "w"))
    make_figure(mn, mm, stages)
    print("LAYER PROBE DONE", flush=True)
