"""Analysis of the training-label-noise suite: quarantine + memorization reversion.

For each noise kind/rate we compare GRAM (full / ablate5 / core_only profiles)
against a dense model trained on the identical noisy labels, on:
  - clean test metrics (recall 5, FP->5, acc excluding 5)
  - the memorization probe: the trained model's predictions ON the mislabeled
    training images, scored against their noisy label vs their true label.
"""

import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEEDS = [0, 1, 2]
RATES = [0.1, 0.3]


def load(name, seed):
    js = json.load(open(f"results/{name}_seed{seed}.json"))
    npz = np.load(f"results/{name}_seed{seed}.npz")
    return js, npz


def fp_into(conf, d):
    conf = np.array(conf)
    return (conf[:, d].sum() - conf[d, d]) / (conf.sum() - conf[d].sum())


def acc_excl(conf, digits):
    conf = np.array(conf)
    keep = [c for c in range(10) if c not in digits]
    return sum(conf[c, c] for c in keep) / conf[keep].sum()


def stat(vals):
    return np.mean(vals), np.std(vals)


def fmt(v, pct=True):
    m, s = v
    return f"{100*m:.2f}±{100*s:.2f}" if pct else f"{m:.3f}±{s:.3f}"


def probe_rates(npz, profile):
    """On the noised train images: (frac predicted = noisy label, = true label)."""
    lg = npz[f"noise_logits_{profile}"].astype(np.float32)
    pred = lg.argmax(1)
    return (pred == npz["noise_y_noisy"]).mean(), (pred == npz["noise_y_true"]).mean()


def main():
    lines = []
    say = lambda t="": (print(t), lines.append(t))

    say("# Training-label noise: quarantine and memorization reversion\n")

    for kind, blurb in [
        ("fp5", "FP labels: non-5 images mislabeled '5' (routed into module-5)"),
        ("fn5", "FN labels: true 5s mislabeled as other digits (diverted away from module-5)"),
        ("swap57", "swap: 5s and 7s trade labels with prob r"),
    ]:
        say(f"\n## {kind} — {blurb}\n")
        say("### Clean test set\n")
        say("| rate | model/profile | recall 5 | recall 7 | FP->5 | FP->7 | acc excl 5,7 |")
        say("|---|---|---|---|---|---|---|")
        for r in RATES:
            rows = [
                (f"noise_{kind}_r{r}_dense", "core_only", "dense (noisy labels)"),
                (f"noise_{kind}_r{r}", "full", "GRAM full"),
                (f"noise_{kind}_r{r}", "ablate5", "GRAM ablate-5"),
                (f"noise_{kind}_r{r}", "ablate7", "GRAM ablate-7"),
                (f"noise_{kind}_r{r}", "core_only", "GRAM core-only"),
            ]
            for name, prof, label in rows:
                r5, r7, f5, f7, ax = [], [], [], [], []
                for s in SEEDS:
                    js, _ = load(name, s)
                    m = js[prof]
                    r5.append(m["recall"]["5"]); r7.append(m["recall"]["7"])
                    f5.append(fp_into(m["confusion"], 5)); f7.append(fp_into(m["confusion"], 7))
                    ax.append(acc_excl(m["confusion"], {5, 7}))
                say(f"| {r} | {label} | {fmt(stat(r5))} | {fmt(stat(r7))} | {fmt(stat(f5))} | "
                    f"{fmt(stat(f7))} | {fmt(stat(ax))} |")

        say("\n### Memorization probe (the mislabeled train images themselves)\n")
        say("| rate | model/profile | pred = noisy label | pred = TRUE label |")
        say("|---|---|---|---|")
        for r in RATES:
            for name, prof, label in [
                (f"noise_{kind}_r{r}_dense", "core_only", "dense (noisy labels)"),
                (f"noise_{kind}_r{r}", "full", "GRAM full"),
                (f"noise_{kind}_r{r}", "ablate5", "GRAM ablate-5"),
                (f"noise_{kind}_r{r}", "ablate7", "GRAM ablate-7"),
                (f"noise_{kind}_r{r}", "core_only", "GRAM core-only"),
            ]:
                nz, tr = [], []
                for s in SEEDS:
                    _, npz = load(name, s)
                    a, b = probe_rates(npz, prof)
                    nz.append(a); tr.append(b)
                say(f"| {r} | {label} | {fmt(stat(nz))} | {fmt(stat(tr))} |")

    # fn5 special split: mislabeled-as-7 (routed to module-7) vs mislabeled-as-other
    say("\n## fn5 split: true 5s relabeled '7' (routed to module-7) vs relabeled other\n")
    say("| rate | profile | subset | pred=noisy | pred=5 (true) |")
    say("|---|---|---|---|---|")
    for r in RATES:
        for prof in ["full", "ablate7", "core_only"]:
            for lab7 in [True, False]:
                nz, tr = [], []
                for s in SEEDS:
                    _, npz = load(f"noise_fn5_r{r}", s)
                    sel = (npz["noise_y_noisy"] == 7) == lab7
                    lg = npz[f"noise_logits_{prof}"].astype(np.float32)[sel]
                    pred = lg.argmax(1)
                    nz.append((pred == npz["noise_y_noisy"][sel]).mean())
                    tr.append((pred == npz["noise_y_true"][sel]).mean())
                say(f"| {r} | {prof} | {'->7' if lab7 else '->other'} | {fmt(stat(nz))} | {fmt(stat(tr))} |")

    # ---------- figures ----------
    # quarantine: test FP->5 under fp5 noise
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    width = 0.2
    xs = np.arange(len(RATES))
    for i, (name_fn, prof, label, color) in enumerate([
            (lambda r: f"noise_fp5_r{r}_dense", "core_only", "dense (noisy)", "#888888"),
            (lambda r: f"noise_fp5_r{r}", "full", "GRAM full", "#b3423a"),
            (lambda r: f"noise_fp5_r{r}", "ablate5", "GRAM ablate-5", "#1f8a70")]):
        ms, ss = [], []
        for r in RATES:
            vals = [fp_into(load(name_fn(r), s)[0][prof]["confusion"], 5) for s in SEEDS]
            ms.append(np.mean(vals)); ss.append(np.std(vals))
        ax.bar(xs + (i - 1) * width, np.array(ms) * 100, width, yerr=np.array(ss) * 100,
               capsize=3, label=label, color=color)
    ax.set_xticks(xs); ax.set_xticklabels([f"r={r}" for r in RATES])
    ax.set_ylabel("clean-test FP rate into 5 (%)")
    ax.set_title("FP-label noise: is the damage removable?")
    ax.legend()

    ax = axes[1]
    for i, (name_fn, prof, label, color) in enumerate([
            (lambda r: f"noise_fp5_r{r}_dense", "core_only", "dense (noisy)", "#888888"),
            (lambda r: f"noise_fp5_r{r}", "full", "GRAM full", "#b3423a"),
            (lambda r: f"noise_fp5_r{r}", "ablate5", "GRAM ablate-5", "#1f8a70")]):
        ms, ss = [], []
        for r in RATES:
            vals = []
            for s in SEEDS:
                _, npz = load(name_fn(r), s)
                vals.append(probe_rates(npz, prof)[1])  # pred == TRUE label
            ms.append(np.mean(vals)); ss.append(np.std(vals))
        ax.bar(xs + (i - 1) * width, np.array(ms) * 100, width, yerr=np.array(ss) * 100,
               capsize=3, label=label, color=color)
    ax.set_xticks(xs); ax.set_xticklabels([f"r={r}" for r in RATES])
    ax.set_ylabel("mislabeled train images recovered to TRUE label (%)")
    ax.set_title("Memorization reversion on the mislabeled images")
    ax.legend()
    plt.tight_layout(); plt.savefig("figures/noise_quarantine.png", dpi=150); plt.close()

    with open("results/summary_noise.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote results/summary_noise.md and figures/noise_quarantine.png")


if __name__ == "__main__":
    main()
