"""Analysis of GRAM MNIST experiments: FP/FN structure, absorption, survival.

For 'dense'-mode runs the aux modules are untrained (never active in fwd, never
updated), so the only meaningful profile is 'core_only'; gram runs use all four.
"""

import glob
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEEDS = [0, 1, 2]
FRACS = [0.03, 0.1, 0.3]


def load_run(name, seed):
    with open(f"results/{name}_seed{seed}.json") as f:
        js = json.load(f)
    npz = np.load(f"results/{name}_seed{seed}.npz")
    return js, npz


def canonical_profile(name):
    return "core_only" if any(
        name.startswith(p) for p in ("dense", "filter")
    ) else "full"


def conf_mean(name, profile):
    """Mean confusion matrix over seeds (counts)."""
    mats = []
    for s in SEEDS:
        js, _ = load_run(name, s)
        mats.append(np.array(js[profile]["confusion"], dtype=float))
    return np.mean(mats, axis=0), np.std([m for m in mats], axis=0)


def seed_stat(name, profile, fn):
    vals = []
    for s in SEEDS:
        js, npz = load_run(name, s)
        vals.append(fn(js[profile], npz, profile))
    return np.mean(vals), np.std(vals)


def recall_of(digit):
    return lambda js, npz, prof: js["recall"][str(digit)]


def fp_into(digit):
    def f(js, npz, prof):
        conf = np.array(js["confusion"])
        return (conf[:, digit].sum() - conf[digit, digit]) / (conf.sum() - conf[digit].sum())
    return f


def overall_acc_excluding(digits):
    def f(js, npz, prof):
        conf = np.array(js["confusion"])
        keep = [c for c in range(10) if c not in digits]
        return sum(conf[c, c] for c in keep) / conf[keep].sum()
    return f


def fmt(m, s, pct=True):
    return f"{100*m:.2f}±{100*s:.2f}" if pct else f"{m:.3f}±{s:.3f}"


def main():
    lines = []
    say = lambda t="": (print(t), lines.append(t))

    # ---------- headline table ----------
    say("## Headline: capability removal via module ablation\n")
    say("| model / profile | overall acc | recall 5 | recall 7 | acc excl {5,7} | FP rate into 5 | FP rate into 7 |")
    say("|---|---|---|---|---|---|---|")
    rows = [
        ("dense_all", "core_only", "dense baseline"),
        ("gram_main", "full", "GRAM full (core+5+7)"),
        ("gram_main", "ablate5", "GRAM ablate module-5"),
        ("gram_main", "ablate7", "GRAM ablate module-7"),
        ("gram_main", "core_only", "GRAM core only"),
        ("filter_no5", "core_only", "filtered (no 5s)"),
        ("filter_no7", "core_only", "filtered (no 7s)"),
        ("filter_no57", "core_only", "filtered (no 5s,7s)"),
    ]
    for name, prof, label in rows:
        acc = seed_stat(name, prof, lambda js, npz, p: js["acc"])
        r5 = seed_stat(name, prof, recall_of(5))
        r7 = seed_stat(name, prof, recall_of(7))
        ax = seed_stat(name, prof, overall_acc_excluding({5, 7}))
        f5 = seed_stat(name, prof, fp_into(5))
        f7 = seed_stat(name, prof, fp_into(7))
        say(f"| {label} | {fmt(*acc)} | {fmt(*r5)} | {fmt(*r7)} | {fmt(*ax)} | {fmt(*f5)} | {fmt(*f7)} |")

    # ---------- FN destinations ----------
    say("\n## Where do the 5s go? (FN destinations, mean counts over 892 test 5s)\n")
    say("| model | " + " | ".join(f"->{c}" for c in range(10)) + " |")
    say("|---" * 11 + "|")
    dest = {}
    for name, prof, label in [
        ("gram_main", "ablate5", "GRAM ablate-5"),
        ("gram_main", "core_only", "GRAM core-only"),
        ("filter_no5", "core_only", "filtered no-5"),
        ("filter_no57", "core_only", "filtered no-5,7"),
    ]:
        m, s = conf_mean(name, prof)
        dest[label] = m[5]
        say(f"| {label} | " + " | ".join(f"{m[5,c]:.0f}" for c in range(10)) + " |")
    say("\n(same, for true 7s under ablate-7 vs filtered no-7)\n")
    say("| model | " + " | ".join(f"->{c}" for c in range(10)) + " |")
    say("|---" * 11 + "|")
    for name, prof, label in [("gram_main", "ablate7", "GRAM ablate-7"),
                              ("filter_no7", "core_only", "filtered no-7")]:
        m, s = conf_mean(name, prof)
        say(f"| {label} | " + " | ".join(f"{m[7,c]:.0f}" for c in range(10)) + " |")

    # cosine similarity between FN destination profiles (excluding the true class)
    def dest_sim(a, b, digit=5):
        va = np.delete(a, digit); vb = np.delete(b, digit)
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
    say(f"\nFN-destination cosine similarity (true-5 row, excl. col 5): "
        f"GRAM-ablate5 vs filtered-no5 = {dest_sim(dest['GRAM ablate-5'], dest['filtered no-5']):.3f}")

    # ---------- absorption sweep ----------
    say("\n## Absorption: fraction f of 5s routed/filtered\n")
    say("Unlabeled-5 treatment arms: all-active = paper's partial-labeling rule; "
        "core = declared core data (p_cr side channel still reaches module-5); "
        "isolated = never activates any module (clean control).\n")
    say("ablate5 = deployment profile (module-7 active); core-only = both modules off, "
        "isolating what the core itself knows about 5s (absorption per se).\n")
    say("| f | ablate5 (all-active) | ablate5 (core+p_cr) | ablate5 (isolated) | core-only (all-active) | core-only (isolated) | partial-filter | GRAM full |")
    say("|---|---|---|---|---|---|---|---|")
    curves = defaultdict(list)
    for f in FRACS:
        g = seed_stat(f"gram_f{f}", "ablate5", recall_of(5))
        gc = seed_stat(f"gram_f{f}_coreunl", "ablate5", recall_of(5))
        gi = seed_stat(f"gram_f{f}_coreiso", "ablate5", recall_of(5))
        gco = seed_stat(f"gram_f{f}", "core_only", recall_of(5))
        gio = seed_stat(f"gram_f{f}_coreiso", "core_only", recall_of(5))
        fl = seed_stat(f"filter5_f{f}", "core_only", recall_of(5))
        gf = seed_stat(f"gram_f{f}", "full", recall_of(5))
        curves["gram"].append(g); curves["gram_coreunl"].append(gc); curves["gram_coreiso"].append(gi)
        curves["gram_coreonly"].append(gco); curves["gram_iso_coreonly"].append(gio)
        curves["filter"].append(fl); curves["gram_full"].append(gf)
        say(f"| {f} | {fmt(*g)} | {fmt(*gc)} | {fmt(*gi)} | {fmt(*gco)} | {fmt(*gio)} | {fmt(*fl)} | {fmt(*gf)} |")
    g1 = seed_stat("gram_main", "ablate5", recall_of(5))
    gco1 = seed_stat("gram_main", "core_only", recall_of(5))
    fl1 = seed_stat("filter_no5", "core_only", recall_of(5))
    gf1 = seed_stat("gram_main", "full", recall_of(5))
    curves["gram"].append(g1); curves["gram_coreonly"].append(gco1)
    curves["gram_coreunl"].append((np.nan, np.nan)); curves["gram_coreiso"].append((np.nan, np.nan))
    curves["gram_iso_coreonly"].append((np.nan, np.nan))
    curves["filter"].append(fl1); curves["gram_full"].append(gf1)
    say(f"| 1.0 | {fmt(*g1)} | (n/a) | (n/a) | {fmt(*gco1)} | (n/a) | {fmt(*fl1)} | {fmt(*gf1)} |")

    # ---------- per-example survival ----------
    say("\n## Per-example: which 5s survive module-5 ablation?\n")
    surv_stats = []
    for s in SEEDS:
        _, npz = load_run("gram_main", s)
        y = npz["labels"]
        lf = npz["logits_full"].astype(np.float32)
        la = npz["logits_ablate5"].astype(np.float32)
        i5 = y == 5
        contrib5 = lf[i5, 5] - la[i5, 5]           # module-5 push on the 5-logit
        surv = la[i5].argmax(1) == 5
        surv_stats.append((surv.mean(), contrib5[surv].mean() if surv.any() else np.nan,
                           contrib5[~surv].mean(), contrib5))
    say(f"survival rate of true 5s under ablate-5: "
        f"{np.mean([x[0] for x in surv_stats]):.3f}±{np.std([x[0] for x in surv_stats]):.3f}")
    say(f"module-5 logit-5 contribution — survivors: "
        f"{np.nanmean([x[1] for x in surv_stats]):.2f}, casualties: {np.mean([x[2] for x in surv_stats]):.2f}")

    # ---------- delta-logit heatmaps ----------
    dl5, dl7 = [], []
    for s in SEEDS:
        _, npz = load_run("gram_main", s)
        y = npz["labels"]
        lf = npz["logits_full"].astype(np.float32)
        d5 = lf - npz["logits_ablate5"].astype(np.float32)
        d7 = lf - npz["logits_ablate7"].astype(np.float32)
        dl5.append(np.stack([d5[y == c].mean(0) for c in range(10)]))
        dl7.append(np.stack([d7[y == c].mean(0) for c in range(10)]))
    dl5, dl7 = np.mean(dl5, axis=0), np.mean(dl7, axis=0)

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (name, prof, label) in zip(axes, [
            ("gram_main", "full", "GRAM full profile"),
            ("gram_main", "ablate5", "GRAM ablate module-5"),
            ("filter_no5", "core_only", "Data-filtered (no 5s)")]):
        m, _ = conf_mean(name, prof)
        mn = m / m.sum(1, keepdims=True)
        ax.imshow(mn, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(label); ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        for i in range(10):
            for j in range(10):
                if mn[i, j] > 0.005:
                    ax.text(j, i, f"{100*mn[i,j]:.0f}", ha="center", va="center",
                            color="white" if mn[i, j] < 0.6 else "black", fontsize=7)
    plt.tight_layout(); plt.savefig("figures/confusions.png", dpi=150); plt.close()

    xs = FRACS + [1.0]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for key, label, style in [("gram", "GRAM ablate-5 (unlabeled: all-active)", "o-"),
                              ("gram_coreunl", "GRAM ablate-5 (unlabeled: core + p_cr)", "s--"),
                              ("gram_iso_coreonly", "GRAM core-only (unlabeled: isolated)", "v--"),
                              ("filter", "partial data filtering", "^-"),
                              ("gram_full", "GRAM full profile", "d:")]:
        m = np.array([c[0] for c in curves[key]]); sd = np.array([c[1] for c in curves[key]])
        ax.errorbar(xs, m, yerr=sd, fmt=style, capsize=3, label=label)
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("fraction of 5s labeled (routed / filterable)")
    ax.set_ylabel("test recall on digit 5")
    ax.set_title("Absorption: ablation removes more than filtering can")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("figures/absorption.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, dl, label in [(axes[0], dl5, "module-5"), (axes[1], dl7, "module-7")]:
        v = np.abs(dl).max()
        im = ax.imshow(dl, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"{label} mean logit contribution")
        ax.set_xlabel("output logit"); ax.set_ylabel("true class")
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig("figures/module_logit_contrib.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    allc, allmargin, allsurv, alldest = [], [], [], []
    for s in SEEDS:
        _, npz = load_run("gram_main", s)
        y = npz["labels"]
        lf = npz["logits_full"].astype(np.float32)
        la = npz["logits_ablate5"].astype(np.float32)
        i5 = y == 5
        allc.append(lf[i5, 5] - la[i5, 5])
        others = np.delete(la[i5], 5, axis=1)
        allmargin.append(la[i5, 5] - others.max(1))  # ablated margin for class 5
        allsurv.append(la[i5].argmax(1) == 5)
        alldest.append(la[i5].argmax(1))
    allc, allmargin = np.concatenate(allc), np.concatenate(allmargin)
    allsurv, alldest = np.concatenate(allsurv), np.concatenate(alldest)
    ax = axes[0]
    ax.scatter(allc[~allsurv], allmargin[~allsurv], s=4, alpha=0.3, color="crimson",
               label="killed by ablation")
    ax.scatter(allc[allsurv], allmargin[allsurv], s=10, alpha=0.8, color="seagreen",
               label="survive ablation")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("module-5 contribution to logit 5 (full − ablated)")
    ax.set_ylabel("ablated-model margin for class 5")
    ax.set_title("Per-example: module contribution vs ablated margin (true 5s)")
    ax.legend()
    r = np.corrcoef(allc, allmargin)[0, 1]
    ax.annotate(f"r = {r:.2f}", xy=(0.03, 0.93), xycoords="axes fraction")
    say(f"\ncorr(module-5 contribution, ablated margin) over true 5s: r = {r:.3f}")
    ax = axes[1]
    dests, counts = np.unique(alldest, return_counts=True)
    order = np.argsort(-counts)
    ax.boxplot([allc[alldest == d] for d in dests[order]], labels=[str(d) for d in dests[order]])
    ax.set_xlabel("ablated-model prediction (true 5s), by frequency")
    ax.set_ylabel("module-5 contribution to logit 5")
    ax.set_title("Module reliance by FN destination")
    plt.tight_layout(); plt.savefig("figures/survival.png", dpi=150); plt.close()

    # ---------- FN destination montage: what do the misrouted 5s look like ----------
    from torchvision import datasets
    te = datasets.MNIST("data", train=False)
    imgs = te.data.numpy()
    _, npz = load_run("gram_main", 0)
    y = npz["labels"]
    la = npz["logits_ablate5"].astype(np.float32)
    i5 = np.where(y == 5)[0]
    pred5 = la[i5].argmax(1)
    dests, counts = np.unique(pred5, return_counts=True)
    top = dests[np.argsort(-counts)][:4]
    fig, axes = plt.subplots(len(top), 8, figsize=(8, 1.05 * len(top)))
    for r, d in enumerate(top):
        # most confident examples of each destination
        cand = i5[pred5 == d]
        conf = la[cand, d]
        cand = cand[np.argsort(-conf)][:8]
        for c in range(8):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(cand):
                ax.imshow(imgs[cand[c]], cmap="gray")
            if c == 0:
                ax.set_title(f"5 -> {d} (n={counts[dests == d][0]})", fontsize=8, loc="left")
    fig.suptitle("True 5s by ablated-model prediction (seed 0, most confident)", fontsize=10)
    plt.tight_layout(); plt.savefig("figures/fn_montage.png", dpi=150); plt.close()

    with open("results/summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote results/summary.md and figures/*.png")


if __name__ == "__main__":
    main()
