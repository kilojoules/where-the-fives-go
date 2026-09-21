"""Track 4 analysis: entailment vs removal — with all review-mandated controls.

- Behavioral: target-b accuracy per arm/profile against the BEST-RETAINED-MIMIC
  baseline (max over retained b' of P[a mod b' = a mod tgt]) and on the strict
  disagreement set {a : every retained b' disagrees with tgt on a}.
- Target probes: linear probe on core-mask features of (a, tgt) inputs, split
  over a's (70/30, fixed), CHANCE-NORMALIZED (acc-1/k)/(1-1/k), with symmetric
  cross-controls (each filter arm probed for the OTHER target = trained-task
  probe-power control).
- Compositional-leakage probes: decode a mod 5 (and mod 13) from carrier-task
  features (entailing carriers {10,15,20} vs non-entailing {7,11,17,19}), at
  two stages (post-embed and post-block2), across arms — the a-embedding
  confound is addressed by the post-embed baseline and the symmetric decode.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

import track4_modfam as t4

SEEDS = [0, 1, 2]
RETAINED = {5: [b for b in t4.BS if b != 5], 13: [b for b in t4.BS if b != 13]}
ENTAIL = {5: [10, 15, 20], 13: []}
NON_ENTAIL_CARRIERS = [7, 11, 17, 19]
A = np.arange(t4.A_MAX + 1)


def mimic_baseline(tgt):
    return max(float(np.mean((A % bp) == (A % tgt))) for bp in RETAINED[tgt])


def strict_set(tgt):
    mask = np.ones(len(A), dtype=bool)
    for bp in RETAINED[tgt]:
        mask &= (A % bp) != (A % tgt)
    return A[mask]


def inputs_for(b):
    xs = np.zeros((len(A), t4.N_IN), dtype=np.float32)
    for i in range(t4.N_BITS):
        xs[:, i] = (A >> i) & 1
    xs[:, t4.N_BITS + t4.BS.index(b)] = 1
    return torch.tensor(xs)


def probe(feats_tr, y_tr, feats_te, y_te, n_classes, seed, epochs=200):
    torch.manual_seed(seed)
    head = torch.nn.Linear(feats_tr.shape[1], n_classes)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(feats_tr), generator=gen)
        for i in range(0, len(feats_tr), 256):
            b = perm[i:i + 256]
            loss = F.cross_entropy(head(feats_tr[b]), y_tr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    with torch.no_grad():
        return (head(feats_te).argmax(1) == y_te).float().mean().item()


def cn(acc, k):
    return (acc - 1 / k) / (1 - 1 / k)


def stages(model, x):
    with torch.no_grad():
        h0 = torch.relu(model.embed(x))
        h = h0
        for blk in model.blocks:
            h = h + blk(h, (0,))
    return {"embed": h0, "block2": h}


def a_split(seed=0, frac=0.7):
    rng = np.random.RandomState(777 + seed)
    perm = rng.permutation(len(A))
    n = int(frac * len(A))
    return perm[:n], perm[n:]


def main():
    lines = []
    say = lambda t="": (print(t), lines.append(t))
    say("# Track 4: entailment vs removal\n")
    say(f"Best-retained-mimic baselines: mod-5 = {100*mimic_baseline(5):.1f}% "
        f"(chance 20%), mod-13 = {100*mimic_baseline(13):.1f}% (chance 7.7%).")
    s5, s13 = strict_set(5), strict_set(13)
    say(f"Strict disagreement sets: |mod-5| = {len(s5)}, |mod-13| = {len(s13)}\n")

    # ---------- behavioral ----------
    say("## Behavioral (held-out pairs; strict set in parens; vs mimic baseline)\n")
    say("| arm | profile | target acc | strict-set acc | mimic baseline |")
    say("|---|---|---|---|---|")
    for tgt in (5, 13):
        strict = strict_set(tgt)
        xs_t = inputs_for(tgt)
        for arm, prof_list in [(f"t4_filter{tgt}", ["core"]),
                               (f"t4_gram{tgt}", ["full", "core"]),
                               ("t4_dense", ["core"])]:
            for prof in prof_list:
                accs, saccs = [], []
                for s in SEEDS:
                    js = json.load(open(f"results/{arm}_seed{s}.json"))
                    accs.append(js[prof][str(tgt)])
                    m = t4.Net(); m.load_state_dict(torch.load(f"results/{arm}_seed{s}.pt")); m.eval()
                    mask = (1,) if (prof == "full" and "gram" in arm) else (0,)
                    with torch.no_grad():
                        pred = m(xs_t[strict], mask=mask).argmax(1)
                    saccs.append((pred == torch.tensor(strict % tgt)).float().mean().item())
                say(f"| {arm} | {prof} | {100*np.mean(accs):.1f}±{100*np.std(accs):.1f} | "
                    f"{100*np.mean(saccs):.1f}±{100*np.std(saccs):.1f} | {100*mimic_baseline(tgt):.1f} |")

    # ---------- target probes (chance-normalized, a-split, cross-controls) ----------
    say("\n## Target probes on core-mask features of (a, target-b) inputs")
    say("(chance-normalized; probe split over a's; 'cross' = other target, trained — probe-power control)\n")
    say("| arm | decode | raw acc | chance-norm |")
    say("|---|---|---|---|")
    probe_res = {}
    for arm in ["t4_dense", "t4_filter5", "t4_gram5", "t4_filter13", "t4_gram13"]:
        for tgt in (5, 13):
            raws = []
            for s in SEEDS:
                m = t4.Net(); m.load_state_dict(torch.load(f"results/{arm}_seed{s}.pt")); m.eval()
                feats = stages(m, inputs_for(tgt))["block2"]
                tr, te = a_split(s)
                y = torch.tensor(A % tgt)
                raws.append(probe(feats[tr], y[tr], feats[te], y[te], tgt, s))
            probe_res[(arm, tgt)] = (np.mean(raws), np.std(raws))
            say(f"| {arm} | mod-{tgt} | {100*np.mean(raws):.1f}±{100*np.std(raws):.1f} | "
                f"{100*cn(np.mean(raws), tgt):.1f} |")

    # ---------- compositional leakage ----------
    say("\n## Compositional leakage: decode target residue from CARRIER-task features")
    say("(carriers never equal the target; entailing = {10,15,20} for mod-5; "
        "non-entailing controls = {7,11,17,19}; both stages)\n")
    say("| arm | decode | carriers | stage | chance-norm acc |")
    say("|---|---|---|---|---|")
    for arm in ["t4_filter5", "t4_filter13", "t4_dense"]:
        for tgt in (5, 13):
            for cname, carriers in [("entailing(5)", ENTAIL[5]), ("non-entailing", NON_ENTAIL_CARRIERS)]:
                for stage in ["embed", "block2"]:
                    raws = []
                    for s in SEEDS:
                        m = t4.Net(); m.load_state_dict(torch.load(f"results/{arm}_seed{s}.pt")); m.eval()
                        fs, ys = [], []
                        for b in carriers:
                            f_ = stages(m, inputs_for(b))[stage]
                            fs.append(f_); ys.append(torch.tensor(A % tgt))
                        feats = torch.cat(fs); y = torch.cat(ys)
                        tr, te = a_split(s)
                        tr_idx = np.concatenate([tr + i * len(A) for i in range(len(carriers))])
                        te_idx = np.concatenate([te + i * len(A) for i in range(len(carriers))])
                        raws.append(probe(feats[tr_idx], y[tr_idx], feats[te_idx], y[te_idx], tgt, s))
                    say(f"| {arm} | mod-{tgt} | {cname} | {stage} | {100*cn(np.mean(raws), tgt):.1f}±{100*np.std([cn(r,tgt) for r in raws]):.1f} |")

    with open("results/summary_t4.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote results/summary_t4.md")


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
