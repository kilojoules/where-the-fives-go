"""The gate: which of the two AUCs actually tracks capability?

INSTRUMENT.md shows softmax P(5) and raw logit 5 disagreeing by up to 0.75 AUC on the same
checkpoint. That is a reason to distrust both until one is validated, and this repo's own rule
(experiments/lisp-capability-eval/README.md, "The gate") says how: give the statistic the SAME
model at several KNOWN capability levels and require it to put them in order.

`filter_frac5` is that dose. A dense model keeping a k fraction of correctly-labelled 5s spans
the whole range between this repo's two endpoints -- k=1.0 is `dense_all`, k=0.0 is
`filter_no5` -- and more 5s is monotonically more 5-capability, by construction rather than by
assumption. No label noise here: the dose must be clean for the gate to mean anything.

A statistic that fails to order this series is not measuring capability on this setup, and no
number it produced elsewhere can be read as capability. Fix the measure; do not reinterpret.

Reports Spearman rho against the dose for each statistic, and prints GATE: PASS / FAIL.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from absorb3 import auc, p5, raw5, acc_others, TARGET
from mnist_data import load_mnist
from model import GramNet

# Fraction of correctly-labelled 5s retained. The first pass used
# [1.0, 0.5, 0.2, 0.05, 0.01, 0.0] and was a bad gate: 54 fives (k=0.01) already saturate
# both statistics at ~0.99, so five of six levels sat on the ceiling and Spearman was decided
# by sub-0.01 wobble there. results/gate5.json keeps that run. This grid spends its levels
# where the statistics actually move, between 0 and ~50 fives.
KEEP = [0.05, 0.01, 0.004, 0.002, 0.001, 0.0005, 0.0]
MASK = (0, 0)                              # dense ships core-only


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v, kind='mergesort')
        r = np.empty(len(v))
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--out', type=Path, default=Path('results/gate5.json'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    dev = torch.device(a.device)
    xte, yte = load_mnist(dev)[2:]
    pos = (yte.cpu().numpy() == TARGET)

    from train import Config, run as train_run

    out = {'keep': KEEP, 'seeds': a.seeds, 'cells': {}, 'gate': {}}
    for seed in a.seeds:
        for k in KEEP:
            tag = f'gate5_k{k}'
            wp = Path('results') / f'{tag}_seed{seed}.pt'
            if not wp.exists():
                train_run(Config(name=tag, seed=seed, device=str(dev), mode='dense',
                                 filter_frac5=1.0 - k, save_weights=True), out_dir='results')
            m = GramNet().to(dev)
            m.load_state_dict(torch.load(wp, map_location=dev))
            rec = dict(keep=k, seed=seed,
                       n_fives=int(round(k * 5421)),
                       auc_softmax=auc(p5(m, xte, MASK), pos),
                       auc_rawlogit=auc(raw5(m, xte, MASK), pos),
                       acc_others=acc_others(m, xte, yte, MASK))
            out['cells'][f'k{k}_seed{seed}'] = rec
            print(json.dumps(dict(event='cell', **{
                k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in rec.items()})),
                flush=True)

    # gate: per seed, does the statistic order the dose?
    for stat in ('auc_softmax', 'auc_rawlogit'):
        rhos = []
        for seed in a.seeds:
            vals = [out['cells'][f'k{k}_seed{seed}'][stat] for k in KEEP]
            rhos.append(spearman(KEEP, vals))
        verdict = 'PASS' if min(rhos) > 0.9 else 'FAIL'
        out['gate'][stat] = dict(spearman_per_seed=[round(r, 4) for r in rhos],
                                 min_rho=round(min(rhos), 4), verdict=verdict)
        print(f'GATE {stat}: {verdict}  (spearman per seed '
              f'{[round(r, 3) for r in rhos]})', flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print('GATE5_DONE', flush=True)


if __name__ == '__main__':
    main()
