"""Does core adaptation's leak come from the teacher? A causal test, not a map.

ABSORPTION.md reports that core adaptation is the only removal route whose logit-5 ranking
RISES when true 5s are mislabeled (+0.157, 8/8 seeds against every other route, p=0.008).
The proposed cause is one term. core_adapt5.py's retain-OFF loss pins the SHIPPED core to the
donor teacher's output on retain data, and a mislabeled 5 sits in that retain set because the
split reads the label (core_adapt5.py:166). The teacher reads the image correctly whatever
the label says, so the term re-teaches the capability on exactly the examples the label
filter missed.

If that is the cause, changing only that term changes only that effect. Three variants, same
donor, same seed, same noise realisation:

  teacher        KL to the donor over all ten logits -- the recipe as written
  teacher_mask5  KL over the nine NON-target logits only. The term's job is to hold the other
                 nine in place; it has no need to carry logit 5. Predicted: leak collapses,
                 retention survives. This is a fix, not an ablation.
  label          cross-entropy on the (noisy) retain labels -- what filtering and GRAM
                 effectively apply. Predicted: behaves like them, logit 5 FALLS under noise.

Donors are shared across the three variants at a given (rate, seed): the donor depends on
neither the variant nor anything downstream, so training it once keeps the comparison paired
and costs 1/3 as much.

Scored on raw logit 5, never recall and never softmax P(5) -- see INSTRUMENT.md.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from absorb3 import auc, raw5, p5, acc_others, forward, TARGET
from mnist_data import load_mnist
from model import GramNet

SHIP = (0, 0)      # core adaptation ships with its module detached
ON = (1, 0)        # module attached -- was the capability kept at all
VARIANTS = ['teacher', 'teacher_mask5', 'label']


@torch.no_grad()
def recall5(model, x, y, aux, bs=2048):
    model.eval()
    pred = torch.cat([forward(model, x[i:i + bs], aux).argmax(1).cpu()
                      for i in range(0, len(x), bs)])
    sel = (y.cpu() == TARGET)
    return float((pred[sel] == TARGET).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rates', type=float, nargs='+', default=[0.0, 0.1, 0.3])
    p.add_argument('--variants', nargs='+', default=VARIANTS)
    p.add_argument('--kind', default='fn5')
    p.add_argument('--cycles', type=int, default=60)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    dev = torch.device(a.device)
    xte, yte = load_mnist(dev)[2:]
    pos = (yte.cpu().numpy() == TARGET)

    out = {'kind': a.kind, 'seed': a.seed, 'cycles': a.cycles, 'rates': a.rates,
           'ship_mask': list(SHIP), 'cells': {}}
    for rate in a.rates:
        donor = f'results/rt_donor_r{rate}_seed{a.seed}.pt'   # shared by the three variants
        for var in a.variants:
            ck = Path('results') / f'rt_{a.kind}_r{rate}_{var}_seed{a.seed}.pt'
            if not ck.exists():
                cmd = [sys.executable, 'core_adapt5.py', '--device', str(dev),
                       '--cycles', str(a.cycles), '--seed', str(a.seed),
                       '--retain-off-target', var,
                       '--donor', donor, '--save', str(ck),
                       '--out', f'results/rt_{a.kind}_r{rate}_{var}_seed{a.seed}.json']
                if rate > 0:
                    cmd += ['--noise-kind', a.kind, '--noise-rate', str(rate)]
                print('+ ' + ' '.join(cmd), flush=True)
                subprocess.run(cmd, check=True)

            m = GramNet().to(dev)
            m.load_state_dict(torch.load(ck, map_location=dev))
            rec = dict(variant=var, rate=rate,
                       leak_raw=auc(raw5(m, xte, SHIP), pos),
                       leak_softmax=auc(p5(m, xte, SHIP), pos),
                       capability=recall5(m, xte, yte, ON),
                       module_on_raw=auc(raw5(m, xte, ON), pos),
                       ablated_recall5=recall5(m, xte, yte, SHIP),
                       acc_others=acc_others(m, xte, yte, SHIP))
            out['cells'][f'r{rate}_{var}'] = rec
            print(json.dumps(dict(event='cell', **{
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})),
                flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print('RETAIN_TARGET_DONE', a.out, flush=True)


if __name__ == '__main__':
    main()
