"""Core adaptation's two knobs, the analogue of the p_as x p_cr grid.

  retain_off_weight   weight on kl(s_off_r, tr_), which pins the SHIPPED core to the donor
                      teacher on retain data (core_adapt5.py:246). retain_target.py showed
                      changing this term changes the leak in 5/5 seeds, so it is the leak
                      path -- the structural analogue of GRAM's p_as.
  core_lr             how far the core can move per step: removal, and damage.

Donors do not depend on either knob, so one donor is trained per (rate, seed) and reused
across all 15 cells. That is most of the saving; without it this grid is 3x the compute.

The reliability filter differs from the GRAM grid. There the question was whether the module
ever learned 5. Here the capability is copied from a donor that already has it, so what breaks
instead is RETENTION: retain_target.py lost the other nine digits outright on one seed
(other-nine 0.159). So a cell is only readable if the other nine digits survived.

Scored on raw logit 5, never softmax and never recall -- see INSTRUMENT.md.
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

SHIP, ON = (0, 0), (1, 0)


@torch.no_grad()
def recall5(model, x, y, aux, bs=2048):
    model.eval()
    pred = torch.cat([forward(model, x[i:i + bs], aux).argmax(1).cpu()
                      for i in range(0, len(x), bs)])
    sel = (y.cpu() == TARGET)
    return float((pred[sel] == TARGET).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--row', type=float, nargs='+', default=[0.0, 0.5, 1.0, 2.0, 4.0],
                   help='retain_off_weight values')
    p.add_argument('--lr', type=float, nargs='+', default=[3e-6, 1e-5, 3e-5],
                   help='core_lr values')
    p.add_argument('--rates', type=float, nargs='+', default=[0.0, 0.3])
    p.add_argument('--kind', default='fn5')
    p.add_argument('--cycles', type=int, default=60)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    dev = torch.device(a.device)
    xte, yte = load_mnist(dev)[2:]
    pos = (yte.cpu().numpy() == TARGET)

    out = {'kind': a.kind, 'seed': a.seed, 'cycles': a.cycles, 'row': a.row, 'lr': a.lr,
           'rates': a.rates, 'ship_mask': list(SHIP), 'cells': {}}
    for rate in a.rates:
        donor = f'results/ca_donor_r{rate}_seed{a.seed}.pt'   # shared by all 15 cells
        for row in a.row:
            for lr in a.lr:
                tag = f'cagrid_{a.kind}_r{rate}_w{row}_lr{lr:g}_seed{a.seed}'
                ck = Path('results') / f'{tag}.pt'
                if not ck.exists():
                    cmd = [sys.executable, 'core_adapt5.py', '--device', str(dev),
                           '--cycles', str(a.cycles), '--seed', str(a.seed),
                           '--retain-off-weight', str(row), '--core-lr', str(lr),
                           '--donor', donor, '--save', str(ck),
                           '--out', f'results/{tag}.json']
                    if rate > 0:
                        cmd += ['--noise-kind', a.kind, '--noise-rate', str(rate)]
                    subprocess.run(cmd, check=True)
                m = GramNet().to(dev)
                m.load_state_dict(torch.load(ck, map_location=dev))
                rec = dict(retain_off_weight=row, core_lr=lr, rate=rate,
                           leak_raw=auc(raw5(m, xte, SHIP), pos),
                           leak_softmax=auc(p5(m, xte, SHIP), pos),
                           capability=recall5(m, xte, yte, ON),
                           module_on_raw=auc(raw5(m, xte, ON), pos),
                           ablated_recall5=recall5(m, xte, yte, SHIP),
                           acc_others=acc_others(m, xte, yte, SHIP))
                out['cells'][f'r{rate}_w{row}_lr{lr:g}'] = rec
                print(json.dumps(dict(event='cell', **{
                    k: (round(v, 5) if isinstance(v, float) else v)
                    for k, v in rec.items()})), flush=True)
                ck.unlink()        # 225 checkpoints is 760MB and nothing rescores them

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print('CA_GRID_DONE', a.out, flush=True)


if __name__ == '__main__':
    main()
