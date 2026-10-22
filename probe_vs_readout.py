"""Two ways the 5 can survive removal, swept together: what a probe recovers, and what the
model emits.

The readout and the probe are different threat models. A user who only sees outputs gets the
READOUT: AUC over the shipped model's own raw logit 5. An attacker who can reach activations
and fit a linear head gets the PROBE. A method can do well on one and badly on the other --
figures/probe_depth.png shows GRAM detached emitting 0.851 while its logits probe reads 0.701,
and core adaptation doing the reverse -- so minimising one says nothing about the other.

This sweeps both objectives across each method's own hyperparameter grid:

  gram        p_as x p_cr, shipped mask (0,1)
  coreadapt   retain_off_weight x core_lr, shipped mask (0,0), 60 cycles

PROBE is a binary 5-vs-rest linear probe fitted on the LOGITS -- the same place the readout
reads from. The two therefore differ only in what is allowed to look at them: the readout is
logit 5 alone, the probe is a learned combination of all ten. Because "use logit 5" is a
special case of that combination, the probe is a strict upper bound on what any linear reader
can get from the output layer, and the gap between them is exactly what a method leaves
lying in the other nine logits.

Trunk depths are recorded too but are not the axis: the best trunk probe is always `embed`,
which sits barely above the 0.956 raw-pixel floor, so maximising over depth measures MNIST's
own separability rather than anything a removal method did.

Each cell also records retention and capability so the same reliability filters used elsewhere
can be applied: GRAM cells where the module never learned 5, core-adaptation cells where the
other nine digits did not survive.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from absorb3 import auc, raw5, acc_others, forward, TARGET
from mnist_data import load_mnist
from model import GramNet
from scale_study import binary_probe_auc, stages_of

LOGITS = "logits"
TRUNK = ["embed", "block1", "block2"]


@torch.no_grad()
def recall5(m, x, y, aux, bs=2048):
    m.eval()
    pred = torch.cat([forward(m, x[i:i+bs], aux).argmax(1).cpu() for i in range(0, len(x), bs)])
    sel = (y.cpu() == TARGET)
    return float((pred[sel] == TARGET).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['gram', 'coreadapt'])
    p.add_argument('--a', type=float, required=True, help='p_as, or retain_off_weight')
    p.add_argument('--b', type=float, required=True, help='p_cr, or core_lr')
    p.add_argument('--rate', type=float, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--kind', default='fn5')
    p.add_argument('--cycles', type=int, default=60)
    p.add_argument('--probe-n', type=int, default=8000)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        print('exists, skipping'); return
    dev = torch.device(a.device)
    xtr, ytr, xte, yte = load_mnist(dev)
    pos = (yte.cpu().numpy() == TARGET)

    tag = f'pvr_{a.method}_a{a.a:g}_b{a.b:g}_r{a.rate}_s{a.seed}'
    wp = Path('results') / f'{tag}.pt'
    if a.method == 'gram':
        mask, on = (0, 1), (1, 1)
        from train import Config, run as train_run
        train_run(Config(name=tag, seed=a.seed, device=str(dev), mode='gram',
                         p_as=a.a, p_cr=a.b, save_weights=True,
                         noise_kind=(a.kind if a.rate > 0 else ''), noise_rate=a.rate),
                  out_dir='results')
        wp = Path('results') / f'{tag}_seed{a.seed}.pt'
    else:
        mask, on = (0, 0), (1, 0)
        donor = f'results/ca_donor_r{a.rate}_seed{a.seed}.pt'
        cmd = [sys.executable, 'core_adapt5.py', '--device', str(dev), '--cycles', str(a.cycles),
               '--seed', str(a.seed), '--retain-off-weight', str(a.a), '--core-lr', str(a.b),
               '--donor', donor, '--save', str(wp), '--out', f'results/{tag}_ca.json']
        if a.rate > 0:
            cmd += ['--noise-kind', a.kind, '--noise-rate', str(a.rate)]
        subprocess.run(cmd, check=True)

    m = GramNet().to(dev)
    m.load_state_dict(torch.load(wp, map_location=dev))

    rng = np.random.RandomState(a.seed + 7)
    sub = rng.choice(len(xtr), a.probe_n, replace=False)
    str_ = stages_of(m, xtr[sub], mask)
    ste = stages_of(m, xte, mask)
    ybin_tr = (ytr[sub] == TARGET).long()
    ybin_te = (yte == TARGET).long()
    probe = {s: binary_probe_auc(str_[s].cpu(), ybin_tr.cpu(), ste[s].cpu(), ybin_te.cpu(),
                                 a.seed) for s in TRUNK + [LOGITS]}
    readout = auc(raw5(m, xte, mask), pos)

    rec = dict(method=a.method, a=a.a, b=a.b, rate=a.rate, seed=a.seed,
               probe_by_depth=probe,
               probe_logits=probe[LOGITS],
               probe_logits_info=abs(probe[LOGITS] - 0.5),
               probe_best_trunk=max(probe[s] for s in TRUNK),
               readout_auc=readout, readout_info=abs(readout - 0.5),
               capability=recall5(m, xte, yte, on),
               acc_others=acc_others(m, xte, yte, mask))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in rec.items() if k != 'probe_by_depth'}), flush=True)
    wp.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
