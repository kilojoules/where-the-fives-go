"""Is the Pareto trade-off in figures/pareto.png a capacity limit?

Prior: probably not. GramNet at dim=256 is 861k parameters for MNIST, where a plain
784-100-10 MLP (~79k) already reaches ~98%. The model is ~11x over-served for the task. And
the leak is not obviously capacity-bound: p_as delivers gradient to the core on routed-5
sub-batches (train.py:188), so the core learns 5-features because it is trained on 5s. More
parameters do not stop a gradient arriving.

The capacity-adjacent hypothesis that IS worth testing: the aux module is small, so maybe the
core has to encode 5-features for the module to do its job, and a bigger module would not need
that. Note the confound this creates in the naive test -- scaling dim with aux_hidden=dim/4
also moves the aux share of parameters from 9.2% to 18.6%, so "bigger model" and "bigger
module relative to core" are not separable that way.

Two arms, therefore:
  'both'      dim scaled with core_hidden=2*dim, aux_hidden=dim/4  -- the naive 'bigger model'
  'core_only' dim and core_hidden scaled, aux_hidden PINNED at 64  -- bigger core, same module

If the front improves under 'both' but not 'core_only', it is the module's capacity that
mattered, not the model's.

Two settings per size: the front knee from the GRAM grid and the long-standing baseline.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from absorb3 import auc, raw5, acc_others, forward, TARGET
from mnist_data import load_mnist
from model import GramNet

SHIP, ON = (0, 1), (1, 1)
SETTINGS = [(0.75, 1.0, "knee"), (0.5, 0.5, "baseline")]


@torch.no_grad()
def recall5(model, x, y, aux, bs=2048):
    model.eval()
    pred = torch.cat([forward(model, x[i:i + bs], aux).argmax(1).cpu()
                      for i in range(0, len(x), bs)])
    sel = (y.cpu() == TARGET)
    return float((pred[sel] == TARGET).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dims', type=int, nargs='+', default=[128, 256, 512])
    p.add_argument('--arm', default='both', choices=['both', 'core_only'])
    p.add_argument('--rates', type=float, nargs='+', default=[0.0, 0.3])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--kind', default='fn5')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    dev = torch.device('cpu')
    xte, yte = load_mnist(dev)[2:]
    pos = (yte.cpu().numpy() == TARGET)
    from train import Config, run as train_run

    out = {'arm': a.arm, 'seed': a.seed, 'dims': a.dims, 'rates': a.rates, 'cells': {}}
    for dim in a.dims:
        ch = 2 * dim
        ah = dim // 4 if a.arm == 'both' else 64
        for rate in a.rates:
            for pas, pcr, tag in SETTINGS:
                name = f'size_{a.arm}_d{dim}_r{rate}_{tag}'
                res = train_run(Config(name=name, seed=a.seed, device='cpu', mode='gram',
                                       dim=dim, core_hidden=ch, aux_hidden=ah,
                                       p_as=pas, p_cr=pcr, save_weights=True,
                                       noise_kind=(a.kind if rate > 0 else ''),
                                       noise_rate=rate), out_dir='results')
                wp = Path('results') / f'{name}_seed{a.seed}.pt'
                m = GramNet(dim=dim, core_hidden=ch, aux_hidden=ah).to(dev)
                m.load_state_dict(torch.load(wp, map_location=dev))
                nparam = sum(q.numel() for q in m.parameters())
                naux = sum(q.numel() for n, q in m.named_parameters() if '.aux.' in n)
                lk = auc(raw5(m, xte, SHIP), pos)
                rec = dict(dim=dim, aux_hidden=ah, params=nparam,
                           aux_frac=round(naux / nparam, 4), rate=rate, setting=tag,
                           p_as=pas, p_cr=pcr, leak_raw=lk, info=abs(lk - 0.5),
                           capability=recall5(m, xte, yte, ON),
                           acc_others=acc_others(m, xte, yte, SHIP))
                out['cells'][f'd{dim}_r{rate}_{tag}'] = rec
                print(json.dumps(dict(event='cell', **{
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in rec.items()})), flush=True)
                wp.unlink()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print('SIZE_PILOT_DONE', a.out, flush=True)


if __name__ == '__main__':
    main()
