"""p_as x p_cr, under label noise. Does p_cr substitute for p_as once labels are wrong?

The two GRAM knobs feed the core and the modules through different doors:

  p_as (train.py:188)  on a ROUTED-5 sub-batch, probability the core gets gradient. The only
                       channel teaching the core "5-pixels -> 5", and therefore the leak.
  p_cr (train.py:180)  on a CORE sub-batch, probability a random module is forward-activated
                       AND updated. train.py's own docstring calls this the absorption side
                       channel.

A true 5 wearing another digit's label is never routed -- it sits in the core sub-batch. So
p_as cannot reach it and p_cr can. The hypothesis: under fn5 noise p_cr partially SUBSTITUTES
for p_as, because the mislabeled 5s arriving through the core sub-batch are what module 5
learns from. results/pas_absorb_seed*.json already shows the shadow of this -- at p_as=0,
r=0.3, three of eight seeds learn 5s despite the routed path being gradient-free to the core.

Weights are not saved: the shared volume is near full and every statistic is recorded inline,
so there is nothing to rescore. Both leak statistics are logged because they disagree in sign
under label noise (INSTRUMENT.md).

A mean over this grid is a trap. Capability is BIMODAL in several cells -- p_as=0.25 at r=0.3
is 0.527+-0.341, some seeds learning 5s and some never -- so a cell rendering its mean states
something true of no single run. `usable_frac` (seeds with module-ON recall > 0.8) is carried
for exactly this reason, and a leak number from a config that never learned 5s is meaningless.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from absorb3 import auc, p5, raw5, acc_others, forward, TARGET
from mnist_data import load_mnist
from model import GramNet

SHIP = (0, 1)   # module 5 detached, module 7 attached -- what GRAM ships
ON = (1, 1)     # both attached -- was the capability ever learned


@torch.no_grad()
def recall5(model, x, y, aux, bs=2048):
    model.eval()
    pred = torch.cat([forward(model, x[i:i + bs], aux).argmax(1).cpu()
                      for i in range(0, len(x), bs)])
    sel = (y.cpu() == TARGET)
    return float((pred[sel] == TARGET).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pas', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument('--pcr', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument('--rates', type=float, nargs='+', default=[0.0, 0.1, 0.3])
    p.add_argument('--kind', default='fn5')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    dev = torch.device(a.device)

    xte, yte = load_mnist(dev)[2:]
    pos = (yte.cpu().numpy() == TARGET)
    from train import Config, run as train_run

    out = {'kind': a.kind, 'seed': a.seed, 'pas': a.pas, 'pcr': a.pcr, 'rates': a.rates,
           'ship_mask': list(SHIP), 'cells': {}}
    done = 0
    for rate in a.rates:
        for pas in a.pas:
            for pcr in a.pcr:
                tag = f'pcrgrid_{a.kind}_r{rate}_pas{pas}_pcr{pcr}'
                res = train_run(Config(name=tag, seed=a.seed, device=str(dev), mode='gram',
                                       p_as=pas, p_cr=pcr, save_weights=False,
                                       noise_kind=(a.kind if rate > 0 else ''),
                                       noise_rate=rate), out_dir='results')
                # train_run writes per-profile metrics; rescore from the live model is not
                # available, so reload nothing -- recompute from the logits it saved.
                npz = np.load(f'results/{tag}_seed{a.seed}.npz')
                lg_ship = torch.tensor(npz['logits_ablate5'].astype(np.float32))
                lg_on = torch.tensor(npz['logits_full'].astype(np.float32))
                cap = res['full']['recall'][TARGET]
                rec = dict(p_as=pas, p_cr=pcr, rate=rate,
                           leak_raw=auc(lg_ship[:, TARGET].numpy(), pos),
                           leak_softmax=auc(lg_ship.softmax(-1)[:, TARGET].numpy(), pos),
                           module_on_raw=auc(lg_on[:, TARGET].numpy(), pos),
                           capability=cap,
                           ablated_recall5=res['ablate5']['recall'][TARGET],
                           acc_others=float(np.mean([v for k, v in res['ablate5']['recall'].items()
                                                     if k != TARGET])))
                out['cells'][f'r{rate}_pas{pas}_pcr{pcr}'] = rec
                done += 1
                print(json.dumps(dict(event='cell', done=done, **{
                    k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})),
                    flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print('PCR_GRID_DONE', a.out, flush=True)


if __name__ == '__main__':
    main()
