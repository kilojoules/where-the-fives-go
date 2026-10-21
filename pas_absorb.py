"""Does p_as still buy anything once the labels are noisy?

p_as exists only in GRAM (train.py:65, applied at train.py:188): on a routed-5 sub-batch the
core receives gradient with probability p_as. It is the knob that governs how much of the 5
leaks into the core through the ROUTED path, and results/pas_depth.json shows it reaching all
the way down into the trunk, not just the readout.

But a true 5 wearing another digit's label is never routed. It sits in the CORE sub-batch,
which gets full core gradient at every p_as (train.py's `core_sel` group, unconditional). So
the routed leak is tunable and the mislabeled leak is not, and crossing the two asks which one
binds.

Only 0.25 and 0.5 are working settings: results/pas_sweep.json has module-ON recall 0.0168 at
p_as=0 and 0.0527 at 0.1 (the capability never got learned, so there was nothing to remove)
and ablated recall 0.6289 at 1.0 (ablation stopped suppressing). module_on_recall5 and
ablated_recall5 are carried here so a cell cannot be read as a win when it is a broken run.

AUC over P(5), never recall, for the leak numbers themselves.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from absorb3 import auc, p5, raw5, acc_others, leaked_set, forward, TARGET
from mnist_data import load_mnist
from model import GramNet

SHIP = (0, 1)        # module 5 detached, module 7 still attached -- what GRAM ships
ON = (1, 1)          # both modules attached -- is the capability there at all?


@torch.no_grad()
def recall5(model, x, y, aux, bs=1024):
    model.eval()
    pred = torch.cat([forward(model, x[i:i + bs], aux).argmax(1).cpu()
                      for i in range(0, len(x), bs)])
    sel = (y.cpu() == TARGET)
    return float((pred[sel] == TARGET).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pas', type=float, nargs='+', default=[0.0, 0.25, 0.5, 1.0])
    p.add_argument('--rates', type=float, nargs='+', default=[0.0, 0.1, 0.3])
    p.add_argument('--kind', default='fn5', choices=['fn5', 'fp5', 'swap57'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=Path('results/pas_absorb.json'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    dev = torch.device(a.device)

    xtr, ytr, xte, yte = load_mnist(dev)
    y_clean_np = ytr.cpu().numpy()
    pos_te = (yte.cpu().numpy() == TARGET)

    from train import Config, run as train_run

    out = {'kind': a.kind, 'seed': a.seed, 'pas': a.pas, 'rates': a.rates,
           'ship_mask': list(SHIP), 'module_on_mask': list(ON), 'cells': {}}

    for rate in a.rates:
        leaked, neg, _ = leaked_set(y_clean_np, rate, a.kind, a.seed)
        x_leak = xtr[torch.tensor(leaked, device=dev)] if len(leaked) else None
        x_neg = xtr[torch.tensor(neg, device=dev)]

        for pas in a.pas:
            tag = f'pasabs_{a.kind}_r{rate}_pas{pas}'
            wp = Path('results') / f'{tag}_seed{a.seed}.pt'
            if not wp.exists():
                train_run(Config(name=tag, seed=a.seed, device=str(dev), mode='gram',
                                 p_as=pas, save_weights=True,
                                 noise_kind=(a.kind if rate > 0 else ''), noise_rate=rate),
                          out_dir='results')
            m = GramNet().to(dev)
            m.load_state_dict(torch.load(wp, map_location=dev))

            rec = dict(p_as=pas, rate=rate, n_relabeled=len(leaked),
                       auc_ship=auc(p5(m, xte, SHIP), pos_te),
                       auc_ship_rawlogit=auc(raw5(m, xte, SHIP), pos_te),
                       auc_module_on=auc(p5(m, xte, ON), pos_te),
                       ablated_recall5=recall5(m, xte, yte, SHIP),
                       module_on_recall5=recall5(m, xte, yte, ON),
                       acc_others=acc_others(m, xte, yte, SHIP))
            if x_leak is not None:
                sc = np.concatenate([p5(m, x_leak, SHIP), p5(m, x_neg, SHIP)])
                lab = np.concatenate([np.ones(len(leaked), bool), np.zeros(len(neg), bool)])
                rec['auc_leaked'] = auc(sc, lab)
            out['cells'][f'r{rate}_pas{pas}'] = rec
            print(json.dumps(dict(event='cell', **{
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})),
                flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print('PAS_ABSORB_DONE', a.out, flush=True)


if __name__ == '__main__':
    main()
