"""Is "bigger does not help" actually "bigger at the wrong learning rate does not help"?

size_pilot.py trained every width at train.py's default lr=1e-3 for 10 epochs. Holding the
learning rate fixed across a 33x width change is the standard way a scaling study lies, and
MNIST accuracy saturates over a wide LR range, so the flat acc_others (0.975-0.978 at every
size) is NOT evidence the LR was right -- only that the model trained.

Protocol, to avoid the circularity that would make this worthless: the learning rate is chosen
per (width, noise) by TASK performance -- module-ON recall on 5s, which is what the model is
for -- and the leak is then reported at that choice. Selecting the LR by the leak itself would
just find whichever LR happened to suppress logit 5, which is the number under test.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from absorb3 import auc, raw5, acc_others, forward, TARGET
from mnist_data import load_mnist
from model import GramNet

PAS, PCR = 0.75, 1.0      # the front knee from the p_as x p_cr grid


@torch.no_grad()
def recall5(m, x, y, aux, bs=2048):
    m.eval()
    pred = torch.cat([forward(m, x[i:i+bs], aux).argmax(1).cpu() for i in range(0, len(x), bs)])
    sel = (y.cpu() == TARGET)
    return float((pred[sel] == TARGET).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dim', type=int, required=True)
    p.add_argument('--lrs', type=float, nargs='+', default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    p.add_argument('--rates', type=float, nargs='+', default=[0.0, 0.3])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--kind', default='fn5')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    dev = torch.device(a.device)
    xte, yte = load_mnist(dev)[2:]
    pos = (yte.cpu().numpy() == TARGET)
    ch, ah = 2 * a.dim, max(16, a.dim // 4)
    from train import Config, run as train_run

    out = {'dim': a.dim, 'seed': a.seed, 'lrs': a.lrs, 'rates': a.rates, 'cells': {}}
    for rate in a.rates:
        for lr in a.lrs:
            name = f'lrs_d{a.dim}_lr{lr:g}_r{rate}'
            wp = Path('results') / f'{name}_seed{a.seed}.pt'
            if not wp.exists():
                train_run(Config(name=name, seed=a.seed, device=str(dev), mode='gram',
                                 dim=a.dim, core_hidden=ch, aux_hidden=ah, lr=lr,
                                 p_as=PAS, p_cr=PCR, save_weights=True,
                                 noise_kind=(a.kind if rate > 0 else ''),
                                 noise_rate=rate), out_dir='results')
            m = GramNet(dim=a.dim, core_hidden=ch, aux_hidden=ah).to(dev)
            m.load_state_dict(torch.load(wp, map_location=dev))
            lk = auc(raw5(m, xte, (0, 1)), pos)
            rec = dict(dim=a.dim, lr=lr, rate=rate, leak_raw=lk, info=abs(lk - 0.5),
                       capability=recall5(m, xte, yte, (1, 1)),
                       acc_others=acc_others(m, xte, yte, (0, 1)))
            out['cells'][f'r{rate}_lr{lr:g}'] = rec
            print(json.dumps(dict(event='cell', **{k: (round(v, 4) if isinstance(v, float)
                                                       else v) for k, v in rec.items()})),
                  flush=True)
            wp.unlink()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print('LR_SWEEP_DONE', a.out, flush=True)


if __name__ == '__main__':
    main()
