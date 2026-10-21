"""Three ways to not recognize 5s, measured the same way.

  filter_no5   the 5 was never in the training data
  gram_main    the 5 was routed into an aux module during training (ablate it to ship)
  core_adapt5  the 5 was moved out of a core that had already learned it

Plus dense_all as the ceiling: a model that still knows 5.

Each arm is scored in the configuration it SHIPS in, which is not the same mask for all of
them, and that is the point:

  dense_all    core only            -- nothing was removed
  filter_no5   core only            -- there is nothing to remove
  gram_main    ablate5 = (0, 1)     -- module 5 detached, module 7 still attached
  core_adapt5  (0, 0)               -- the adapted module detached

The headline number is recall on the 892 held-out 5s; the control is mean recall on the other
nine digits, which says whether a method bought its forgetting by damaging the model.
"""
import argparse
import json
from pathlib import Path

import torch

from model import GramNet
from train import Config, load_mnist, metrics_from_logits, run as train_run

TARGET = 5
ARMS = [
    ('dense_all',  dict(mode='dense'),                              (0, 0), 'knows 5 (ceiling)'),
    ('filter_no5', dict(mode='dense', filter_digits=(5,)),          (0, 0), 'never in the data'),
    ('gram_main',  dict(mode='gram'),                               (0, 1), 'routed away in training'),
]


def forward(model, x, aux):
    h = model.embed(x)
    for block in model.blocks:
        out = block.core(h)
        for m, mod in zip(aux, block.aux):
            if m:
                out = out + m * mod(h)
        h = h + out
    return model.head(h)


def score(model, xte, yte, aux, dev):
    model.eval()
    with torch.no_grad():
        lg = torch.cat([forward(model, xte[i:i + 1024], aux).cpu() for i in range(0, len(xte), 1024)])
    m = metrics_from_logits(lg, yte)
    return dict(acc=m['acc'], recall5=m['recall'][TARGET],
                recall_others=sum(v for k, v in m['recall'].items() if k != TARGET) / 9,
                recall_per_digit={str(k): v for k, v in m['recall'].items()},
                ship_mask=list(aux))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--adapted', type=Path, default=Path('results/core_adapt5_c12.pt'))
    p.add_argument('--adapted-cycles', type=int, default=12)
    p.add_argument('--out', type=Path, default=Path('results/compare3.json'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    dev = torch.device(a.device)
    xte, yte = load_mnist(dev)[2:]

    out = {}
    for name, kw, mask, blurb in ARMS:
        tag = f'{name}_seed{a.seed}'
        wp = Path('results') / f'{tag}.pt'
        if not wp.exists():
            train_run(Config(name=name, seed=a.seed, device=str(dev), save_weights=True, **kw),
                      out_dir='results')
        m = GramNet().to(dev)
        m.load_state_dict(torch.load(wp, map_location=dev))
        out[name] = dict(blurb=blurb, **score(m, xte, yte, mask, dev))
        print(json.dumps(dict(event='arm', arm=name, ship_mask=list(mask),
                              recall5=round(out[name]['recall5'], 4),
                              recall_others=round(out[name]['recall_others'], 4))), flush=True)

    if a.adapted.exists():
        m = GramNet().to(dev)
        m.load_state_dict(torch.load(a.adapted, map_location=dev))
        out['core_adapt5'] = dict(blurb=f'moved out after training ({a.adapted_cycles} cycles)',
                                  **score(m, xte, yte, (0, 0), dev))
        print(json.dumps(dict(event='arm', arm='core_adapt5', ship_mask=[0, 0],
                              recall5=round(out['core_adapt5']['recall5'], 4),
                              recall_others=round(out['core_adapt5']['recall_others'], 4))), flush=True)
    else:
        print(f'WARNING: {a.adapted} missing; run core_adapt5.py first', flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(target_digit=TARGET, seed=a.seed, arms=out), indent=1))
    print('COMPARE3_DONE', a.out)


if __name__ == '__main__':
    main()
