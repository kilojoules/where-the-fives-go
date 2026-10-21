"""Is the noise-driven rise in P(5) a capability, or leftover softmax mass?

results/absorb3.json and results/pas_absorb.json both show AUC over softmax P(5) climbing
toward 0.99 as true 5s get mislabeled -- for every route, and at every p_as. Two cells say
that cannot all be capability:

  filter_no5 at r=0.3 keeps ZERO label-5 examples, so logit 5 is never once a positive
  training target, yet P(5) ranks held-out 5s at 0.9917.

  p_as=0 at r=0.1 scores auc_ship 0.8693 with module_on_recall5 0.0000 -- nothing was
  learned, so there is nothing to leak.

Softmax is normalised, so P(5) rises whenever the OTHER nine logits fall. A 5 mislabeled '3'
teaches 'not 3' to every other example and drives a different random wrong digit each time,
so the nine trained logits disagree on 5-shaped pixels while logit 5, untrained, is left as
the residual. That inflates P(5) without logit 5 knowing anything.

Three readouts on the same checkpoints, no retraining:

  auc_softmax   AUC over P(5)                 -- what the sweeps reported
  auc_rawlogit  AUC over logit 5 alone        -- immune to what the other nine do
  auc_margin    AUC over logit5 - max(others) -- the quantity argmax actually compares, so
                it answers 'does the model rank a 5 above its own best alternative'

If the rise lives in auc_softmax and not in auc_rawlogit, the sweeps measured the instrument.
"""
import json
from pathlib import Path

import numpy as np
import torch

from absorb3 import auc, forward, TARGET
from mnist_data import load_mnist
from model import GramNet

RATES = [0.0, 0.05, 0.1, 0.3]
PAS = [0.0, 0.1, 0.25, 0.5, 1.0]
ARM_MASK = {'dense_all': (0, 0), 'filter_no5': (0, 0),
            'gram_main': (0, 1), 'core_adapt5': (0, 0)}


@torch.no_grad()
def logits_of(model, x, aux, bs=2048):
    model.eval()
    return torch.cat([forward(model, x[i:i + bs], aux).cpu() for i in range(0, len(x), bs)])


def three(lg, pos):
    raw = lg[:, TARGET].numpy()
    other = lg.clone()
    other[:, TARGET] = -float('inf')
    margin = (lg[:, TARGET] - other.max(1).values).numpy()
    return dict(auc_softmax=auc(lg.softmax(-1)[:, TARGET].numpy(), pos),
                auc_rawlogit=auc(raw, pos),
                auc_margin=auc(margin, pos))


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xte, yte = load_mnist(dev)[2:]
    pos = (yte.cpu().numpy() == TARGET)
    out = {'routes': {}, 'pas': {}}

    def load(p):
        m = GramNet().to(dev)
        m.load_state_dict(torch.load(p, map_location=dev))
        return m

    for rate in RATES:
        for arm, mask in ARM_MASK.items():
            p = (Path('results') / (f'absorb_fn5_r{rate}_core_adapt5.pt' if arm == 'core_adapt5'
                                    else f'absorb_fn5_r{rate}_{arm}_seed0.pt'))
            if not p.exists():
                print(f'missing {p}', flush=True)
                continue
            rec = three(logits_of(load(p), xte, mask), pos)
            out['routes'].setdefault(arm, {})[str(rate)] = rec
            print(json.dumps(dict(event='route', arm=arm, rate=rate,
                                  **{k: round(v, 4) for k, v in rec.items()})), flush=True)

    for rate in [0.0, 0.1, 0.3]:
        for pas in PAS:
            p = Path('results') / f'pasabs_fn5_r{rate}_pas{pas}_seed0.pt'
            if not p.exists():
                print(f'missing {p}', flush=True)
                continue
            rec = three(logits_of(load(p), xte, (0, 1)), pos)
            out['pas'][f'r{rate}_pas{pas}'] = rec
            print(json.dumps(dict(event='pas', p_as=pas, rate=rate,
                                  **{k: round(v, 4) for k, v in rec.items()})), flush=True)

    Path('results').mkdir(exist_ok=True)
    Path('results/absorb_raw.json').write_text(json.dumps(out, indent=1))
    print('ABSORB_RAW_DONE', flush=True)


if __name__ == '__main__':
    main()
