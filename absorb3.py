"""What mislabeled 5s do to each removal route.

Every route in this repo decides what to remove by reading the LABEL:

  filter_no5   drops examples whose label is 5
  gram_main    routes examples whose label is 5 into module 5
  core_adapt5  splits target/retain on whether the label is 5 (core_adapt5.py: is_t = ytr == 5)

So the label is the oracle for provenance, and all three trust it equally. Under 'fn5' noise
-- a fraction of true 5s carrying some other digit's label -- every route therefore absorbs
the SAME mislabeled images into the part it ships. The question this script asks is not how
many leak but what each route DOES with the ones that leak, because the three mechanisms give
those images very different gradients:

  filter_no5   ordinary CE toward a WRONG digit     -> 5-shaped pixels, label says 'not 5'
  gram_main    ordinary core CE toward a WRONG digit (p_cr module exposure aside)
  core_adapt5  the retain-OFF term pins the shipped core to the TEACHER's output on these
               images (core_adapt5.py: kl(s_off_r, tr_), weight 2.0), and the teacher is the
               donor, which still recognizes 5s. Core adaptation is the only route whose
               mechanism re-teaches the capability on a mislabeled example.

Scored by AUC over P(5), never recall: recall at a fixed argmax is trivially zero whenever
logit 5 sits below the best competitor, which is exactly what suppression arranges, and it
moves under a bias shift. See the retraction in the Lisp campaign's README.

Two readouts per arm:
  auc_clean     clean held-out test set, 892 true 5s vs the other 9108 -- does the SHIPPED
                model still rank 5s? This is the one that matters to a defender.
  auc_leaked    the mislabeled train images themselves (true 5s, wrong label) as positives
                against a matched sample of true non-5 train images -- did the shipped model
                absorb the specific examples that slipped past the label filter?
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import layer_probe
from mnist_data import load_mnist, apply_label_noise
from model import GramNet

TARGET = 5
# name, train.py kwargs, shipped aux mask, blurb
ARMS = [
    ('dense_all',  dict(mode='dense'),                       (0, 0), 'knows 5 (ceiling)'),
    ('filter_no5', dict(mode='dense', filter_digits=(5,)),   (0, 0), 'never in the data'),
    ('gram_main',  dict(mode='gram'),                        (0, 1), 'routed away in training'),
]
CORE_ADAPT = ('core_adapt5', (0, 0), 'moved out of a core that knew 5')


def forward(model, x, aux):
    h = model.embed(x)
    for block in model.blocks:
        out = block.core(h)
        for m, mod in zip(aux, block.aux):
            if m:
                out = out + m * mod(h)
        h = h + out
    return model.head(h)


@torch.no_grad()
def p5(model, x, aux, bs=1024):
    """Softmax P(5) under a given aux mask -- the shipped readout."""
    model.eval()
    return torch.cat([forward(model, x[i:i + bs], aux).softmax(-1)[:, TARGET].cpu()
                      for i in range(0, len(x), bs)]).numpy()


def auc(scores, pos):
    """Mann-Whitney AUC with MIDRANKS for ties.

    layer_probe.auroc, used by the other instruments here, breaks ties by argsort order
    instead, which biases a readout whose P(5) saturates to the same float for many inputs
    -- exactly what strong suppression produces. Both are reported per arm (auc_clean and
    auc_clean_argsort_ties) so the two can be compared directly; they agree wherever ties
    are absent.
    """
    pos = np.asarray(pos, dtype=bool)
    r = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(scores, kind='mergesort')
    s = np.asarray(scores)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def raw5(model, x, aux, bs=1024):
    """Logit 5 alone, unnormalised. Softmax P(5) rises whenever the other nine logits fall,
    so the two disagree by up to 0.75 AUC on these checkpoints (results/absorb_raw.json);
    both are recorded so neither can be mistaken for the other."""
    model.eval()
    return torch.cat([forward(model, x[i:i + bs], aux)[:, TARGET].cpu()
                      for i in range(0, len(x), bs)]).numpy()


@torch.no_grad()
def acc_others(model, x, y, aux, bs=1024):
    model.eval()
    pred = torch.cat([forward(model, x[i:i + bs], aux).argmax(1).cpu()
                      for i in range(0, len(x), bs)])
    sel = (y.cpu() != TARGET)
    return float((pred[sel] == y.cpu()[sel]).float().mean())


def leaked_set(y_clean_np, rate, kind, seed):
    """Absolute train indices of the true 5s that ended up with a wrong label, plus a
    matched sample of true non-5s to score them against. Same realization as every arm."""
    y_noisy = apply_label_noise(y_clean_np.copy(), np.ones(len(y_clean_np), dtype=bool),
                               kind, rate, seed)
    leaked = np.where((y_clean_np == TARGET) & (y_noisy != TARGET))[0]
    rng = np.random.RandomState(seed + 31)
    neg_pool = np.where(y_clean_np != TARGET)[0]
    neg = rng.choice(neg_pool, size=min(len(neg_pool), max(1, 10 * len(leaked))), replace=False)
    return leaked, neg, y_noisy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rates', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.3])
    p.add_argument('--kind', default='fn5', choices=['fn5', 'fp5', 'swap57'])
    p.add_argument('--cycles', type=int, default=60, help='core-adaptation operating point')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=Path('results/absorb3.json'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    dev = torch.device(a.device)

    xtr, ytr, xte, yte = load_mnist(dev)
    y_clean_np = ytr.cpu().numpy()
    pos_te = (yte.cpu().numpy() == TARGET)

    # train.py pulls in torchvision-free data via mnist_data; import late so a missing
    # torchvision cannot break the core-adapt-only path.
    from train import Config, run as train_run

    out = {'kind': a.kind, 'seed': a.seed, 'cycles': a.cycles, 'rates': a.rates, 'arms': {}}

    for rate in a.rates:
        leaked, neg, y_noisy = leaked_set(y_clean_np, rate, a.kind, a.seed)
        x_leak = xtr[torch.tensor(leaked, device=dev)] if len(leaked) else None
        x_neg = xtr[torch.tensor(neg, device=dev)]
        n_lab5 = int((y_noisy == TARGET).sum())
        print(json.dumps(dict(event='rate', rate=rate, relabeled=len(leaked),
                              label_5s_remaining=n_lab5)), flush=True)

        runs = []
        for name, kw, mask, blurb in ARMS:
            tag = f'absorb_{a.kind}_r{rate}_{name}'
            wp = Path('results') / f'{tag}_seed{a.seed}.pt'
            if not wp.exists():
                train_run(Config(name=tag, seed=a.seed, device=str(dev), save_weights=True,
                                 noise_kind=(a.kind if rate > 0 else ''), noise_rate=rate,
                                 **kw), out_dir='results')
            runs.append((name, wp, mask, blurb))

        # core adaptation: donor and cycles both on the noisy labels
        ck = Path('results') / f'absorb_{a.kind}_r{rate}_core_adapt5_seed{a.seed}.pt'
        if not ck.exists():
            cmd = [sys.executable, 'core_adapt5.py', '--device', str(dev),
                   '--cycles', str(a.cycles), '--seed', str(a.seed),
                   '--donor', f'results/absorb_{a.kind}_r{rate}_donor_seed{a.seed}.pt',
                   '--save', str(ck),
                   '--out',
                   f'results/absorb_{a.kind}_r{rate}_core_adapt5_seed{a.seed}.json']
            if rate > 0:
                cmd += ['--noise-kind', a.kind, '--noise-rate', str(rate)]
            print('+ ' + ' '.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        runs.append((CORE_ADAPT[0], ck, CORE_ADAPT[1], CORE_ADAPT[2]))

        for name, wp, mask, blurb in runs:
            m = GramNet().to(dev)
            m.load_state_dict(torch.load(wp, map_location=dev))
            rec = dict(blurb=blurb, ship_mask=list(mask),
                       auc_clean=auc(p5(m, xte, mask), pos_te),
                       auc_clean_rawlogit=auc(raw5(m, xte, mask), pos_te),
                       auc_clean_argsort_ties=float(
                           layer_probe.auroc(p5(m, xte, mask), pos_te.astype(int))),
                       acc_others=acc_others(m, xte, yte, mask),
                       n_relabeled=len(leaked))
            if name in ('gram_main', 'core_adapt5'):   # module back on: is it still there?
                on = (1, 1) if name == 'gram_main' else (1, 0)
                rec['auc_module_on'] = auc(p5(m, xte, on), pos_te)
                rec['module_on_mask'] = list(on)
            if x_leak is not None:
                sc = np.concatenate([p5(m, x_leak, mask), p5(m, x_neg, mask)])
                lab = np.concatenate([np.ones(len(leaked), bool), np.zeros(len(neg), bool)])
                rec['auc_leaked'] = auc(sc, lab)
            out['arms'].setdefault(name, {})[str(rate)] = rec
            print(json.dumps(dict(event='arm', rate=rate, arm=name,
                                  **{k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in rec.items() if k != 'blurb'})), flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print('ABSORB3_DONE', a.out, flush=True)


if __name__ == '__main__':
    main()
