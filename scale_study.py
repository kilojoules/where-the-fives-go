"""Forgetting against model size and label noise: behaviour, probes, and a fine-tune attack.

Every result in this repo so far was measured at dim=256 (861k params) with clean or noisy
labels but one architecture. This sweeps size as a control parameter and runs three
instruments on the SAME checkpoint at each (arm, size, noise, seed):

  1. BEHAVIOUR   AUC over raw logit 5 on held-out 5s, shipped mask. Plus capability
                 (module-ON recall) and other-nine retention. Never recall for the leak and
                 never softmax P(5) -- see INSTRUMENT.md, where those two disagree in SIGN.

  2. PROBE       A BINARY 5-vs-rest linear probe at each depth (input/embed/block1/block2/
                 logits), scored by AUC. layer_probe.probe_fit trains a 10-way CE head and
                 reading column 5 off it was shown unreliable here -- it lands below raw for
                 GRAM and above it for core adaptation, because it optimises classification,
                 not ranking. A one-logit probe trained on the ranking question is the right
                 instrument, and the input stage is the floor that cannot lie: raw pixels
                 already give AUC ~0.96, so trunk numbers must be read against it.

  3. ATTACK      elicit.py's fine-tune attack, re-implemented for variable model size (the
                 original hardcodes GramNet() and a fixed checkpoint registry). k labeled 5s
                 plus 4k non-5s, attacker trains only the partitions it would actually own.
                 Reported as attacker-useful AUC: the best over the trajectory subject to
                 non-5 accuracy staying >= 95%, so an attack that wins by wrecking the model
                 does not count.

Core adaptation is included despite costing ~30x the other arms per cell. It is the arm where
these two instruments matter most: it is the only route whose leak RISES with mislabeling, and
it carries the largest instrument discrepancy on record here -- probe_on_logits read 0.9431
against a raw logit of 0.7519 on the same checkpoint. Whether its re-learned 5 is recoverable
by an attacker, or merely present, is what decides how much the teacher leak actually matters.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from absorb3 import auc, raw5, acc_others, forward, TARGET
from mnist_data import load_mnist
from model import GramNet

STAGES = ["input", "embed", "block1", "block2", "logits"]
ARMS = {  # name -> (train kwargs, shipped mask, partitions an attacker owns)
    "dense_all":  (dict(mode="dense"), (0, 0), ("core",)),
    "filter_no5": (dict(mode="dense", filter_digits=(5,)), (0, 0), ("core",)),
    "gram_main":  (dict(mode="gram"), (0, 1), ("core", "aux7")),
    # core adaptation ships core-only with its module detached, so an attacker owns the core
    "core_adapt5": (None, (0, 0), ("core",)),
}
CA_CYCLES = 60          # the operating point established in CORE_ADAPTATION.md
ATTACK_STEPS, ATTACK_BATCH = 300, 32


def stages_of(model, x, mask, bs=2048):
    """Representations at each depth, batched so dim=1024 does not blow memory."""
    outs = {s: [] for s in STAGES}
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = x[i:i + bs]
            outs["input"].append(xb.flatten(1))
            h = model.embed(xb)
            outs["embed"].append(h.clone())
            for j, b in enumerate(model.blocks):
                h = h + b(h, mask)
                outs[f"block{j+1}"].append(h.clone())
            outs["logits"].append(model.head(h))
    return {k: torch.cat(v) for k, v in outs.items()}


def binary_probe_auc(ftr, ytr_bin, fte, yte_bin, seed, iters=200):
    """One-logit 5-vs-rest probe, fitted with LBFGS and scored by AUC.

    This was SGD for a fixed epoch budget, and that was wrong twice over. Fitting a linear
    head is a small CONVEX problem, so the answer should not depend on how long a first-order
    optimiser was allowed to run -- and it did: on the same checkpoint the probe read 0.8364
    at 15 epochs, 0.8849 at 60 and 0.9319 at 200, still climbing. Worse, 0.8364 is BELOW the
    model's own raw logit-5 AUC of 0.8370, which is impossible for a converged probe, since
    'use logit 5 alone' is one of the linear combinations available to it. An underfit probe
    silently understates what an attacker can recover, which is the one direction an
    elicitation instrument must never err in.

    LBFGS solves it properly in one shot and is faster than the epoch loop it replaces.
    """
    torch.manual_seed(seed)
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6
    ftr, fte = (ftr - mu) / sd, (fte - mu) / sd
    head = torch.nn.Linear(ftr.shape[1], 1)
    pos_w = torch.tensor([(ytr_bin == 0).sum() / max(1, (ytr_bin == 1).sum())], dtype=torch.float)
    y = ytr_bin.float()
    opt = torch.optim.LBFGS(head.parameters(), max_iter=iters, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head(ftr).squeeze(1), y, pos_weight=pos_w)
        loss = loss + 1e-4 * head.weight.pow(2).sum()        # keeps it finite if separable
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return auc(head(fte).squeeze(1).numpy(), yte_bin.numpy().astype(bool))


def partition_of(name):
    if ".aux.0." in name:
        return "aux5"
    if ".aux.1." in name:
        return "aux7"
    return "core"


def attack(model, parts, mask, k, xtr, ytr, xte, yte, seed, lr=1e-3):
    """Fine-tune on k labeled 5s + 4k others. Returns attacker-useful AUC: best over the
    trajectory subject to non-5 accuracy >= 95%, so wrecking the model is not a win."""
    rng = np.random.RandomState(seed + 999)
    y = ytr.cpu().numpy()
    idx = np.concatenate([rng.choice(np.where(y == TARGET)[0], k, replace=False),
                          rng.choice(np.where(y != TARGET)[0], 4 * k, replace=False)])
    xa, ya = xtr[idx], ytr[idx]
    ps = [p for n, p in model.named_parameters() if partition_of(n) in parts]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in ps:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(ps, lr=lr)
    gen = torch.Generator().manual_seed(seed)
    pos = (yte.cpu().numpy() == TARGET)
    best = -1.0
    for step in range(ATTACK_STEPS + 1):
        if step % 50 == 0:
            a = auc(raw5(model, xte, mask), pos)
            if acc_others(model, xte, yte, mask) >= 0.95:
                best = max(best, a)
        b = torch.randint(len(xa), (ATTACK_BATCH,), generator=gen)
        loss = F.cross_entropy(forward(model, xa[b], mask), ya[b])
        opt.zero_grad(); loss.backward(); opt.step()
    return float(best)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dim', type=int, required=True)
    p.add_argument('--rates', type=float, nargs='+', default=[0.0, 0.1, 0.3])
    p.add_argument('--arms', nargs='+', default=list(ARMS))
    p.add_argument('--ks', type=int, nargs='+', default=[10, 50])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--kind', default='fn5')
    p.add_argument('--probe-n', type=int, default=8000)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    dev = torch.device(a.device)
    xtr, ytr, xte, yte = load_mnist(dev)
    pos = (yte.cpu().numpy() == TARGET)
    ch, ah = 2 * a.dim, max(16, a.dim // 4)
    from train import Config, run as train_run

    out = {'dim': a.dim, 'core_hidden': ch, 'aux_hidden': ah, 'seed': a.seed,
           'kind': a.kind, 'cells': {}}
    rng = np.random.RandomState(a.seed + 7)
    sub = rng.choice(len(xtr), a.probe_n, replace=False)     # probe train split
    xp, yp = xtr[sub], ytr[sub]

    for rate in a.rates:
        for arm in a.arms:
            kw, mask, parts = ARMS[arm]
            name = f'scale_{arm}_d{a.dim}_r{rate}'
            wp = Path('results') / f'{name}_seed{a.seed}.pt'
            if not wp.exists():
                if arm == 'core_adapt5':
                    cmd = [sys.executable, 'core_adapt5.py', '--device', str(dev),
                           '--cycles', str(CA_CYCLES), '--seed', str(a.seed),
                           '--dim', str(a.dim), '--core-hidden', str(ch),
                           '--aux-hidden', str(ah),
                           '--donor', f'results/{name}_donor_seed{a.seed}.pt',
                           '--save', str(wp), '--out', f'results/{name}_seed{a.seed}.json']
                    if rate > 0:
                        cmd += ['--noise-kind', a.kind, '--noise-rate', str(rate)]
                    subprocess.run(cmd, check=True)
                else:
                    train_run(Config(name=name, seed=a.seed, device=str(dev), dim=a.dim,
                                     core_hidden=ch, aux_hidden=ah, save_weights=True,
                                     noise_kind=(a.kind if rate > 0 else ''),
                                     noise_rate=rate, **kw), out_dir='results')

            def fresh():
                m = GramNet(dim=a.dim, core_hidden=ch, aux_hidden=ah).to(dev)
                m.load_state_dict(torch.load(wp, map_location=dev))
                return m

            m = fresh()
            rec = dict(arm=arm, rate=rate, dim=a.dim,
                       params=sum(q.numel() for q in m.parameters()),
                       leak_raw=auc(raw5(m, xte, mask), pos),
                       acc_others=acc_others(m, xte, yte, mask))
            if arm in ('gram_main', 'core_adapt5'):
                on = (1, 1) if arm == 'gram_main' else (1, 0)
                pred = torch.cat([forward(m, xte[i:i+2048], on).argmax(1).cpu()
                                  for i in range(0, len(xte), 2048)])
                sel = (yte.cpu() == TARGET)
                rec['capability'] = float((pred[sel] == TARGET).float().mean())

            str_ = stages_of(m, xp, mask)
            ste = stages_of(m, xte, mask)
            ybin_tr = (yp == TARGET).long()
            ybin_te = (yte == TARGET).long()
            rec['probe'] = {s: binary_probe_auc(str_[s].cpu(), ybin_tr.cpu(), ste[s].cpu(),
                                               ybin_te.cpu(), a.seed)
                            for s in STAGES}

            rec['attack'] = {}
            for k in a.ks:
                rec['attack'][str(k)] = attack(fresh(), parts, mask, k,
                                               xtr, ytr, xte, yte, a.seed)

            out['cells'][f'r{rate}_{arm}'] = rec
            print(json.dumps(dict(event='cell', arm=arm, rate=rate, dim=a.dim,
                                  leak=round(rec['leak_raw'], 4),
                                  probe={s: round(v, 3) for s, v in rec['probe'].items()},
                                  attack={k: round(v, 4) for k, v in rec['attack'].items()})),
                  flush=True)
            wp.unlink()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print('SCALE_STUDY_DONE', a.out, flush=True)


if __name__ == '__main__':
    main()
