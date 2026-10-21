"""Hillary Sanders' core adaptation (AIAO-121 "No-MLP C5"), ported to MNIST to forget 5.

The GRAM arms already in this repo route gradients DURING training: the aux module is told
about 5 from the start, so the core never learns it. Core adaptation asks a harder question —
**can you take a model that already knows 5, and move that knowledge out of its core
afterwards?** That is what AIAO-121 does to a 200M language model, and it is the method the
Lisp campaign calls `cycled-GRAM-OFF`.

Why MNIST is the better testbed for it: at 200M the Lisp capability sits at behavioural chance,
so every claim there is distributional. Here "did it forget 5" has an unambiguous answer —
recall on the 5 class, against 892 held-out fives.

RECIPE, transcribed from experiments/AIAO-121-core-adaptation/current_method.json (version 5,
arm `fullreplace-c5`), with the language-model shapes mapped onto this MLP:

  donor            a trained model whose CORE knows 5 (here: mode='dense' run)
  teacher          an independent frozen copy of the donor
  core scope       the block core MLPs only; embed and head frozen
                   (hers: "26 MLP weight matrices; other core parameters frozen")
  cycles           5
  per cycle:
    copy phase     512 aux updates with the core DROPPED (core_dropout 1.0, addition_gate 0).
                   The module must reproduce the teacher's 5-behaviour by itself.
    second stage   8 blocks x (64 aux updates, then 64 core updates), alternating,
                   addition_gate 1, dropout 0.
  core loss        KL_5(teacher || core+aux) + KL_retain(teacher || core+aux)
                   + 2 * KL_retain(teacher || core alone)        <- the retain-OFF term
  aux loss         CE+KL on 5 through the live route, + 0.25*(CE+KL) on retain
  lr               aux 2e-4, core 1e-5          (hers, used as published and NOT retuned)

The retain-OFF term is the mechanism: it pins the core to teacher behaviour on non-5 digits
*with the module off*, which is exactly the state the model ships in.

Reported at the end, all on the held-out test set:
  ablate5 recall on 5   the number that says whether it forgot      (donor ~0.98)
  ablate5 accuracy      whether the other nine digits survived
  full recall on 5      whether the capability still exists with the module ON
A method that merely damages the model shows both numbers falling together.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GramNet
from mnist_data import load_mnist, apply_label_noise   # shared with train.py

TARGET = 5

def metrics_from_logits(logits, y):
    """Copied from train.py so this script does not import it (torchvision)."""
    pred = logits.argmax(1)
    y = y.cpu()
    acc = (pred == y).float().mean().item()
    conf = torch.zeros(10, 10, dtype=torch.long)
    for t, pr in zip(y.tolist(), pred.tolist()):
        conf[t, pr] += 1
    recall = {c: (conf[c, c].item() / max(1, conf[c].sum().item())) for c in range(10)}
    return {'acc': acc, 'recall': recall}


def train_donor(xtr, ytr, dev, gen, seed, epochs=10, bs=256, lr=1e-3, wd=1e-4):
    """train.py's dense arm: core-only forward, plain CE. Same hyperparameters."""
    torch.manual_seed(seed)
    m = GramNet().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    n = len(xtr)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen, device=dev)
        tot = 0.0
        for i in range(0, n - bs + 1, bs):
            j = perm[i:i + bs]
            loss = F.cross_entropy(forward(m, xtr[j], True, (0, 0)), ytr[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
        print(json.dumps(dict(event='donor-epoch', epoch=ep + 1,
                              loss=round(tot / max(1, n // bs), 4))), flush=True)
    return m


def forward(model, x, use_core=True, aux=(0, 0)):
    """One forward with explicit control of the core and each aux module.

    use_core=False is the copy phase's core_dropout=1.0: the module alone must carry the
    capability. aux=(1,0) is the deployed-with-module state; aux=(0,0) is what ships.
    """
    h = model.embed(x)
    for block in model.blocks:
        out = block.core(h) if use_core else torch.zeros_like(h)
        for m, mod in zip(aux, block.aux):
            if m:
                out = out + m * mod(h)
        h = h + out
    return model.head(h)


def kl(student_logits, teacher_logits):
    return F.kl_div(student_logits.log_softmax(-1), teacher_logits.log_softmax(-1),
                    reduction='batchmean', log_target=True)


def batches(x, y, idx, bs, gen):
    perm = idx[torch.randperm(len(idx), generator=gen, device=idx.device)]
    for i in range(0, len(perm) - bs + 1, bs):
        j = perm[i:i + bs]
        yield x[j], y[j]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--donor', type=Path, help='donor .pt; trained here if absent')
    p.add_argument('--out', type=Path, default=Path('results/core_adapt5.json'))
    p.add_argument('--save', type=Path, default=Path('results/core_adapt5.pt'))
    p.add_argument('--cycles', type=int, default=5)
    p.add_argument('--copy-updates', type=int, default=512)
    p.add_argument('--stage2-blocks', type=int, default=8)
    p.add_argument('--per-block', type=int, default=64)
    p.add_argument('--aux-lr', type=float, default=2e-4)
    p.add_argument('--core-lr', type=float, default=1e-5)
    p.add_argument('--retain-off-weight', type=float, default=2.0)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--noise-kind', default='',
                   help="training-label noise ('fn5'/'fp5'/'swap57'); the target/retain "
                        'split and the donor both follow the NOISY labels')
    p.add_argument('--noise-rate', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()

    dev = torch.device(a.device)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    torch.manual_seed(a.seed)
    xtr, ytr, xte, yte = load_mnist(dev)
    y_clean = ytr.clone()
    if a.noise_kind:
        import numpy as np
        y_np = ytr.cpu().numpy().copy()
        apply_label_noise(y_np, np.ones(len(y_np), dtype=bool), a.noise_kind, a.noise_rate, a.seed)
        ytr = torch.tensor(y_np, device=dev)
        n_bad = int((ytr != y_clean).sum())
        print(json.dumps(dict(event='noise', kind=a.noise_kind, rate=a.noise_rate,
                              relabeled=n_bad,
                              true_5s_left_in_retain=int(((y_clean == TARGET) &
                                                          (ytr != TARGET)).sum()))), flush=True)

    # ---- donor: a model whose CORE already knows 5 -------------------------------------
    if a.donor and a.donor.exists():
        donor = GramNet().to(dev)
        donor.load_state_dict(torch.load(a.donor, map_location=dev))
        print(json.dumps(dict(event='donor', source=str(a.donor))), flush=True)
    else:
        print(json.dumps(dict(event='donor', source='training core-only dense on all digits')), flush=True)
        donor = train_donor(xtr, ytr, dev, gen, a.seed)
        # Save where --donor asked, so a sweep over noise rates does not overwrite one
        # rate's donor with another's.
        dest = a.donor or Path('results/donor_dense_all.pt')
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(donor.state_dict(), dest)

    teacher = copy.deepcopy(donor).eval().requires_grad_(False)
    model = copy.deepcopy(donor).to(dev)

    # core scope: block MLPs only. embed and head frozen, as in the 200M recipe.
    core_params, aux_params = [], []
    for n, prm in model.named_parameters():
        if '.aux.0.' in n:
            aux_params.append(prm)
        elif '.core.' in n:
            core_params.append(prm)
        else:
            prm.requires_grad_(False)
    for n, prm in model.named_parameters():
        if '.aux.1.' in n:
            prm.requires_grad_(False)          # aux7 unused here
    opt_aux = torch.optim.AdamW(aux_params, lr=a.aux_lr)
    opt_core = torch.optim.AdamW(core_params, lr=a.core_lr)

    is_t = ytr == TARGET
    idx_t = torch.nonzero(is_t).squeeze(1)
    idx_r = torch.nonzero(~is_t).squeeze(1)
    print(json.dumps(dict(event='data', target=TARGET, train_target=len(idx_t),
                          train_retain=len(idx_r), test_target=int((yte == TARGET).sum()))), flush=True)

    def report(tag):
        model.eval()
        with torch.no_grad():
            out = {}
            for pname, mask in (('ablate5', (0, 0)), ('full', (1, 0))):
                lg = torch.cat([forward(model, xte[i:i + 1024], True, mask).cpu()
                                for i in range(0, len(xte), 1024)])
                m = metrics_from_logits(lg, yte)
                out[pname] = dict(acc=m['acc'], recall5=m['recall'][TARGET],
                                  recall_others=sum(v for k, v in m['recall'].items()
                                                    if k != TARGET) / 9,
                                  recall_per_digit={str(k): v for k, v in m['recall'].items()})
        model.train()
        print(json.dumps(dict(event='eval', tag=tag, **{
            k: {kk: round(vv, 4) for kk, vv in v.items() if kk != 'recall_per_digit'}
            for k, v in out.items()})), flush=True)
        return out

    history = [dict(tag='donor', **report('donor'))]
    started = time.time()

    for cycle in range(1, a.cycles + 1):
        # ---- copy phase: core dropped, the module alone must carry 5 -------------------
        for prm in core_params:
            prm.requires_grad_(False)
        it = batches(xtr, ytr, idx_t, a.batch, gen)
        for step in range(a.copy_updates):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = batches(xtr, ytr, idx_t, a.batch, gen)
                xb, yb = next(it)
            with torch.no_grad():
                tl = teacher(xb, mask=(0, 0))
            sl = forward(model, xb, use_core=False, aux=(1, 0))
            loss = F.cross_entropy(sl, yb) + kl(sl, tl)
            opt_aux.zero_grad(set_to_none=True)
            loss.backward()
            opt_aux.step()
        for prm in core_params:
            prm.requires_grad_(True)

        # ---- second stage: alternate aux and core, module ON ---------------------------
        it_t = batches(xtr, ytr, idx_t, a.batch, gen)
        it_r = batches(xtr, ytr, idx_r, a.batch, gen)

        def nxt(it, idx):
            try:
                return next(it), it
            except StopIteration:
                it = batches(xtr, ytr, idx, a.batch, gen)
                return next(it), it

        for blk in range(a.stage2_blocks):
            for step in range(a.per_block):          # aux updates
                (xb, yb), it_t = nxt(it_t, idx_t)
                (xr, yr), it_r = nxt(it_r, idx_r)
                with torch.no_grad():
                    tl, tr_ = teacher(xb, mask=(0, 0)), teacher(xr, mask=(0, 0))
                sl = forward(model, xb, True, (1, 0))
                sr = forward(model, xr, True, (1, 0))
                loss = (F.cross_entropy(sl, yb) + kl(sl, tl)
                        + 0.25 * (F.cross_entropy(sr, yr) + kl(sr, tr_)))
                opt_aux.zero_grad(set_to_none=True)
                loss.backward()
                opt_aux.step()
            for step in range(a.per_block):          # core updates
                (xb, yb), it_t = nxt(it_t, idx_t)
                (xr, yr), it_r = nxt(it_r, idx_r)
                with torch.no_grad():
                    tl, tr_ = teacher(xb, mask=(0, 0)), teacher(xr, mask=(0, 0))
                s_on_t = forward(model, xb, True, (1, 0))
                s_on_r = forward(model, xr, True, (1, 0))
                s_off_r = forward(model, xr, True, (0, 0))      # the shipped state
                loss = (kl(s_on_t, tl) + kl(s_on_r, tr_)
                        + a.retain_off_weight * kl(s_off_r, tr_))
                opt_core.zero_grad(set_to_none=True)
                loss.backward()
                opt_core.step()
        history.append(dict(tag=f'cycle{cycle}', **report(f'cycle{cycle}')))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.save)
    a.out.write_text(json.dumps(dict(
        method='AIAO-121 No-MLP C5 core adaptation, ported to MNIST',
        source='experiments/AIAO-121-core-adaptation/current_method.json v5 (fullreplace-c5)',
        target_digit=TARGET, seed=a.seed,
        recipe=dict(cycles=a.cycles, copy_updates=a.copy_updates,
                    stage2_blocks=a.stage2_blocks, per_block=a.per_block,
                    aux_lr=a.aux_lr, core_lr=a.core_lr,
                    retain_off_weight=a.retain_off_weight, batch=a.batch,
                    core_scope='block core MLPs; embed and head frozen'),
        history=history, seconds=round(time.time() - started, 1),
        checkpoint=str(a.save)), indent=1))
    print('CORE_ADAPT5_DONE', a.out)


if __name__ == '__main__':
    main()
