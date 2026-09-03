"""Feature-level knowledge probe (converged version, post-review).

For each checkpoint and feature mask, extract penultimate features, train a
fresh linear readout to convergence (cosine-decayed AdamW, saturation check),
and measure test recall/AUROC on 5s. Masks: (0,0) = core-only features;
(1,0) = core+module-5 features (gram checkpoints only) — the probe gap between
the two measures 5-information located in module-5 directly.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

from model import GramNet
from train import load_mnist

EPOCHS = 100


@torch.no_grad()
def features(model, x, mask, bs=2048):
    out = []
    for i in range(0, len(x), bs):
        h = model.embed(x[i:i + bs])
        for b in model.blocks:
            h = h + b(h, mask)
        out.append(h)
    return torch.cat(out)


def auroc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def probe_ckpt(prefix, seed, xtr, ytr, xte, yte, mask):
    model = GramNet()
    model.load_state_dict(torch.load(f"results/{prefix}_seed{seed}.pt"))
    model.eval()
    ftr = features(model, xtr, mask)
    fte = features(model, xte, mask)

    torch.manual_seed(seed)
    head = torch.nn.Linear(ftr.shape[1], 10)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    gen = torch.Generator().manual_seed(seed)
    prev_loss = None
    for ep in range(EPOCHS):
        perm = torch.randperm(len(ftr), generator=gen)
        tot = 0.0
        for i in range(0, len(ftr), 512):
            b = perm[i:i + 512]
            loss = F.cross_entropy(head(ftr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        prev_loss, delta = tot / len(ftr), (None if prev_loss is None else abs(prev_loss - tot / len(ftr)))
    with torch.no_grad():
        lg = head(fte)
        tr_acc = (head(ftr).argmax(1) == ytr).float().mean().item()
    pred = lg.argmax(1)
    i5 = yte == 5
    return {
        "recall5": (pred[i5] == 5).float().mean().item(),
        "auroc5": float(auroc(lg[:, 5].numpy(), i5.numpy().astype(int))),
        "acc": (pred == yte).float().mean().item(),
        "train_acc": tr_acc,
        "final_delta": delta,
    }


if __name__ == "__main__":
    torch.set_num_threads(8)
    xtr, ytr, xte, yte = load_mnist(torch.device("cpu"))
    gram = ["ck_gram_main", "ck_gram_f0.03"] + [f"dose_k{k}" for k in [0.03, 0.1, 0.4, 1.0]]
    jobs = [(p, (0, 0)) for p in ["ck_dense_all", "ck_filter_no5"] + gram]
    jobs += [(p, (1, 0)) for p in gram]
    out = {}
    for prefix, mask in jobs:
        rs = [probe_ckpt(prefix, s, xtr, ytr, xte, yte, mask) for s in [0, 1, 2]]
        out[f"{prefix}|{mask}"] = rs
        m = lambda key: (np.mean([r[key] for r in rs]), np.std([r[key] for r in rs]))
        print(f"{prefix:16s} mask{mask} recall5 {100*m('recall5')[0]:.2f}±{100*m('recall5')[1]:.2f} | "
              f"AUROC5 {m('auroc5')[0]:.4f}±{m('auroc5')[1]:.4f} | train_acc {100*m('train_acc')[0]:.2f} | "
              f"dloss {m('final_delta')[0]:.2e}", flush=True)
    with open("results/probe.json", "w") as f:
        json.dump(out, f)
    print("PROBE DONE", flush=True)
