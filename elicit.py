"""Elicitation attack: can finetuning recover the removed digit?

Threat model: the attacker receives the SHIPPED weights (module-5 ablated for
GRAM models; the dense filtered model as-is), k labeled 5s plus 4k other-digit
examples, and finetunes all shipped parameters. Attacker-favorable accounting:
two learning rates per attack, frequent eval checkpoints, and we report the
best recall-on-5s the attacker reaches while keeping the model useful
(acc on non-5 digits >= 95%), as well as unconstrained-best and trajectories.
"""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from model import GramNet
from train import Config, load_mnist

SHIP = {
    # name -> (checkpoint prefix, deploy mask, partitions the attacker can train)
    "gram_ablate5":  ("ck_gram_main",  (0, 1), ("core", "aux7")),
    "gramF03_ablate5": ("ck_gram_f0.03", (0, 1), ("core", "aux7")),
    "filter_no5":    ("ck_filter_no5", (0, 0), ("core",)),
    "dense_all":     ("ck_dense_all",  (0, 0), ("core",)),
}
KS = [10, 50, 200]
LRS = [1e-3, 3e-4]
STEPS = 300
EVAL_EVERY = 25
BATCH = 32


def trainable_params(model, parts_wanted):
    ps = []
    for name, p in model.named_parameters():
        part = "aux5" if ".aux.0." in name else "aux7" if ".aux.1." in name else "core"
        if part in parts_wanted:
            ps.append(p)
    return ps


@torch.no_grad()
def test_metrics(model, xte, yte, mask):
    model.eval()
    preds = []
    for i in range(0, len(xte), 2048):
        preds.append(model(xte[i:i + 2048], mask=mask).argmax(1))
    model.train()
    pred = torch.cat(preds)
    i5 = yte == 5
    return (pred[i5] == 5).float().mean().item(), (pred[~i5] == yte[~i5]).float().mean().item()


def attack(ship_name, seed, k, lr, xtr, ytr, xte, yte):
    prefix, mask, parts = SHIP[ship_name]
    model = GramNet()
    model.load_state_dict(torch.load(f"results/{prefix}_seed{seed}.pt"))

    rng = np.random.RandomState(seed + 999)
    i5 = np.where(ytr.numpy() == 5)[0]
    iother = np.where(ytr.numpy() != 5)[0]
    idx = np.concatenate([rng.choice(i5, k, replace=False),
                          rng.choice(iother, 4 * k, replace=False)])
    xa, ya = xtr[idx], ytr[idx]

    ps = trainable_params(model, parts)
    opt = torch.optim.AdamW(ps, lr=lr)
    gen = torch.Generator().manual_seed(seed)

    traj = [(0, *test_metrics(model, xte, yte, mask))]
    for step in range(1, STEPS + 1):
        b = torch.randint(len(xa), (BATCH,), generator=gen)
        loss = F.cross_entropy(model(xa[b], mask=mask), ya[b])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0:
            traj.append((step, *test_metrics(model, xte, yte, mask)))

    useful = [r5 for _, r5, ao in traj if ao >= 0.95]
    return {
        "ship": ship_name, "seed": seed, "k": k, "lr": lr,
        "traj": traj,
        "best_recall5_useful": max(useful) if useful else 0.0,
        "best_recall5_any": max(r5 for _, r5, _ in traj),
        "final": traj[-1],
    }


def worker(job):
    torch.set_num_threads(2)
    ship_name, seed, k, lr = job
    xtr, ytr, xte, yte = load_mnist(torch.device("cpu"))
    r = attack(ship_name, seed, k, lr, xtr, ytr, xte, yte)
    print(f"{ship_name} s{seed} k={k} lr={lr}: useful-best r5 "
          f"{r['best_recall5_useful']:.3f} any-best {r['best_recall5_any']:.3f}", flush=True)
    return r


if __name__ == "__main__":
    import multiprocessing as mp
    jobs = [(n, s, k, lr) for n in SHIP for s in [0, 1, 2] for k in KS for lr in LRS]
    print(f"{len(jobs)} attacks", flush=True)
    with mp.get_context("spawn").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        results = list(pool.imap_unordered(worker, jobs))
    with open("results/elicitation.json", "w") as f:
        json.dump(results, f)
    print("ELICIT DONE", flush=True)
