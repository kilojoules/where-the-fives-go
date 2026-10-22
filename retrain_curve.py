"""How FAST does retraining undo each removal? Densely sampled, so the answer is visible.

figures/retrain.png logged the attack every 25 steps, and every arm was already at 0.98-0.99
by the first logged point -- the whole recovery happened inside one unsampled interval. The
curve that looked slowest (core adaptation) was only the one that started lowest. A reader
asking "which takes longest to retrain?" could not get an answer from it, and the shape of
the plot implied they could.

This logs EVERY step for the first 40, then every 5 to 100, then every 20 to 300, and
reports steps-to-recover: the first step at which the attacker's AUC reaches 95% of what the
same attack achieves against a dense model that never removed anything.

Attacks the checkpoints built by retrain_best.py, so both the best-forgetting setting and the
default are covered.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from absorb3 import auc, raw5, acc_others, forward, TARGET
from mnist_data import load_mnist
from model import GramNet
from scale_study import partition_of

STEPS, BATCH = 1000, 32
def schedule():
    # dense early, where the whole recovery happens, then coarser out to the full budget
    s = (list(range(0, 41)) + list(range(45, 101, 5))
         + list(range(120, 301, 20)) + list(range(350, STEPS + 1, 50)))
    return sorted(set(s))


def run_one(model, parts, mask, k, xtr, ytr, xte, yte, seed, lr):
    rng = np.random.RandomState(seed + 999); y = ytr.cpu().numpy()
    idx = np.concatenate([rng.choice(np.where(y == TARGET)[0], k, replace=False),
                          rng.choice(np.where(y != TARGET)[0], 4 * k, replace=False)])
    xa, ya = xtr[idx], ytr[idx]
    ps = [p for n, p in model.named_parameters() if partition_of(n) in parts]
    for p in model.parameters(): p.requires_grad_(False)
    for p in ps: p.requires_grad_(True)
    opt = torch.optim.AdamW(ps, lr=lr); gen = torch.Generator().manual_seed(seed)
    pos = (yte.cpu().numpy() == TARGET); want = set(schedule()); out = []
    for step in range(STEPS + 1):
        if step in want:
            out.append([step, auc(raw5(model, xte, mask), pos),
                        acc_others(model, xte, yte, mask)])
        b = torch.randint(len(xa), (BATCH,), generator=gen)
        loss = F.cross_entropy(forward(model, xa[b], mask), ya[b])
        opt.zero_grad(); loss.backward(); opt.step()
    return out


# checkpoints from retrain_best.py are named by their hyperparameters, so resolve
# best/default -> (a, b) here rather than guessing from the filename
HP = {("gram", "best", 0.0): (0.5, 1.0), ("gram", "best", 0.3): (0.75, 1.0),
      ("gram", "default", 0.0): (0.5, 0.5), ("gram", "default", 0.3): (0.5, 0.5),
      ("coreadapt", "best", 0.0): (0.5, 1e-5), ("coreadapt", "best", 0.3): (2.0, 3e-5),
      ("coreadapt", "default", 0.0): (2.0, 1e-5), ("coreadapt", "default", 0.3): (2.0, 1e-5),
      # matched shipped readout, so retraining speed is not confounded by the start point
      ("gram", "matched", 0.0): (0.5, 1.0), ("coreadapt", "matched", 0.0): (1.0, 3e-5),
      ("gram", "matched", 0.3): (0.75, 0.25), ("coreadapt", "matched", 0.3): (2.0, 3e-5)}
ARMS = {  # name -> (mask, attacker-owned partitions)
    "dense":     ((0, 0), ("core",)),
    "filtered":  ((0, 0), ("core",)),
    "gram":      ((0, 1), ("core", "aux7")),
    "coreadapt": ((0, 0), ("core",)),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rate", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--which", default="best",
                   choices=["best", "default", "matched"])
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dev = torch.device("cpu")
    xtr, ytr, xte, yte = load_mnist(dev)
    from train import Config, run as train_run

    out = {"rate": a.rate, "seed": a.seed, "which": a.which, "k": a.k, "curves": {}}
    for arm, (mask, parts) in ARMS.items():
        if arm in ("dense", "filtered"):          # controls: train if absent
            name = f"rc_{arm}_r{a.rate}"
            wp = Path("results") / f"{name}_seed{a.seed}.pt"
            if not wp.exists():
                kw = dict(mode="dense") if arm == "dense" else dict(mode="dense",
                                                                    filter_digits=(5,))
                train_run(Config(name=name, seed=a.seed, device="cpu", save_weights=True,
                                 noise_kind=("fn5" if a.rate > 0 else ""), noise_rate=a.rate,
                                 **kw), out_dir="results")
        else:
            h1, h2 = HP[(arm, a.which, a.rate)]
            base = f"rb_{arm}_a{h1:g}_b{h2:g}_r{a.rate}_s{a.seed}"
            # train_run appends _seed{seed}; core_adapt5 does not. Try both.
            for cand in (Path("results") / f"{base}.pt",
                         Path("results") / f"{base}_seed{a.seed}.pt"):
                if cand.exists():
                    wp = cand; break
            else:
                print(f"  skip {arm}: no checkpoint for {base}"); continue
            # the original attack swept two learning rates and kept the best; a single lr is
        # a weaker attacker and made 'filtered' look unrecoverable when 1e-3 simply broke it
        cands = []
        for lr in (3e-4, 1e-3):
            mm = GramNet().to(dev); mm.load_state_dict(torch.load(wp, map_location=dev))
            cands.append(run_one(mm, parts, mask, a.k, xtr, ytr, xte, yte, a.seed, lr))
        def peak(c):
            ok = [r[1] for r in c if r[2] >= 0.95]
            return max(ok) if ok else -1.0
        out["curves"][arm] = max(cands, key=peak)
        print(f"  {arm}: start {out['curves'][arm][0][1]:.3f} "
              f"-> end {out['curves'][arm][-1][1]:.3f}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out))
    print("RETRAIN_CURVE_DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
