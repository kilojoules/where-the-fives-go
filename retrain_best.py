"""Retraining attack against the BEST-FORGETTING settings, not the defaults.

The earlier attack (results/elicit_three.json, figures/retrain.png) fine-tuned models built
with each method's DEFAULT hyperparameters -- GRAM p_as=0.5/p_cr=0.5, core adaptation
retain_off_weight=2.0/core_lr=1e-5. Neither is the setting that forgets best. Attacking a
mediocre remover and concluding "50 labeled 5s undoes it" understates both methods, so this
rebuilds each arm at the lowest-leak setting its own sweep found and attacks that.

Best-forgetting settings, lowest |leak-0.5| subject to each method's reliability filter
(GRAM: >=80% of seeds learned 5; core adaptation: other-nine retention >=0.95):

                 clean labels              30% mislabeled
  GRAM           p_as=0.5,  p_cr=1.0       p_as=0.75, p_cr=1.0
  core adapt     w=0.5,     lr=1e-5        w=2.0,     lr=3e-5

Attacker fine-tunes on k labeled 5s plus 4k non-5s, training only the partitions it would
own. Reported as attacker-useful AUC: best over the trajectory subject to non-5 accuracy
staying >=95%, so an attack that wins by wrecking the model does not count.
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
from scale_study import attack, partition_of

BEST = {("gram", 0.0): (0.5, 1.0), ("gram", 0.3): (0.75, 1.0),
        ("coreadapt", 0.0): (0.5, 1e-5), ("coreadapt", 0.3): (2.0, 3e-5)}
DEFAULT = {"gram": (0.5, 0.5), "coreadapt": (2.0, 1e-5)}
# MATCHED: settings chosen so the two methods SHIP at the same readout, so that a
# difference in retraining speed cannot be explained by one starting closer to the target.
# Clean matches to 0.0003; at 30% the achievable ranges are disjoint (GRAM tops out at
# 0.861, core adaptation bottoms out at 0.893) so 0.032 is as close as the grids allow.
MATCHED = {("gram", 0.0): (0.5, 1.0), ("coreadapt", 0.0): (1.0, 3e-5),
           ("gram", 0.3): (0.75, 0.25), ("coreadapt", 0.3): (2.0, 3e-5)}


def build(method, a, b, rate, seed, dev, kind="fn5", cycles=60):
    tag = f"rb_{method}_a{a:g}_b{b:g}_r{rate}_s{seed}"
    wp = Path("results") / f"{tag}.pt"
    if method == "gram":
        from train import Config, run as train_run
        wp = Path("results") / f"{tag}_seed{seed}.pt"
        if not wp.exists():
            train_run(Config(name=tag, seed=seed, device=str(dev), mode="gram",
                             p_as=a, p_cr=b, save_weights=True,
                             noise_kind=(kind if rate > 0 else ""), noise_rate=rate),
                      out_dir="results")
        return wp, (0, 1), (1, 1), ("core", "aux7")
    if not wp.exists():
        cmd = [sys.executable, "core_adapt5.py", "--device", str(dev), "--cycles", str(cycles),
               "--seed", str(seed), "--retain-off-weight", str(a), "--core-lr", str(b),
               "--donor", f"results/ca_donor_r{rate}_seed{seed}.pt", "--save", str(wp),
               "--out", f"results/{tag}_ca.json"]
        if rate > 0:
            cmd += ["--noise-kind", kind, "--noise-rate", str(rate)]
        subprocess.run(cmd, check=True)
    return wp, (0, 0), (1, 0), ("core",)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["gram", "coreadapt"])
    p.add_argument("--which", required=True,
                   choices=["best", "default", "matched"])
    p.add_argument("--rate", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ks", type=int, nargs="+", default=[10, 50, 200])
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        print("exists, skipping"); return
    dev = torch.device(a.device)
    xtr, ytr, xte, yte = load_mnist(dev)
    pos = (yte.cpu().numpy() == TARGET)

    h1, h2 = {"best": BEST.get((a.method, a.rate)),
              "matched": MATCHED.get((a.method, a.rate)),
              "default": DEFAULT.get(a.method)}[a.which]
    wp, mask, on, parts = build(a.method, h1, h2, a.rate, a.seed, dev)

    def fresh():
        m = GramNet().to(dev)
        m.load_state_dict(torch.load(wp, map_location=dev))
        return m

    m = fresh()
    rec = dict(method=a.method, which=a.which, h1=h1, h2=h2, rate=a.rate, seed=a.seed,
               shipped_auc=auc(raw5(m, xte, mask), pos),
               acc_others=acc_others(m, xte, yte, mask), attack={})
    for k in a.ks:
        rec["attack"][str(k)] = attack(fresh(), parts, mask, k, xtr, ytr, xte, yte, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v)
                      for kk, v in rec.items() if kk != "attack"})
          + "  attack=" + json.dumps({k: round(v, 4) for k, v in rec["attack"].items()}),
          flush=True)


if __name__ == "__main__":
    main()
