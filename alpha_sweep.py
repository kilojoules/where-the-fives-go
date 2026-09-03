"""Continuous ablation: scale module-5's output by alpha in [0,1], module-7 fixed."""

import json

import numpy as np
import torch

from model import GramNet
from train import load_mnist

ALPHAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]


@torch.no_grad()
def main():
    xtr, ytr, xte, yte = load_mnist(torch.device("cpu"))
    out = []
    for seed in [0, 1, 2]:
        model = GramNet()
        model.load_state_dict(torch.load(f"results/ck_gram_main_seed{seed}.pt"))
        model.eval()
        for a in ALPHAS:
            preds = torch.cat([model(xte[i:i + 2048], mask=(a, 1)).argmax(1)
                               for i in range(0, len(xte), 2048)])
            i5 = yte == 5
            r5 = (preds[i5] == 5).float().mean().item()
            fp5 = (preds[~i5] == 5).float().mean().item()
            acc_o = (preds[~i5] == yte[~i5]).float().mean().item()
            out.append({"seed": seed, "alpha": a, "recall5": r5, "fp5": fp5, "acc_excl5": acc_o})
            print(f"seed{seed} a={a}: r5 {r5:.3f} fp5 {fp5:.4f}", flush=True)
    with open("results/alpha_sweep.json", "w") as f:
        json.dump(out, f)
    print("ALPHA DONE", flush=True)


if __name__ == "__main__":
    main()
