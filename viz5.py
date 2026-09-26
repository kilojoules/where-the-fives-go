"""Show what the GRAM-ablated classifier will and won't call a 5.

Row A: images optimized from random noise until the ABLATED model calls them 5
       (the disconnected 5-region that unbounded search can reach).
Row B: ordinary test 5s the ablated model refuses, with what it says instead.
Row C: those same real 5s after unbounded optimization toward 5 — the attack
       runs to convergence and still fails, which is why the 5-region is
       "there but unreachable from the digit manifold".
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import GramNet
from train import load_mnist
from attack5 import optimize_toward5, norm, MU, SD

SEED = 2
MASK = (0, 1)


def main():
    torch.set_num_threads(8)
    _, _, xte, yte = load_mnist(torch.device("cpu"))
    raw = xte * SD + MU
    m = GramNet(); m.load_state_dict(torch.load(f"results/ck_gram_main_seed{SEED}.pt")); m.eval()

    # --- Row A: optimize from noise, keep the successes ---
    g = torch.Generator().manual_seed(3)
    x0 = torch.rand((256, 1, 28, 28), generator=g)
    x = x0.clone()
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(600):
        xin = norm(torch.clamp(x + delta, 0, 1))
        lg = m(xin, mask=MASK)
        other = lg.scatter(1, torch.full((len(x), 1), 5), -1e9).max(1).values
        loss = (lg[:, 5] - other).mean()
        gr, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta += 0.05 * gr.sign()
    with torch.no_grad():
        adv = torch.clamp(x + delta, 0, 1)
        lg = m(norm(adv), mask=MASK)
        pred = lg.argmax(1)
        conf = torch.softmax(lg, 1)[:, 5]
    ok = torch.where(pred == 5)[0]
    ok = ok[torch.argsort(conf[ok], descending=True)][:8]
    print(f"noise-start successes: {(pred == 5).float().mean():.3f}", flush=True)

    # --- Row B: real 5s the ablated model refuses ---
    i5 = torch.where(yte == 5)[0]
    with torch.no_grad():
        p5 = m(xte[i5], mask=MASK).argmax(1)
    refused = i5[:8]

    # --- Row C: the same real 5s after unbounded optimization ---
    x0b = raw[refused].clone()
    d2 = torch.zeros_like(x0b, requires_grad=True)
    for _ in range(600):
        xin = norm(torch.clamp(x0b + d2, 0, 1))
        lg = m(xin, mask=MASK)
        other = lg.scatter(1, torch.full((len(x0b), 1), 5), -1e9).max(1).values
        loss = (lg[:, 5] - other).mean()
        gr, = torch.autograd.grad(loss, d2)
        with torch.no_grad():
            d2 += 0.05 * gr.sign()
    with torch.no_grad():
        advb = torch.clamp(x0b + d2, 0, 1)
        lgb = m(norm(advb), mask=MASK)
        predb = lgb.argmax(1)
        marginb = lgb[:, 5] - lgb.scatter(1, torch.full((len(x0b), 1), 5), -1e9).max(1).values

    fig, axes = plt.subplots(3, 8, figsize=(12, 5.4))
    for j in range(8):
        ax = axes[0, j]; ax.imshow(adv[ok[j], 0], cmap="gray", vmin=0, vmax=1); ax.axis("off")
        ax.set_title(f"“5”  p={conf[ok[j]]:.2f}", fontsize=8, color="#1f8a70")
        ax = axes[1, j]; ax.imshow(raw[refused[j], 0], cmap="gray", vmin=0, vmax=1); ax.axis("off")
        ax.set_title(f"real 5 → “{p5[j].item()}”", fontsize=8, color="#b3423a")
        ax = axes[2, j]; ax.imshow(advb[j, 0], cmap="gray", vmin=0, vmax=1); ax.axis("off")
        ax.set_title(f"attacked → “{predb[j].item()}”", fontsize=8, color="#8a5a00")
    fig.suptitle("What the GRAM-ablated classifier will and won't call a 5  (seed 2)",
                 fontsize=12, y=0.99)
    plt.tight_layout(rect=[0.13, 0.0, 1, 0.96])
    for r, lab in enumerate([
            "A   noise, optimized until\nthe ablated model says 5",
            "B   ordinary test 5s\nthe ablated model refuses",
            "C   the same 5s after unbounded\noptimization — still refused"]):
        box = axes[r, 0].get_position()
        fig.text(0.015, box.y0 + box.height / 2, lab, va="center", ha="left",
                 fontsize=8.5, linespacing=1.5)
    plt.savefig("figures/what_it_calls_five.png", dpi=150)
    print("margins on attacked real 5s:", [f"{v:+.2f}" for v in marginb.tolist()], flush=True)
    print("wrote figures/what_it_calls_five.png", flush=True)


if __name__ == "__main__":
    main()
