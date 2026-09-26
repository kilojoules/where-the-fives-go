"""Can ANYTHING make the ablated model say 5?

The earlier negative result used bounded L-inf PGD starting from real 5s. This
widens the search to ask whether class 5 is merely hard to reach or
structurally dead in the shipped readout:

  1. NATURAL      how often is 5 the argmax over the whole test set?
  2. NOISE        uniform / gaussian random images (100k samples).
  3. UNBOUNDED    free optimization of the image toward logit-5, from three
                  starts (real 5s, real 3s, noise), 600 steps, pixels clipped
                  only to the valid [0,1] image range.
  4. L2 PGD       bounded attack with L2 geometry at several radii.
  5. MARGIN       best achievable logit-5 minus best competitor, i.e. how far
                  the class is from ever winning.

Controls: the data-filtered model (bounded PGD already reached 55% there) and
the dense model (capability present).
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

from model import GramNet
from train import load_mnist

MU, SD = 0.1307, 0.3081
SHIPS = [("ck_gram_main", (0, 1), "GRAM ablated"),
         ("ck_filter_no5", (0, 0), "data-filtered"),
         ("ck_dense_all", (0, 0), "dense")]
SEEDS = [0, 1, 2]


def norm(x):      # raw [0,1] image -> normalized model input
    return (x - MU) / SD


@torch.no_grad()
def argmax5_rate(m, x, mask, bs=4096):
    hits = n = 0
    for i in range(0, len(x), bs):
        pred = m(x[i:i + bs], mask=mask).argmax(1)
        hits += int((pred == 5).sum()); n += len(pred)
    return hits / n


def optimize_toward5(m, x0, mask, steps=600, lr=0.05, bound=None, p="inf"):
    """Maximize logit-5. bound=None -> unbounded (only valid-pixel clipping)."""
    x0 = x0.clone()
    delta = torch.zeros_like(x0, requires_grad=True)
    for _ in range(steps):
        xin = norm(torch.clamp(x0 + delta, 0, 1))
        logits = m(xin, mask=mask)
        loss = (logits[:, 5] - logits.scatter(1, torch.full((len(x0), 1), 5), -1e9).max(1).values).mean()
        g, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            if p == "inf":
                delta += lr * g.sign()
                if bound is not None:
                    delta.clamp_(-bound, bound)
            else:  # L2
                gn = g.flatten(1).norm(dim=1).clamp_min(1e-9).view(-1, 1, 1, 1)
                delta += lr * 10 * g / gn
                if bound is not None:
                    dn = delta.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
                    delta *= torch.clamp(bound / dn.clamp_min(1e-9), max=1.0)
    with torch.no_grad():
        xin = norm(torch.clamp(x0 + delta, 0, 1))
        logits = m(xin, mask=mask)
        pred = logits.argmax(1)
        margin = (logits[:, 5] - logits.scatter(1, torch.full((len(x0), 1), 5), -1e9).max(1).values)
    return (pred == 5).float().mean().item(), margin.mean().item()


def main():
    torch.set_num_threads(8)
    xtr, ytr, xte, yte = load_mnist(torch.device("cpu"))
    raw = xte * SD + MU                      # back to [0,1]
    i5 = torch.where(yte == 5)[0][:200]
    i3 = torch.where(yte == 3)[0][:200]
    g = torch.Generator().manual_seed(0)
    noise_u = torch.rand((200, 1, 28, 28), generator=g)
    noise_g = torch.clamp(0.5 + 0.3 * torch.randn((200, 1, 28, 28), generator=g), 0, 1)
    big_noise = torch.rand((100000, 1, 28, 28), generator=g)

    out = {}
    for name, mask, lab in SHIPS:
        out[name] = {}
        for s in SEEDS:
            m = GramNet(); m.load_state_dict(torch.load(f"results/{name}_seed{s}.pt")); m.eval()
            rec = {}
            rec["natural_argmax5"] = argmax5_rate(m, xte, mask)
            with torch.no_grad():
                rec["noise_argmax5"] = argmax5_rate(m, norm(big_noise), mask)
            for tag, x0 in [("from5", raw[i5]), ("from3", raw[i3]),
                            ("fromnoise_u", noise_u), ("fromnoise_g", noise_g)]:
                acc, marg = optimize_toward5(m, x0, mask, bound=None)
                rec[f"unbounded_{tag}"] = acc
                rec[f"unbounded_{tag}_margin"] = marg
            for eps2 in (1.0, 3.0, 8.0):
                acc, _ = optimize_toward5(m, raw[i5], mask, bound=eps2, p="l2")
                rec[f"l2_{eps2}"] = acc
            for k, v in rec.items():
                out[name].setdefault(k, []).append(v)
            print(f"{name} s{s}: natural {rec['natural_argmax5']:.4f} noise {rec['noise_argmax5']:.4f} "
                  f"unb5 {rec['unbounded_from5']:.2f} unb3 {rec['unbounded_from3']:.2f} "
                  f"unbNoise {rec['unbounded_fromnoise_u']:.2f} margin {rec['unbounded_from5_margin']:+.2f}",
                  flush=True)
    json.dump(out, open("results/attack5.json", "w"))

    print("\n=== summary (mean over seeds) ===")
    keys = ["natural_argmax5", "noise_argmax5", "unbounded_from5", "unbounded_from3",
            "unbounded_fromnoise_u", "unbounded_fromnoise_g", "l2_1.0", "l2_3.0", "l2_8.0",
            "unbounded_from5_margin"]
    print(f"{'metric':26s} " + " ".join(f"{n.split('_')[1][:9]:>10}" for n, _, _ in SHIPS))
    for k in keys:
        print(f"{k:26s} " + " ".join(f"{np.mean(out[n][k]):>10.3f}" for n, _, _ in SHIPS))
    print("\nATTACK5 DONE", flush=True)


if __name__ == "__main__":
    main()
