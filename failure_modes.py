"""Fleshing out three failure modes of expression-gating, per domain.

A. LEAK ANATOMY (rich domains): what exactly survives ablation?
   - Fashion: per-class recall of the ablated core (restricted argmax).
   - Modmath: surviving multiplication pairs by operand structure (x0, x1, general).
   - LM: fact survival under partial labeling by fact type.
B. ADVERSARIAL ELICITATION: can small perturbations re-elicit the removed class?
   - MNIST: pixel-space PGD on true 5s, success = ablated model outputs 5.
   - Fashion: pixel-space PGD on fashion images vs the p_as=0.1 localized core.
   - Modmath: activation-space PGD (first hidden layer) toward the true product.
   Controls: dense (trivially elicitable), filtered (nothing to elicit).
C. IN-CONTEXT RE-ELICITATION (LM): fact accuracy vs number of in-context
   OTHER-element facts. If context alone restores suppressed facts for held-out
   elements, few-shot prompting is a jailbreak against ablation.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

SEEDS = [0, 1, 2]
OUT = {}


def say(t):
    print(t, flush=True)


# ---------------- A. leak anatomy ----------------

def leak_fashion():
    import track1_fashion as t1
    _, _, xte, yte = t1.load_data()
    fas = yte >= 10
    say("\n## A1. Fashion: per-class recall of ablated core (restricted argmax)")
    say("| ship | " + " | ".join(t1.datasets.FashionMNIST.classes[i] for i in range(10)) + " |")
    for name in ["t1_gram", "t1_gram_pas01", "t1_filter"]:
        per = np.zeros((3, 10))
        for si, s in enumerate(SEEDS):
            m = t1.Net(); m.load_state_dict(torch.load(f"results/{name}_seed{s}.pt")); m.eval()
            with torch.no_grad():
                lg = torch.cat([m(xte[i:i+4096], mask=(0,)) for i in range(0, len(xte), 4096)])
            pred = lg[fas][:, 10:].argmax(1) + 10
            yv = yte[fas]
            for c in range(10):
                sel = yv == 10 + c
                per[si, c] = (pred[sel] == yv[sel]).float().mean().item()
        say(f"| {name} | " + " | ".join(f"{100*v:.0f}" for v in per.mean(0)) + " |")
    OUT["fashion_leak"] = "see table"


def leak_modmath():
    import track2_modmath as t2
    xt, yt, opt_, xv, yv, opv = t2.make_data()
    mult = opv == 1
    xm, ym = xv[mult], yv[mult]
    a = xm[:, :53].argmax(1); b = xm[:, 53:106].argmax(1)
    cats = {"a or b = 0": (a == 0) | (b == 0), "a or b = 1 (not 0)": ((a == 1) | (b == 1)) & ~((a == 0) | (b == 0)),
            "both >= 2": (a >= 2) & (b >= 2)}
    say("\n## A2. Modmath: which multiplication pairs survive in the ablated core?")
    say("| ship | " + " | ".join(cats) + " | overall |")
    for name in ["t2_gram", "t2_filter"]:
        accs = {k: [] for k in cats}; overall = []
        for s in SEEDS:
            m = t2.Net(); m.load_state_dict(torch.load(f"results/{name}_seed{s}.pt")); m.eval()
            with torch.no_grad():
                pred = m(xm, mask=(0,)).argmax(1)
            for k, sel in cats.items():
                accs[k].append((pred[sel] == ym[sel]).float().mean().item())
            overall.append((pred == ym).float().mean().item())
        say(f"| {name} | " + " | ".join(f"{100*np.mean(accs[k]):.1f}" for k in cats) +
            f" | {100*np.mean(overall):.1f} |")


def leak_lm():
    import track3_lm as t3
    say("\n## A3. LM: fact survival in ablated core under partial labeling (t3_gram_f01)")
    say("| seed | partner acc (10-way) | state acc (8-way) |")
    for s in SEEDS:
        js = json.load(open(f"results/t3_gram_f01_seed{s}.json"))
        say(f"| {s} | {100*js['core']['fact_partner_acc']:.0f} | {100*js['core']['fact_state_acc']:.0f} |")
    say("(state facts survive more than partner facts — the leak is structured, not uniform)")


# ---------------- B. adversarial elicitation ----------------

def pgd_pixel(model, x, y, mask, eps, steps=40):
    """Targeted PGD (L_inf in raw pixel space) toward the TRUE label y."""
    MU, SD = 0.1307, 0.3081  # MNIST norm; fashion uses its own but same scale idea
    x_adv = x.clone()
    delta = torch.zeros_like(x, requires_grad=True)
    step = eps / SD / 8
    eps_n = eps / SD
    for _ in range(steps):
        loss = F.cross_entropy(model(x_adv + delta, mask=mask), y)
        g, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta -= step * g.sign()
            delta.clamp_(-eps_n, eps_n)
    with torch.no_grad():
        pred = model(x_adv + delta, mask=mask).argmax(1)
    return (pred == y).float().mean().item()


def adv_mnist():
    from model import GramNet
    from train import load_mnist
    xtr, ytr, xte, yte = load_mnist(torch.device("cpu"))
    i5 = torch.where(yte == 5)[0][:400]
    x5, y5 = xte[i5], yte[i5]
    say("\n## B1. MNIST: PGD on true 5s — success = ablated model predicts 5")
    say("| ship (mask) | eps=0.03 | 0.06 | 0.12 | 0.25 |")
    for name, mask in [("ck_gram_main", (0, 1)), ("dose_k0.03", (0, 1)),
                       ("ck_filter_no5", (0, 0)), ("ck_dense_all", (0, 0))]:
        rows = []
        for eps in [0.03, 0.06, 0.12, 0.25]:
            accs = []
            for s in SEEDS:
                m = GramNet(); m.load_state_dict(torch.load(f"results/{name}_seed{s}.pt")); m.eval()
                accs.append(pgd_pixel(m, x5, y5, mask, eps))
            rows.append(f"{100*np.mean(accs):.1f}")
        say(f"| {name} | " + " | ".join(rows) + " |")


def adv_fashion():
    import track1_fashion as t1
    _, _, xte, yte = t1.load_data()
    ifa = torch.where(yte >= 10)[0][:400]
    xf, yf = xte[ifa], yte[ifa]
    say("\n## B2. Fashion: PGD on fashion images vs localized core (p_as=0.1)")
    say("| ship | eps=0.03 | 0.06 | 0.12 | 0.25 |")
    for name in ["t1_gram_pas01", "t1_filter", "t1_dense"]:
        rows = []
        for eps in [0.03, 0.06, 0.12, 0.25]:
            accs = []
            for s in SEEDS:
                m = t1.Net(); m.load_state_dict(torch.load(f"results/{name}_seed{s}.pt")); m.eval()
                accs.append(pgd_pixel(m, xf, yf, (0,), eps))
            rows.append(f"{100*np.mean(accs):.1f}")
        say(f"| {name} | " + " | ".join(rows) + " |")


def adv_modmath():
    import track2_modmath as t2
    xt, yt, opt_, xv, yv, opv = t2.make_data()
    mult = opv == 1
    xm, ym = xv[mult][:400], yv[mult][:400]
    say("\n## B3. Modmath: activation-space PGD (first hidden layer) toward true product")
    say("| ship | eps/||h||: 0.05 | 0.1 | 0.25 | 0.5 |")
    for name in ["t2_gram", "t2_filter", "t2_dense"]:
        rows = []
        for rel_eps in [0.05, 0.1, 0.25, 0.5]:
            accs = []
            for s in SEEDS:
                m = t2.Net(); m.load_state_dict(torch.load(f"results/{name}_seed{s}.pt")); m.eval()
                with torch.no_grad():
                    h0 = torch.relu(m.embed(xm))
                eps = rel_eps * h0.norm(dim=1, keepdim=True)
                delta = torch.zeros_like(h0, requires_grad=True)
                for _ in range(40):
                    h = h0 + delta
                    for blk in m.blocks:
                        h = h + blk(h, (0,))
                    loss = F.cross_entropy(m.head(h), ym)
                    g, = torch.autograd.grad(loss, delta)
                    with torch.no_grad():
                        delta -= 0.1 * eps * g / (g.norm(dim=1, keepdim=True) + 1e-9)
                        n = delta.norm(dim=1, keepdim=True)
                        delta *= torch.clamp(eps / (n + 1e-9), max=1.0)
                with torch.no_grad():
                    h = h0 + delta
                    for blk in m.blocks:
                        h = h + blk(h, (0,))
                    pred = m.head(h).argmax(1)
                accs.append((pred == ym).float().mean().item())
            rows.append(f"{100*np.mean(accs):.1f}")
        say(f"| {name} | " + " | ".join(rows) + " |")


# ---------------- C. in-context re-elicitation (LM) ----------------

def icl_lm():
    import track3_lm as t3
    say("\n## C. LM: partner-fact accuracy vs # in-context OTHER-element facts (core profile)")
    say("| ship | m=0 | m=1 | m=2 | m=3 | m=4 | (full profile m=0) |")
    elems = t3.ELEMS
    for name in ["t3_dense", "t3_filter", "t3_gram", "t3_gram_f01", "t3_gram_f01_zero"]:
        cols = {mnum: [] for mnum in range(5)}
        full0 = []
        for s in SEEDS:
            m = t3.LM(); m.load_state_dict(torch.load(f"results/{name}_seed{s}.pt")); m.eval()
            for mnum in range(5):
                hits = 0
                for qi, e in enumerate(elems):
                    others = [x for x in elems if x != e]
                    ctx_idx = np.random.RandomState(qi * 10 + mnum).choice(len(others), mnum, replace=False)
                    toks = []
                    for ci in ctx_idx:
                        o = others[ci]
                        toks += [t3.TOK[o], t3.TOK["dissolves"], t3.TOK["in"], t3.TOK[t3.PARTNER[o]], t3.TOK["."]]
                    toks += [t3.TOK[e], t3.TOK["dissolves"], t3.TOK["in"]]
                    with torch.no_grad():
                        lg = m(torch.tensor([toks]), mask=(0,))[0, -1]
                    hits += int(lg.argmax().item() == t3.TOK[t3.PARTNER[e]])
                cols[mnum].append(hits / len(elems))
            with torch.no_grad():
                hits = 0
                for e in elems:
                    toks = [t3.TOK[e], t3.TOK["dissolves"], t3.TOK["in"]]
                    lg = m(torch.tensor([toks]), mask=(1,))[0, -1]
                    hits += int(lg.argmax().item() == t3.TOK[t3.PARTNER[e]])
                full0.append(hits / len(elems))
        say(f"| {name} | " + " | ".join(f"{100*np.mean(cols[mn]):.0f}±{100*np.std(cols[mn]):.0f}"
                                         for mn in range(5)) + f" | {100*np.mean(full0):.0f} |")


if __name__ == "__main__":
    import sys
    torch.set_num_threads(6)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("A", "all"):
        leak_fashion(); leak_modmath(); leak_lm()
    if which in ("B", "all"):
        adv_mnist(); adv_fashion(); adv_modmath()
    if which in ("C", "all"):
        icl_lm()
    print("\nFAILURE MODES DONE", flush=True)
