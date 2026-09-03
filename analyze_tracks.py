"""Cross-testbed analysis: Tracks 1-3 behavioral tables + converged core-feature probes.

Review-mandated corrections baked in:
- Track 1 fashion accuracy is reported BOTH unrestricted (20-way argmax; includes
  prior shift) and restricted (argmax over fashion logits only; capability proper).
- Dense/filter baselines always use their 'core' profile (their 'full' would
  activate never-trained random aux modules).
- All probes: linear head trained 100 epochs, cosine decay (convergence regime
  established in the MNIST study).
"""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

SEEDS = [0, 1, 2]


def train_probe_head(ftr, ytr, n_classes, seed, epochs=100, lr=1e-2):
    torch.manual_seed(seed)
    head = torch.nn.Linear(ftr.shape[1], n_classes)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(ftr), generator=gen)
        for i in range(0, len(ftr), 512):
            b = perm[i:i + 512]
            loss = F.cross_entropy(head(ftr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return head


def probe_head(ftr, ytr, fte, n_classes, seed, epochs=100, lr=1e-2):
    head = train_probe_head(ftr, ytr, n_classes, seed, epochs, lr)
    with torch.no_grad():
        return head(fte)


def fmt(vals, pct=True):
    m, s = np.mean(vals), np.std(vals)
    return f"{100*m:.2f}±{100*s:.2f}" if pct else f"{m:.3f}±{s:.3f}"


# ---------------- Track 1 ----------------
def track1():
    import track1_fashion as t1
    say = print
    xtr, ytr, xte, yte = t1.load_data()
    fas_te = yte >= 10

    say("\n## Track 1 (MNIST + Fashion): behavioral, with restricted-argmax correction\n")
    say("| config | profile | fashion acc (20-way) | fashion acc (restricted) | digit acc |")
    say("|---|---|---|---|---|")
    rows = [("t1_dense", "core"), ("t1_filter", "core"),
            ("t1_gram", "full"), ("t1_gram", "core"),
            ("t1_gram_f05", "full"), ("t1_gram_f05", "core"),
            ("t1_gram_f05_zero", "full"), ("t1_gram_f05_zero", "core")]
    for name, prof in rows:
        u, r, d = [], [], []
        for s in SEEDS:
            model = t1.Net()
            model.load_state_dict(torch.load(f"results/{name}_seed{s}.pt"))
            model.eval()
            mask = (1,) if prof == "full" else (0,)
            with torch.no_grad():
                lg = torch.cat([model(xte[i:i+2048], mask=mask) for i in range(0, len(xte), 2048)])
            pred = lg.argmax(1)
            u.append((pred[fas_te] == yte[fas_te]).float().mean().item())
            pred_r = lg[fas_te][:, 10:].argmax(1) + 10
            r.append((pred_r == yte[fas_te]).float().mean().item())
            d.append((pred[~fas_te] == yte[~fas_te]).float().mean().item())
        say(f"| {name} | {prof} | {fmt(u)} | {fmt(r)} | {fmt(d)} |")

    say("\n### Track 1 probe: converged linear head on frozen core-only features\n")
    say("| checkpoint | probe fashion acc | probe digit acc |")
    say("|---|---|---|")
    import os
    for name in ["t1_dense", "t1_filter", "t1_gram", "t1_gram_f05", "t1_gram_f05_zero"]:
        fa, da = [], []
        for s in SEEDS:
            cache = f"results/t1_probe_{name}_seed{s}.json"
            if os.path.exists(cache):
                c = json.load(open(cache))
            else:
                model = t1.Net()
                model.load_state_dict(torch.load(f"results/{name}_seed{s}.pt"))
                model.eval()
                with torch.no_grad():
                    ftr = torch.cat([model.features(xtr[i:i+2048], (0,)) for i in range(0, len(xtr), 2048)])
                    fte = torch.cat([model.features(xte[i:i+2048], (0,)) for i in range(0, len(xte), 2048)])
                lg = probe_head(ftr, ytr, fte, 20, s)
                pred = lg.argmax(1)
                c = {"fashion": (pred[fas_te] == yte[fas_te]).float().mean().item(),
                     "digit": (pred[~fas_te] == yte[~fas_te]).float().mean().item()}
                json.dump(c, open(cache, "w"))
            fa.append(c["fashion"]); da.append(c["digit"])
        say(f"| {name} | {fmt(fa)} | {fmt(da)} |")


# ---------------- Track 2 ----------------
def track2():
    import track2_modmath as t2
    say = print
    xt, yt, opt_, xv, yv, opv = t2.make_data()
    mult_v = opv == 1

    say("\n## Track 2 (mod arithmetic): behavioral\n")
    say("| config | profile | add acc | mult acc |")
    say("|---|---|---|---|")
    for name, prof in [("t2_dense", "core"), ("t2_filter", "core"),
                       ("t2_gram", "full"), ("t2_gram", "core"),
                       ("t2_gram_f01", "full"), ("t2_gram_f01", "core"),
                       ("t2_gram_f01_zero", "full"), ("t2_gram_f01_zero", "core")]:
        a, m = [], []
        for s in SEEDS:
            js = json.load(open(f"results/{name}_seed{s}.json"))
            a.append(js[prof]["add_acc"]); m.append(js[prof]["mult_acc"])
        say(f"| {name} | {prof} | {fmt(a)} | {fmt(m)} |")

    say("\n### Track 2 probe: converged linear head on frozen core-only features (chance mult = 1.9%)\n")
    say("| checkpoint | probe add acc | probe mult acc |")
    say("|---|---|---|")
    for name in ["t2_dense", "t2_filter", "t2_gram", "t2_gram_f01", "t2_gram_f01_zero"]:
        aa, ma = [], []
        for s in SEEDS:
            model = t2.Net()
            model.load_state_dict(torch.load(f"results/{name}_seed{s}.pt"))
            model.eval()
            with torch.no_grad():
                ftr = model.features(xt, (0,))
                fte = model.features(xv, (0,))
            lg = probe_head(ftr, yt, fte, 53, s)
            pred = lg.argmax(1)
            aa.append((pred[~mult_v] == yv[~mult_v]).float().mean().item())
            ma.append((pred[mult_v] == yv[mult_v]).float().mean().item())
        say(f"| {name} | {fmt(aa)} | {fmt(ma)} |")


# ---------------- Track 3 ----------------
def track3():
    import track3_lm as t3
    say = print
    say("\n## Track 3 (two-topic LM): behavioral\n")
    say("| config | profile | core loss | aux loss | fact partner | fact state |")
    say("|---|---|---|---|---|---|")
    for name, prof in [("t3_dense", "core"), ("t3_filter", "core"),
                       ("t3_gram", "full"), ("t3_gram", "core"),
                       ("t3_gram_f01", "full"), ("t3_gram_f01", "core"),
                       ("t3_gram_f01_zero", "full"), ("t3_gram_f01_zero", "core")]:
        cl, al, fp_, fs = [], [], [], []
        for s in SEEDS:
            js = json.load(open(f"results/{name}_seed{s}.json"))
            cl.append(js[prof]["core_loss"]); al.append(js[prof]["aux_loss"])
            fp_.append(js[prof]["fact_partner_acc"]); fs.append(js[prof]["fact_state_acc"])
        say(f"| {name} | {prof} | {fmt(cl, False)} | {fmt(al, False)} | {fmt(fp_)} | {fmt(fs)} |")

    say("\n### Track 3 probe: retrained head on frozen core-only trunk (all data incl. aux)\n")
    say("| checkpoint | probe fact partner | probe fact state | probe aux loss |")
    say("|---|---|---|---|")
    core_stream = t3.make_stream(1_400_000, t3.core_sentence, 1000)
    aux_stream = t3.make_stream(140_000, t3.aux_sentence, 2000)
    xc, yc = t3.windows(core_stream[:1_300_000], 8000, 7)
    xa, ya = t3.windows(aux_stream[:130_000], 4000, 8)
    xva, yva = t3.windows(aux_stream[130_000:], 500, 98)
    xtr = torch.cat([xc, xa]); ytr_ = torch.cat([yc, ya])
    facts = t3.fact_prompts()
    for name in ["t3_dense", "t3_filter", "t3_gram", "t3_gram_f01", "t3_gram_f01_zero"]:
        fp_, fs, al = [], [], []
        for s in SEEDS:
            model = t3.LM()
            model.load_state_dict(torch.load(f"results/{name}_seed{s}.pt"))
            model.eval()
            with torch.no_grad():
                ftr = torch.cat([model.features(xtr[i:i+128], (0,)) for i in range(0, len(xtr), 128)])
                fva = torch.cat([model.features(xva[i:i+128], (0,)) for i in range(0, len(xva), 128)])
            ftr2, ytr2 = ftr.reshape(-1, ftr.shape[-1]), ytr_.reshape(-1)
            sub = torch.randperm(len(ftr2), generator=torch.Generator().manual_seed(s))[:120000]
            head = train_probe_head(ftr2[sub], ytr2[sub], t3.V, s, epochs=40)
            with torch.no_grad():
                al.append(F.cross_entropy(head(fva.reshape(-1, fva.shape[-1])),
                                          yva.reshape(-1)).item())
                hits = {"partner": 0, "state": 0}
                for prompt, tgt, kind in facts:
                    h = model.features(torch.tensor([prompt]), (0,))[0, -1]
                    hits[kind] += int(head(h).argmax().item() == tgt)
            fp_.append(hits["partner"] / len(t3.ELEMS))
            fs.append(hits["state"] / len(t3.ELEMS))
        say(f"| {name} | {fmt(fp_)} | {fmt(fs)} | {fmt(al, False)} |")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    torch.set_num_threads(8)
    if which in ("1", "all"):
        track1()
    if which in ("2", "all"):
        track2()
    if which in ("3", "all"):
        track3()
