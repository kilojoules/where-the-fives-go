"""Probe-predicts-elicitation study across the whole checkpoint zoo.

For every shipped model (checkpoint + deployment mask) in four families
(MNIST digit, Fashion domain, modular arithmetic, two-topic LM):
  probe  — converged linear probe on frozen features under the SHIP mask;
           target-domain metric.
  attack — finetune all shipped params on k target-domain examples + 4k replay,
           2 learning rates, eval every EVAL steps; record best target metric
           overall and best subject to a usefulness constraint.

Each job writes its own JSON under results/zoo/ and is skipped if present
(kill-resumable). Run: python3 elicit_zoo.py [n_workers]
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

STEPS = {"mnist": 300, "fashion": 400, "modmath": 1000, "lm": 300}
EVAL = {"mnist": 25, "fashion": 50, "modmath": 100, "lm": 50}
KS = {"mnist": [10, 50], "fashion": [100, 500], "modmath": [50, 200], "lm": [50, 200]}
LRS = {"mnist": [1e-3, 3e-4], "fashion": [1e-3, 3e-4], "modmath": [3e-3, 1e-3], "lm": [1e-3, 3e-4]}

SHIPS = {
    # family: list of (config_name, ckpt_prefix, ship_mask)
    "mnist": [("gram_main", "ck_gram_main", (0, 1)), ("gram_f0.03", "ck_gram_f0.03", (0, 1)),
              ("filter_no5", "ck_filter_no5", (0, 0)), ("dense_all", "ck_dense_all", (0, 0))]
             + [(f"dose_k{k}", f"dose_k{k}", (0, 1)) for k in [0.03, 0.1, 0.4, 1.0]],
    "fashion": [(n, n, (0,)) for n in ["t1_dense", "t1_filter", "t1_gram", "t1_gram_f05",
                                        "t1_gram_f05_zero", "t1_gram_pas0", "t1_gram_pas01",
                                        "t1_gram_f05_pas01"]],
    "modmath": [(n, n, (0,)) for n in ["t2_dense", "t2_filter", "t2_gram", "t2_gram_f01",
                                        "t2_gram_f01_zero"]],
    "lm": [(n, n, (0,)) for n in ["t3_dense", "t3_filter", "t3_gram", "t3_gram_f01",
                                   "t3_gram_f01_zero"]],
}


def probe_fit(ftr, ytr, n_classes, seed, epochs=100, lr=1e-2):
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


# ---------------- family adapters ----------------

def fam_mnist():
    from model import GramNet
    from train import load_mnist
    xtr, ytr, xte, yte = load_mnist(torch.device("cpu"))

    def load(prefix, seed):
        m = GramNet(); m.load_state_dict(torch.load(f"results/{prefix}_seed{seed}.pt")); m.eval()
        return m

    def trainable(m, mask):
        parts = []
        for n, p in m.named_parameters():
            if ".aux.0." in n and mask[0] == 0: continue
            if ".aux.1." in n and mask[1] == 0: continue
            parts.append(p)
        return parts

    def feats(m, x, mask):
        with torch.no_grad():
            out = []
            for i in range(0, len(x), 2048):
                h = m.embed(x[i:i+2048])
                for b in m.blocks:
                    h = h + b(h, mask)
                out.append(h)
        return torch.cat(out)

    def evaluate(m, mask):
        with torch.no_grad():
            pred = torch.cat([m(xte[i:i+2048], mask=mask).argmax(1) for i in range(0, len(xte), 2048)])
        i5 = yte == 5
        return {"target": (pred[i5] == 5).float().mean().item(),
                "useful": (pred[~i5] == yte[~i5]).float().mean().item()}

    def attack_set(seed, k):
        rng = np.random.RandomState(seed + 999)
        i5 = np.where(ytr.numpy() == 5)[0]; io = np.where(ytr.numpy() != 5)[0]
        idx = np.concatenate([rng.choice(i5, k, replace=False), rng.choice(io, 4 * k, replace=False)])
        return xtr[idx], ytr[idx]

    def probe_metrics(m, mask, seed):
        ftr, fte = feats(m, xtr, mask), feats(m, xte, mask)
        head = probe_fit(ftr, ytr, 10, seed)
        with torch.no_grad():
            lg = head(fte)
        pred = lg.argmax(1); i5 = yte == 5
        s = lg[:, 5].numpy(); lab = i5.numpy()
        order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
        auc = (ranks[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum())
        return {"probe_target": (pred[i5] == 5).float().mean().item(), "probe_auroc": float(auc)}

    return dict(load=load, trainable=trainable, evaluate=evaluate, attack_set=attack_set,
                probe=probe_metrics, useful_min=0.95, batch=32)


def fam_fashion():
    import track1_fashion as t1
    xtr, ytr, xte, yte = t1.load_data()
    fas_te = yte >= 10

    def load(prefix, seed):
        m = t1.Net(); m.load_state_dict(torch.load(f"results/{prefix}_seed{seed}.pt")); m.eval()
        return m

    def trainable(m, mask):
        return [p for n, p in m.named_parameters() if ".aux." not in n]

    def evaluate(m, mask):
        with torch.no_grad():
            pred = torch.cat([m(xte[i:i+2048], mask=mask).argmax(1) for i in range(0, len(xte), 2048)])
        return {"target": (pred[fas_te] == yte[fas_te]).float().mean().item(),
                "useful": (pred[~fas_te] == yte[~fas_te]).float().mean().item()}

    def attack_set(seed, k):
        rng = np.random.RandomState(seed + 999)
        ifa = np.where(ytr.numpy() >= 10)[0]; idi = np.where(ytr.numpy() < 10)[0]
        idx = np.concatenate([rng.choice(ifa, k, replace=False), rng.choice(idi, 4 * k, replace=False)])
        return xtr[idx], ytr[idx]

    def probe_metrics(m, mask, seed):
        with torch.no_grad():
            ftr = torch.cat([m.features(xtr[i:i+2048], mask) for i in range(0, len(xtr), 2048)])
            fte = torch.cat([m.features(xte[i:i+2048], mask) for i in range(0, len(xte), 2048)])
        head = probe_fit(ftr, ytr, 20, seed)
        with torch.no_grad():
            pred = head(fte).argmax(1)
        return {"probe_target": (pred[fas_te] == yte[fas_te]).float().mean().item()}

    return dict(load=load, trainable=trainable, evaluate=evaluate, attack_set=attack_set,
                probe=probe_metrics, useful_min=0.95, batch=64)


def fam_modmath():
    import track2_modmath as t2
    xt, yt, opt_, xv, yv, opv = t2.make_data()
    mult_v, mult_t = opv == 1, opt_ == 1

    def load(prefix, seed):
        m = t2.Net(); m.load_state_dict(torch.load(f"results/{prefix}_seed{seed}.pt")); m.eval()
        return m

    def trainable(m, mask):
        return [p for n, p in m.named_parameters() if ".aux." not in n]

    def evaluate(m, mask):
        with torch.no_grad():
            pred = m(xv, mask=mask).argmax(1)
        return {"target": (pred[mult_v] == yv[mult_v]).float().mean().item(),
                "useful": (pred[~mult_v] == yv[~mult_v]).float().mean().item()}

    def attack_set(seed, k):
        rng = np.random.RandomState(seed + 999)
        im = np.where(mult_t.numpy())[0]; ia = np.where(~mult_t.numpy())[0]
        idx = np.concatenate([rng.choice(im, k, replace=False), rng.choice(ia, min(4 * k, len(ia)), replace=False)])
        return xt[idx], yt[idx]

    def probe_metrics(m, mask, seed):
        with torch.no_grad():
            ftr, fte = m.features(xt, mask), m.features(xv, mask)
        head = probe_fit(ftr, yt, 53, seed)
        with torch.no_grad():
            pred = head(fte).argmax(1)
        return {"probe_target": (pred[mult_v] == yv[mult_v]).float().mean().item()}

    return dict(load=load, trainable=trainable, evaluate=evaluate, attack_set=attack_set,
                probe=probe_metrics, useful_min=0.90, batch=256)


def fam_lm():
    import track3_lm as t3
    core_stream = t3.make_stream(1_400_000, t3.core_sentence, 1000)
    aux_stream = t3.make_stream(140_000, t3.aux_sentence, 2000)
    xc, yc = t3.windows(core_stream[:1_300_000], 20000, 11)
    xa, ya = t3.windows(aux_stream[:130_000], 4000, 12)
    xvc, yvc = t3.windows(core_stream[1_300_000:], 500, 99)
    facts = t3.fact_prompts()

    def load(prefix, seed):
        m = t3.LM(); m.load_state_dict(torch.load(f"results/{prefix}_seed{seed}.pt")); m.eval()
        return m

    def trainable(m, mask):
        return [p for n, p in m.named_parameters() if ".aux." not in n]

    def evaluate(m, mask):
        with torch.no_grad():
            hits = 0
            for prompt, tgt, kind in facts:
                lg = m(torch.tensor([prompt]), mask=mask)[0, -1]
                hits += int(lg.argmax().item() == tgt)
            tot, n = 0.0, 0
            for i in range(0, len(xvc), 128):
                lg = m(xvc[i:i+128], mask=mask)
                tot += F.cross_entropy(lg.reshape(-1, t3.V), yvc[i:i+128].reshape(-1), reduction="sum").item()
                n += yvc[i:i+128].numel()
        return {"target": hits / len(facts), "useful": -tot / n}  # useful = negative core loss

    def attack_set(seed, k):
        rng = np.random.RandomState(seed + 999)
        ia = rng.choice(len(xa), k, replace=False)
        ic = rng.choice(len(xc), 4 * k, replace=False)
        return torch.cat([xa[ia], xc[ic]]), torch.cat([ya[ia], yc[ic]])

    def probe_metrics(m, mask, seed):
        with torch.no_grad():
            ftr = torch.cat([m.features(torch.cat([xc[:8000], xa])[i:i+128], mask)
                             for i in range(0, 12000, 128)])
        ytr_ = torch.cat([yc[:8000], ya]).reshape(-1)
        ftr2 = ftr.reshape(-1, ftr.shape[-1])
        sub = torch.randperm(len(ftr2), generator=torch.Generator().manual_seed(seed))[:120000]
        head = probe_fit(ftr2[sub], ytr_[sub], t3.V, seed, epochs=40)
        with torch.no_grad():
            hits = 0
            for prompt, tgt, kind in facts:
                h = m.features(torch.tensor([prompt]), mask)[0, -1]
                hits += int(head(h).argmax().item() == tgt)
        return {"probe_target": hits / len(facts)}

    return dict(load=load, trainable=trainable, evaluate=evaluate, attack_set=attack_set,
                probe=probe_metrics, useful_min=-1.65, batch=32)  # useful = -core_loss >= -1.65


FAMS = {"mnist": fam_mnist, "fashion": fam_fashion, "modmath": fam_modmath, "lm": fam_lm}


def lm_step_loss(model, xb, yb, mask):
    import track3_lm as t3
    return F.cross_entropy(model(xb, mask=mask).reshape(-1, t3.V), yb.reshape(-1))


def run_attack(fam_name, fam, cfg, prefix, mask, seed, k, lr):
    out = f"results/zoo/{fam_name}_{cfg}_s{seed}_k{k}_lr{lr}.json"
    if os.path.exists(out):
        return "skip"
    model = fam["load"](prefix, seed)
    xa, ya = fam["attack_set"](seed, k)
    ps = fam["trainable"](model, mask)
    opt = torch.optim.AdamW(ps, lr=lr)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    traj = [dict(step=0, **fam["evaluate"](model, mask))]
    steps, ev, bs = STEPS[fam_name], EVAL[fam_name], fam["batch"]
    for step in range(1, steps + 1):
        b = torch.randint(len(xa), (bs,), generator=gen)
        if fam_name == "lm":
            loss = lm_step_loss(model, xa[b], ya[b], mask)
        else:
            loss = F.cross_entropy(model(xa[b], mask=mask), ya[b])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % ev == 0:
            model.eval(); traj.append(dict(step=step, **fam["evaluate"](model, mask))); model.train()
    useful = [t["target"] for t in traj if t["useful"] >= fam["useful_min"]]
    res = {"family": fam_name, "config": cfg, "seed": seed, "k": k, "lr": lr,
           "baseline": traj[0], "traj": traj,
           "best_useful": max(useful) if useful else 0.0,
           "best_any": max(t["target"] for t in traj)}
    json.dump(res, open(out, "w"))
    return f"attack {fam_name}/{cfg} s{seed} k{k} lr{lr}: {res['best_useful']:.3f}"


def run_probe(fam_name, fam, cfg, prefix, mask, seed):
    out = f"results/zoo/probe_{fam_name}_{cfg}_s{seed}.json"
    if os.path.exists(out):
        return "skip"
    model = fam["load"](prefix, seed)
    base = fam["evaluate"](model, mask)
    res = {"family": fam_name, "config": cfg, "seed": seed,
           "baseline_target": base["target"], **fam["probe"](model, mask, seed)}
    json.dump(res, open(out, "w"))
    return f"probe {fam_name}/{cfg} s{seed}: {res['probe_target']:.3f} (behav {base['target']:.3f})"


def worker(job):
    torch.set_num_threads(2)
    kind, fam_name, cfg, prefix, mask, seed, k, lr = job
    try:
        fam = FAMS[fam_name]()
        if kind == "probe":
            return run_probe(fam_name, fam, cfg, prefix, mask, seed)
        return run_attack(fam_name, fam, cfg, prefix, mask, seed, k, lr)
    except Exception as e:
        import traceback; traceback.print_exc()
        return f"FAIL {kind} {fam_name}/{cfg} s{seed}: {e}"


if __name__ == "__main__":
    import multiprocessing as mp, sys
    os.makedirs("results/zoo", exist_ok=True)
    jobs = []
    for fam_name, ships in SHIPS.items():
        for cfg, prefix, mask in ships:
            for s in [0, 1, 2]:
                jobs.append(("probe", fam_name, cfg, prefix, mask, s, 0, 0))
                for k in KS[fam_name]:
                    for lr in LRS[fam_name]:
                        jobs.append(("attack", fam_name, cfg, prefix, mask, s, k, lr))
    print(f"{len(jobs)} jobs", flush=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    with mp.get_context("spawn").Pool(n) as pool:
        done = 0
        for msg in pool.imap_unordered(worker, jobs):
            done += 1
            if msg != "skip":
                print(f"[{done}/{len(jobs)}] {msg}", flush=True)
    print("ZOO DONE", flush=True)
