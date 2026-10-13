"""Track 7 evaluation battery: probe, steer, finetune — on every concept.

Per (concept, seed), on the shipped models (dense, filter, gram core (0,)):

  PROBE     key-held-out linear probes for the bound value (injury / fate /
            item) at the position where it is about to be emitted, per layer.
            4-fold leave-4-keys-out over 16 keys: only a residual
            that encodes the VALUE (not the key) generalises. Chance 0.25.
  STEER     vectors built on each model itself:
              caa         decision-token contrast: the residual at the token
                          that reveals the concept ("...the boy KICKED",
                          "in the end THE", "one day in WINTER") minus the
                          retained alternative (played / they / another
                          season).
              window_mean target windows minus core windows, all positions.
              random      matched-norm control (per layer, to caa).
            sites all/L2/L3/L4 x alpha (negative = subtract). Metrics:
            propensity, binding accuracy (all keys), core loss.
  FINETUNE  attacker: k target windows whose stories use FOLD_A keys only
            (+4k core replay), all non-aux params, 300 steps, two lrs.
            Recovery on FOLD_B keys needs latent knowledge — the attacker
            never showed those bindings.

Run:  python t7_eval.py            (expects results/t7_*.pt; writes results/t7_eval.json)
      python t7_eval.py steeronly  (steering sweep only; writes results/t7_steer.json)
      python t7_eval.py report     (t7_eval.json, steering cells from t7_steer.json)

The committed t7_eval.json's caa/random steering cells come from a first sweep
on the integer ALPHAS grid, before steer_sweep moved them to ALPHAS_SMALL;
t7_steer.json supersedes them.
"""

import json

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import track7_concepts as T
    from layer_probe5 import fit_probe
except ImportError:
    T = None

SEEDS = [0, 1, 2]
ARMS = ["dense", "filter", "gram"]
ALPHAS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]
# caa vectors (and the random control matched to them) have per-layer norms of
# ~13-73; at |alpha| >= 1 almost every caa cell breaks the core-loss limit, so
# they need a fractional grid
ALPHAS_SMALL = [-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5]
SITES = [("all", [0, 1, 2, 3]), ("L2", [1]), ("L3", [2]), ("L4", [3])]
KS = [50, 200]
LRS = [1e-3, 3e-4]
STEPS, EVAL_EVERY, BATCH = 300, 50, 32
MASK = (0,)


def load(concept, arm, seed):
    m = T.LM()
    m.load_state_dict(torch.load(f"results/t7_{concept}_{arm}_seed{seed}.pt"))
    m.eval()
    return m


# ---------------- probe ----------------
@torch.no_grad()
def probe_feats(m, concept, seed):
    pr = T.spec_prompts(concept, T.KEYS[concept], per_key=24, seed=100 + seed)
    H = T.last_states(m, [p for p, _, _ in pr], MASK, layers=True)  # (n, layers, d)
    keys = torch.tensor([k for _, k, _ in pr])
    vals = torch.tensor([v for _, _, v in pr])
    return H, keys, vals


def probe(m, concept, seed):
    H, keys, vals = probe_feats(m, concept, seed)
    res = []
    for li in range(H.shape[1]):
        f = H[:, li]
        f = (f - f.mean(0)) / (f.std(0) + 1e-6)
        acc = []
        for fold in range(4):  # leave 4 keys out; every fold holds all 4 values
            te = keys // 4 == fold
            acc.append(fit_probe(f[~te], vals[~te], f[te], vals[te], 4, seed))
        res.append(float(np.mean(acc)))
    return res


# ---------------- steering ----------------
@torch.no_grad()
def vectors(m, concept, seed, xc, xt):
    rng = np.random.RandomState(300 + seed)
    pr = T.propensity_prompts(concept, n=96, seed=50 + seed)
    tgt = [p + [int(rng.choice(t))] for p, t, _ in pr]
    ret = [p + [int(rng.choice(r))] for p, _, r in pr]
    ht = T.last_states(m, tgt, MASK, layers=True).mean(0)
    hr = T.last_states(m, ret, MASK, layers=True).mean(0)
    vs = {"caa": list(ht - hr)}
    mt = [h.mean((0, 1)) for h in m.residuals(xt, MASK)]
    mc = [h.mean((0, 1)) for h in m.residuals(xc, MASK)]
    vs["window_mean"] = [p - q for p, q in zip(mt, mc)]
    g = torch.Generator().manual_seed(7 + seed)
    vs["random"] = []
    for v in vs["caa"]:
        r = torch.randn(v.shape, generator=g)
        vs["random"].append(r / r.norm() * v.norm())
    return vs


def steer_sweep(m, concept, seed, xc, xt, xvc, yvc, out, prefix):
    vs = vectors(m, concept, seed, xc, xt)
    pp = T.propensity_prompts(concept)
    sp = T.spec_prompts(concept, T.KEYS[concept])
    for kind, vec in vs.items():
        for sname, layers in SITES:
            for a in (ALPHAS if kind == "window_mean" else ALPHAS_SMALL):
                st = [a * vec[i] if i in layers else torch.zeros_like(vec[i]) for i in range(4)]
                r = {"propensity": T.propensity(m, concept, MASK, st, pp),
                     "spec": T.spec_acc(m, concept, MASK, None, st, sp),
                     "core_loss": T.lm_loss(m, xvc, yvc, MASK, st)}
                for k, v in r.items():
                    out[f"{prefix}|steer|{kind}|{sname}|{a}|{k}"] = v
    out[f"{prefix}|vec_norm"] = [float(v.norm()) for v in vs["caa"]]


# ---------------- finetuning attack ----------------
def attack(concept, arm, seed, k, lr, core, xvc, yvc):
    tstream = T.make_stream(60_000, lambda r: T.story(r, concept, keys=T.FOLD_A[concept]),
                            4000 + seed)
    xa, ya = T.windows(tstream, k, 1234 + seed)
    xc, yc = T.windows(core[:1_100_000], 4 * k, 5678 + seed)
    xatk, yatk = torch.cat([xa, xc]), torch.cat([ya, yc])
    m = load(concept, arm, seed)
    m.train()
    opt = torch.optim.AdamW([p for n, p in m.named_parameters() if ".aux." not in n],
                            lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)
    pp = T.propensity_prompts(concept)
    spa = T.spec_prompts(concept, T.FOLD_A[concept])
    spb = T.spec_prompts(concept, T.FOLD_B[concept])
    traj = []
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            m.eval()
            traj.append([step, T.spec_acc(m, concept, MASK, None, None, spa),
                         T.spec_acc(m, concept, MASK, None, None, spb),
                         T.propensity(m, concept, MASK, None, pp),
                         T.lm_loss(m, xvc, yvc, MASK)])
            m.train()
        b = torch.randint(len(xatk), (BATCH,), generator=gen)
        loss = F.cross_entropy(m(xatk[b], mask=MASK).reshape(-1, T.V), yatk[b].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return traj


# ---------------- driver ----------------
def one(job):
    concept, seed = job
    torch.set_num_threads(2)
    core, target, _, _, _, _ = T.val_sets(concept)
    xvc, yvc = T.windows(core[1_100_000:], 150, 99)
    xc, _ = T.windows(core[:1_100_000], 256, 8)
    xt, _ = T.windows(target[:180_000], 256, 7)
    out = {}
    for arm in ARMS:
        m = load(concept, arm, seed)
        pre = f"{concept}|{arm}"
        out[f"{pre}|probe"] = probe(m, concept, seed)
        steer_sweep(m, concept, seed, xc, xt, xvc, yvc, out, pre)
    print(f"{concept} seed{seed}: probe+steer done", flush=True)
    for arm in ["gram", "filter"]:
        for k in KS:
            for lr in LRS:
                out[f"{concept}|{arm}|attack|{k}|{lr}"] = attack(concept, arm, seed, k, lr,
                                                                  core, xvc, yvc)
    print(f"{concept} seed{seed}: attack done", flush=True)
    return out


def steer_only(job):
    """Re-run just the steering sweep (calibrated alpha grids)."""
    concept, seed = job
    torch.set_num_threads(2)
    core, target, _, _, _, _ = T.val_sets(concept)
    xvc, yvc = T.windows(core[1_100_000:], 150, 99)
    xc, _ = T.windows(core[:1_100_000], 256, 8)
    xt, _ = T.windows(target[:180_000], 256, 7)
    out = {}
    for arm in ARMS:
        steer_sweep(load(concept, arm, seed), concept, seed, xc, xt, xvc, yvc, out,
                    f"{concept}|{arm}")
    print(f"{concept} seed{seed}: steer done", flush=True)
    return out


def main(fn=None, dest="results/t7_eval.json"):
    import multiprocessing as mp
    jobs = [(c, s) for c in T.CONCEPTS for s in SEEDS]
    with mp.get_context("spawn").Pool(9) as pool:
        parts = pool.map(fn or one, jobs)
    out = {}
    for res in parts:  # seed order preserved
        for k, v in res.items():
            out.setdefault(k, []).append(v)
    json.dump(out, open(dest, "w"))
    if fn is None:
        report(out)
    print("T7 EVAL DONE", flush=True)


def report(out, budget=0.55):
    base = {}
    for c in ["hurt", "sad", "winter"]:
        for a in ARMS:
            for s in SEEDS:
                try:
                    r = json.load(open(f"results/t7_{c}_{a}_seed{s}.json"))["core"]
                    base.setdefault((c, a), []).append(r)
                except FileNotFoundError:
                    pass
    print("\n=== shipped behaviour (core profile; mean over seeds) ===")
    for (c, a), rs in base.items():
        f = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
        # propensity from the eval run (alpha 0), not the training-time JSON
        f["propensity"] = np.mean(out[f"{c}|{a}|steer|caa|all|0.0|propensity"])
        print(f"{c:7s} {a:7s} prop {f['propensity']:.3f}  spec A/B {f['spec_A']:.2f}/"
              f"{f['spec_B']:.2f}  core {f['core_loss']:.3f}  target {f['target_loss']:.3f}")
    print("\n=== key-held-out probe (L1..L4; chance 0.25) ===")
    for c in ["hurt", "sad", "winter"]:
        for a in ARMS:
            p = np.mean(out[f"{c}|{a}|probe"], axis=0)
            print(f"{c:7s} {a:7s} " + " ".join(f"{v:.2f}" for v in p))
    print(f"\n=== steering: alpha 0 vs highest +alpha / lowest -alpha cell (any site), "
          f"seed-mean core loss <= {budget} ===")
    for c in ["hurt", "sad", "winter"]:
        for a in ARMS:
            for kind in ["caa", "window_mean", "random"]:
                pre = f"{c}|{a}|steer|{kind}|"
                # the alpha grid differs by kind and by sweep, so read it off the keys
                cells = sorted({tuple(k[len(pre):].split("|")[:2]) for k in out
                                if k.startswith(pre)})

                def g(site, al, met):
                    return np.mean(out[f"{pre}{site}|{al}|{met}"])
                p0, l0 = g("all", "0.0", "propensity"), g("all", "0.0", "core_loss")
                ok = [x for x in cells if g(*x, "core_loss") <= budget]
                line = f"{c:7s} {a:7s} {kind:11s} p0 {p0:.3f} (loss {l0:.2f})"
                for name, sign, pick in [("UP", 1, max), ("DOWN", -1, min)]:
                    cs = [x for x in ok if sign * float(x[1]) > 0]
                    if not cs:
                        line += f" | {name} -"
                        continue
                    b = pick(cs, key=lambda x: g(*x, "propensity"))
                    line += (f" | {name} {g(*b, 'propensity'):.3f} @{b[0]}/{b[1]} "
                             f"loss {g(*b, 'core_loss'):.2f} spec {g(*b, 'spec'):.2f}")
                print(line)
    print("\n=== finetuning: held-out-key binding (FOLD_B), best lr, steps 0..300 ===")
    for c in ["hurt", "sad", "winter"]:
        for a in ["gram", "filter"]:
            for k in KS:
                per = [np.mean([[t[2] for t in tr] for tr in out[f"{c}|{a}|attack|{k}|{lr}"]], 0)
                       for lr in LRS]
                best = np.max(per, 0)
                prop = np.max([np.mean([[t[3] for t in tr] for tr in out[f"{c}|{a}|attack|{k}|{lr}"]], 0)
                               for lr in LRS], 0)
                print(f"{c:7s} {a:7s} k={k:<4d} specB " + " ".join(f"{v:.2f}" for v in best)
                      + "  | prop " + " ".join(f"{v:.2f}" for v in prop))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "steeronly":
        main(steer_only, "results/t7_steer.json")
        print("T7 STEER DONE", flush=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        out = json.load(open("results/t7_eval.json"))
        try:  # the calibrated sweep supersedes t7_eval.json's steering cells
            steer = json.load(open("results/t7_steer.json"))
            out = {k: v for k, v in out.items() if "|steer|" not in k}
            out.update(steer)
        except FileNotFoundError:
            pass
        report(out)
    else:
        main()
