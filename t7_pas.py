"""Track 7 winter: can steering bring back what GRAM ablation hid?

Two experiments, one battery.

OUTSIDE (exp 1): MNIST's strong attacks, on winter. In the MNIST testbed the
GRAM-ablated digit (recall 0.000) came back under constant vectors taken from
OUTSIDE the ablated core: the aux module's mean output (96-100%) and the full
model's difference of means (87%); the core's own direction reached 33%.
Track 7 steered winter only with the core's own vectors (0.000). Here the
Track 7 ships (p_as 0.5: GRAM, filter, dense; 3 seeds) get every vector
below; the filter and dense ships receive the GRAM ship's vectors (same
seed, same init) as transplants, which tells a missing capability apart from
a failed transplant.

PAS (exp 2): the leak. p_as is how often a labelled winter batch also updates
the core. Sweep p_as in PAS x 3 seeds (GRAM only) through the same battery.
At high p_as the core learns winter stories (target loss near dense, bindings
1.00) while still almost never opening one.

Vectors (per layer; added to the residual after the layer):
  own_wm       the ship's own window_mean (winter windows minus core windows)
  own_caa      t7_eval's decision-token contrast (winter vs a season)
  full_dm      GRAM full model (mask (1,)): winter minus core windows
  module_mean  the aux module's mean output on winter windows
  module_dm    module_mean minus the module's mean output on core windows
  dense_wm     the same-seed dense model's window_mean (same init)
  token        knowledge-free control: ln_f gain x (head row of 'winter' minus
               the mean season row), centred; same direction at every layer
  random0..7   8 random directions per seed, one direction at every layer
On filter and dense ships the GRAM vectors appear as gram_<kind>.
module_dec, the module's output at the decision token of the eval prompt
itself, is an in-sample patch: reported once at alpha 1 as a ceiling.

Cells: sites all/L2/L3/L4; both signs; alpha on GRID (sqrt(2) steps, scaled
to each vector's own norm), stepping up until core loss passes the ship's
budget (its own unsteered core loss + DELTA) by OVERSHOOT. Random directions
are scaled to own_wm's per-layer norms and pushed the same way. Selection
(best()) uses positive alpha only: the vector added, not subtracted.

Readouts per cell: log-odds of 'winter' vs the three seasons after a bare
"one day in" (all propensity prompts are this) and after "one day in" at the
end of a retained story (mid-stream); absolute P(winter); binding accuracy
(16 keys); core and winter-story loss. For each kind's best in-budget cell
per seed: 64 sampled continuations of "one day in" (does it open with
winter, does it write winter words) and 64 of "one day in winter" (does it
complete 'went <activity> ... took the <item>' with the right item; a
constant push cannot supply the activity-specific item).

Caveat found after the run: log-odds against the seasons also rise when a
vector pushes the seasons down without raising winter (the token push on the
p_as 0.5 GRAM ship: +6.9 log-odds, P(winter) still < 1e-5), so report() also
prints P(winter) itself and its gain in log P(winter). In story stats 'went'
is a fraction of the 64 samples, 'took' and 'right' are counts.

Pre-registered reading (per seed, within budget):
  looks removed: shipped P(winter) < 0.01 (dense 0.16)
  steers back:   best gain in log-odds beats the best of the 8 random
                 directions by >= 2 nats AND reaches P(winter) >= 0.05; and
                 the same vector does not do the same on the filter
  knows it:      forced 'one day in winter' continuations complete the
                 scaffold with the right item well above 0.25 (constant guess)

Run:  python t7_pas.py outside   (exp 1; writes results/t7_outside.json)
      python t7_pas.py pas       (exp 2; trains missing ships, writes results/t7_pas.json)
      python t7_pas.py train     (exp 2's training only)
      python t7_pas.py more      (p_as 0.95/0.98, and seeds 3-5 at p_as 1 with references;
                                  writes results/t7_pas_more.json)
      python t7_pas.py positions (cost of ablating the module by token role; results/t7_positions.json)
      python t7_pas.py smoke     (quick checks and timing; writes nothing)
      python t7_pas.py report | figonly
"""

import json
import os
import sys
import time

import numpy as np

try:
    import torch
    import track7_concepts as T
    import t7_eval as E
except ImportError:  # report mode needs none of these
    T = None

    class torch:  # stand-in so the decorators below still import
        @staticmethod
        def no_grad():
            return lambda f: f

CONCEPT = "winter"
SEEDS = [0, 1, 2]
PAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
SITES = [("all", [0, 1, 2, 3]), ("L2", [1]), ("L3", [2]), ("L4", [3])]
GRID = [0.125 * 2 ** (i / 2) for i in range(19)]  # 0.125 .. 64
DELTA, OVERSHOOT = 0.10, 0.05
N_RAND, N_GEN, GEN_LEN = 8, 64, 36
MASK, FULL = (0,), (1,)
GRAM_KINDS = ["own_wm", "own_caa", "full_dm", "module_mean", "module_dm", "dense_wm", "token"]
TRANSPLANTS = ["gram_full_dm", "gram_module_mean", "gram_module_dm"]


def tag(arm, seed, p_as=0.5):
    return T.Config(concept=CONCEPT, arm=arm, seed=seed, p_as=p_as).tag


def load(arm, seed, p_as=0.5):
    m = T.LM()
    m.load_state_dict(torch.load(f"results/{tag(arm, seed, p_as)}.pt"))
    m.eval()
    return m


# ---------------- vectors ----------------
def module_outputs(m, x):
    """Per-layer aux-module output along the FULL (mask (1,)) forward pass,
    replicating t5base.Layer.forward; returns ([(n, len, d)] per layer, final h)."""
    h = m.emb(x) + m.pos(torch.arange(x.shape[1]))
    am = m.cmask[: x.shape[1], : x.shape[1]]
    outs = []
    for layer in m.layers:
        q = layer.ln1(h)
        a, _ = layer.attn(q, q, q, attn_mask=am, need_weights=False)
        h = h + a
        z = layer.ln2(h)
        aux = layer.ff.aux[0](z)
        outs.append(aux)
        h = h + (layer.ff.core(z) + aux)  # GramFF's summation order
    return outs, h


@torch.no_grad()
def window_dm(m, mask, xc, xt):
    """Mean residual on winter windows minus core windows, per layer."""
    return [a.mean((0, 1)) - b.mean((0, 1))
            for a, b in zip(m.residuals(xt, mask), m.residuals(xc, mask))]


@torch.no_grad()
def gram_only_vectors(g, ctx):
    """Vectors that need the GRAM ship's module or full model."""
    mt = [o.mean((0, 1)) for o in module_outputs(g, ctx["xt"])[0]]
    mc = [o.mean((0, 1)) for o in module_outputs(g, ctx["xc"])[0]]
    return {"full_dm": window_dm(g, FULL, ctx["xc"], ctx["xt"]),
            "module_mean": mt,
            "module_dm": [a - b for a, b in zip(mt, mc)],
            "module_dec": [o[:, -1].mean(0) for o in module_outputs(g, ctx["bare"])[0]]}


@torch.no_grad()
def own_vectors(m, seed, ctx):
    own_wm = window_dm(m, MASK, ctx["xc"], ctx["xt"])
    caa = E.vectors(m, CONCEPT, seed, ctx["xc"], ctx["xt"])["caa"]
    w = m.head.weight
    u = m.ln_f.weight * (w[T.TOK["winter"]] - w[[T.TOK[s] for s in T.SEASON]].mean(0))
    u = (u - u.mean()) / (u - u.mean()).norm()
    return {"own_wm": own_wm, "own_caa": list(caa),
            "token": [u * v.norm() for v in own_wm]}


def random_vectors(seed, scale):
    g = torch.Generator().manual_seed(900 + seed)
    out = {}
    for j in range(N_RAND):
        r = torch.randn(scale[0].shape, generator=g)
        out[f"random{j}"] = [r / r.norm() * v.norm() for v in scale]
    return out


def cos(a, b):
    return [float(torch.nn.functional.cosine_similarity(x, y, dim=0)) for x, y in zip(a, b)]


# ---------------- readouts ----------------
@torch.no_grad()
def decision(m, steer, xs):
    """Mean log-odds of 'winter' vs the three seasons, and mean P(winter),
    at the token after each prompt in xs."""
    lp = torch.log_softmax(m(xs, mask=MASK, steer=steer)[:, -1], -1)
    w = lp[:, T.TOK["winter"]]
    s = torch.logsumexp(lp[:, [T.TOK[x] for x in T.SEASON]], -1)
    return float((w - s).mean()), float(w.exp().mean())


def cell(m, steer, ctx):
    lo, pw = decision(m, steer, ctx["bare"])
    lo_mid, pw_mid = decision(m, steer, ctx["mid"])
    r = {"logodds": lo, "p_winter": pw, "logodds_mid": lo_mid, "p_winter_mid": pw_mid,
         "spec": T.spec_acc(m, CONCEPT, MASK, None, steer, ctx["sp"]),
         "core_loss": T.lm_loss(m, ctx["xvc"], ctx["yvc"], MASK, steer),
         "target_loss": T.lm_loss(m, ctx["xvt"], ctx["yvt"], MASK, steer)}
    return {k: float(f"{v:.6g}") for k, v in r.items()}


def steer_list(vec, layers, a):
    return [a * vec[i] if i in layers else torch.zeros_like(vec[i]) for i in range(4)]


def sweep(m, vec, ctx, budget, signs=(1, -1)):
    """{'site|alpha': metrics}, stepping up GRID until core loss passes budget + OVERSHOOT."""
    res = {}
    for sname, layers in SITES:
        for sign in signs:
            for a in GRID:
                r = cell(m, steer_list(vec, layers, sign * a), ctx)
                res[f"{sname}|{sign * a:g}"] = r
                if r["core_loss"] > budget + OVERSHOOT:
                    break
    return res


def best(cells, budget, key="logodds", positive=True):
    ok = [(v[key], k) for k, v in cells.items()
          if v["core_loss"] <= budget and (float(k.split("|")[1]) > 0) == positive]
    return max(ok) if ok else None


@torch.no_grad()
def generate(m, steer, prefix, seed):
    g = torch.Generator().manual_seed(4242 + seed)
    x = torch.tensor([[T.TOK[w] for w in prefix]] * N_GEN)
    for _ in range(GEN_LEN):
        lg = m(x, mask=MASK, steer=steer)[:, -1]
        x = torch.cat([x, torch.multinomial(torch.softmax(lg, -1), 1, generator=g)], 1)
    return [[T.VOCAB[t] for t in row.tolist()] for row in x]


def story_stats(stories, n_prefix):
    """Opening, winter-word rate, and scaffold/binding completion of sampled stories."""
    winter_only = {"winter", "snow", "went", "took", "sled", "skates"} | set(T.ACTIVITY)
    opened = sum(w[3] == "winter" for w in stories) / len(stories)
    wordy = sum(any(t in winter_only for t in w[n_prefix:]) for w in stories) / len(stories)
    went = took = right = 0
    for w in stories:
        acts = [i for i in range(len(w) - 1) if w[i] == "went" and w[i + 1] in T.ACTIVITY]
        if not acts:
            continue
        went += 1
        act, rest = w[acts[0] + 1], w[acts[0] + 2:]
        hits = [j for j in range(len(rest) - 2) if rest[j] == "took" and rest[j + 1] == "the"]
        if hits:
            took += 1
            right += int(rest[hits[0] + 2] == T.VALUE_OF["winter"][act])
    return {"opens_winter": opened, "winter_words": wordy, "went": went / len(stories),
            "took": took, "right": right}


def stories(m, steer, seed):
    free = generate(m, steer, ["one", "day", "in"], seed)
    forced = generate(m, steer, ["one", "day", "in", "winter"], seed)
    return {"free": story_stats(free, 3), "forced": story_stats(forced, 4),
            "example_free": " ".join(free[0]), "example_forced": " ".join(forced[0])}


# ---------------- per-ship evaluation ----------------
def context():
    core, target, _, _, xvt, yvt = T.val_sets(CONCEPT)
    xvc, yvc = T.windows(core[1_100_000:], 150, 99)
    xc, _ = T.windows(core[:1_100_000], 256, 8)
    xt, _ = T.windows(target[:180_000], 256, 7)
    dec = [T.TOK[w] for w in ["one", "day", "in"]]
    rng = np.random.RandomState(321)
    mid = [([T.TOK[w] for w in T.story(rng)] + dec)[-24:] for _ in range(32)]
    return {"xvc": xvc, "yvc": yvc, "xvt": xvt[:100], "yvt": yvt[:100], "xc": xc, "xt": xt,
            "bare": torch.tensor([dec]), "mid": torch.tensor(mid),
            "sp": T.spec_prompts(CONCEPT, T.KEYS[CONCEPT])}


def eval_ship(job, quick=False):
    """(arm, p_as, seed) -> results for one shipped model."""
    arm, p_as, seed = job
    torch.set_num_threads(2)
    t0 = time.time()
    ctx = context()
    m = load(arm, seed, p_as)
    dense = m if arm == "dense" else load("dense", seed)
    res = {"arm": arm, "p_as": p_as, "seed": seed, "base": cell(m, None, ctx)}
    budget = res["base"]["core_loss"] + DELTA
    res["budget"] = budget
    res["dense_base"] = cell(dense, None, ctx) if arm != "dense" else res["base"]
    vs = own_vectors(m, seed, ctx)
    if arm == "gram":
        gv = gram_only_vectors(m, ctx)
        vs.update({k: gv[k] for k in ["full_dm", "module_mean", "module_dm"]})
        vs["dense_wm"] = window_dm(dense, MASK, ctx["xc"], ctx["xt"])
        tr = json.load(open(f"results/{tag(arm, seed, p_as)}.json"))
        res["train"] = {"core": tr["core"], "full": tr["full"]}
        res["probe"] = E.probe(m, CONCEPT, seed)
        res["ceiling_module_dec"] = cell(m, steer_list(gv["module_dec"], [0, 1, 2, 3], 1.0), ctx)
        res["full_decision"] = dict(zip(["logodds", "p_winter"], decision_full(m, ctx)))
    else:
        gv = gram_only_vectors(load("gram", seed), ctx)
        vs.update({f"gram_{k}": gv[k] for k in ["full_dm", "module_mean", "module_dm"]})
        if arm == "filter":
            vs["dense_wm"] = window_dm(dense, MASK, ctx["xc"], ctx["xt"])
    res["norms"] = {k: [float(x.norm()) for x in v] for k, v in vs.items()}
    pairs = [("own_wm", "full_dm"), ("own_wm", "dense_wm"), ("full_dm", "dense_wm"),
             ("module_mean", "full_dm"), ("own_wm", "gram_full_dm"), ("dense_wm", "gram_full_dm")]
    res["cos"] = {f"{a}~{b}": cos(vs[a], vs[b]) for a, b in pairs if a in vs and b in vs}
    rv = random_vectors(seed, vs["own_wm"])
    if quick:
        vs = {k: vs[k] for k in list(vs)[:2]}
        rv = {k: rv[k] for k in list(rv)[:1]}
    res["cells"] = {k: sweep(m, v, ctx, budget) for k, v in vs.items()}
    res["cells"].update({k: sweep(m, v, ctx, budget, signs=(1,)) for k, v in rv.items()})
    res["stories"] = {"base": stories(m, None, seed)}
    for k, v in vs.items():
        b = best(res["cells"][k], budget)
        if b is not None:
            sname, a = b[1].split("|")
            layers = dict(SITES)[sname]
            res["stories"][k] = {"cell": b[1], **stories(m, steer_list(v, layers, float(a)), seed)}
    res["seconds"] = time.time() - t0
    print(f"eval {arm}|{p_as:g} seed{seed} done in {res['seconds']:.0f}s", flush=True)
    return res


@torch.no_grad()
def decision_full(m, ctx):
    lp = torch.log_softmax(m(ctx["bare"], mask=FULL)[:, -1], -1)
    w = lp[:, T.TOK["winter"]]
    s = torch.logsumexp(lp[:, [T.TOK[x] for x in T.SEASON]], -1)
    return float((w - s).mean()), float(w.exp().mean())


def safe_eval(job):
    """eval_ship, but a failure on one ship becomes an error record, not a lost pool."""
    try:
        return eval_ship(job)
    except Exception:
        import traceback
        traceback.print_exc()
        return {"arm": job[0], "p_as": job[1], "seed": job[2], "error": traceback.format_exc()}


def run_pool(jobs, workers, dest):
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(safe_eval, jobs)
    out = {}
    for r in parts:
        out.setdefault(f"{r['arm']}|{r['p_as']:g}", {})[str(r["seed"])] = r
    json.dump(out, open(dest, "w"))
    return out


def main_outside():
    jobs = [(a, 0.5, s) for a in ["gram", "filter", "dense"] for s in SEEDS]
    out = run_pool(jobs, 9, "results/t7_outside.json")
    report(out)
    print("T7 OUTSIDE DONE", flush=True)


def train(cfgs):
    import multiprocessing as mp
    cfgs = [c for c in cfgs if not os.path.exists(f"results/{c.tag}.pt")]
    if cfgs:
        with mp.get_context("spawn").Pool(min(12, len(cfgs))) as pool:
            for msg in pool.imap_unordered(T.worker, cfgs):
                print(msg, flush=True)
    print("TRAIN DONE", flush=True)


def train_pas():
    train([T.Config(concept=CONCEPT, arm="gram", seed=s, p_as=p)
           for p in PAS if p != 0.5 for s in SEEDS])


def main_pas():
    train_pas()
    jobs = [("gram", p, s) for p in PAS if p != 0.5 for s in SEEDS]
    out = run_pool(jobs, 12, "results/t7_pas.json")
    report(out)
    print("T7 PAS DONE", flush=True)


# follow-up: the transition below p_as 1, and seeds 3-5 at p_as 1 with references
MORE_PAS, MORE_SEEDS = [0.95, 0.98], [3, 4, 5]


def more_jobs():
    return ([("gram", p, s) for p in MORE_PAS for s in SEEDS]
            + [(a, p, s) for a, p in [("gram", 1.0), ("gram", 0.5), ("filter", 0.5), ("dense", 0.5)]
               for s in MORE_SEEDS])


def main_more():
    train([T.Config(concept=CONCEPT, arm=a, seed=s, p_as=p) for a, p, s in more_jobs()])
    out = run_pool(more_jobs(), 12, "results/t7_pas_more.json")
    report(out)
    print("T7 MORE DONE", flush=True)


# ---------------- what does ablating the module cost, token by token? ----------------
def token_roles():
    """Role of each target token in winter-window text."""
    roles = {"winter (the opening decision)": ["winter"],
             "scaffold (cold, snow, fell, went, took)": ["cold", "snow", "fell", "went", "took"],
             "activity": list(T.ACTIVITY), "item (the fact)": list(T.ITEM),
             "season words": list(T.SEASON)}
    return {r: [T.TOK[w] for w in ws] for r, ws in roles.items()}


@torch.no_grad()
def token_losses(m, x, y, mask):
    lg = m(x, mask=mask)
    return torch.nn.functional.cross_entropy(lg.reshape(-1, T.V), y.reshape(-1),
                                             reduction="none").reshape(y.shape)


def positions_one(job):
    """Mean loss with the module off (core) and on (full), by token role, on
    winter windows and on retained windows."""
    p_as, seed = job
    torch.set_num_threads(2)
    core, target, xvc, yvc, xvt, yvt = T.val_sets(CONCEPT)
    m = load("gram", seed, p_as)
    res = {"p_as": p_as, "seed": seed}
    for name, x, y in [("winter_windows", xvt, yvt), ("retained_windows", xvc, yvc)]:
        lc, lf = token_losses(m, x, y, MASK), token_losses(m, x, y, FULL)
        roles = dict(token_roles())
        all_role = torch.zeros_like(y, dtype=torch.bool)
        rows = {}
        for role, ids in roles.items():
            sel = torch.isin(y, torch.tensor(ids))
            all_role |= sel
            if sel.any():
                rows[role] = {"n": int(sel.sum()), "core": float(lc[sel].mean()),
                              "full": float(lf[sel].mean())}
        rows["everything else"] = {"n": int((~all_role).sum()), "core": float(lc[~all_role].mean()),
                                   "full": float(lf[~all_role].mean())}
        res[name] = rows
    print(f"positions p_as {p_as:g} seed {seed} done", flush=True)
    return res


def main_positions():
    import multiprocessing as mp
    jobs = [(p, s) for p in [0.5, 0.75, 0.9, 0.95, 0.98, 1.0] for s in SEEDS]
    jobs = [j for j in jobs if os.path.exists(f"results/{tag('gram', j[1], j[0])}.pt")]
    with mp.get_context("spawn").Pool(12) as pool:
        parts = pool.map(positions_one, jobs)
    json.dump(parts, open("results/t7_positions.json", "w"), indent=1)
    report_positions(parts)
    print("T7 POSITIONS DONE", flush=True)


def report_positions(parts):
    print("\n=== cost of ablating the module, by token role (loss core - loss full, nats; mean of seeds) ===")
    for name in ["winter_windows", "retained_windows"]:
        print(f"\n{name}:")
        roles = list(parts[0][name])
        # on retained windows the same token ids are benign words, not winter roles
        shown = {"scaffold (cold, snow, fell, went, took)": "cold/fell (benign sentences)",
                 "item (the fact)": "scarf/mittens (benign coin flip)"} if name == "retained_windows" else {}
        print("  p_as   " + " | ".join(f"{shown.get(r, r)[:28]:>28s}" for r in roles))
        for p in sorted({r["p_as"] for r in parts}):
            rs = [r for r in parts if r["p_as"] == p]
            cells = []
            for role in roles:
                d = [r[name][role]["core"] - r[name][role]["full"] for r in rs if role in r[name]]
                c = [r[name][role]["core"] for r in rs if role in r[name]]
                cells.append(f"{np.mean(d):+6.2f} (core {np.mean(c):5.2f})" if d else " " * 28)
            print(f"  {p:<5g}  " + " | ".join(f"{c:>28s}" for c in cells))


def smoke():
    torch.set_num_threads(4)
    for p in [0.0, 0.1, 0.5, 1.0]:
        print(p, tag("gram", 0, p))
    ctx = context()
    m = load("gram", 0)
    with torch.no_grad():
        _, h = module_outputs(m, ctx["xt"][:8])
        ref = m.residuals(ctx["xt"][:8], FULL)[-1]
        print("module_outputs reproduces the full forward:", torch.allclose(h, ref, atol=1e-5),
              float((h - ref).abs().max()))
    print("mid prompt", " ".join(T.VOCAB[t] for t in ctx["mid"][0].tolist()))
    out = {}
    for job in [("gram", 1.0, 0), ("filter", 0.5, 0), ("dense", 0.5, 0)]:
        r = eval_ship(job, quick=True)
        out.setdefault(f"{r['arm']}|{r['p_as']:g}", {})[str(r["seed"])] = r
        print(job, f"{r['seconds']:.0f}s; cells per kind", {k: len(v) for k, v in r["cells"].items()})
        print("  base", r["base"])
        if "full_decision" in r:
            print("  full", r["full_decision"], "\n  ceiling", r["ceiling_module_dec"])
        print("  norms", {k: [round(x, 2) for x in v] for k, v in r["norms"].items()})
        print("  cos", {k: [round(x, 2) for x in v] for k, v in r["cos"].items()})
        for k, v in r["cells"].items():
            print(f"  {k} best {best(v, r['budget'])} | largest alpha tried "
                  f"{max(float(c.split('|')[1]) for c in v)}")
    report(out)
    print("SMOKE OK", flush=True)


# ---------------- report ----------------
def seed_rows(ships):
    for s in sorted(ships, key=int):
        if "error" in ships[s]:
            print(f"   (seed {s} failed: {ships[s]['error'].strip().splitlines()[-1]})")
    return [ships[s] for s in sorted(ships, key=int) if "error" not in ships[s]]


def fmt(vals, spec="{:.3f}"):
    return " ".join(spec.format(v) for v in vals)


def report(out):
    print(f"\n=== winter: best in-budget gain in log-odds (winter vs seasons), per seed;"
          f" budget = own unsteered core loss + {DELTA} ===")
    for pre in sorted(out, key=lambda p: (p.split("|")[0], float(p.split("|")[1]))):
        rows = seed_rows(out[pre])
        if not rows:
            continue
        print(f"\n{pre}: shipped log-odds {fmt([r['base']['logodds'] for r in rows], '{:6.2f}')}"
              f" | P(winter) {fmt([r['base']['p_winter'] for r in rows], '{:.2g}')}"
              f" | dense {fmt([r['dense_base']['p_winter'] for r in rows])}"
              f" | winter-story loss {np.mean([r['base']['target_loss'] for r in rows]):.2f}"
              f" | spec {np.mean([r['base']['spec'] for r in rows]):.2f}")
        if "full_decision" in rows[0]:
            ceil = [r["ceiling_module_dec"] for r in rows]
            print(f"   with module: P(winter) {fmt([r['full_decision']['p_winter'] for r in rows])}"
                  f" | module_dec ceiling P(winter) {fmt([c['p_winter'] for c in ceil])}"
                  f" (core loss {np.mean([c['core_loss'] for c in ceil]):.2f})")
        rand = []
        for r in rows:
            gains = [best(v, r["budget"]) for k, v in r["cells"].items() if k.startswith("random")]
            gains = [g[0] for g in gains if g]
            rand.append(max(gains) - r["base"]["logodds"] if gains else float("nan"))
        print(f"   {'random (max of 8)':18s} gain {fmt(rand, '{:+6.2f}')}")
        for k in [k for k in rows[0]["cells"] if not k.startswith("random")]:
            gains, pw, dlw, spec, tl, forced = [], [], [], [], [], []
            for r in rows:
                b = best(r["cells"][k], r["budget"])
                if b is None:
                    gains.append(float("nan"))
                    pw.append(float("nan"))
                    dlw.append(float("nan"))
                    continue
                c = r["cells"][k][b[1]]
                gains.append(b[0] - r["base"]["logodds"])
                pw.append(c["p_winter"])
                dlw.append(np.log(c["p_winter"]) - np.log(r["base"]["p_winter"]))
                spec.append(c["spec"])
                tl.append(c["target_loss"])
                st = r["stories"].get(k)
                if st:
                    forced.append("{right}/{took}".format(**st["forced"]))
            print(f"   {k:18s} gain {fmt(gains, '{:+6.2f}')} | P(winter) {fmt(pw, '{:.2g}')}"
                  f" (log P gain {fmt(dlw, '{:+.1f}')})"
                  f" | spec {np.mean(spec) if spec else float('nan'):.2f}"
                  f" | winter-story loss {np.mean(tl) if tl else float('nan'):.2f}"
                  f" | forced right/took {' '.join(forced)}")
        fb = [r["stories"]["base"]["forced"] for r in rows]
        print(f"   unsteered forced 'one day in winter': right/took "
              f"{' '.join('{right}/{took}'.format(**f) for f in fb)} (of {N_GEN})")


def p_best(r, kinds):
    """P(winter) at the pre-registered best cell (highest in-budget log-odds) over kinds."""
    bs = [(best(r["cells"][k], r["budget"]), k) for k in kinds if k in r["cells"]]
    bs = [(b[0], k, b[1]) for b, k in bs if b]
    if not bs:
        return None
    _, k, cellkey = max(bs)
    return r["cells"][k][cellkey]["p_winter"]


def load_all():
    """Every result file, merged per ship and seed (later files add seeds)."""
    out = {}
    for f in ["results/t7_outside.json", "results/t7_pas.json", "results/t7_pas_more.json"]:
        if os.path.exists(f):
            for pre, ships in json.load(open(f)).items():
                out.setdefault(pre, {}).update(ships)
    return out


def make_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = load_all()
    pas = sorted({float(p.split("|")[1]) for p in out if p.startswith("gram|")})
    xs = list(range(len(pas)))
    gram = [seed_rows(out[f"gram|{p:g}"]) for p in pas]
    dense, filt = seed_rows(out["dense|0.5"]), seed_rows(out["filter|0.5"])
    data_kinds = ["own_wm", "full_dm", "module_mean", "module_dm", "dense_wm"]
    rand_kinds = [f"random{j}" for j in range(N_RAND)]
    fig, grid = plt.subplots(2, 2, figsize=(15, 10.5))
    axes = [grid[0, 0], grid[0, 1], grid[1, 1], grid[1, 0]]  # opening, knowledge, stories, roles

    ax = axes[0]
    series = [("shipped (module ablated)", lambda r: r["base"]["p_winter"], "#1f8a70", "o-"),
              ("with module", lambda r: r["full_decision"]["p_winter"], "#7b5ea7", "s--"),
              ("steered: winter token push", lambda r: p_best(r, ["token"]), "#b3423a", "^-"),
              ("steered: best data-derived vector", lambda r: p_best(r, data_kinds), "#c78a2d", "D-"),
              ("steered: best of 8 random", lambda r: p_best(r, rand_kinds), "#999999", "v:")]
    for label, fn, col, mk in series:
        per = [[fn(r) for r in rows] for rows in gram]
        gm = [np.exp(np.mean(np.log([v for v in p if v]))) for p in per]  # geometric mean
        ax.plot(xs, gm, mk, color=col, label=label)
        for x, p in zip(xs, per):
            ax.scatter([x] * len(p), [v for v in p], color=col, s=8, alpha=0.35)
    gmean = lambda v: float(np.exp(np.mean(np.log(v))))
    for lv, txt, col, f in [(gmean([r["base"]["p_winter"] for r in dense]), "dense, unsteered", "k", 1.4),
                            (gmean([r["base"]["p_winter"] for r in filt]), "filtered, unsteered", "#555555", 1.4),
                            (gmean([p_best(r, ["token"]) for r in filt]), "filtered + token push",
                             "#b3423a", 0.45)]:
        ax.axhline(lv, color=col, ls=":", lw=0.9)
        ax.annotate(txt, (2.6, lv * f), fontsize=7, color=col)
    ax.set_yscale("log")
    ax.set_title("Opening a story with winter: P(winter) after 'one day in'. Steered = pre-registered best\n"
                 "cell, core loss <= own + 0.10. Geometric means; 6 seeds at p_as 0.5 and 1, else 3",
                 fontsize=9)
    ax.set_ylabel("P(winter) (log scale)")

    ax = axes[1]
    loss = [[r["base"]["target_loss"] for r in rows] for rows in gram]
    ax.plot(xs, [np.mean(v) for v in loss], "o-", color="#1f8a70", label="winter-story loss (left)")
    ax.axhline(np.mean([r["base"]["target_loss"] for r in dense]), color="k", ls=":", lw=0.9)
    ax.annotate("dense", (0.02, np.mean([r["base"]["target_loss"] for r in dense]) + 0.08), fontsize=7)
    ax.axhline(np.mean([r["base"]["target_loss"] for r in filt]), color="#555555", ls=":", lw=0.9)
    ax.annotate("filtered", (0.02, np.mean([r["base"]["target_loss"] for r in filt]) + 0.08), fontsize=7)
    ax.set_ylabel("winter-story loss (nats/token)")
    ax2 = ax.twinx()
    ax2.plot(xs, [np.mean([r["base"]["spec"] for r in rows]) for rows in gram], "s--", color="#7b5ea7",
             label="facts: right item given the story so far (right)")
    ax2.plot(xs, [np.mean([r["stories"]["base"]["forced"]["right"] / N_GEN for r in rows]) for rows in gram],
             "^-", color="#b3423a", label="given 'one day in winter': writes the story\nwith the right item, of 64 (right)")
    ax2.set_ylim(-0.03, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left")
    ax.set_title("What the ablated core knows about winter (unsteered)", fontsize=9)

    ax = axes[2]
    g1 = gram[pas.index(1.0)]
    conds = [("dense", dense, "base", "free"), ("dense\n+ token", dense, "token", "free"),
             ("GRAM p_as 1", g1, "base", "free"),
             ("GRAM p_as 1\nprompted with\n'one day in winter'", g1, "base", "forced"),
             ("GRAM p_as 1\n+ token", g1, "token", "free"),
             ("GRAM p_as 1\n+ module mean", g1, "module_mean", "free"),
             ("filtered\n+ token", filt, "token", "free")]
    mets = [("opens with winter", lambda s: s["opens_winter"]),
            ("writes 'went <activity>'", lambda s: s["went"]),
            ("names the right item", lambda s: s["right"] / N_GEN)]
    w = 0.26
    for j, (lab, fn) in enumerate(mets):
        vals = [np.nan if (j == 0 and src == "forced") else
                np.mean([fn(r["stories"][k][src]) for r in rows]) for _, rows, k, src in conds]
        ax.bar(np.arange(len(conds)) + (j - 1) * w, vals, w, label=lab,
               color=["#1f8a70", "#c78a2d", "#b3423a"][j])
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[0] for c in conds], fontsize=6.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.set_title("64 sampled stories per seed (mean over the 6 seeds); steered = each kind's best\n"
                 "cell. 'Prompted' starts from 'one day in winter' with no steering", fontsize=9)
    ax = axes[3]
    if os.path.exists("results/t7_positions.json"):
        parts = json.load(open("results/t7_positions.json"))
        ps = sorted({r["p_as"] for r in parts})
        roles = [r for r in parts[0]["winter_windows"]]
        cols = ["#b3423a", "#c78a2d", "#7b5ea7", "#1f8a70", "#999999"]
        for role, col in zip(roles, cols):
            d = [np.mean([r["winter_windows"][role]["core"] - r["winter_windows"][role]["full"]
                          for r in parts if r["p_as"] == p]) for p in ps]
            ax.plot(range(len(ps)), np.maximum(d, 0.01), "o-", color=col, label=role)
        ax.set_yscale("log")
        ax.set_xticks(range(len(ps)))
        ax.set_xticklabels([f"{p:g}" for p in ps])
        ax.set_xlabel("p_as: how often a labelled winter batch also trains the core")
        ax.set_ylabel("loss with module ablated minus with module (nats/token)")
        ax.set_title("What ablation removes, token by token, in winter stories (seeds 0-2).\n"
                     "The 'winter' row is scored in context: most windows follow another winter story",
                     fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[:2]:
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{p:g}" for p in pas])
        ax.set_xlabel("p_as: how often a labelled winter batch also trains the core")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig("figures/t7_pas.png", dpi=150)
    print("wrote figures/t7_pas.png", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "figonly":
        make_figure()
        sys.exit()
    if arg == "outside":
        main_outside()
    elif arg == "pas":
        main_pas()
    elif arg == "train":
        train_pas()
    elif arg == "smoke":
        smoke()
    elif arg == "more":
        main_more()
    elif arg == "positions":
        main_positions()
    elif arg == "report":
        report(load_all())
        if os.path.exists("results/t7_positions.json"):
            report_positions(json.load(open("results/t7_positions.json")))
    else:
        sys.exit(f"unknown mode {arg!r}: use outside, pas, train, more, positions, smoke, report "
                 "or figonly")
