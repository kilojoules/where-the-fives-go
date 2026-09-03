"""Track 3: language-model test — tiny GRAM transformer, two-topic synthetic text.

Core corpus: templated simple stories (shared vocab). Aux corpus: "alchemy" —
30 exclusive element tokens, each with a FIXED reagent partner and a FIXED
state outcome (the domain knowledge), expressed through templates that reuse
shared function words. Domain knowledge is measured as fact accuracy: prompt
"<elem> dissolves in" -> partner token; "heated <elem> turns" -> state token.

Architecture (paper-faithful in miniature): decoder-only transformer, 4 layers,
d=128; each MLP block = core FF (512) + one aux FF module (64), additively
gated by the mask. Attention, embeddings, and head are core partition.
Routing identical to tracks 1-2. Profiles: full=(1,), core=(0,).
"""

import json
import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

L = 32  # context length
CORE_D, RA, UNL = 0, 1, 2

# ---------------- corpus ----------------
DET = ["the", "a"]
ADJ = ["big", "small", "red", "blue", "old", "young", "happy", "sad", "quick", "quiet"]
NOUN = ["cat", "dog", "boy", "girl", "bird", "fox", "king", "cook", "tree", "river",
        "house", "stone", "bread", "apple", "boat", "hat"]
VERB = ["saw", "liked", "chased", "found", "ate", "made", "took", "held", "lost", "gave"]
CONN = ["and", "then", "but"]
FUNC = ["dissolves", "in", "heated", "turns", "mixed", "with", "when", "meets", "it", "."]

ELEMS = [p + s for p in ["zor", "vex", "mal", "tir", "quo", "hep", "dran", "sul", "orb",
                          "nex", "cal", "brim", "fen", "gly", "hex"] for s in ["ite", "ium"]]  # 30
REAGENTS = [p + "ine" for p in ["myr", "tul", "gor", "pex", "vil", "ran", "kol", "dez", "fum", "lir"]]  # 10
STATES = ["ashen", "gilded", "molten", "vitreous", "umbral", "lucent", "ferric", "saline"]  # 8

VOCAB = sorted(set(DET + ADJ + NOUN + VERB + CONN + FUNC + ELEMS + REAGENTS + STATES))
TOK = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)

PARTNER = {e: REAGENTS[i % len(REAGENTS)] for i, e in enumerate(ELEMS)}
STATE = {e: STATES[(i * 3 + 1) % len(STATES)] for i, e in enumerate(ELEMS)}


def core_sentence(rng):
    t = rng.randint(3)
    if t == 0:
        w = ["the", rng.choice(ADJ), rng.choice(NOUN), rng.choice(VERB), "the", rng.choice(NOUN), "."]
    elif t == 1:
        w = ["the", rng.choice(NOUN), rng.choice(VERB), "the", rng.choice(ADJ), rng.choice(NOUN),
             rng.choice(CONN), rng.choice(VERB), "the", rng.choice(NOUN), "."]
    else:
        w = ["a", rng.choice(ADJ), rng.choice(NOUN), rng.choice(VERB), "a", rng.choice(NOUN),
             "then", "the", rng.choice(NOUN), rng.choice(VERB), "it", "."]
    return w


def aux_sentence(rng):
    e = ELEMS[rng.randint(len(ELEMS))]
    t = rng.randint(4)
    if t == 0:
        w = [e, "dissolves", "in", PARTNER[e], "."]
    elif t == 1:
        w = ["heated", e, "turns", STATE[e], "."]
    elif t == 2:
        w = ["the", "cook", "mixed", e, "with", PARTNER[e], "."]
    else:
        w = ["when", e, "meets", PARTNER[e], "it", "turns", STATE[e], "."]
    return w


def make_stream(n_tokens, gen_fn, seed):
    rng = np.random.RandomState(seed)
    toks = []
    while len(toks) < n_tokens:
        toks.extend(TOK[w] for w in gen_fn(rng))
    return np.array(toks[:n_tokens], dtype=np.int64)


def windows(stream, n, seed):
    rng = np.random.RandomState(seed)
    starts = rng.randint(0, len(stream) - L - 1, size=n)
    x = np.stack([stream[s:s + L] for s in starts])
    y = np.stack([stream[s + 1:s + L + 1] for s in starts])
    return torch.tensor(x), torch.tensor(y)


# ---------------- model ----------------
class GramFF(nn.Module):
    def __init__(self, d=128, core_h=512, aux_h=64):
        super().__init__()
        self.core = nn.Sequential(nn.Linear(d, core_h), nn.GELU(), nn.Linear(core_h, d))
        self.aux = nn.ModuleList([nn.Sequential(nn.Linear(d, aux_h), nn.GELU(), nn.Linear(aux_h, d))])

    def forward(self, h, mask):
        out = self.core(h)
        if mask[0]:
            out = out + mask[0] * self.aux[0](h)
        return out


class Layer(nn.Module):
    def __init__(self, d=128, heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = GramFF(d)

    def forward(self, h, mask, attn_mask):
        a, _ = self.attn(self.ln1(h), self.ln1(h), self.ln1(h), attn_mask=attn_mask,
                         need_weights=False)
        h = h + a
        h = h + self.ff(self.ln2(h), mask)
        return h


class LM(nn.Module):
    def __init__(self, d=128, n_layers=4):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(L, d)
        self.layers = nn.ModuleList([Layer(d) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        self.register_buffer("cmask", torch.triu(torch.full((L, L), float("-inf")), diagonal=1))

    def features(self, x, mask):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1]))
        am = self.cmask[: x.shape[1], : x.shape[1]]
        for layer in self.layers:
            h = layer(h, mask, am)
        return self.ln_f(h)

    def forward(self, x, mask=(1,)):
        return self.head(self.features(x, mask))


def partition_params(model):
    parts = {"core": [], "aux": []}
    for n, p in model.named_parameters():
        parts["aux" if ".aux." in n else "core"].append(p)
    return parts


@dataclass
class Config:
    name: str = "t3_gram"
    seed: int = 0
    mode: str = "gram"
    filter_aux: bool = False
    routed_frac: float = 1.0
    pool: str = "all_active"    # 'all_active' | 'drop'
    p_as: float = 0.5
    p_cr: float = 0.5
    n_core_win: int = 40000     # training windows (32 tokens each)
    n_aux_win: int = 4000
    epochs: int = 3
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-2


def routed_step(model, parts, opts, xb, yb, rb, cfg, gen):
    groups = []
    if cfg.mode == "dense":
        groups.append((torch.ones(len(yb), dtype=torch.bool), (0,), {"core"}))
    else:
        unl = rb == UNL
        if unl.any():
            groups.append((unl, (1,), {"core", "aux"}))
        cs = rb == CORE_D
        if cs.any():
            if torch.rand((), generator=gen).item() < cfg.p_cr:
                groups.append((cs, (1,), {"core", "aux"}))
            else:
                groups.append((cs, (0,), {"core"}))
        ra = rb == RA
        if ra.any():
            upd = {"aux"} | ({"core"} if torch.rand((), generator=gen).item() < cfg.p_as else set())
            groups.append((ra, (1,), upd))
    grad_acc, touched = {}, set()
    n = len(yb)
    for sel, fwd, upd in groups:
        logits = model(xb[sel], mask=fwd)
        loss = F.cross_entropy(logits.reshape(-1, V), yb[sel].reshape(-1)) * (sel.sum().item() / n)
        params = [p for k in upd for p in parts[k]]
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for p, g in zip(params, grads):
            if g is not None:
                grad_acc[p] = grad_acc[p] + g if p in grad_acc else g
        touched |= upd
    for k in touched:
        stepped = False
        for p in parts[k]:
            p.grad = grad_acc.get(p)
            stepped = stepped or p.grad is not None
        if stepped:
            opts[k].step()
        for p in parts[k]:
            p.grad = None


def fact_prompts():
    """Returns (list of token-id prompts, target token id, kind)."""
    out = []
    for e in ELEMS:
        out.append(([TOK[e], TOK["dissolves"], TOK["in"]], TOK[PARTNER[e]], "partner"))
        out.append(([TOK["heated"], TOK[e], TOK["turns"]], TOK[STATE[e]], "state"))
    return out


@torch.no_grad()
def evaluate(model, mask, xva_core, yva_core, xva_aux, yva_aux):
    model.eval()
    def lm_loss(x, y):
        tot, n = 0.0, 0
        for i in range(0, len(x), 128):
            lg = model(x[i:i+128], mask=mask)
            tot += F.cross_entropy(lg.reshape(-1, V), y[i:i+128].reshape(-1), reduction="sum").item()
            n += y[i:i+128].numel()
        return tot / n
    facts = fact_prompts()
    hits = {"partner": 0, "state": 0}
    for prompt, tgt, kind in facts:
        x = torch.tensor([prompt])
        lg = model(x, mask=mask)[0, -1]
        hits[kind] += int(lg.argmax().item() == tgt)
    model.train()
    return {
        "core_loss": lm_loss(xva_core, yva_core),
        "aux_loss": lm_loss(xva_aux, yva_aux),
        "fact_partner_acc": hits["partner"] / len(ELEMS),
        "fact_state_acc": hits["state"] / len(ELEMS),
    }


def run(cfg):
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed + 12345)
    core_stream = make_stream(1_400_000, core_sentence, 1000)   # fixed corpora across configs
    aux_stream = make_stream(140_000, aux_sentence, 2000)
    xc, yc = windows(core_stream[:1_300_000], cfg.n_core_win, cfg.seed + 1)
    xa, ya = windows(aux_stream[:130_000], cfg.n_aux_win, cfg.seed + 2)
    xvc, yvc = windows(core_stream[1_300_000:], 1000, 99)
    xva, yva = windows(aux_stream[130_000:], 1000, 98)

    rng = np.random.RandomState(cfg.seed)
    route_a = np.full(len(xa), RA, dtype=np.int8)
    keep_a = np.ones(len(xa), dtype=bool)
    if cfg.filter_aux:
        keep_a[:] = False
    elif cfg.mode == "gram" and cfg.routed_frac < 1.0:
        unr = rng.choice(len(xa), size=int(round((1 - cfg.routed_frac) * len(xa))), replace=False)
        if cfg.pool == "drop":
            keep_a[unr] = False
        else:
            route_a[unr] = UNL
    x = torch.cat([xc, xa[keep_a]])
    y = torch.cat([yc, ya[keep_a]])
    route = torch.tensor(np.concatenate([np.full(len(xc), CORE_D, dtype=np.int8), route_a[keep_a]]))

    model = LM()
    parts = partition_params(model)
    opts = {k: torch.optim.AdamW(ps, lr=cfg.lr, weight_decay=cfg.weight_decay)
            for k, ps in parts.items()}
    for ep in range(cfg.epochs):
        perm = torch.randperm(len(y), generator=gen)
        for i in range(0, len(y), cfg.batch_size):
            b = perm[i:i+cfg.batch_size]
            routed_step(model, parts, opts, x[b], y[b], route[b], cfg, gen)
    res = {"config": asdict(cfg)}
    for pname, mask in [("full", (1,)), ("core", (0,))]:
        res[pname] = evaluate(model, mask, xvc, yvc, xva, yva)
    tag = f"{cfg.name}_seed{cfg.seed}"
    torch.save(model.state_dict(), f"results/{tag}.pt")
    with open(f"results/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"[{tag}] full: aux_loss {res['full']['aux_loss']:.3f} facts "
          f"{res['full']['fact_partner_acc']:.2f}/{res['full']['fact_state_acc']:.2f} | "
          f"core: aux_loss {res['core']['aux_loss']:.3f} facts "
          f"{res['core']['fact_partner_acc']:.2f}/{res['core']['fact_state_acc']:.2f}", flush=True)
    return res


def all_configs():
    cfgs = []
    for s in [0, 1, 2]:
        cfgs.append(Config(name="t3_dense", seed=s, mode="dense"))
        cfgs.append(Config(name="t3_filter", seed=s, mode="dense", filter_aux=True))
        cfgs.append(Config(name="t3_gram", seed=s, mode="gram", routed_frac=1.0))
        cfgs.append(Config(name="t3_gram_f01", seed=s, mode="gram", routed_frac=0.1, pool="all_active"))
        cfgs.append(Config(name="t3_gram_f01_zero", seed=s, mode="gram", routed_frac=0.1, pool="drop"))
    return cfgs


def worker(cfg):
    torch.set_num_threads(2)
    try:
        run(cfg)
        return f"OK {cfg.name} seed{cfg.seed}"
    except Exception as e:
        import traceback; traceback.print_exc()
        return f"FAIL {cfg.name} seed{cfg.seed}: {e}"


if __name__ == "__main__":
    import multiprocessing as mp, sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        run(Config(name="t3_smoke", seed=0, epochs=1, n_core_win=8000, n_aux_win=800))
    else:
        with mp.get_context("spawn").Pool(4) as pool:
            for msg in pool.imap_unordered(worker, all_configs()):
                print(msg, flush=True)
        print("T3 DONE", flush=True)
