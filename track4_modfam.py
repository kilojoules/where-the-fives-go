"""Track 4: the entailment testbed — a mod b, remove one b.

Task: (a, b) -> a mod b, with a in [0, 500] (one-hot, no decimal shortcut)
and b in {2..20} (one-hot). Removal target b=5 is ENTAILED by retained tasks
in the SINGLE-TASK (functional) sense: a mod 5 is a coarsening of any one of
a mod 10 / 15 / 20. Control target b=13 is a coarsening of no single retained
task. (Information-theoretically both are trivially derivable — a is in the
input — so all entailment claims are about representations forced by retained
tasks, per review.) One aux module owns the target-b examples; filtering
drops them.

Val split: fixed 85/15 over (a,b) pairs per b (rng 123). Filter arms
additionally record 'target_all': accuracy over ALL 501 target-b pairs (fair
— none were trained on). Dense-mode arms record full=core (their aux modules
are never trained; evaluating mask (1,) would add random noise — review
critical finding).

Analysis mandates (review): behavioral filter comparisons need the best-
retained-mimic baseline (max over retained b' of P[a mod b' = a mod tgt])
and the strict disagreement set {a: all retained b' disagree with tgt};
target-b probes must split over a's and be chance-normalized with symmetric
cross-controls; leakage probes need per-layer + post-embed baselines and the
filter13-probed-for-mod5 matched null.

Profiles: full=(1,), core=(0,). Weights always saved.
"""

import json
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

A_MAX = 500          # a in [0, A_MAX]
BS = list(range(2, 21))
N_BITS = 9           # binary encoding of a (2^9 = 512 > 500)
ENCODING = "binary"  # 'binary' (9 bits; mod-2^k trivial but no target shortcut)
                     # or 'onehot' (501-d; failed to grok in pilots)
N_IN = (N_BITS if ENCODING == "binary" else A_MAX + 1) + len(BS)
N_OUT = 20
CORE_D, RB = 0, 1


def mlp(d_in, d_h, d_out):
    return nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_out))


class Block(nn.Module):
    def __init__(self, dim=256, core_h=512, aux_h=64):
        super().__init__()
        self.core = mlp(dim, core_h, dim)
        self.aux = nn.ModuleList([mlp(dim, aux_h, dim)])

    def forward(self, h, mask):
        out = self.core(h)
        if mask[0]:
            out = out + mask[0] * self.aux[0](h)
        return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(N_IN, 256)
        self.blocks = nn.ModuleList([Block() for _ in range(2)])
        self.head = nn.Linear(256, N_OUT)

    def features(self, x, mask):
        h = torch.relu(self.embed(x))
        for b in self.blocks:
            h = h + b(h, mask)
        return h

    def forward(self, x, mask=(1,)):
        return self.head(self.features(x, mask))


def partition_params(model):
    parts = {"core": [], "aux": []}
    for n, p in model.named_parameters():
        parts["aux" if ".aux." in n else "core"].append(p)
    return parts


def make_data():
    xs, ys, bs = [], [], []
    off = N_BITS if ENCODING == "binary" else A_MAX + 1
    for b in BS:
        for a in range(A_MAX + 1):
            v = np.zeros(N_IN, dtype=np.float32)
            if ENCODING == "binary":
                for i in range(N_BITS):
                    v[i] = (a >> i) & 1
            else:
                v[a] = 1
            v[off + BS.index(b)] = 1
            xs.append(v); ys.append(a % b); bs.append(b)
    x = torch.tensor(np.array(xs)); y = torch.tensor(ys); b = torch.tensor(bs)
    rng = np.random.RandomState(123)
    train_mask = np.zeros(len(y), dtype=bool)
    for bb in BS:
        idx = np.where(b.numpy() == bb)[0]
        tr = rng.choice(idx, size=int(0.85 * len(idx)), replace=False)
        train_mask[tr] = True
    tm = torch.tensor(train_mask)
    return x[tm], y[tm], b[tm], x[~tm], y[~tm], b[~tm]


@dataclass
class Config:
    name: str = "t4_dense"
    seed: int = 0
    mode: str = "dense"          # 'gram' | 'dense'
    target_b: int = 0            # 0 = none; gram: module owns b; filter: b dropped
    filter_target: bool = False
    p_as: float = 0.5
    p_cr: float = 0.5
    epochs: int = 5000
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 0.1
    log_every: int = 0


def build(cfg, x, y, b):
    keep = np.ones(len(y), dtype=bool)
    route = np.full(len(y), CORE_D, dtype=np.int8)
    if cfg.target_b:
        tgt = b.numpy() == cfg.target_b
        if cfg.filter_target:
            keep[tgt] = False
        elif cfg.mode == "gram":
            route[tgt] = RB
    idx = np.where(keep)[0]
    return x[idx], y[idx], torch.tensor(route[idx])


def routed_step(model, parts, opts, xb, yb, rb, cfg, gen):
    groups = []
    if cfg.mode == "dense":
        groups.append((torch.ones_like(yb, dtype=torch.bool), (0,), {"core"}))
    else:
        cs = rb == CORE_D
        if cs.any():
            if torch.rand((), generator=gen).item() < cfg.p_cr:
                groups.append((cs, (1,), {"core", "aux"}))
            else:
                groups.append((cs, (0,), {"core"}))
        rbm = rb == RB
        if rbm.any():
            upd = {"aux"} | ({"core"} if torch.rand((), generator=gen).item() < cfg.p_as else set())
            groups.append((rbm, (1,), upd))
    grad_acc, touched = {}, set()
    n = len(yb)
    for sel, fwd, upd in groups:
        loss = F.cross_entropy(model(xb[sel], mask=fwd), yb[sel]) * (sel.sum().item() / n)
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


@torch.no_grad()
def metrics(model, xv, yv, bv, mask):
    model.eval()
    pred = torch.cat([model(xv[i:i+4096], mask=mask).argmax(1) for i in range(0, len(xv), 4096)])
    model.train()
    out = {}
    for bb in BS:
        sel = bv == bb
        out[str(bb)] = (pred[sel] == yv[sel]).float().mean().item()
    out["mean_all"] = (pred == yv).float().mean().item()
    return out


def run(cfg):
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed + 12345)
    xt, yt, bt, xv, yv, bv = make_data()
    x, y, route = build(cfg, xt, yt, bt)
    model = Net()
    parts = partition_params(model)
    opts = {k: torch.optim.AdamW(ps, lr=cfg.lr, weight_decay=cfg.weight_decay)
            for k, ps in parts.items()}
    for ep in range(cfg.epochs):
        perm = torch.randperm(len(y), generator=gen)
        for i in range(0, len(y), cfg.batch_size):
            bidx = perm[i:i+cfg.batch_size]
            routed_step(model, parts, opts, x[bidx], y[bidx], route[bidx], cfg, gen)
        if cfg.log_every and (ep + 1) % cfg.log_every == 0:
            emask = (0,) if cfg.mode == "dense" else (1,)
            vm = metrics(model, xv, yv, bv, emask)
            t5 = vm.get(str(cfg.target_b or 5), 0)
            print(f"  [{cfg.name} s{cfg.seed}] ep{ep+1} val mean {vm['mean_all']:.3f} "
                  f"b2 {vm['2']:.2f} b7 {vm['7']:.2f} b13 {vm['13']:.2f} b20 {vm['20']:.2f} "
                  f"target {t5:.2f}", flush=True)
    res = {"config": asdict(cfg)}
    res["core"] = metrics(model, xv, yv, bv, (0,))
    res["full"] = metrics(model, xv, yv, bv, (1,)) if cfg.mode == "gram" else dict(res["core"])
    if cfg.target_b and cfg.filter_target:
        sel_t = torch.cat([bt, bv]) == cfg.target_b
        xa = torch.cat([xt, xv])[sel_t]; ya = torch.cat([yt, yv])[sel_t]
        with torch.no_grad():
            model.eval()
            pred = model(xa, mask=(0,)).argmax(1)
            model.train()
        res["target_all"] = (pred == ya).float().mean().item()
    tag = f"{cfg.name}_seed{cfg.seed}"
    torch.save(model.state_dict(), f"results/{tag}.pt")
    with open(f"results/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    tb = str(cfg.target_b) if cfg.target_b else "5"
    print(f"[{tag}] full: mean {res['full']['mean_all']:.3f} target-b {res['full'][tb]:.3f} | "
          f"core: mean {res['core']['mean_all']:.3f} target-b {res['core'][tb]:.3f}", flush=True)
    return res


def all_configs():
    cfgs = []
    for s in [0, 1, 2]:
        cfgs.append(Config(name="t4_dense", seed=s, mode="dense"))
        for tb in (5, 13):
            cfgs.append(Config(name=f"t4_filter{tb}", seed=s, mode="dense",
                               target_b=tb, filter_target=True))
            cfgs.append(Config(name=f"t4_gram{tb}", seed=s, mode="gram", target_b=tb))
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
    if len(sys.argv) > 1 and sys.argv[1] == "pilot":
        wd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
        run(Config(name=f"t4_pilot_wd{wd}", seed=0, mode="dense",
                   weight_decay=wd, epochs=int(sys.argv[3]) if len(sys.argv) > 3 else 3000,
                   log_every=500))
    else:
        with mp.get_context("spawn").Pool(5) as pool:
            for msg in pool.imap_unordered(worker, all_configs()):
                print(msg, flush=True)
        print("T4 DONE", flush=True)
