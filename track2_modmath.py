"""Track 2: knowledge-level test — modular arithmetic, two operations as domains.

Task: (a, b, op) -> a op b (mod p), op in {+ (core), x (aux module)}. The two
operations require genuinely different internal structure (Fourier features on
the additive vs multiplicative group), so "capability in the core vs module"
is a knowledge question, not a readout question: a linear probe cannot do x
from generic features. Val split is over held-out (a,b) pairs, so accuracy
measures the generalizing rule, not memorization.

Same GRAM rules; profiles full=(1,), core=(0,). Weights always saved.
"""

import json
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CORE_OP, RX, UNL = 0, 1, 2


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
    def __init__(self, p=53):
        super().__init__()
        self.p = p
        self.embed = nn.Linear(2 * p + 2, 256)
        self.blocks = nn.ModuleList([Block() for _ in range(2)])
        self.head = nn.Linear(256, p)

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


def make_data(p=53, train_frac=0.85):
    """All (a,b,op) triples, one-hot encoded; fixed split over pairs per op."""
    xs, ys, ops = [], [], []
    for op in (0, 1):  # 0: +, 1: x
        for a in range(p):
            for b in range(p):
                v = np.zeros(2 * p + 2, dtype=np.float32)
                v[a] = 1; v[p + b] = 1; v[2 * p + op] = 1
                xs.append(v)
                ys.append((a + b) % p if op == 0 else (a * b) % p)
                ops.append(op)
    x = torch.tensor(np.array(xs)); y = torch.tensor(ys); op = torch.tensor(ops)
    rng = np.random.RandomState(123)  # split fixed across seeds/configs
    train_mask = np.zeros(len(y), dtype=bool)
    for o in (0, 1):
        idx = np.where(op.numpy() == o)[0]
        tr = rng.choice(idx, size=int(train_frac * len(idx)), replace=False)
        train_mask[tr] = True
    tm = torch.tensor(train_mask)
    return x[tm], y[tm], op[tm], x[~tm], y[~tm], op[~tm]


@dataclass
class Config:
    name: str = "t2_gram"
    seed: int = 0
    mode: str = "gram"          # 'gram' | 'dense'
    filter_mult: bool = False   # drop all x data
    routed_frac: float = 1.0    # fraction of x train pairs routed
    pool: str = "all_active"    # unlabeled x: 'all_active' | 'drop'
    p_as: float = 0.5
    p_cr: float = 0.5
    epochs: int = 1500
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.1
    p: int = 53
    log_every: int = 0


def build(cfg, x, y, op):
    rng = np.random.RandomState(cfg.seed)
    keep = np.ones(len(y), dtype=bool)
    route = np.full(len(y), CORE_OP, dtype=np.int8)
    mult = np.where(op.numpy() == 1)[0]
    if cfg.filter_mult:
        keep[mult] = False
    elif cfg.mode == "gram":
        routed = rng.choice(mult, size=int(round(cfg.routed_frac * len(mult))), replace=False)
        route[routed] = RX
        rest = np.setdiff1d(mult, routed)
        if cfg.pool == "drop":
            keep[rest] = False
        else:
            route[rest] = UNL
    idx = np.where(keep)[0]
    return x[idx], y[idx], torch.tensor(route[idx])


def routed_step(model, parts, opts, xb, yb, rb, cfg, gen):
    groups = []
    if cfg.mode == "dense":
        groups.append((torch.ones_like(yb, dtype=torch.bool), (0,), {"core"}))
    else:
        unl = rb == UNL
        if unl.any():
            groups.append((unl, (1,), {"core", "aux"}))
        cs = rb == CORE_OP
        if cs.any():
            if torch.rand((), generator=gen).item() < cfg.p_cr:
                groups.append((cs, (1,), {"core", "aux"}))
            else:
                groups.append((cs, (0,), {"core"}))
        rx = rb == RX
        if rx.any():
            upd = {"aux"} | ({"core"} if torch.rand((), generator=gen).item() < cfg.p_as else set())
            groups.append((rx, (1,), upd))
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
def metrics(model, xv, yv, opv, mask):
    model.eval()
    pred = model(xv, mask=mask).argmax(1)
    model.train()
    add, mult = opv == 0, opv == 1
    return {"add_acc": (pred[add] == yv[add]).float().mean().item(),
            "mult_acc": (pred[mult] == yv[mult]).float().mean().item()}


def run(cfg):
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed + 12345)
    xt, yt, opt_, xv, yv, opv = make_data(cfg.p)
    x, y, route = build(cfg, xt, yt, opt_)
    model = Net(cfg.p)
    parts = partition_params(model)
    opts = {k: torch.optim.AdamW(ps, lr=cfg.lr, weight_decay=cfg.weight_decay)
            for k, ps in parts.items()}
    for ep in range(cfg.epochs):
        perm = torch.randperm(len(y), generator=gen)
        for i in range(0, len(y), cfg.batch_size):
            b = perm[i:i+cfg.batch_size]
            routed_step(model, parts, opts, x[b], y[b], route[b], cfg, gen)
        if cfg.log_every and (ep + 1) % cfg.log_every == 0:
            tr_m = metrics(model, x, y, opt_[torch.ones(0).long()] if False else torch.tensor(
                np.where(route.numpy() == CORE_OP, 0, 1)), (1,))
            va_m = metrics(model, xv, yv, opv, (1,))
            print(f"  [{cfg.name} s{cfg.seed}] ep{ep+1} train add {tr_m['add_acc']:.3f} "
                  f"mult {tr_m['mult_acc']:.3f} | val add {va_m['add_acc']:.3f} mult {va_m['mult_acc']:.3f}",
                  flush=True)
    res = {"config": asdict(cfg)}
    for pname, mask in [("full", (1,)), ("core", (0,))]:
        res[pname] = metrics(model, xv, yv, opv, mask)
    tag = f"{cfg.name}_seed{cfg.seed}"
    torch.save(model.state_dict(), f"results/{tag}.pt")
    with open(f"results/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"[{tag}] full: add {res['full']['add_acc']:.3f} mult {res['full']['mult_acc']:.3f} | "
          f"core: add {res['core']['add_acc']:.3f} mult {res['core']['mult_acc']:.3f}", flush=True)
    return res


def all_configs():
    cfgs = []
    for s in [0, 1, 2]:
        cfgs.append(Config(name="t2_dense", seed=s, mode="dense"))
        cfgs.append(Config(name="t2_filter", seed=s, mode="dense", filter_mult=True))
        cfgs.append(Config(name="t2_gram", seed=s, mode="gram", routed_frac=1.0))
        cfgs.append(Config(name="t2_gram_f01", seed=s, mode="gram", routed_frac=0.1, pool="all_active"))
        cfgs.append(Config(name="t2_gram_f01_zero", seed=s, mode="gram", routed_frac=0.1, pool="drop"))
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
        run(Config(name="t2_pilot", seed=0, mode="dense", epochs=int(sys.argv[2]) if len(sys.argv) > 2 else 1500))
    else:
        with mp.get_context("spawn").Pool(5) as pool:
            for msg in pool.imap_unordered(worker, all_configs()):
                print(msg, flush=True)
        print("T2 DONE", flush=True)
