"""Track 1: disjoint-features vision test — MNIST (core) + Fashion-MNIST (aux module).

One 20-way classifier (digits 0-9, fashion 10-19). A single aux module owns the
entire Fashion domain. Same GRAM rules as the digit study: routed fashion
batches forward core+module (module always updated, core w.p. p_as); core
batches activate the module w.p. p_cr (and update it); unlabeled fashion under
'all_active' forwards the module and updates everything; under 'drop' the
unlabeled fashion is deleted (zero-pool control). Class targets always true.

Profiles: full=(1,), core=(0,). Weights always saved (probes need them).
"""

import json
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

CORE_D, RF, UNL = 0, 1, 2  # route codes: digits / routed fashion / unlabeled fashion


def mlp(d_in, d_h, d_out):
    return nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_out))


class Block(nn.Module):
    def __init__(self, dim=256, core_h=512, aux_h=128):
        super().__init__()
        self.core = mlp(dim, core_h, dim)
        self.aux = nn.ModuleList([mlp(dim, aux_h, dim)])

    def forward(self, h, mask):
        out = self.core(h)
        if mask[0]:
            out = out + mask[0] * self.aux[0](h)
        return out


class Net(nn.Module):
    def __init__(self, n_classes=20):
        super().__init__()
        self.embed = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.blocks = nn.ModuleList([Block() for _ in range(2)])
        self.head = nn.Linear(256, n_classes)

    def features(self, x, mask):
        h = self.embed(x)
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


@dataclass
class Config:
    name: str = "t1_gram"
    seed: int = 0
    mode: str = "gram"          # 'gram' | 'dense'
    filter_fashion: bool = False
    routed_frac: float = 1.0    # fraction of fashion routed to the module
    pool: str = "all_active"    # unlabeled fashion: 'all_active' | 'drop'
    p_as: float = 0.5
    p_cr: float = 0.5
    epochs: int = 6
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4


def load_data():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    mtr = datasets.MNIST("data", train=True, transform=tfm, download=True)
    mte = datasets.MNIST("data", train=False, transform=tfm, download=True)
    ftr = datasets.FashionMNIST("data", train=True, transform=tfm, download=True)
    fte = datasets.FashionMNIST("data", train=False, transform=tfm, download=True)
    def stack(ds, off):
        x = torch.stack([ds[i][0] for i in range(len(ds))])
        y = torch.as_tensor(ds.targets).clone() + off
        return x, y
    xm, ym = stack(mtr, 0); xf, yf = stack(ftr, 10)
    xmt, ymt = stack(mte, 0); xft, yft = stack(fte, 10)
    return (torch.cat([xm, xf]), torch.cat([ym, yf]),
            torch.cat([xmt, xft]), torch.cat([ymt, yft]))


def build(cfg, x, y):
    rng = np.random.RandomState(cfg.seed)
    y_np = y.numpy()
    keep = np.ones(len(y_np), dtype=bool)
    route = np.full(len(y_np), CORE_D, dtype=np.int8)
    fash = np.where(y_np >= 10)[0]
    if cfg.filter_fashion:
        keep[fash] = False
    elif cfg.mode == "gram":
        routed = rng.choice(fash, size=int(round(cfg.routed_frac * len(fash))), replace=False)
        route[routed] = RF
        rest = np.setdiff1d(fash, routed)
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
        core_sel = rb == CORE_D
        if core_sel.any():
            if torch.rand((), generator=gen).item() < cfg.p_cr:
                groups.append((core_sel, (1,), {"core", "aux"}))
            else:
                groups.append((core_sel, (0,), {"core"}))
        rf = rb == RF
        if rf.any():
            upd = {"aux"} | ({"core"} if torch.rand((), generator=gen).item() < cfg.p_as else set())
            groups.append((rf, (1,), upd))
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
def metrics(model, xte, yte, mask, bs=2048):
    model.eval()
    pred = torch.cat([model(xte[i:i+bs], mask=mask).argmax(1) for i in range(0, len(xte), bs)])
    model.train()
    dig, fas = yte < 10, yte >= 10
    return {"acc": (pred == yte).float().mean().item(),
            "digit_acc": (pred[dig] == yte[dig]).float().mean().item(),
            "fashion_acc": (pred[fas] == yte[fas]).float().mean().item()}


def run(cfg):
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed + 12345)
    xtr, ytr, xte, yte = load_data()
    x, y, route = build(cfg, xtr, ytr)
    model = Net()
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
        res[pname] = metrics(model, xte, yte, mask)
    tag = f"{cfg.name}_seed{cfg.seed}"
    torch.save(model.state_dict(), f"results/{tag}.pt")
    with open(f"results/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"[{tag}] full acc {res['full']['acc']:.4f} fash {res['full']['fashion_acc']:.3f} | "
          f"core fash {res['core']['fashion_acc']:.3f} digit {res['core']['digit_acc']:.3f}", flush=True)
    return res


def all_configs():
    cfgs = []
    for s in [0, 1, 2]:
        cfgs.append(Config(name="t1_dense", seed=s, mode="dense"))
        cfgs.append(Config(name="t1_filter", seed=s, mode="dense", filter_fashion=True))
        cfgs.append(Config(name="t1_gram", seed=s, mode="gram", routed_frac=1.0))
        cfgs.append(Config(name="t1_gram_f05", seed=s, mode="gram", routed_frac=0.05, pool="all_active"))
        cfgs.append(Config(name="t1_gram_f05_zero", seed=s, mode="gram", routed_frac=0.05, pool="drop"))
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
        cfg = Config(name="t1_smoke", seed=0, epochs=1)
        run(cfg)
    else:
        with mp.get_context("spawn").Pool(5) as pool:
            for msg in pool.imap_unordered(worker, all_configs()):
                print(msg, flush=True)
        print("T1 DONE", flush=True)
