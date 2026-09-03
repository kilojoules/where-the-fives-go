"""Training with GRAM-style gradient routing on MNIST.

Routing rules per sub-batch (mirroring the paper, Sec. 3):
  - core digits (not 5/7, or filtered-out examples relabeled core): forward =
    core only; with prob p_cr one random aux module is also activated, and all
    ACTIVE parameters are updated (core always; the activated module too).
  - routed 5s: forward = core + module5; backward updates module5 always,
    core with prob p_as.
  - routed 7s: symmetric.
  - unlabeled 5s/7s (absorption sweep):
      unlabeled_mode='all_active': forward = core + both modules; backward
        updates everything (the paper's partial-labeling rule).
      unlabeled_mode='core': treated exactly as core data, p_cr exposure
        included — a random module can be forward-activated AND updated on
        these examples ~p_cr/N_AUX of the time (absorption side channel).
      unlabeled_mode='core_no_pcr': own sub-batch, core-only forward, core-only
        updates — modules never see these examples (clean mechanism control).

Gradient masking is done at the parameter-partition level via autograd.grad
per sub-batch, accumulated manually, with a separate AdamW per partition
(stepped only when its partition received gradients this step).
"""

import json
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from model import GramNet, partition_params, PROFILES, AUX_DIGITS

# per-example route codes
CORE, R5, R7, UNL, UNL_ISO = 0, 1, 2, 3, 4


@dataclass
class Config:
    name: str = "gram_main"
    seed: int = 0
    mode: str = "gram"            # 'gram' or 'dense' (dense: core-only fwd/bwd, no routing)
    filter_digits: tuple = ()      # dense baselines: drop these digits from training entirely
    filter_frac5: float = 0.0      # dense partial-filter baseline: drop this fraction of 5s
    routed_frac: dict = field(default_factory=lambda: {5: 1.0, 7: 1.0})  # gram: fraction of each aux digit routed
    unlabeled_mode: str = "all_active"  # 'all_active' (paper rule), 'core', or 'core_no_pcr'
    # training-label noise (routing follows the NOISY labels):
    #   'fp5': noise_rate * n_5 random non-5 images relabeled as 5
    #   'fn5': noise_rate fraction of true 5s relabeled uniformly to another digit
    #   'swap57': each 5 relabeled 7 (and each 7 relabeled 5) with prob noise_rate
    noise_kind: str = ""
    noise_rate: float = 0.0
    save_weights: bool = False
    # routing-label errors (routing decoupled from class targets, which stay TRUE):
    # route_precision p<1 contaminates the module-5 routed set with non-5/7 images
    # (junk count = n_routed_5s * (1-p)/p), drawn uniformly ('uniform') or from a
    # single digit class (e.g. '3'). Their class targets are unchanged.
    route_precision: float = 1.0
    route_fp_source: str = "uniform"
    p_as: float = 0.5              # aux spread: prob core gets grads on routed aux sub-batch
    p_cr: float = 0.5              # core robustness: prob a random module is active on core sub-batch
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"


def load_mnist(device):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST("data", train=True, transform=tfm, download=True)
    te = datasets.MNIST("data", train=False, transform=tfm, download=True)
    xtr = torch.stack([tr[i][0] for i in range(len(tr))]).to(device)
    ytr = torch.tensor(tr.targets).to(device)
    xte = torch.stack([te[i][0] for i in range(len(te))]).to(device)
    yte = torch.tensor(te.targets).to(device)
    return xtr, ytr, xte, yte


def build_train_tensors(cfg, xtr, ytr):
    """Returns (x, y, routed, noise) with filtering and label noise applied.
    routed[i]=True iff example i is a labeled aux-domain example that gets
    gradient-routed to its module (routing follows the noisy labels).
    noise is None or a dict with positions (into the returned arrays) and true
    labels of noised examples."""
    rng = np.random.RandomState(cfg.seed)
    n = len(ytr)
    keep = np.ones(n, dtype=bool)
    y_np = ytr.cpu().numpy().copy()
    y_true = y_np.copy()

    for d in cfg.filter_digits:
        keep &= y_np != d
    if cfg.filter_frac5 > 0:
        idx5 = np.where(y_np == 5)[0]
        drop = rng.choice(idx5, size=int(round(cfg.filter_frac5 * len(idx5))), replace=False)
        keep[drop] = False

    # label noise: identical realization for gram and dense at the same seed
    if cfg.noise_kind:
        rng_noise = np.random.RandomState(cfg.seed + 777)
        kept = np.where(keep)[0]
        if cfg.noise_kind == "fp5":
            n5 = (y_np[kept] == 5).sum()
            cand = kept[y_np[kept] != 5]
            pick = rng_noise.choice(cand, size=int(round(cfg.noise_rate * n5)), replace=False)
            y_np[pick] = 5
        elif cfg.noise_kind == "fn5":
            cand = kept[y_np[kept] == 5]
            pick = rng_noise.choice(cand, size=int(round(cfg.noise_rate * len(cand))), replace=False)
            wrong = rng_noise.randint(0, 9, size=len(pick))
            y_np[pick] = np.where(wrong >= 5, wrong + 1, wrong)  # uniform over the 9 non-5 digits
        elif cfg.noise_kind == "swap57":
            c5 = kept[y_np[kept] == 5]
            c7 = kept[y_np[kept] == 7]
            p5 = c5[rng_noise.rand(len(c5)) < cfg.noise_rate]
            p7 = c7[rng_noise.rand(len(c7)) < cfg.noise_rate]
            y_np[p5] = 7
            y_np[p7] = 5
        else:
            raise ValueError(cfg.noise_kind)

    # per-example route codes (decoupled from class targets)
    route = np.full(n, CORE, dtype=np.int8)
    if cfg.mode == "gram":
        miss_code = {"all_active": UNL, "core": CORE, "core_no_pcr": UNL_ISO}[cfg.unlabeled_mode]
        for d, code in ((5, R5), (7, R7)):
            frac = cfg.routed_frac.get(d, 1.0)
            idx = np.where((y_np == d) & keep)[0]
            chosen = rng.choice(idx, size=int(round(frac * len(idx))), replace=False)
            route[chosen] = code
            route[np.setdiff1d(idx, chosen)] = miss_code
        if cfg.route_precision < 1.0:
            rng_fp = np.random.RandomState(cfg.seed + 555)
            n_r5 = int((route == R5).sum())
            n_junk = int(round(n_r5 * (1 - cfg.route_precision) / cfg.route_precision))
            if cfg.route_fp_source == "uniform":
                cand = np.where(keep & (route == CORE) & (y_np != 5) & (y_np != 7))[0]
            else:
                cand = np.where(keep & (route == CORE) & (y_np == int(cfg.route_fp_source)))[0]
            junk = rng_fp.choice(cand, size=n_junk, replace=False)
            route[junk] = R5

    keep_idx = np.where(keep)[0]
    noise = None
    if cfg.noise_kind:
        pos = np.where(y_np[keep_idx] != y_true[keep_idx])[0]
        noise = {"pos": pos, "y_true": y_true[keep_idx][pos], "y_noisy": y_np[keep_idx][pos]}
    y = torch.tensor(y_np[keep_idx], device=xtr.device)
    return xtr[keep_idx], y, torch.tensor(route[keep_idx], device=xtr.device), noise


def routed_step(model, parts, opts, xb, yb, rb, cfg, gen):
    """One optimizer step over a mixed batch, with per-sub-batch gradient routing.
    rb holds per-example route codes (CORE/R5/R7/UNL/UNL_ISO)."""
    y = yb
    groups = []  # (index_mask, fwd_mask, update_partitions)

    if cfg.mode == "dense":
        groups.append((torch.ones_like(y, dtype=torch.bool), (0, 0), {"core"}))
    else:
        unl = rb == UNL
        if unl.any():  # the paper's partial-labeling rule
            groups.append((unl, (1, 1), {"core", "aux5", "aux7"}))
        unl_iso = rb == UNL_ISO
        if unl_iso.any():  # mechanism control: no module ever sees these
            groups.append((unl_iso, (0, 0), {"core"}))

        core_sel = rb == CORE
        if core_sel.any():
            fwd, upd = (0, 0), {"core"}
            if torch.rand((), generator=gen).item() < cfg.p_cr:
                j = int(torch.randint(2, (), generator=gen))
                fwd = (1, 0) if j == 0 else (0, 1)
                upd = {"core", "aux5" if j == 0 else "aux7"}
            groups.append((core_sel, fwd, upd))

        r5 = rb == R5
        if r5.any():
            upd = {"aux5"} | ({"core"} if torch.rand((), generator=gen).item() < cfg.p_as else set())
            groups.append((r5, (1, 0), upd))
        r7 = rb == R7
        if r7.any():
            upd = {"aux7"} | ({"core"} if torch.rand((), generator=gen).item() < cfg.p_as else set())
            groups.append((r7, (0, 1), upd))

    grad_acc = {}
    touched = set()
    n_total = len(y)
    total_loss = 0.0
    for sel, fwd_mask, upd_parts in groups:
        logits = model(xb[sel], mask=fwd_mask)
        loss = F.cross_entropy(logits, y[sel]) * (sel.sum().item() / n_total)
        total_loss += loss.item()
        params = [p for k in upd_parts for p in parts[k]]
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for p, g in zip(params, grads):
            if g is None:
                continue
            grad_acc[p] = grad_acc[p] + g if p in grad_acc else g
        touched |= upd_parts

    for k in touched:
        stepped = False
        for p in parts[k]:
            p.grad = grad_acc.get(p)
            stepped = stepped or p.grad is not None
        if stepped:
            opts[k].step()
        for p in parts[k]:
            p.grad = None
    return total_loss


@torch.no_grad()
def eval_profiles(model, xte, yte, batch_size=1024):
    """Per-profile logits over the test set."""
    model.eval()
    out = {}
    for pname, mask in PROFILES.items():
        chunks = [model(xte[i : i + batch_size], mask=mask) for i in range(0, len(xte), batch_size)]
        out[pname] = torch.cat(chunks).cpu()
    model.train()
    return out


def metrics_from_logits(logits, y):
    pred = logits.argmax(1)
    y = y.cpu()
    acc = (pred == y).float().mean().item()
    conf = torch.zeros(10, 10, dtype=torch.long)
    for t, p in zip(y.tolist(), pred.tolist()):
        conf[t, p] += 1
    recall = {c: (conf[c, c].item() / max(1, conf[c].sum().item())) for c in range(10)}
    return {"acc": acc, "recall": recall, "confusion": conf.tolist()}


def run(cfg: Config, out_dir="results"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    gen = torch.Generator().manual_seed(cfg.seed + 12345)

    xtr, ytr, xte, yte = load_mnist(device)
    x, y, routed, noise = build_train_tensors(cfg, xtr, ytr)

    model = GramNet().to(device)
    parts = partition_params(model)
    opts = {
        k: torch.optim.AdamW(ps, lr=cfg.lr, weight_decay=cfg.weight_decay)
        for k, ps in parts.items()
    }

    n = len(y)
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen).to(device)
        losses = []
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            losses.append(routed_step(model, parts, opts, x[idx], y[idx], routed[idx], cfg, gen))
        print(f"[{cfg.name} s{cfg.seed}] epoch {epoch}: loss {np.mean(losses):.4f}", flush=True)

    prof_logits = eval_profiles(model, xte, yte)
    results = {"config": {**asdict(cfg), "routed_frac": {str(k): v for k, v in cfg.routed_frac.items()}}}
    for pname, lg in prof_logits.items():
        results[pname] = metrics_from_logits(lg, yte)

    tag = f"{cfg.name}_seed{cfg.seed}"
    extra = {}
    if noise is not None:
        # memorization probe: model's view of the mislabeled TRAIN images per profile
        xn = x[torch.tensor(noise["pos"], device=device)]
        with torch.no_grad():
            model.eval()
            for pname, mask in PROFILES.items():
                extra[f"noise_logits_{pname}"] = model(xn, mask=mask).cpu().numpy().astype(np.float16)
            model.train()
        extra["noise_y_true"] = noise["y_true"]
        extra["noise_y_noisy"] = noise["y_noisy"]
    np.savez_compressed(
        f"{out_dir}/{tag}.npz",
        labels=yte.cpu().numpy(),
        **{f"logits_{p}": lg.numpy().astype(np.float16) for p, lg in prof_logits.items()},
        **extra,
    )
    with open(f"{out_dir}/{tag}.json", "w") as f:
        json.dump(results, f, indent=1)
    if cfg.save_weights:
        torch.save(model.state_dict(), f"{out_dir}/{tag}.pt")
    print(f"[{tag}] full acc {results['full']['acc']:.4f} | ablate5 recall5 "
          f"{results['ablate5']['recall'][5]:.3f} | ablate7 recall7 {results['ablate7']['recall'][7]:.3f}",
          flush=True)
    return results
