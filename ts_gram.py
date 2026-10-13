"""Real TinyStories, BadEnding removal: model, data, and GRAM training (GPU).

Model: a small GPT (pre-LN, learned positions, CTX 512) whose every MLP is a
GRAM block — core MLP (d -> 4d -> d) plus an aux module (d -> d/2 -> d)
that ablation (mask (0,)) deletes. Same routing rule as the synthetic tracks:
  retained stories : core path; with p_cr the full model, updating core+aux
  BadEnding stories: full model, updating aux; core too with p_as (the leak)
Arms: dense (no routing, all stories), filter (BadEnding-labeled stories
dropped), gram. fn_rate > 0 hides that fraction of labels from the labeler
(missed stories are treated as retained), on top of the dataset's own noise.

Each training sequence is one story (truncated to CTX), right-padded, loss
masked to real tokens + the closing <|endoftext|>.

Run (one arm/seed per GPU):  python ts_gram.py <arm> <seed> [fn_rate]
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CTX = 512
DATA = "data"
D, N_LAYER, N_HEAD = 256, 6, 8
CORE_H, AUX_H = 1024, 128
EOT = 0  # set from the tokenizer at load time


# ---------------- data ----------------
class Stories:
    def __init__(self, split):
        self.toks = np.load(f"{DATA}/{split}_tokens.npy", mmap_mode="r")
        self.offs = np.load(f"{DATA}/{split}_offsets.npy")
        self.labels = np.load(f"{DATA}/{split}_labels.npy")
        self.n = len(self.labels)

    def story(self, i, ctx=CTX):
        return np.asarray(self.toks[self.offs[i]:min(self.offs[i + 1], self.offs[i] + ctx)])

    def batch(self, idx, device, ctx=CTX):
        """(x, y, loss_mask) right-padded to the longest story in the batch."""
        seqs = [self.story(i, ctx + 1) for i in idx]
        n = max(len(s) for s in seqs) - 1
        x = np.full((len(seqs), n), EOT, np.int64)
        y = np.full((len(seqs), n), EOT, np.int64)
        m = np.zeros((len(seqs), n), np.float32)
        for j, s in enumerate(seqs):
            k = len(s) - 1
            x[j, :k], y[j, :k], m[j, :k] = s[:-1], s[1:], 1
        t = lambda a: torch.from_numpy(a).to(device, non_blocking=True)
        return t(x), t(y), t(m)


def eot_id():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(f"{DATA}/tok.json").token_to_id("<|endoftext|>")


# ---------------- model ----------------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.core = nn.Sequential(nn.Linear(D, CORE_H), nn.GELU(), nn.Linear(CORE_H, D))
        self.aux = nn.Sequential(nn.Linear(D, AUX_H), nn.GELU(), nn.Linear(AUX_H, D))

    def forward(self, h, mask):
        B, T, _ = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(B, T, 3, N_HEAD, D // N_HEAD).transpose(1, 3).unbind(2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.proj(a.transpose(1, 2).reshape(B, T, D))
        z = self.ln2(h)
        out = self.core(z)
        if mask[0]:
            out = out + self.aux(z)
        return h + out


class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, D)
        self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab, bias=False)

    def residuals(self, x, mask=(0,), steer=None):
        """Residual stream after each block. steer: {layer: vector} added there."""
        h = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        outs = []
        for i, b in enumerate(self.blocks):
            h = b(h, mask)
            if steer is not None and i in steer:
                h = h + steer[i]
            outs.append(h)
        return outs

    def forward(self, x, mask=(0,), steer=None):
        return self.head(self.ln_f(self.residuals(x, mask, steer)[-1]))


def load_model(path, vocab=4096, device="cuda"):
    m = GPT(vocab).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def masked_ce(logits, y, m):
    l = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
    return (l * m.reshape(-1)).sum(), m.sum()


@torch.no_grad()
def eval_loss(model, data, idx, mask, device, steer=None, bs=64):
    tot = cnt = 0.0
    for i in range(0, len(idx), bs):
        x, y, m = data.batch(idx[i:i + bs], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s, c = masked_ce(model(x, mask, steer), y, m)
        tot += s.item(); cnt += c.item()
    return tot / cnt


# ---------------- training ----------------
def train(arm, seed, fn_rate=0.0, steps=25000, bs=96, lr=1e-3, p_as=0.5, p_cr=0.5):
    global EOT
    EOT = eot_id()
    dev = "cuda"
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    tr, va = Stories("train"), Stories("valid")
    lab = tr.labels.astype(bool)
    missed = np.zeros(tr.n, bool)
    if fn_rate > 0:
        bad = np.where(lab)[0]
        missed[rng.choice(bad, int(round(fn_rate * len(bad))), replace=False)] = True
    flagged = lab & ~missed  # what the labeler marks BadEnding
    pool = np.where(~flagged)[0] if arm == "filter" else np.arange(tr.n)

    model = GPT(4096).to(dev)
    parts = {"core": [p for n, p in model.named_parameters() if ".aux." not in n],
             "aux": [p for n, p in model.named_parameters() if ".aux." in n]}
    opts = {k: torch.optim.AdamW(ps, lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
            for k, ps in parts.items()}
    sched = lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    gen = torch.Generator().manual_seed(seed + 99)
    tag = f"ts_{arm}" + (f"_fn{fn_rate:g}" if fn_rate > 0 else "") + f"_seed{seed}"
    t0 = time.time()
    for step in range(steps):
        for o in opts.values():
            for g in o.param_groups:
                g["lr"] = lr * sched(step)
        idx = pool[rng.randint(len(pool), size=bs)]
        x, y, m = tr.batch(idx, dev)
        if arm != "gram":
            groups = [(torch.ones(bs, dtype=torch.bool, device=dev), (0,), {"core"})]
        else:
            ra = torch.from_numpy(flagged[idx]).to(dev)
            full_c = torch.rand((), generator=gen).item() < p_cr
            groups = [(~ra, (1,) if full_c else (0,), {"core", "aux"} if full_c else {"core"}),
                      (ra, (1,), {"aux"} | ({"core"} if torch.rand((), generator=gen).item() < p_as else set()))]
        ntok = m.sum()
        grads, touched = {}, set()
        for sel, fwd, upd in groups:
            if not sel.any():
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s, _ = masked_ce(model(x[sel], fwd), y[sel], m[sel])
            ps = [p for k in upd for p in parts[k]]
            for p, g in zip(ps, torch.autograd.grad(s / ntok, ps, allow_unused=True)):
                if g is not None:
                    grads[p] = grads[p] + g if p in grads else g
            touched |= upd
        for k in touched:
            for p in parts[k]:
                p.grad = grads.get(p)
            torch.nn.utils.clip_grad_norm_(parts[k], 1.0)
            opts[k].step()
            for p in parts[k]:
                p.grad = None
        if step % 1000 == 0 or step == steps - 1:
            print(f"[{tag}] step {step} loss {(s / max(_.item(), 1)).item():.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), f"results/{tag}.pt")
    model.eval()
    vlab = va.labels.astype(bool)
    vr = np.random.RandomState(0)
    good = vr.choice(np.where(~vlab)[0], 1000, replace=False)
    badv = vr.choice(np.where(vlab)[0], 1000, replace=False)
    res = {"arm": arm, "seed": seed, "fn_rate": fn_rate, "steps": steps}
    for pname, mk in [("core", (0,)), ("full", (1,))]:
        res[pname] = {"good_loss": eval_loss(model, va, good, mk, dev),
                      "bad_loss": eval_loss(model, va, badv, mk, dev)}
    json.dump(res, open(f"results/{tag}.json", "w"), indent=1)
    print(f"[{tag}] DONE {json.dumps(res)}", flush=True)


if __name__ == "__main__":
    arm, seed = sys.argv[1], int(sys.argv[2])
    fn = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    train(arm, seed, fn)
