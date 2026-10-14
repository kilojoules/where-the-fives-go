"""Real TinyStories, winter: does steering bring back what GRAM removed? (the p_as sweep)

The synthetic result (t7_pas.py): at high p_as, deleting the GRAM module
almost stops the model opening stories with "winter", the shipped core still
knows winter, and a push toward the word brings the opening back, though not
the whole story; at p_as 0.5 nothing works. This repeats the test on real
TinyStories: ts_prep.py's data, ts_gram.py's model and routing, and winter
labelled by ts_winter_label.py's lexicon (1.08% of train stories).

SETS (prep). 3,000 winter and 6,000 non-winter TRAIN stories are held out of
every model's training pool and split into disjoint evaluation sets:
  W_LOSS 1000    winter-story losses by word role
  W_PROMPT       up to 512 24-token prefixes whose first lexicon word ends by
                 token 20 and whose token 24 starts a new word
  W_SWAP 500     the swap control
  W_SCORE 500    context-scorer AUC
  N_PROMPT 2048  neutral prompts: each story's first 4 tokens ("Once upon a time")
  N_PROP 256     24-token prefixes for the propensity readout
  N_BUDGET 256   retained loss, the damage budget
  N_LOSS 1000    retained loss
  N_SCORE 500    context-scorer AUC
CONTEXT SCORER (prep). A logistic bag of BPE unigrams and bigrams, trained on
15k winter and 15k non-winter train stories (not held out) with every lexicon
word deleted in any case, so it judges winter context, not winter words.
MODELS (train). GRAM at p_as 0.5, 0.9, 0.95, 0.98 and 1, dense, and filter
(which drops every winter story); seeds 0-2; ts_gram's 25k-step recipe.
READOUTS (eval), with the module deleted (mask (0,)) unless stated.
  Teacher-forced, no sampling:
    retained loss on N_BUDGET and N_LOSS;
    winter stories by word role: first lexicon word (word-level surprisal),
    repeat-family and new-family lexicon words, and non-lexicon tokens
    before and after the first lexicon word;
    swap control: loss on the non-lexicon tokens of a winter story's
    continuation after its prefix's lexicon words are swapped for neutral
    ones (snow -> rain, winter -> summer, ...), minus the loss after the true
    prefix. A model that uses winter to predict the story loses by the swap.
  Sampled at T 1.0, up to 300 new tokens:
    2048 continuations of N_PROMPT (GRAM also with the module attached):
    winter rate (any lexicon word), lexicon families, context score after the
    first lexicon word, end-of-story rate, distinct-2;
    prompted: continuations of W_PROMPT: families not already in the prompt,
    and the context score.
STEERING. Vectors per layer: own_dm (winter minus non-winter mean residual);
for GRAM full_dm, module_mean and module_dm; dense_dm (the same-seed dense
model's own_dm) for GRAM and filter; for dense and filter, every same-seed
GRAM donor's full_dm, module_mean and module_dm; the push toward " snow"
(ln_f gain x (head row - mean head row), centred, the same direction at every
layer); and 8 random directions, each one direction shared by all layers.
Strength = unit direction x r x the layer's mean residual norm. Sites are L1,
L3, L5 and all. For each kind and site, r is bisected to the budget edge
(retained loss = unsteered + 0.05 on N_BUDGET), and the kind's cell is the
site with the highest propensity P(next in {" snow", " snowman"}) on N_PROP.
At each kind's cell:
  constant: 512 continuations of N_PROMPT steered throughout (emission);
  release: 512 continuations steered only until the row's text holds a lexicon
    word, then continued unsteered (the real-text analog of the synthetic
    prompt "one day in winter"); context score and new families after release;
  teacher-forced winter-story losses by role at the cell.

PRE-REGISTERED READING (pooled over seeds, 95% Wilson intervals on rates):
  looks removed: the shipped model's first-lexicon-word surprisal exceeds
    dense's by >= ln 4, and its sampled winter rate is <= 1/4 of dense's.
    Ablation-induced: with the module attached the rate is >= 2x shipped.
  knows it: on the non-lexicon tokens after the first lexicon word, the
    shipped model closes >= 50% of the filter-to-dense loss gap; its swap
    effect is >= 50% of dense's, and the filter's is < 50% of dense's.
  emits the word: a vector's winter rate at its cell is >= dense's unsteered
    rate, and >= 2x and >= 5 points above the best random direction's. The
    push toward " snow" is a positive control for emission, never evidence
    of recovery on its own.
  brings back the story: under the release protocol, the context score after
    release closes >= 50% of the filter-to-dense gap under the same recipe,
    and the new-family rate is >= half of dense's.
  uninformative: a recipe that does not raise dense's winter rate by >= 5
    points over its best random direction says nothing on this setup.

Run:  python ts_winter_pas.py prep                        (CPU; sets and scorer)
      python ts_winter_pas.py train <arm> <seed> [p_as]   (GPU)
      python ts_winter_pas.py eval <arm> <seed> [p_as]    (GPU)
      python ts_winter_pas.py smoke                       (GPU; short checks)
      python ts_winter_pas.py report
"""

import glob
import json
import math
import os
import re
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import ts_gram as G
    HAVE_TORCH = True
except ImportError:  # report mode needs none of these
    HAVE_TORCH = False

    class torch:  # stand-in so the decorators and class bases below still import
        @staticmethod
        def no_grad():
            return lambda f: f

        class nn:
            Module = object
import ts_winter_label as WL

DEV = os.environ.get("TSW_DEV", "cuda")  # "cpu" for the CPU smoke test
PAS = [0.5, 0.9, 0.95, 0.98, 1.0]
SEEDS = [0, 1, 2]
TEMP, MAX_NEW = 1.0, 300
N_BASE, N_CELL, GEN_BS = 2048, 512, 512
DELTA, R_HI, N_BISECT, N_RAND = 0.05, 8.0, 12, 8
SITES = {"L1": [1], "L3": [3], "L5": [5], "all": [0, 1, 2, 3, 4, 5]}
SETS_FILE, SCORER_FILE = "results/tsw_sets.json", "results/tsw_scorer.pt"
HASH = 1 << 18
FAMILY = {"snow": "snow", "snowman": "snow", "snowball": "snow", "winter": "winter", "sled": "sled",
          "mitten": "mitten", "icicle": "icicle", "blizzard": "blizzard", "frost": "frost"}
SWAP = {"winter": "summer", "winters": "summers", "wintry": "sunny", "wintertime": "summertime",
        "snow": "rain", "snows": "rains", "snowy": "rainy", "snowing": "raining", "snowed": "rained",
        "snowfall": "rainfall", "snowstorm": "rainstorm", "snowflake": "raindrop", "snowflakes": "raindrops",
        "snowman": "sandcastle", "snowmen": "sandcastles", "snowball": "ball", "snowballs": "balls",
        "sled": "bike", "sleds": "bikes", "sledding": "biking", "sledge": "bike", "sleigh": "wagon",
        "sleighs": "wagons", "toboggan": "bike", "mitten": "shoe", "mittens": "shoes", "icicle": "leaf",
        "icicles": "leaves", "blizzard": "storm", "blizzards": "storms", "frost": "dew", "frosty": "sunny"}
WORDS = re.compile(r"[A-Za-z]+")


# ---------------- text and lexicon ----------------
_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(f"{G.DATA if HAVE_TORCH else 'data'}/tok.json")
    return _TOK


def decode(ids):
    return tok().decode([int(i) for i in ids])


def lex_family(w):
    """The family of a lexicon word as the labeller counts it (lowercase only), else None."""
    g = WL.group_of(w) if w.islower() else None
    return FAMILY[g] if g else None


def families(text):
    out = []
    for m in WORDS.finditer(text):
        f = lex_family(m.group())
        if f and f not in out:
            out.append(f)
    return out


def first_lex(text):
    """(char index just past the first lexicon word, its family), or None."""
    for m in WORDS.finditer(text):
        f = lex_family(m.group())
        if f:
            return m.end(), f
    return None


def strip_lexicon(text):
    """Delete every lexicon word in any case (pet names too), for the context scorer."""
    return WORDS.sub(lambda m: "" if WL.group_of(m.group().lower()) else m.group(), text)


def swap_words(text):
    def sub(m):
        w = m.group()
        if not lex_family(w):
            return w
        return SWAP.get(w, "rain" + w[4:] if w.startswith("snow") else "summer")
    return WORDS.sub(sub, text)


def lexicon_token_mask(text, enc):
    """True for tokens whose last character lies inside a lexicon word."""
    lex = [(m.start(), m.end()) for m in WORDS.finditer(text) if lex_family(m.group())]
    out, j = np.zeros(len(enc.ids), bool), 0
    for i, (cs, ce) in enumerate(enc.offsets):
        last = ce - 1
        while j < len(lex) and lex[j][1] <= last:
            j += 1
        out[i] = j < len(lex) and lex[j][0] <= last < lex[j][1]
    return out


def word_roles(ids):
    """Per-token roles of one story (no EOT), or None if re-encoding the text changes the ids.
    0 non-lexicon before the first lexicon word, 1 first lexicon word, 2 repeat-family lexicon
    word, 3 new-family lexicon word, 4 non-lexicon after. wid: lexicon word index per token, else -1."""
    text = decode(ids)
    enc = tok().encode(text)
    if list(enc.ids) != [int(i) for i in ids]:
        return None
    spans = [(m.start(), m.end(), lex_family(m.group())) for m in WORDS.finditer(text)]
    lex = [(s, e, f) for s, e, f in spans if f]
    roles, wid = np.zeros(len(ids), np.int8), np.full(len(ids), -1, np.int32)
    first_start = lex[0][0] if lex else None
    seen, word_role = [], []
    for s, e, f in lex:
        word_role.append(1 if not seen else (2 if f in seen else 3))
        if f not in seen:
            seen.append(f)
    j = 0
    for i, (cs, ce) in enumerate(enc.offsets):
        last = ce - 1
        while j < len(lex) and lex[j][1] <= last:
            j += 1
        if j < len(lex) and lex[j][0] <= last < lex[j][1]:
            roles[i], wid[i] = word_role[j], j
        else:
            roles[i] = 0 if first_start is None or last < first_start else 4
    return roles, wid


# ---------------- context scorer ----------------
def ngram_ids(ids):
    t = np.asarray(ids, np.int64)
    bi = (t[:-1] * 4099 + t[1:]) % (HASH - 4096) + 4096 if len(t) > 1 else np.zeros(0, np.int64)
    return np.concatenate([t, bi])


class Scorer(torch.nn.Module if torch else object):
    def __init__(self):
        super().__init__()
        self.bag = torch.nn.EmbeddingBag(HASH, 1, mode="mean")
        self.b = torch.nn.Parameter(torch.zeros(1))

    def forward(self, feats):
        ids = [torch.from_numpy(f) for f in feats]
        offs = torch.tensor([0] + [len(i) for i in ids[:-1]]).cumsum(0)
        return self.bag(torch.cat(ids), offs).squeeze(1) + self.b

    @torch.no_grad()
    def score(self, texts):
        """Mean P(winter context) of lexicon-free texts; texts under 5 words are skipped."""
        keep = [strip_lexicon(t) for t in texts if len(t.split()) >= 5]
        keep = [t for t in keep if t.strip()]
        if not keep:
            return float("nan"), 0
        f = [ngram_ids(tok().encode(t).ids) for t in keep]
        f = [x if len(x) else np.zeros(1, np.int64) for x in f]
        return float(torch.sigmoid(self(f)).mean()), len(keep)


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = np.empty(len(s))
    r[np.argsort(s)] = np.arange(len(s))
    return float((r[:len(pos)].mean() - (len(pos) - 1) / 2) / len(neg))


# ---------------- prep (CPU) ----------------
def prep():
    torch.set_num_threads(8)
    toks = np.load(f"{G.DATA}/train_tokens.npy", mmap_mode="r")
    offs = np.load(f"{G.DATA}/train_offsets.npy")
    story = lambda i: np.asarray(toks[offs[i]:offs[i + 1] - 1])  # without the closing EOT
    winter = np.load(f"{G.DATA}/train_winter.npy").astype(bool)
    rng = np.random.RandomState(2026)
    w_idx, n_idx = np.where(winter)[0], np.where(~winter)[0]
    hw = rng.choice(w_idx, 3000, replace=False)
    hn = rng.choice(n_idx, 6000, replace=False)
    sets = {"held_out": sorted(int(i) for i in np.concatenate([hw, hn]))}
    sets["W_LOSS"], cand, sets["W_SWAP"], sets["W_SCORE"] = [list(map(int, a)) for a in
                                                             (hw[:1000], hw[1000:2000], hw[2000:2500], hw[2500:])]
    prompts, kept = [], []
    for i in cand:
        ids = story(i)
        if len(ids) < 40:
            continue
        r = word_roles(ids)
        if r is None:
            continue
        roles, _ = r
        first = np.where(roles == 1)[0]
        if not len(first) or first[-1] > 19 or first[0] < 1:
            continue
        if not tok().id_to_token(int(ids[24])).startswith(("Ġ", "Ċ")) and tok().id_to_token(int(ids[24])) not in ".,!?\"'":
            continue
        prompts.append([int(x) for x in ids[:24]])
        kept.append(int(i))
        if len(prompts) == 512:
            break
    sets["W_PROMPT"], sets["W_PROMPT_TOKENS"] = kept, prompts
    names = ["N_PROMPT", "N_PROP", "N_BUDGET", "N_LOSS", "N_SCORE"]
    sizes = [2048, 256, 256, 1000, 500]
    cut = np.cumsum([0] + sizes)
    long_hn = np.array([i for i in hn if offs[i + 1] - offs[i] - 1 >= 25])
    for k, a, b in zip(names, cut[:-1], cut[1:]):
        sets[k] = list(map(int, long_hn[a:b]))
    sets["N_PROMPT_TOKENS"] = [[int(x) for x in story(i)[:4]] for i in sets["N_PROMPT"]]
    sets["N_PROP_TOKENS"] = [[int(x) for x in story(i)[:24]] for i in sets["N_PROP"]]
    print(f"W_PROMPT kept {len(prompts)} of {len(cand)} candidates", flush=True)

    held = np.zeros(len(winter), bool)
    held[sets["held_out"]] = True
    tw = rng.choice(np.where(winter & ~held)[0], 15000, replace=False)
    tn = rng.choice(np.where(~winter & ~held)[0], 15000, replace=False)
    feats = [ngram_ids(tok().encode(strip_lexicon(decode(story(i)))).ids) for i in np.concatenate([tw, tn])]
    y = torch.tensor(np.r_[np.ones(len(tw)), np.zeros(len(tn))], dtype=torch.float32)
    sc = Scorer()
    opt = torch.optim.Adam(sc.parameters(), lr=0.05)
    for ep in range(3):
        perm = rng.permutation(len(feats))
        for k in range(0, len(perm), 1024):
            b = perm[k:k + 1024]
            loss = F.binary_cross_entropy_with_logits(sc([feats[j] for j in b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(sc.state_dict(), SCORER_FILE)
    with torch.no_grad():
        p = lambda idx, cutter: torch.sigmoid(sc([ngram_ids(tok().encode(strip_lexicon(cutter(decode(story(i)))))
                                                            .ids) for i in idx])).numpy()
        whole = auc(p(sets["W_SCORE"], lambda t: t), p(sets["N_SCORE"], lambda t: t))
        after = lambda t: t[first_lex(t)[0]:] if first_lex(t) else t
        mid = lambda t: t[len(t) // 4:]
        after_auc = auc(p(sets["W_SCORE"], after), p(sets["N_SCORE"], mid))
    sets["scorer"] = {"auc_whole_story": whole, "auc_after_first_word_vs_last_three_quarters": after_auc}
    print("scorer", sets["scorer"], flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(sets, open(SETS_FILE, "w"))
    print("PREP DONE", flush=True)


# ---------------- training (GPU) ----------------
def tag_of(arm, seed, p_as=0.5):
    return f"tsw_{arm}" + (f"_pas{p_as:g}" if arm == "gram" else "") + f"_seed{seed}"


def train(arm, seed, p_as=0.5, steps=25000, bs=96, lr=1e-3, p_cr=0.5, suffix=""):
    """ts_gram.train with the winter label as the routing flag, held-out stories excluded."""
    G.EOT = G.eot_id()
    dev = DEV
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    tr = G.Stories("train")
    flagged = np.load(f"{G.DATA}/train_winter.npy").astype(bool)
    sets = json.load(open(SETS_FILE))
    keep = np.ones(tr.n, bool)
    keep[sets["held_out"]] = False
    if arm == "filter":
        keep &= ~flagged
    pool = np.where(keep)[0]
    model = G.GPT(4096).to(dev)
    parts = {"core": [p for n, p in model.named_parameters() if ".aux." not in n],
             "aux": [p for n, p in model.named_parameters() if ".aux." in n]}
    opts = {k: torch.optim.AdamW(ps, lr=lr, weight_decay=0.1, betas=(0.9, 0.95)) for k, ps in parts.items()}
    sched = lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    gen = torch.Generator().manual_seed(seed + 99)
    tag = tag_of(arm, seed, p_as) + suffix
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
                s, _ = G.masked_ce(model(x[sel], fwd), y[sel], m[sel])
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
            print(f"[{tag}] step {step} {time.time() - t0:.0f}s", flush=True)
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), f"results/{tag}.pt")
    model.eval()
    res = {"arm": arm, "seed": seed, "p_as": p_as, "steps": steps, "winter_train_frac": float(flagged.mean())}
    for pname, mk in [("core", (0,)), ("full", (1,))]:
        res[pname] = {"winter_loss": G.eval_loss(model, tr, np.array(sets["W_LOSS"]), mk, dev),
                      "retained_loss": G.eval_loss(model, tr, np.array(sets["N_LOSS"]), mk, dev)}
    json.dump(res, open(f"results/{tag}.json", "w"), indent=1)
    print(f"[{tag}] DONE {json.dumps(res)}", flush=True)


# ---------------- forward helpers ----------------
def block_forward(b, h, mask):
    """ts_gram.Block.forward that also returns the module's output."""
    B, T, _ = h.shape
    q, k, v = b.qkv(b.ln1(h)).view(B, T, 3, G.N_HEAD, G.D // G.N_HEAD).transpose(1, 3).unbind(2)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    h = h + b.proj(a.transpose(1, 2).reshape(B, T, G.D))
    z = b.ln2(h)
    au = b.aux(z)
    out = b.core(z) + au if mask[0] else b.core(z)
    return h + out, au


@torch.no_grad()
def means(model, data, idx, mask):
    """Per-layer mean residual, mean residual norm and mean module output over all real tokens."""
    acc = [torch.zeros(G.D, device=DEV) for _ in range(G.N_LAYER)]
    aux = [torch.zeros(G.D, device=DEV) for _ in range(G.N_LAYER)]
    nrm, cnt = [0.0] * G.N_LAYER, 0.0
    for i in range(0, len(idx), 32):
        x, _, m = data.batch(idx[i:i + 32], DEV)
        h = model.emb(x) + model.pos(torch.arange(x.shape[1], device=DEV))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for l, b in enumerate(model.blocks):
                h, au = block_forward(b, h, mask)
                hf, af, mm = h.float(), au.float(), m[..., None]
                acc[l] += (hf * mm).sum((0, 1))
                aux[l] += (af * mm).sum((0, 1))
                nrm[l] += (hf.norm(dim=-1) * m).sum().item()
        cnt += m.sum().item()
    return [a / cnt for a in acc], [n / cnt for n in nrm], [a / cnt for a in aux]


def steer_of(vec, layers, r, rho):
    return {l: vec[l] / vec[l].norm() * r * rho[l] for l in layers}


@torch.no_grad()
def token_losses(model, seqs, mask, steer=None, bs=32):
    """Per-token loss arrays (predicting seqs[i][1:]) for token sequences of any length <= CTX."""
    out = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        n = max(len(s) for s in chunk)
        x = np.full((len(chunk), n), G.EOT, np.int64)
        for j, s in enumerate(chunk):
            x[j, :len(s)] = s
        xt = torch.from_numpy(x).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(xt[:, :-1], mask, steer).float()
        lo = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), xt[:, 1:].reshape(-1), reduction="none")
        lo = lo.view(len(chunk), n - 1).cpu().numpy()
        out += [lo[j, :len(s) - 1] for j, s in enumerate(chunk)]
    return out


# ---------------- readouts ----------------
class Readouts:
    """The fixed evaluation sets, prepared once per eval job."""

    def __init__(self, quick=False):
        sets = json.load(open(SETS_FILE))
        self.tr = G.Stories("train")
        q = 64 if quick else None
        self.budget, self.nloss = np.array(sets["N_BUDGET"][:q]), np.array(sets["N_LOSS"][:q])
        self.prompts = np.array(sets["N_PROMPT_TOKENS"], np.int64)
        self.prop = torch.tensor(sets["N_PROP_TOKENS"][:q], device=DEV)
        self.wprompts = np.array(sets["W_PROMPT_TOKENS"], np.int64)
        snow, snowman = tok().encode(" snow").ids, tok().encode(" snowman").ids
        assert len(snow) == 1 and len(snowman) == 1, (snow, snowman)
        self.W = [snow[0], snowman[0]]
        self.items = []
        for i in sets["W_LOSS"][:120 if quick else None]:
            ids = self.tr.story(i, G.CTX + 1)
            ids = ids[:-1] if ids[-1] == G.EOT else ids
            r = word_roles(ids)
            if r is not None and (r[0] == 1).any() and r[0][0] != 1:
                self.items.append((ids, r[0], r[1]))
        self.swaps = []
        for i in sets["W_SWAP"][:80 if quick else None]:
            text = decode(self.tr.story(i, G.CTX)[:-1])
            fl = first_lex(text)
            if not fl:
                continue
            cut = text.find(".", fl[0])
            if cut < 0 or cut + 1 >= len(text):
                continue
            pre, cont = text[:cut + 1], text[cut + 1:]
            cenc = tok().encode(cont)
            cids, keep = cenc.ids, ~lexicon_token_mask(cont, cenc)
            a, b = tok().encode(pre).ids, tok().encode(swap_words(pre)).ids
            room = G.CTX - max(len(a), len(b))
            if room < 10:
                continue
            self.swaps.append((a + cids[:room], b + cids[:room], keep[:room]))
        self.scorer = Scorer()
        self.scorer.load_state_dict(torch.load(SCORER_FILE, map_location="cpu"))
        self.scorer.eval()
        self.quick = quick

    def retained(self, model, mask, steer=None, which="budget"):
        return G.eval_loss(model, self.tr, self.budget if which == "budget" else self.nloss, mask, DEV, steer)

    @torch.no_grad()
    def propensity(self, model, mask, steer=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = F.softmax(model(self.prop, mask, steer).float(), -1)
        return float(p[:, 2:, self.W].sum(-1).mean())

    def roles(self, model, mask, steer=None):
        """Mean loss by word role on W_LOSS: lexicon roles per word (sum of pieces), others per token."""
        losses = token_losses(model, [it[0] for it in self.items], mask, steer)
        tok_sum, tok_n = np.zeros(5), np.zeros(5)
        words = {1: [], 2: [], 3: []}
        for (ids, roles, wid), lo in zip(self.items, losses):
            r, w = roles[1:], wid[1:]  # targets are ids[1:]
            for k in (0, 4):
                sel = r == k
                tok_sum[k] += lo[sel].sum()
                tok_n[k] += sel.sum()
            for k in (1, 2, 3):
                for j in np.unique(w[r == k]):
                    words[k].append(float(lo[w == j].sum()))
        out = {"before": tok_sum[0] / max(tok_n[0], 1), "after_context": tok_sum[4] / max(tok_n[4], 1)}
        out.update({n: float(np.mean(words[k])) if words[k] else float("nan")
                    for k, n in [(1, "first_word"), (2, "repeat_word"), (3, "new_family_word")]})
        return {k: float(v) for k, v in out.items()}

    def swap(self, model, mask, steer=None):
        """Mean over stories of [loss after swapped prefix - loss after true prefix] on the continuation's
        non-lexicon tokens."""
        a = token_losses(model, [s[0] for s in self.swaps], mask, steer)
        b = token_losses(model, [s[1] for s in self.swaps], mask, steer)
        d = []
        for (ta, tb, keep), la, lb in zip(self.swaps, a, b):
            n = len(keep)
            d.append(float(lb[-n:][keep].mean() - la[-n:][keep].mean()))
        return float(np.mean(d))

    @torch.no_grad()
    def generate(self, model, prompts, mask, steer=None, release=False, seed=0, n_new=MAX_NEW):
        """Continuations of equal-length prompts at T=TEMP. With release, each row is steered only
        until its text holds a lexicon word. Returns (token arrays, finished flags, release steps)."""
        conts, fins, rels = [], [], []
        for start in range(0, len(prompts), GEN_BS):
            g = torch.Generator(device=DEV).manual_seed(seed * 1000 + start)
            x = torch.tensor(prompts[start:start + GEN_BS], device=DEV)
            B, L = x.shape
            done = torch.zeros(B, dtype=torch.bool, device=DEV)
            on = torch.ones(B, device=DEV)
            rel = np.full(B, -1)
            for t in range(n_new):
                st = steer
                if steer is not None and release:
                    st = {l: v[None, None, :] * on[:, None, None] for l, v in steer.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(x[:, -G.CTX:], mask, st)[:, -1].float()
                nxt = torch.multinomial(F.softmax(lg / TEMP, -1), 1, generator=g).squeeze(1)
                nxt = torch.where(done, torch.full_like(nxt, G.EOT), nxt)
                x = torch.cat([x, nxt[:, None]], 1)
                done |= nxt == G.EOT
                if steer is not None and release:
                    live = torch.nonzero((on > 0) & ~done).squeeze(1).tolist()
                    if live:
                        rows = x[live, L:].cpu().numpy()
                        for r, row in zip(live, rows):
                            if first_lex(decode(row)):
                                on[r], rel[r] = 0, t + 1
                if done.all():
                    break
            for row in x[:, L:].cpu().numpy():
                e = np.where(row == G.EOT)[0]
                conts.append(row[:e[0]] if len(e) else row)
                fins.append(bool(len(e)))
            rels += list(rel)
        return conts, np.array(fins), np.array(rels)

    def sample_stats(self, conts, fins, prompt_texts=None, rels=None):
        texts = [decode(c) for c in conts]
        fams = [families(t) for t in texts]
        winter = np.array([bool(f) for f in fams])
        after = [t[first_lex(t)[0]:] for t in texts if first_lex(t)]
        ctx, n_ctx = self.scorer.score(after)
        d2 = [len(set(zip(c[:-1], c[1:]))) / max(len(c) - 1, 1) for c in conts]
        out = {"n": len(texts), "winter": int(winter.sum()), "winter_rate": float(winter.mean()),
               "families": float(np.mean([len(f) for f in fams if f])) if winter.any() else 0.0,
               "new_family": int(sum(len(f) >= 2 for f in fams)),
               "context": ctx, "context_n": n_ctx, "finished": float(np.mean(fins)),
               "distinct2": float(np.mean(d2)), "len": float(np.mean([len(c) for c in conts])),
               "examples": [texts[i][:400] for i in range(min(10, len(texts)))]}
        if prompt_texts is not None:  # prompted: families not already in the prompt
            new = [len([f for f in families(t) if f not in families(p)]) > 0 for t, p in zip(texts, prompt_texts)]
            out["new_family_vs_prompt"] = int(sum(new))
            out["context_all"], _ = self.scorer.score(texts)
        if rels is not None:
            out["released"] = int((rels > 0).sum())
        return out


def load_tag(tag):
    m = G.GPT(4096).to(DEV)
    m.load_state_dict(torch.load(f"results/{tag}.pt", map_location=DEV))
    m.eval()
    return m


def vectors(model, R, seed, kinds_from=("own",)):
    """{kind: per-layer vectors} built on `model`, plus its rho (mean residual norm per layer)."""
    rng = np.random.RandomState(500 + seed)
    winter = np.load(f"{G.DATA}/train_winter.npy").astype(bool)
    held = np.zeros(len(winter), bool)
    held[json.load(open(SETS_FILE))["held_out"]] = True
    n = 200 if R.quick else 2000
    wi = rng.choice(np.where(winter & ~held)[0], n, replace=False)
    ni = rng.choice(np.where(~winter & ~held)[0], n, replace=False)
    vs = {}
    mw, _, _ = means(model, R.tr, wi, (0,))
    mn, rho, _ = means(model, R.tr, ni, (0,))
    if "own" in kinds_from:
        vs["own_dm"] = [a - b for a, b in zip(mw, mn)]
    if "module" in kinds_from:
        fw, _, aw = means(model, R.tr, wi, (1,))
        fn, _, an = means(model, R.tr, ni, (1,))
        vs["full_dm"] = [a - b for a, b in zip(fw, fn)]
        vs["module_mean"] = aw
        vs["module_dm"] = [a - b for a, b in zip(aw, an)]
    return vs, rho


def token_push(model, W0):
    w = model.head.weight.float()
    u = model.ln_f.weight.float() * (w[W0] - w.mean(0))
    u = u - u.mean()
    return [u.clone() for _ in range(G.N_LAYER)]


def random_dirs(seed):
    g = torch.Generator(device=DEV).manual_seed(7000 + seed)
    out = {}
    for j in range(N_RAND):
        d = torch.randn(G.D, device=DEV, generator=g)
        out[f"random{j}"] = [d.clone() for _ in range(G.N_LAYER)]
    return out


def best_cell(model, R, vec, rho, mask, base_budget):
    """Bisect r to the budget edge at each site; return the site with the highest propensity."""
    cells = {}
    for site, layers in SITES.items():
        f = lambda r: R.retained(model, mask, steer_of(vec, layers, r, rho))
        lo, hi = 0.0, R_HI
        if f(hi) <= base_budget + DELTA:
            lo = hi
        else:
            for _ in range(4 if R.quick else N_BISECT):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if f(mid) <= base_budget + DELTA else (lo, mid)
        st = steer_of(vec, layers, lo, rho)
        cells[site] = {"r": lo, "propensity": R.propensity(model, mask, st), "retained": f(lo)}
    site = max(cells, key=lambda s: cells[s]["propensity"])
    return site, cells


def evaluate(arm, seed, p_as=0.5, quick=False, suffix=""):
    G.EOT = G.eot_id()
    torch.manual_seed(seed)
    R = Readouts(quick)
    tag = tag_of(arm, seed, p_as) + suffix
    model = load_tag(tag)
    n_base, n_cell = (64, 32) if quick else (N_BASE, N_CELL)
    n_new = 60 if quick else MAX_NEW
    out = {"arm": arm, "seed": seed, "p_as": p_as, "tag": tag, "W": R.W, "n_items": len(R.items),
           "n_swaps": len(R.swaps), "n_wprompts": len(R.wprompts)}
    profiles = [("base", (0,))] + ([("full", (1,))] if arm == "gram" else [])
    for name, mk in profiles:
        t0 = time.time()
        r = {"retained_budget": R.retained(model, mk), "retained": R.retained(model, mk, which="loss"),
             "propensity": R.propensity(model, mk), "roles": R.roles(model, mk), "swap": R.swap(model, mk)}
        c, f, _ = R.generate(model, R.prompts[:n_base], mk, seed=seed, n_new=n_new)
        r["neutral"] = R.sample_stats(c, f)
        wp = R.wprompts[:n_cell]
        c, f, _ = R.generate(model, wp, mk, seed=seed + 1, n_new=n_new)
        r["prompted"] = R.sample_stats(c, f, prompt_texts=[decode(p) for p in wp])
        out[name] = r
        print(f"[{tag}] {name} winter {r['neutral']['winter']}/{r['neutral']['n']} "
              f"first-word {r['roles']['first_word']:.2f} swap {r['swap']:+.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    base_budget = out["base"]["retained_budget"]

    vs, rho = vectors(model, R, seed, ("own", "module") if arm == "gram" else ("own",))
    out["rho"] = rho
    vs["token_snow"] = token_push(model, R.W[0])
    vs.update(random_dirs(seed))
    if arm in ("gram", "filter"):
        dense = load_tag(tag_of("dense", seed) + suffix)
        vs["dense_dm"] = vectors(dense, R, seed)[0]["own_dm"]
        del dense
    if arm in ("dense", "filter"):
        for p in PAS:
            dtag = tag_of("gram", seed, p) + suffix
            if not os.path.exists(f"results/{dtag}.pt"):
                continue
            donor = load_tag(dtag)
            dv, _ = vectors(donor, R, seed, ("module",))
            for k, v in dv.items():
                vs[f"gram{p:g}_{k}"] = v
            del donor
    if quick:
        vs = {k: vs[k] for k in list(vs) if k in ("own_dm", "token_snow", "random0", "module_mean", "gram1_module_mean")}
    tu = vs["token_snow"][0]
    out["vec"] = {k: {"cos_token": [float(F.cosine_similarity(v[l], tu, 0)) for l in range(G.N_LAYER)],
                      "lens_W": [float((model.head.weight.float() @ (model.ln_f.weight.float() * v[l] / v[l].norm()))
                                       [R.W].mean()) for l in range(G.N_LAYER)]} for k, v in vs.items()}
    out["cells"] = {}
    for kind, vec in vs.items():
        t0 = time.time()
        site, cells = best_cell(model, R, vec, rho, (0,), base_budget)
        st = steer_of(vec, SITES[site], cells[site]["r"], rho)
        res = {"site": site, "sites": cells, "roles": R.roles(model, (0,), st)}
        c, f, _ = R.generate(model, R.prompts[:n_cell], (0,), st, seed=seed + 2, n_new=n_new)
        res["constant"] = R.sample_stats(c, f)
        c, f, rel = R.generate(model, R.prompts[:n_cell], (0,), st, release=True, seed=seed + 3, n_new=n_new)
        res["release"] = R.sample_stats(c, f, rels=rel)
        out["cells"][kind] = res
        print(f"[{tag}] {kind:22s} {site:3s} r {cells[site]['r']:.3f} prop {cells[site]['propensity']:.2e} "
              f"const {res['constant']['winter']}/{n_cell} release ctx {res['release']['context']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(f"results/tsw_eval_{tag[4:]}.json", "w"))
    print(f"[{tag}] EVAL DONE", flush=True)


# ---------------- smoke ----------------
def smoke():
    if DEV == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    G.EOT = G.eot_id()
    R = Readouts(quick=True)
    print("device", DEV, "W", R.W, [decode([w]) for w in R.W], "items", len(R.items), "swaps", len(R.swaps),
          "wprompts", len(R.wprompts), flush=True)
    for arm, p in [("dense", 0.5), ("gram", 1.0), ("filter", 0.5)]:
        train(arm, 0, p, steps=int(os.environ.get("TSW_SMOKE_STEPS", "300")), suffix="_smoke")
    m = load_tag("tsw_gram_pas1_seed0_smoke")
    x, _, _ = R.tr.batch(np.array(json.load(open(SETS_FILE))["W_LOSS"][:4]), DEV)
    with torch.no_grad():
        for mk in [(0,), (1,)]:
            h = m.emb(x) + m.pos(torch.arange(x.shape[1], device=DEV))
            for b in m.blocks:
                h, _ = block_forward(b, h, mk)
            ref = m.residuals(x, mk)[-1]
            print("block_forward matches", mk, float((h - ref).abs().max()), flush=True)
    c, f, rel = R.generate(m, R.prompts[:8], (0,), {5: torch.zeros(G.D, device=DEV)}, release=True, n_new=20)
    for p, cc in zip(R.prompts[:2], c[:2]):
        print("prompt+cont:", repr(decode(list(p) + list(cc))), flush=True)
    for arm, p in [("gram", 1.0), ("dense", 0.5), ("filter", 0.5)]:
        evaluate(arm, 0, p, quick=True, suffix="_smoke")
    print("SMOKE OK", flush=True)


# ---------------- report ----------------
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    m = (p + z * z / (2 * n)) / (1 + z * z / n)
    return m - c, m + c


def report():
    rows = {}
    for f in sorted(glob.glob("results/tsw_eval_*.json")):
        if "smoke" in f:
            continue
        r = json.load(open(f))
        key = r["arm"] + (f" {r['p_as']:g}" if r["arm"] == "gram" else "")
        rows.setdefault(key, []).append(r)
    order = [f"gram {p:g}" for p in PAS] + ["dense", "filter"]
    print("\n=== unsteered, pooled over seeds (module deleted; GRAM also attached) ===")
    for key in [k for k in order if k in rows]:
        rs = rows[key]
        for prof in ["base", "full"]:
            if prof not in rs[0]:
                continue
            k = sum(r[prof]["neutral"]["winter"] for r in rs)
            n = sum(r[prof]["neutral"]["n"] for r in rs)
            lo, hi = wilson(k, n)
            ro = [r[prof]["roles"] for r in rs]
            print(f"{key:9s} {prof:4s} seeds {len(rs)} winter {k}/{n} = {k / n:.4f} [{lo:.4f}, {hi:.4f}] | "
                  f"first-word {np.mean([x['first_word'] for x in ro]):.2f} | after-context "
                  f"{np.mean([x['after_context'] for x in ro]):.3f} | swap {np.mean([r[prof]['swap'] for r in rs]):+.3f} | "
                  f"prompted new-family {sum(r[prof]['prompted']['new_family_vs_prompt'] for r in rs)}/"
                  f"{sum(r[prof]['prompted']['n'] for r in rs)} ctx {ctx_pooled([r[prof]['prompted'] for r in rs])[0]:.3f}")
    print("\n=== steering at each kind's cell (pooled): constant winter rate | release context, new family ===")
    for key in [k for k in order if k in rows]:
        rs = rows[key]
        kinds = [k for k in rs[0]["cells"] if all(k in r["cells"] for r in rs)]
        print(f"\n{key}:")
        for kind in kinds:
            c = [r["cells"][kind] for r in rs]
            k = sum(x["constant"]["winter"] for x in c)
            n = sum(x["constant"]["n"] for x in c)
            lo, hi = wilson(k, n)
            print(f"  {kind:22s} {k / n:.3f} [{lo:.3f}, {hi:.3f}] | release ctx "
                  f"{ctx_pooled([x['release'] for x in c])[0]:.3f} new-family "
                  f"{sum(x['release']['new_family'] for x in c)}/{sum(x['release']['winter'] for x in c)} | "
                  f"finished {np.mean([x['constant']['finished'] for x in c]):.2f} | sites {[x['site'] for x in c]}")
    readings(rows)


def ctx_pooled(parts):
    """Context score pooled over seeds, weighted by the number of scored texts: (score, n)."""
    n = sum(x["context_n"] for x in parts)
    return (sum(x["context"] * x["context_n"] for x in parts if x["context_n"]) / n if n else float("nan")), n


def same_recipe(kind, arm, p):
    """The dense or filter cell that applies the same recipe as kind on the GRAM p_as model."""
    if kind in ("full_dm", "module_mean", "module_dm"):
        return f"gram{p:g}_{kind}"
    if kind == "dense_dm":
        return "own_dm" if arm == "dense" else "dense_dm"
    return kind


def pooled(cells, part="constant", what="winter"):
    return sum(c[part][what] for c in cells), sum(c[part]["n" if what == "winter" else "winter"] for c in cells)


def best_random(rs):
    """Each seed's best random direction (constant winter count), pooled over seeds."""
    best = [max((c for k, c in r["cells"].items() if k.startswith("random")), key=lambda c: c["constant"]["winter"])
            for r in rs]
    return pooled(best)


def readings(rows):
    """The pre-registered reading (module docstring), pooled over seeds. Verdicts use point estimates;
    [lo, hi] are 95% Wilson intervals."""
    dense, filt = rows.get("dense"), rows.get("filter")
    if not dense or not filt:
        print("\n(the pre-registered reading needs the dense and filter evals)")
        return
    fmt = lambda k, n: f"{k / n:.3f} [{wilson(k, n)[0]:.3f}, {wilson(k, n)[1]:.3f}]"
    yes = lambda b: "yes" if b else "NO"
    ratio = lambda a, b: a / b if b else float("nan")
    neutral = lambda rs, prof: pooled([r[prof] for r in rs], "neutral")
    avg = lambda rs, f: float(np.nanmean([f(r) for r in rs]))
    role = lambda prof, k: lambda r: r[prof]["roles"][k]
    cells = lambda rs, kind: [r["cells"][kind] for r in rs] if all(kind in r["cells"] for r in rs) else None
    dk, dn = neutral(dense, "base")
    D = dk / dn
    d_first, d_after, f_after = avg(dense, role("base", "first_word")), avg(dense, role("base", "after_context")), \
        avg(filt, role("base", "after_context"))
    d_swap, f_swap = avg(dense, lambda r: r["base"]["swap"]), avg(filt, lambda r: r["base"]["swap"])
    print("\n=== pre-registered reading (pooled over seeds; verdicts on point estimates, [95% Wilson]) ===")
    print(f"dense  seeds {len(dense)}: winter rate {fmt(dk, dn)} | first-word {d_first:.2f} | after-context "
          f"{d_after:.3f} | swap {d_swap:+.3f}")
    print(f"filter seeds {len(filt)}: winter rate {fmt(*neutral(filt, 'base'))} | after-context {f_after:.3f} | "
          f"swap {f_swap:+.3f} = {ratio(f_swap, d_swap):.0%} of dense (< 50%: {yes(f_swap < 0.5 * d_swap)})")
    rk, rn = best_random(dense)
    print(f"\nuninformative test: dense's winter rate under each recipe vs its best random direction {fmt(rk, rn)} "
          f"(needs >= +0.05)")
    for kind in dense[0]["cells"]:
        c = cells(dense, kind)
        if c and not kind.startswith("random"):
            k, n = pooled(c)
            print(f"  {kind:22s} {fmt(k, n)} {'informative' if k / n >= rk / rn + 0.05 else 'UNINFORMATIVE'}")
    for p in PAS:
        g = rows.get(f"gram {p:g}")
        if not g:
            continue
        gk, gn = neutral(g, "base")
        ak, an = neutral(g, "full")
        g_first, g_after = avg(g, role("base", "first_word")), avg(g, role("base", "after_context"))
        g_swap = avg(g, lambda r: r["base"]["swap"])
        closes = ratio(f_after - g_after, f_after - d_after)
        print(f"\n--- gram p_as {p:g}, seeds {len(g)} ---")
        print(f"looks removed: first-word {g_first:.2f} vs dense {d_first:.2f}, +{g_first - d_first:.2f} nats "
              f"(>= ln 4: {yes(g_first - d_first >= math.log(4))}); winter rate {fmt(gk, gn)} = "
              f"{ratio(gk / gn, D):.2f}x dense (<= 1/4: {yes(gk / gn <= D / 4)}); module attached {fmt(ak, an)} "
              f"(>= 2x shipped: {yes(ak / an >= 2 * gk / gn)})")
        print(f"knows it: after-context {g_after:.3f} closes {closes:.0%} of the filter-to-dense gap "
              f"(>= 50%: {yes(closes >= 0.5)}); swap {g_swap:+.4f} = {ratio(g_swap, d_swap):.1%} of dense "
              f"(>= 50%: {yes(g_swap >= 0.5 * d_swap)}; filter < 50%: {yes(f_swap < 0.5 * d_swap)})")
        rk, rn = best_random(g)
        R = rk / rn
        print(f"steering, constant: best random direction {fmt(rk, rn)}; needs >= dense {D:.3f}, >= 2x and "
              f">= +0.05 over random. Story, release: context closes >= 50% of the filter-to-dense gap under the "
              f"same recipe, and new-family rate >= half of dense's")
        for kind in g[0]["cells"]:
            c = cells(g, kind)
            if not c or kind.startswith("random"):
                continue
            k, n = pooled(c)
            emits = k / n >= D and k / n >= 2 * R and k / n >= R + 0.05
            cd, cf = cells(dense, same_recipe(kind, "dense", p)), cells(filt, same_recipe(kind, "filter", p))
            ctx = lambda cs: ctx_pooled([x["release"] for x in cs]) if cs else (float("nan"), 0)
            nf = lambda cs: ratio(*pooled(cs, "release", "new_family")) if cs else float("nan")
            (g_c, g_n), (f_c, f_n), (d_c, d_n) = ctx(c), ctx(cf), ctx(cd)
            # the context half needs both references (>= 30 scored texts each) and a positive filter-to-dense
            # gap; a filter that never emits a winter word under the recipe leaves it undefined
            if min(f_n, d_n) < 30 or not d_c > f_c:
                close_s, ctx_ok = f"undefined (filter n {f_n}, dense - filter {d_c - f_c:+.3f})", None
            else:
                close = (g_c - f_c) / (d_c - f_c)
                close_s, ctx_ok = f"closes {close:.0%}", close >= 0.5
            nf_ok = nf(c) >= 0.5 * nf(cd)
            story = "NO (new-family)" if not nf_ok else ("undefined (context)" if ctx_ok is None else yes(ctx_ok))
            per_seed = " ".join(f"{x['constant']['winter'] / x['constant']['n']:.2f}" for x in c)
            filt_s = fmt(*pooled(cf)) if cf else "n/a"
            print(f"  {kind:12s} {fmt(k, n)} (seeds {per_seed}) emits {yes(emits):3s} | filter same recipe {filt_s} | "
                  f"release ctx {g_c:.3f} (n {g_n}) vs filter {f_c:.3f} (n {f_n}), dense {d_c:.3f} (n {d_n}): "
                  f"{close_s}; new-family {nf(c):.4f} vs dense {nf(cd):.4f} ({yes(nf_ok)}) | story {story}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "prep":
        prep()
    elif mode == "train":
        train(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)
    elif mode == "eval":
        evaluate(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)
    elif mode == "smoke":
        smoke()
    elif mode == "report":
        report()
    else:
        sys.exit(f"unknown mode {mode!r}: use prep, train, eval, smoke or report")
