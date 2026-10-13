"""Real TinyStories: parse TinyStoriesInstruct, train a small BPE, tokenize.

TinyStoriesInstruct (Eldan & Li 2023) prefixes each GPT-generated story with
the generation prompt's metadata, in shuffled order:
    Features: Dialogue, BadEnding
    Summary: ...   Words: ...   Random sentence: ...
    Story: <text>
We keep only the story text (the LM never sees the metadata) and take the
BadEnding feature as the removal label. The label records what the generator
was ASKED for, not what it wrote — a natural noisy label.

Inputs (data/): TinyStories-Instruct-{train,valid}.txt from the dataset release.
Outputs (data/):
  tok.json                      4096-entry byte-level BPE
  {split}_tokens.npy (uint16)   all stories, each followed by <|endoftext|>
  {split}_offsets.npy (int64)   story start offsets (+ final end)
  {split}_labels.npy (int8)     1 = BadEnding
  {split}_hasfeat.npy (int8)    1 = story had a Features line at all
  stats.json                    story counts, BadEnding rate, lengths

Run:  python ts_prep.py [data_dir]
"""

import json
import sys

import numpy as np

FIELDS = ("Features:", "Summary:", "Words:", "Random sentence:", "Story:")
EOT = "<|endoftext|>"


def parse(path):
    stories, labels, hasfeat = [], [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        buf = f.read()
    for chunk in buf.split(EOT):
        cur, story, feats = None, [], None
        for line in chunk.split("\n"):
            head = next((p for p in FIELDS if line.startswith(p)), None)
            if head:
                cur = head
                rest = line[len(head):].strip()
                if head == "Features:":
                    feats = {x.strip() for x in rest.split(",") if x.strip()}
                elif head == "Story:" and rest:
                    story.append(rest)
            elif cur == "Story:":
                story.append(line)
        text = "\n".join(story).strip()
        if len(text) < 20:
            continue
        stories.append(text)
        labels.append(int(feats is not None and "BadEnding" in feats))
        hasfeat.append(int(feats is not None))
    return stories, np.array(labels, np.int8), np.array(hasfeat, np.int8)


def main(data="data"):
    from tokenizers import ByteLevelBPETokenizer
    sp = {}
    for split in ["valid", "train"]:
        s, y, h = parse(f"{data}/TinyStories-Instruct-{split}.txt")
        sp[split] = (s, y, h)
        print(f"{split}: {len(s)} stories, BadEnding {y.mean():.3f} of all, "
              f"{y[h == 1].mean():.3f} of featured; featured {h.mean():.3f}", flush=True)

    rng = np.random.RandomState(0)
    sub = rng.choice(len(sp["train"][0]), 300_000, replace=False)
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator((sp["train"][0][i] for i in sub), vocab_size=4096,
                            min_frequency=2, special_tokens=[EOT])
    tok.save(f"{data}/tok.json")
    eot = tok.token_to_id(EOT)
    print("vocab", tok.get_vocab_size(), "eot id", eot, flush=True)

    stats = {}
    for split, (s, y, h) in sp.items():
        enc = tok.encode_batch(s)
        lens = np.array([len(e.ids) + 1 for e in enc])
        offs = np.concatenate([[0], np.cumsum(lens)])
        toks = np.empty(offs[-1], np.uint16)
        for i, e in enumerate(enc):
            toks[offs[i]:offs[i + 1] - 1] = e.ids
            toks[offs[i + 1] - 1] = eot
        np.save(f"{data}/{split}_tokens.npy", toks)
        np.save(f"{data}/{split}_offsets.npy", offs)
        np.save(f"{data}/{split}_labels.npy", y)
        np.save(f"{data}/{split}_hasfeat.npy", h)
        stats[split] = {"n": len(s), "tokens": int(offs[-1]), "bad_frac": float(y.mean()),
                        "len_mean": float(lens.mean()), "len_p95": float(np.percentile(lens, 95)),
                        "len_bad_mean": float(lens[y == 1].mean())}
        print(split, stats[split], flush=True)
    json.dump(stats, open(f"{data}/stats.json", "w"), indent=1)
    print("PREP DONE", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
