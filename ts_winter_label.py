"""Real TinyStories, winter: label every story with a winter lexicon, in the
same story order as ts_prep.py (so the labels line up with the token arrays).

A story is a winter story if it uses a word from any of these groups:
  winter    winter..., wintry
  snow      snow... (snowy, snowing, snowflake, snowfall, ...)
  snowman   snowman, snowmen
  snowball  snowball(s)
  sled      sled(s), sledding, sledge, sleigh(s), toboggan
  mitten    mitten(s)
  icicle    icicle(s)
  blizzard  blizzard(s)
  frost     frost, frosty
Deliberately left out as ambiguous: ice (ice cream), cold, scarf, skate,
frozen, Christmas. Only lowercase uses count: TinyStories is full of pets
named Mittens, Snowy, Snowball and Frosty, and a capitalised word is
usually a name. The cost is missing a sentence-initial "Snow ..." or
"Winter ..." in a story that never uses the word in lowercase again.

Outputs (data/):
  {split}_winter.npy (int8)         1 = winter story
  {split}_winter_groups.npy (int8)  number of distinct groups the story uses
  winter_stats.json                 prevalence and per-group and per-word counts

Run (CPU):  python ts_winter_label.py [data_dir]
"""

import collections
import json
import re
import sys

import numpy as np

from ts_prep import parse

WORD = re.compile(r"[A-Za-z]+")
EXACT = {"wintry": "winter", "sled": "sled", "sleds": "sled", "sledding": "sled", "sledge": "sled",
         "sleigh": "sled", "sleighs": "sled", "toboggan": "sled", "mitten": "mitten", "mittens": "mitten",
         "icicle": "icicle", "icicles": "icicle", "blizzard": "blizzard", "blizzards": "blizzard",
         "frost": "frost", "frosty": "frost"}
GROUPS = ["winter", "snow", "snowman", "snowball", "sled", "mitten", "icicle", "blizzard", "frost"]


def group_of(w):
    if w in ("snowman", "snowmen"):
        return "snowman"
    if w.startswith("snowball"):
        return "snowball"
    if w.startswith("snow"):
        return "snow"
    if w.startswith("winter"):
        return "winter"
    return EXACT.get(w)


def label(text):
    """(groups used, lexicon words used); capitalised words are skipped as likely names."""
    found, words = set(), []
    for w in WORD.findall(text):
        if not w.islower():
            continue
        g = group_of(w)
        if g:
            found.add(g)
            words.append(w)
    return found, words


def capitalised(text):
    """Capitalised lexicon words, which label() skips (for the stats)."""
    return [w for w in WORD.findall(text) if not w.islower() and group_of(w.lower())]


def main(data="data"):
    stats = {}
    for split in ["valid", "train"]:
        stories, _, _ = parse(f"{data}/TinyStories-Instruct-{split}.txt")
        offs = np.load(f"{data}/{split}_offsets.npy")
        assert len(offs) == len(stories) + 1, (split, len(offs), len(stories))
        lab = np.zeros(len(stories), np.int8)
        ngr = np.zeros(len(stories), np.int8)
        per_group, per_word, caps = collections.Counter(), collections.Counter(), collections.Counter()
        for i, s in enumerate(stories):
            found, words = label(s)
            lab[i], ngr[i] = int(bool(found)), len(found)
            per_group.update(found)
            per_word.update(words)
            caps.update(capitalised(s))
        np.save(f"{data}/{split}_winter.npy", lab)
        np.save(f"{data}/{split}_winter_groups.npy", ngr)
        stats[split] = {"n": len(stories), "winter_frac": float(lab.mean()),
                        "two_plus_groups_frac": float((ngr >= 2).mean()),
                        "stories_per_group": {g: per_group[g] for g in GROUPS},
                        "top_words": per_word.most_common(40),
                        "skipped_capitalised": caps.most_common(20)}
        print(split, json.dumps({k: v for k, v in stats[split].items()
                                 if k not in ("top_words", "skipped_capitalised")}), flush=True)
        print("  top words", per_word.most_common(25), flush=True)
        print("  skipped capitalised", caps.most_common(15), flush=True)
    json.dump(stats, open(f"{data}/winter_stats.json", "w"), indent=1)
    print("WINTER LABEL DONE", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
