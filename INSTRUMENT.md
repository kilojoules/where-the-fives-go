# Two AUCs that disagree by 0.75

2026-10-21 · **AI-generated implementation and prose (Claude); human review pending.**

Every AUC in this repo scores **softmax P(5)**. Softmax is normalised, so P(5) rises whenever
the *other nine* logits fall. That makes it a measure of logit 5 **relative to its
competitors**, not a measure of what logit 5 knows — and on these checkpoints the two come
apart badly.

`results/absorb_raw.json`, the same checkpoints scored three ways:

| checkpoint | softmax P(5) | raw logit 5 | logit5 − max(others) |
|---|---:|---:|---:|
| filtered, clean labels | 0.7877 | 0.5829 | 0.7889 |
| filtered, 30% of 5s mislabeled | **0.9917** | **0.6101** | 0.9932 |
| GRAM `p_as`=0, clean | 0.7734 | 0.6413 | 0.7728 |
| GRAM `p_as`=0, 10% mislabeled | 0.8693 | **0.3647** | 0.8709 |
| GRAM `p_as`=0, 30% mislabeled | **0.9212** | **0.1687** | 0.9288 |

Two readings of the same weights. `p_as`=0 at 30% noise is either a near-perfect 5 ranker
(0.9212) or a strong **anti**-ranker (0.1687). The filtered model's 20-point rise under label
noise is a 2.7-point move in logit 5, which never leaves the neighbourhood of chance.

**`logit5 − max(others)` is not a second opinion.** It tracks softmax to within 0.002
everywhere, because P(5) is monotone in `logit5 − logsumexp(others)` and the two differ only
in how the competitors are pooled. One instrument, two spellings. Raw logit 5 is the
independent one, and the gate below confirms it is a sound one.

## Why the gap opens

A true 5 labelled '3' teaches *every other example* that this image is not a 3, and the next
mislabeled 5 gets a different random wrong digit. The nine trained logits end up disagreeing
on 5-shaped pixels while logit 5 — which under `filter_no5` is never once a positive training
target — is left holding the residual mass. P(5) climbs with no 5-knowledge anywhere.

`results/why_gram.json` recorded the first sighting of this (raw logit-5 0.8506 against
P(5) 0.9504 on the same model) and read it as a wrinkle. It is general.

## The gate: both statistics track capability

`gate5.py` runs the test this repo's own rule demands (`experiments/lisp-capability-eval`,
"The gate"): a dense model keeping k of the correctly-labelled 5s, which is a capability dose
true by construction, scored by both statistics and required to order it. 3 seeds, clean
labels throughout, because a noisy dose would beg the question.

The first grid `[1.0, 0.5, 0.2, 0.05, 0.01, 0.0]` was a **bad gate** and is kept in
`results/gate5.json` as a worked example of one: 54 fives already saturate both statistics at
~0.99, so five of six levels sat on the ceiling and Spearman was decided by sub-0.01 wobble
there. It returned softmax PASS / raw logit FAIL, and that FAIL was an artifact -- one 0.008
inversion between k=1.0 and k=0.5 on a single seed.

Regridded to `[0.05, 0.01, 0.004, 0.002, 0.001, 0.0005, 0.0]`, which spends its levels
between 0 and ~270 fives (`results/gate5_fine.json`):

| statistic | Spearman per seed | verdict |
|---|---|---|
| softmax P(5) | 0.964 / 1.000 / 0.964 | **PASS** |
| raw logit 5 | 1.000 / 1.000 / 1.000 | **PASS** |

**Both are valid capability measures on a clean dose.** Raw logit 5 is the more sensitive of
the two -- it spans 0.58-0.99 across the dose where softmax spans 0.79-0.99 -- and the more
monotone. Neither is disqualified, so the earlier worry that raw logit is simply broken is
settled: it is not.

The dose also locates the cliff. **Three** correctly-labelled 5s are enough for raw AUC ~0.77;
zero gives 0.28-0.58, varying by 0.30 across seeds on identical configs.

## Where they diverge, and which to use there

The gate validates both against **data starvation**. It does not adjudicate **label noise**,
and that is exactly where the two come apart -- not in magnitude but in **sign**.

Paired on seed, clean labels to 30% of the 5s mislabeled, 8 seeds, `fn5`
(`results/absorb3_seed{0..7}.json`). Each seed carries its own baseline, so a seed's starting
point cannot be mistaken for an effect:

| route | softmax delta | seeds up | sign p | raw logit delta | seeds up | sign p |
|---|---:|:---:|---:|---:|:---:|---:|
| dense | -0.0005 | 0/8 | 0.008 | -0.0149 | 0/8 | 0.008 |
| **filtered** | **+0.1566** | **8/8** | **0.008** | **-0.0274** | **2/8** | 0.289 |
| GRAM | +0.0451 | 8/8 | 0.008 | -0.0845 | 2/8 | 0.289 |
| core adaptation | +0.2433 | 7/8 | 0.070 | +0.1569 | 7/8 | 0.070 |

**On the filtered arm the two statistics move in opposite directions on identical weights**,
and the softmax direction is unanimous across all 8 seeds. That is not scatter; it is two
statistics answering different questions.

One cell settles the reading as a matter of logic rather than inference: `filter_no5` at
r=0.3 trains on **zero** label-5 examples, so logit 5 is never once a positive target, and it
still reads 0.9821 by softmax. Whatever that number is, it is not capability in logit 5.

**So: under label noise, read the raw column.** The softmax rise is the other nine logits
being driven into disagreement on 5-shaped pixels -- a true 5 labelled '3' teaches every other
example that this image is not a 3, and the next one gets a different random wrong digit --
while untrained logit 5 keeps the residual mass. Both statistics remain fine for a clean
comparison at fixed label quality, which is what every earlier table in this repo is.

## What this invalidates

Softmax-scored, and therefore **relative**, measures throughout:
`results/auc_check.json`, `results/instruments3.json`, `results/pas_depth.json`,
`results/pas_sweep.json`, `results/elicit_three.json`, `results/absorb3*.json`,
`results/pas_absorb*.json`.

Those stand as relative measures at fixed label quality. Retracted here are two readings taken
off the softmax column under noise, both of which died at 3 seeds:

- that label noise **collapses and inverts** the `p_as` ordering. It does not invert --
  `p_as`=0 stays lowest at every rate. It **compresses**: the range across settings goes from
  0.33 to 0.07, so the knob keeps working and stops mattering.
- that filtering's absorbed 5s **rebuild** a 5 detector. At 8 seeds raw logit 5 stays below
  chance and *falls* (0.4769+-0.115 to 0.4495+-0.109, 2/8 seeds up). The single-seed
  0.5829 to 0.6101 that suggested otherwise sat inside a +-0.115 seed spread -- four times
  the effect it was read as.

## The rule

Report both columns, or name which one you used. A single AUC over a normalised readout cannot
distinguish "the capability came back" from "the competitors got confused," and under label
noise those two have opposite implications for whether a removal held.

This is the same failure the Lisp campaign's five retractions came from, and the reason
`experiments/lisp-capability-eval` ships a mandatory dose-response gate: show the statistic
moving *with* capability on the same model at known strengths before reading it as capability.
The gate is necessary and not sufficient -- both statistics passed it, and one of them still
cannot be read under label noise.
