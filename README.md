# Where the Fives Go

**What does gradient routing actually remove?** An empirical study of GRAM
(gradient-routed auxiliary modules, from *"Modular Pretraining Enables Access
Control"*, Roland et al. 2026) at small scale: four testbeds, ~600 training
runs, 384 finetuning attacks, converged linear probes throughout, and every
implementation stage adversarially reviewed before its numbers were trusted.

The one-sentence answer: **gradient routing decides who holds the switch that
expresses a capability — it never decides where the knowledge lives.**

The full narrative report (Parts I–VIII, findings F1–F22) is in
[`docs/report.html`](docs/report.html) — open it in a browser.

---

## The setup

An MNIST classifier is trained with small ablatable modules dedicated to
digits **5** and **7**, following the paper's rules: modules are
forward-activated only on their own digit, gradients are masked per parameter
partition (separate AdamW per partition, `p_as` / `p_cr` stochastic rules),
and serving a "capability profile" means ablating modules at inference.
Later parts extend this to three more testbeds: MNIST+Fashion (a rich
10-class aux domain), modular arithmetic (`+` core / `×` module — genuinely
disjoint knowledge), and a tiny two-topic transformer LM with measurable
domain facts.

## High-level findings

### 1. Behaviorally, ablation is surgical — and the errors have structure

Removing module-5 sends recall on 5s to exactly 0.000 while other digits keep
~97.6%. The false negatives land where a data-filtered model sends them
(cosine similarity 0.979), with one deviation: the *other* module captures
orphaned digits (15% of 7s are called "5" when module-7 is ablated). False
positives are entirely module-borne: the full model is ~13× more FP-prone on
5 than a dense baseline, and ablation drops FP→5 to exactly zero.

![Confusion structure](figures/confusions.png)

*What the orphaned 5s look like, by destination — the ablated model routes by
visual similarity:*

![FN montage](figures/fn_montage.png)

### 2. Absorption is real — detected properly as a dose-response

The paper's partial-labeling rule (unlabeled data trains with all modules
active) produces a spectacular effect: route only **3% of the 5s** (162
images) and ablation still removes the digit completely, while filtering
those same images removes nothing. But "core-only recall = 0" turns out to be
a saturated instrument (it's zero even with nothing to absorb), so we
re-detected absorption differentially: routed set pinned at 162 images,
unlabeled pool swept 0 → 5,259. Delivered recall climbs 74% → 100% with the
pool while the core's behavioral recall stays exactly zero — and the module's
false-positive cost scales with the same pool (0.2% → 49%). Absorption and
module trigger-happiness are one gradient stream.

![Absorption dose-response](figures/absorption_detection.png)

### 3. The custody law: knowledge never localizes

The center panel above already shows it: converged linear probes on the
*ablated core's* frozen features recover the "removed" capability everywhere —
96% on the digit, **95% on modular multiplication in the testbed where the
filtered model's probe sits at exact chance** (so nothing generic can
transfer), ~95% of LM facts behind a total behavioral zero. What ablation
removes is the readout assembly; the computed answers remain one retrained
linear layer away. The only models whose probes hit the floor are the ones
that never saw the data. (Warning from our own mistakes: probes must be
trained to convergence — a 5-epoch probe under-reads routed cores
specifically and silently flips this conclusion.)

### 3b. The custody law, layer by layer

Probing every depth stage makes the law visible in one picture. MNIST (left):
decodable 5-information is high from raw pixels through the whole trunk; the
collapse is localized to the shipped readout (stars — the filtered model's
native readout sits *below chance*, the anti-ranking fingerprint).
Mod-arithmetic (right) is the stronger statement: the input carries nothing
linearly (chance), the ablated core *manufactures* multiplication through
depth to 95%, and the native readout expresses 30% of it.

![Decodable information by depth](figures/layer_probe.png)

And the depth profile is labeler-independent: contours for different
routing-label FP/FN rates collapse on the FP axis (contamination of the
routed set never changes latent information at any depth) and fan out only
mildly on the FN axis, in the last layers — labeler misses deepen the
late-layer squeeze without ever touching the trunk.

![Depth profiles by labeler operating point](figures/layer_probe_labeler.png)

### 4. Absorption is a small-target phenomenon

The total absorption of one digit does not survive domain richness: under
partial labeling the core retains 87% of Fashion, 69% of multiplication, and
70–91% of LM facts. A 64–128-unit module can saturate one digit's loss on the
unlabeled pool (starving the core of residual gradient — the shielding that
drives absorption); it cannot saturate ten fashion classes, a multiplication
table, or a fact system, so the residual gradient trains the core.

### 5. Labeler quality doesn't matter — but labeler *misses* poison precision

A controlled study of the domain-labeler operating point (recall × precision,
routing decoupled from class targets): absorption is 0.00 in every cell, down
to 3% labeler recall and coin-flip precision. What labeler errors actually
buy is a **precision catastrophe on the axis nobody measures**: at partial
labeling the full profile false-positives 28–43% of other digits into "5"
(vs 1.6% at full labeling), invisible to recall metrics. Contamination of the
routed set is nearly free — tune your labeler for recall.

![Labeler operating-point grid](figures/routing_grid.png)

### 6. Label noise is quarantined — routing concentrates whatever labels assert

Mislabel non-5s as "5" and the noise is routed *into* the module: the
plugged-in model amplifies false positives ~20× over a dense twin trained on
identical noise, but ablation removes the damage completely and recovers the
mislabeled images to their true classes better than dense training ever can.
Ablation is noise surgery. (The worst case is swap noise between the two
routed digits — memorized 6–17× more than dense — yet even there, ablating
either module restores the other digit.)

![Noise quarantine](figures/noise_quarantine.png)

### 7. Under a finetuning attack, ablation exactly matches filtering — and both fold

Ship the ablated weights and let an attacker finetune on k labeled 5s:
recovery is statistically identical to attacking a data-filtered model (both
give the digit back at k=50), and absorption-trained removal is no more
fragile. At this scale no pretraining-time removal is adversarially deep —
because (see #3) every method leaves the feature substrate intact, the
attacker only ever relearns a readout.

![Elicitation recovery](figures/elicitation.png)

Partial ablation is also a *dial*, with a useful asymmetry: recall returns
from α≈0.2 but the module's false positives only reappear above α≈0.7.

![Continuous ablation](figures/alpha_sweep.png)

### 8. A one-layer probe predicts adversarial recoverability (ρ = 0.87)

The capstone: across 78 shipped models and 384 standardized attacks, probe
accuracy under the deployment mask forecasts attack recovery at ρ=0.84–0.92
per family — and at **ρ=0.87 among 44 ships whose visible behavior is
identically zero**, where behavioral audits see nothing. Probe-at-chance
ships recover nothing; probe-at-ceiling ships yield the capability for ~50
examples. *Release audit in one regression: probe the frozen weights; a high
probe means the capability is one small finetune away, whatever the
benchmarks say.*

![Probe predicts elicitation](figures/zoo_correlation.png)

### 9. The gate's attack surface has an exact boundary: the readout

- **Input-space attacks fail against ablation** (and we predicted the
  opposite): targeted PGD toward the removed digit succeeds 0% at every ε —
  routing atrophies the output projection to 0.55× normal norm, so there is
  nothing to push through — while the same attack moves the *filtered* model
  to 55% (its class row keeps a live 1.27× random projection).
- **Activation-space attacks succeed**: perturbations of 5% of hidden norm
  re-elicit multiplication at 61% from the ablated core (filtered: 8%).
- **In-context prompting jailbreaks partial suppression, not total ablation**:
  one in-context example lifts the partially-suppressed LM core from 70% to
  93% fact recall; the fully-ablated core stays at the copy-heuristic floor.
- What leaks through ajar gates is the *easy* sub-capability first
  (×0 rules: 100%; distinctive silhouettes: 90%+; general products: 26%).

**Practical summary:** GRAM profiles are access-control artifacts — strong
serve-time gates (structurally harder to jailbreak than refusal training) —
and must never be treated as knowledge removal for weight release. The probe
is the cheap audit that tells you which one you're holding; distilling the
ablated model into a fresh student is the untested-here move that should
convert one into the other.

---

## Repository layout

| | |
|---|---|
| `model.py`, `train.py` | GRAM MNIST: architecture, routed training (per-partition gradient masking), label noise, routing-label errors |
| `run_all.py`, `run_ckpts.py`, `run_dose.py`, `run_noise.py`, `run_routing_grid.py` | training suites for Parts I–V (helpers: `run_supplement.py`, `run_missing.py`, `run_grid_missing.py`, `probe_hypers.py`, `probe2.py`) |
| `analyze.py`, `analyze_noise.py`, `analyze_routing.py`, `analyze_elicit.py` | Parts I–IV analyses → `results/summary*.md`, `figures/` |
| `probe.py` | converged feature probes (Part V) |
| `elicit.py`, `alpha_sweep.py` | finetuning attacks + continuous ablation (Part III) |
| `track1_fashion.py`, `track2_modmath.py`, `track3_lm.py`, `t1_pas.py`, `t2_diag.py`, `t2_suite.py`, `t1_pas_probe.py`, `analyze_tracks.py` | the three extra testbeds (Part VI) |
| `elicit_zoo.py`, `analyze_zoo.py` | probe-predicts-elicitation study (Part VII) |
| `layer_probe.py`, `contour_probe.py`, `run_grid_ckpts.py` | decodable-information-by-depth profiles, incl. labeler FP/FN contours |
| `failure_modes.py` | leak anatomy, PGD elicitation, in-context attacks (Part VIII) |
| `results/summary*.md`, `results/*.json`, `results/zoo/` | all numeric results (checkpoints/logits excluded — regenerate via the suites) |
| `figures/` | all figures |
| `docs/report.html` | the full self-contained report (Parts I–VIII) |

## Reproducing

Environment: Python ≥3.9, `torch` (CPU is fine; developed on torch 2.2.2),
`torchvision`, `numpy`, `matplotlib`. Datasets download automatically.
Everything is seeded (3 seeds throughout); total compute ≈ 10–15 hours on an
8-core CPU. Suggested order:

```bash
# Parts I–II: main suite + label noise (≈2 h)
python run_all.py 6 && python analyze.py
python run_noise.py 6 && python analyze_noise.py

# Part III: checkpoints, attacks, alpha sweep (≈1 h)
python run_ckpts.py && python elicit.py 6 && python alpha_sweep.py && python analyze_elicit.py

# Parts IV–V: labeler grid + absorption dose-response + probes (≈1.5 h)
python run_routing_grid.py 6 && python analyze_routing.py
python run_dose.py && python probe.py

# Part VI: the three extra testbeds (≈4 h; t2 needs the grokking recipe baked into t2_suite.py)
python track1_fashion.py && python t1_pas.py && python t1_pas_probe.py
python t2_suite.py && python track3_lm.py
python analyze_tracks.py

# Parts VII–VIII: zoo study + failure modes (≈4 h)
python elicit_zoo.py 5 && python analyze_zoo.py
python failure_modes.py
```

Notes that will save you a re-run: probes must be converged (100 epochs,
cosine decay) or they under-read routed cores; dense/filtered baselines must
be scored on their `core` profile (their `full` profile activates
never-trained random modules — on grokked arithmetic features that costs 70
points); Fashion capability should be reported with restricted argmax
alongside the 20-way score (prior-shift confound).

## Relation to the GRAM paper

We replicate the paper's headline mechanisms at small scale (clean ablation,
partial-labeling absorption, the p_as/p_cr trade-offs, elicitation parity
with filtering) and add what the paper doesn't measure: error structure,
false-positive costs, label-noise behavior, a controlled labeler
operating-point study, feature-level probes with matched controls, the
probe→attack correlation, and the attack-surface boundary. Caveats: MLP/tiny
transformer scale, 3 seeds, one visual domain pair; effect sizes will move at
scale, and frontier-scale in-context learning may reach latent knowledge our
1M-parameter LM cannot.
