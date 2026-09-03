## Headline: capability removal via module ablation

| model / profile | overall acc | recall 5 | recall 7 | acc excl {5,7} | FP rate into 5 | FP rate into 7 |
|---|---|---|---|---|---|---|
| dense baseline | 98.03±0.21 | 97.35±1.24 | 96.89±1.01 | 98.25±0.13 | 0.12±0.06 | 0.20±0.07 |
| GRAM full (core+5+7) | 95.28±1.18 | 92.94±7.32 | 97.83±0.75 | 95.21±1.53 | 1.59±1.51 | 1.78±0.69 |
| GRAM ablate module-5 | 88.75±0.37 | 0.00±0.00 | 96.53±0.30 | 97.56±0.42 | 0.00±0.00 | 1.73±0.62 |
| GRAM ablate module-7 | 86.20±1.68 | 97.12±1.79 | 0.00±0.00 | 95.96±2.26 | 3.68±3.08 | 0.00±0.00 |
| GRAM core only | 79.20±0.19 | 0.00±0.00 | 0.00±0.00 | 98.02±0.24 | 0.00±0.00 | 0.00±0.00 |
| filtered (no 5s) | 89.26±0.16 | 0.00±0.00 | 98.70±0.12 | 97.91±0.18 | 0.00±0.00 | 0.52±0.10 |
| filtered (no 7s) | 87.95±0.06 | 97.20±0.56 | 0.00±0.00 | 98.12±0.11 | 0.18±0.02 | 0.00±0.00 |
| filtered (no 5s,7s) | 79.35±0.15 | 0.00±0.00 | 0.00±0.00 | 98.21±0.18 | 0.00±0.00 | 0.00±0.00 |

## Where do the 5s go? (FN destinations, mean counts over 892 test 5s)

| model | ->0 | ->1 | ->2 | ->3 | ->4 | ->5 | ->6 | ->7 | ->8 | ->9 |
|---|---|---|---|---|---|---|---|---|---|---|
| GRAM ablate-5 | 12 | 3 | 2 | 429 | 4 | 0 | 45 | 102 | 230 | 66 |
| GRAM core-only | 12 | 2 | 5 | 451 | 7 | 0 | 36 | 0 | 315 | 64 |
| filtered no-5 | 30 | 2 | 9 | 409 | 12 | 0 | 52 | 14 | 265 | 98 |
| filtered no-5,7 | 12 | 3 | 7 | 371 | 17 | 0 | 32 | 0 | 396 | 54 |

(same, for true 7s under ablate-7 vs filtered no-7)

| model | ->0 | ->1 | ->2 | ->3 | ->4 | ->5 | ->6 | ->7 | ->8 | ->9 |
|---|---|---|---|---|---|---|---|---|---|---|
| GRAM ablate-7 | 1 | 8 | 261 | 259 | 13 | 153 | 1 | 0 | 20 | 313 |
| filtered no-7 | 7 | 37 | 184 | 289 | 23 | 5 | 2 | 0 | 21 | 461 |

FN-destination cosine similarity (true-5 row, excl. col 5): GRAM-ablate5 vs filtered-no5 = 0.979

## Absorption: fraction f of 5s routed/filtered

Unlabeled-5 treatment arms: all-active = paper's partial-labeling rule; core = declared core data (p_cr side channel still reaches module-5); isolated = never activates any module (clean control).

ablate5 = deployment profile (module-7 active); core-only = both modules off, isolating what the core itself knows about 5s (absorption per se).

| f | ablate5 (all-active) | ablate5 (core+p_cr) | ablate5 (isolated) | core-only (all-active) | core-only (isolated) | partial-filter | GRAM full |
|---|---|---|---|---|---|---|---|
| 0.03 | 3.06±4.25 | 96.56±0.67 | 40.02±15.60 | 0.00±0.00 | 96.71±1.17 | 97.38±0.80 | 99.85±0.05 |
| 0.1 | 0.30±0.42 | 97.16±1.32 | 51.91±16.02 | 0.00±0.00 | 98.13±0.29 | 97.65±0.56 | 99.85±0.05 |
| 0.3 | 6.02±5.80 | 97.98±0.64 | 47.23±30.59 | 0.00±0.00 | 98.54±0.42 | 97.35±0.23 | 99.96±0.05 |
| 1.0 | 0.00±0.00 | (n/a) | (n/a) | 0.00±0.00 | (n/a) | 0.00±0.00 | 92.94±7.32 |

## Per-example: which 5s survive module-5 ablation?

survival rate of true 5s under ablate-5: 0.000±0.000
module-5 logit-5 contribution — survivors: nan, casualties: 39.49

corr(module-5 contribution, ablated margin) over true 5s: r = -0.337
