# Training-label noise: quarantine and memorization reversion


## fp5 — FP labels: non-5 images mislabeled '5' (routed into module-5)

### Clean test set

| rate | model/profile | recall 5 | recall 7 | FP->5 | FP->7 | acc excl 5,7 |
|---|---|---|---|---|---|---|
| 0.1 | dense (noisy labels) | 98.32±0.18 | 97.54±0.45 | 0.36±0.13 | 0.27±0.09 | 97.75±0.21 |
| 0.1 | GRAM full | 96.64±3.45 | 67.93±9.72 | 8.36±5.42 | 5.42±5.27 | 84.43±0.87 |
| 0.1 | GRAM ablate-5 | 0.00±0.00 | 95.56±0.87 | 0.00±0.00 | 1.59±1.21 | 96.70±1.50 |
| 0.1 | GRAM ablate-7 | 97.65±1.89 | 0.00±0.00 | 13.13±1.72 | 0.00±0.00 | 92.78±1.72 |
| 0.1 | GRAM core-only | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 97.33±0.81 |
| 0.3 | dense (noisy labels) | 96.86±0.51 | 97.89±0.12 | 0.40±0.16 | 0.25±0.04 | 97.92±0.08 |
| 0.3 | GRAM full | 91.59±8.13 | 95.82±1.96 | 6.05±4.78 | 8.22±2.32 | 78.24±6.66 |
| 0.3 | GRAM ablate-5 | 0.00±0.00 | 95.36±0.46 | 0.00±0.00 | 1.42±0.52 | 97.09±0.53 |
| 0.3 | GRAM ablate-7 | 97.53±1.41 | 0.00±0.00 | 16.50±5.00 | 0.00±0.00 | 91.37±5.07 |
| 0.3 | GRAM core-only | 0.00±0.00 | 0.00±0.00 | 0.06±0.09 | 0.00±0.00 | 97.61±0.21 |

### Memorization probe (the mislabeled train images themselves)

| rate | model/profile | pred = noisy label | pred = TRUE label |
|---|---|---|---|
| 0.1 | dense (noisy labels) | 17.40±4.18 | 79.77±4.32 |
| 0.1 | GRAM full | 13.22±8.96 | 75.77±2.03 |
| 0.1 | GRAM ablate-5 | 0.00±0.00 | 95.26±2.02 |
| 0.1 | GRAM ablate-7 | 17.96±4.92 | 77.61±4.82 |
| 0.1 | GRAM core-only | 0.00±0.00 | 85.85±0.98 |
| 0.3 | dense (noisy labels) | 10.97±1.42 | 86.22±0.89 |
| 0.3 | GRAM full | 6.97±4.42 | 77.59±5.91 |
| 0.3 | GRAM ablate-5 | 0.00±0.00 | 97.09±0.36 |
| 0.3 | GRAM ablate-7 | 18.63±6.57 | 78.84±5.96 |
| 0.3 | GRAM core-only | 0.04±0.06 | 86.70±0.73 |

## fn5 — FN labels: true 5s mislabeled as other digits (diverted away from module-5)

### Clean test set

| rate | model/profile | recall 5 | recall 7 | FP->5 | FP->7 | acc excl 5,7 |
|---|---|---|---|---|---|---|
| 0.1 | dense (noisy labels) | 96.60±0.98 | 96.53±0.26 | 0.23±0.11 | 0.12±0.03 | 98.23±0.12 |
| 0.1 | GRAM full | 97.72±1.83 | 93.77±2.29 | 5.64±5.49 | 0.38±0.21 | 92.54±5.58 |
| 0.1 | GRAM ablate-5 | 0.00±0.00 | 96.89±0.86 | 0.00±0.00 | 3.51±2.35 | 97.66±0.20 |
| 0.1 | GRAM ablate-7 | 97.38±1.87 | 0.00±0.00 | 6.77±7.47 | 0.00±0.00 | 93.09±6.33 |
| 0.1 | GRAM core-only | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 98.14±0.02 |
| 0.3 | dense (noisy labels) | 94.99±0.23 | 96.95±0.28 | 0.20±0.01 | 0.19±0.01 | 98.06±0.09 |
| 0.3 | GRAM full | 90.40±8.85 | 95.72±1.13 | 2.08±1.59 | 1.23±1.20 | 95.58±1.55 |
| 0.3 | GRAM ablate-5 | 0.00±0.00 | 95.95±1.83 | 0.00±0.00 | 6.04±0.49 | 97.43±0.35 |
| 0.3 | GRAM ablate-7 | 96.75±2.02 | 0.00±0.00 | 6.14±4.06 | 0.00±0.00 | 95.35±2.35 |
| 0.3 | GRAM core-only | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 97.88±0.17 |

### Memorization probe (the mislabeled train images themselves)

| rate | model/profile | pred = noisy label | pred = TRUE label |
|---|---|---|---|
| 0.1 | dense (noisy labels) | 16.30±1.49 | 81.24±2.58 |
| 0.1 | GRAM full | 1.60±1.15 | 97.36±2.36 |
| 0.1 | GRAM ablate-5 | 27.92±5.10 | 0.00±0.00 |
| 0.1 | GRAM ablate-7 | 1.78±1.00 | 96.68±2.17 |
| 0.1 | GRAM core-only | 29.27±2.56 | 0.00±0.00 |
| 0.3 | dense (noisy labels) | 13.37±1.34 | 83.48±1.52 |
| 0.3 | GRAM full | 2.85±1.56 | 91.41±7.69 |
| 0.3 | GRAM ablate-5 | 22.14±0.88 | 0.00±0.00 |
| 0.3 | GRAM ablate-7 | 2.07±1.17 | 96.29±2.31 |
| 0.3 | GRAM core-only | 23.66±1.26 | 0.00±0.00 |

## swap57 — swap: 5s and 7s trade labels with prob r

### Clean test set

| rate | model/profile | recall 5 | recall 7 | FP->5 | FP->7 | acc excl 5,7 |
|---|---|---|---|---|---|---|
| 0.1 | dense (noisy labels) | 96.86±0.09 | 97.37±0.40 | 0.19±0.06 | 0.23±0.07 | 97.98±0.28 |
| 0.1 | GRAM full | 65.88±43.94 | 39.17±42.47 | 9.46±6.17 | 3.84±5.13 | 94.58±1.79 |
| 0.1 | GRAM ablate-5 | 0.00±0.00 | 95.85±1.91 | 0.00±0.00 | 10.17±0.25 | 97.26±0.36 |
| 0.1 | GRAM ablate-7 | 97.38±1.61 | 0.00±0.00 | 13.64±3.47 | 0.00±0.00 | 95.07±2.92 |
| 0.1 | GRAM core-only | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 98.01±0.15 |
| 0.3 | dense (noisy labels) | 91.93±2.78 | 94.94±2.98 | 0.29±0.15 | 0.73±0.36 | 97.99±0.29 |
| 0.3 | GRAM full | 35.50±30.71 | 89.23±4.88 | 3.01±3.29 | 6.99±3.65 | 94.73±2.05 |
| 0.3 | GRAM ablate-5 | 0.00±0.00 | 96.69±0.44 | 0.00±0.00 | 10.58±0.08 | 97.24±0.12 |
| 0.3 | GRAM ablate-7 | 97.31±1.27 | 0.00±0.00 | 12.50±1.06 | 0.00±0.00 | 96.73±0.88 |
| 0.3 | GRAM core-only | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 98.14±0.10 |

### Memorization probe (the mislabeled train images themselves)

| rate | model/profile | pred = noisy label | pred = TRUE label |
|---|---|---|---|
| 0.1 | dense (noisy labels) | 2.75±0.26 | 96.55±0.34 |
| 0.1 | GRAM full | 47.43±4.21 | 51.07±5.19 |
| 0.1 | GRAM ablate-5 | 43.44±0.67 | 53.58±1.24 |
| 0.1 | GRAM ablate-7 | 53.04±1.36 | 44.53±1.18 |
| 0.1 | GRAM core-only | 0.00±0.00 | 0.00±0.00 |
| 0.3 | dense (noisy labels) | 7.82±1.22 | 91.39±1.25 |
| 0.3 | GRAM full | 31.40±13.79 | 65.44±11.26 |
| 0.3 | GRAM ablate-5 | 45.97±0.50 | 52.18±0.29 |
| 0.3 | GRAM ablate-7 | 52.29±0.06 | 46.23±0.57 |
| 0.3 | GRAM core-only | 0.00±0.00 | 0.00±0.00 |

## fn5 split: true 5s relabeled '7' (routed to module-7) vs relabeled other

| rate | profile | subset | pred=noisy | pred=5 (true) |
|---|---|---|---|---|
| 0.1 | full | ->7 | 0.53±0.75 | 98.94±1.50 |
| 0.1 | full | ->other | 1.73±1.21 | 97.16±2.49 |
| 0.1 | ablate7 | ->7 | 0.00±0.00 | 98.85±0.82 |
| 0.1 | ablate7 | ->other | 2.00±1.13 | 96.41±2.36 |
| 0.1 | core_only | ->7 | 0.00±0.00 | 0.00±0.00 |
| 0.1 | core_only | ->other | 32.80±2.82 | 0.00±0.00 |
| 0.3 | full | ->7 | 6.89±9.74 | 91.13±9.68 |
| 0.3 | full | ->other | 2.36±1.37 | 91.44±7.45 |
| 0.3 | ablate7 | ->7 | 0.00±0.00 | 97.82±1.59 |
| 0.3 | ablate7 | ->other | 2.33±1.32 | 96.10±2.41 |
| 0.3 | core_only | ->7 | 0.00±0.00 | 0.00±0.00 |
| 0.3 | core_only | ->other | 26.63±1.43 | 0.00±0.00 |
