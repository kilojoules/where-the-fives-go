
## Track 1 (MNIST + Fashion): behavioral, with restricted-argmax correction

| config | profile | fashion acc (20-way) | fashion acc (restricted) | digit acc |
|---|---|---|---|---|
| t1_dense | core | 88.20±0.19 | 88.28±0.20 | 97.56±0.42 |
| t1_filter | core | 0.00±0.00 | 14.52±2.96 | 97.56±0.08 |
| t1_gram | full | 86.98±0.43 | 87.10±0.41 | 97.51±0.28 |
| t1_gram | core | 55.85±8.78 | 76.70±3.90 | 97.59±0.15 |
| t1_gram_f05 | full | 87.94±0.37 | 87.97±0.37 | 97.71±0.09 |
| t1_gram_f05 | core | 86.38±1.24 | 87.42±0.63 | 97.80±0.13 |
| t1_gram_f05_zero | full | 78.20±0.49 | 78.62±0.47 | 97.66±0.29 |
| t1_gram_f05_zero | core | 35.15±8.23 | 62.34±7.05 | 97.71±0.29 |

### Track 1 probe: converged linear head on frozen core-only features

| checkpoint | probe fashion acc | probe digit acc |
|---|---|---|
| t1_dense | 89.20±0.16 | 98.21±0.18 |
| t1_filter | 83.50±0.29 | 97.97±0.10 |
| t1_gram | 88.46±0.05 | 98.18±0.10 |
| t1_gram_f05 | 89.01±0.27 | 98.14±0.03 |
| t1_gram_f05_zero | 85.55±0.31 | 98.17±0.16 |

## Track 2 (mod arithmetic): behavioral

| config | profile | add acc | mult acc |
|---|---|---|---|
| t2_dense | core | 83.81±19.18 | 71.56±19.24 |
| t2_filter | core | 92.89±6.34 | 1.97±0.22 |
| t2_gram | full | 99.84±0.22 | 95.50±5.54 |
| t2_gram | core | 100.00±0.00 | 29.54±22.01 |
| t2_gram_f01 | full | 98.66±1.58 | 93.84±4.19 |
| t2_gram_f01 | core | 99.13±0.78 | 69.35±21.38 |
| t2_gram_f01_zero | full | 98.34±2.18 | 3.16±0.49 |
| t2_gram_f01_zero | core | 98.26±2.46 | 2.53±0.68 |

### Track 2 probe: converged linear head on frozen core-only features (chance mult = 1.9%)

| checkpoint | probe add acc | probe mult acc |
|---|---|---|
| t2_dense | 83.41±19.58 | 71.25±19.19 |
| t2_filter | 69.43±19.83 | 1.90±0.34 |
| t2_gram | 98.10±1.35 | 95.18±3.92 |
| t2_gram_f01 | 97.71±2.60 | 93.44±4.47 |
| t2_gram_f01_zero | 99.21±0.56 | 3.71±0.87 |

## Track 3 (two-topic LM): behavioral

| config | profile | core loss | aux loss | fact partner | fact state |
|---|---|---|---|---|---|
| t3_dense | core | 1.542±0.001 | 0.839±0.004 | 100.00±0.00 | 100.00±0.00 |
| t3_filter | core | 1.540±0.000 | 10.110±0.081 | 0.00±0.00 | 0.00±0.00 |
| t3_gram | full | 1.543±0.000 | 0.900±0.016 | 98.89±1.57 | 90.00±9.43 |
| t3_gram | core | 1.540±0.001 | 6.494±0.438 | 0.00±0.00 | 0.00±0.00 |
| t3_gram_f01 | full | 1.542±0.001 | 0.837±0.005 | 100.00±0.00 | 100.00±0.00 |
| t3_gram_f01 | core | 1.541±0.001 | 1.412±0.226 | 70.00±16.56 | 91.11±6.85 |
| t3_gram_f01_zero | full | 1.542±0.002 | 1.640±0.010 | 13.33±2.72 | 17.78±6.29 |
| t3_gram_f01_zero | core | 1.542±0.002 | 6.259±0.424 | 0.00±0.00 | 0.00±0.00 |

### Track 3 probe: retrained head on frozen core-only trunk (all data incl. aux)

| checkpoint | probe fact partner | probe fact state | probe aux loss |
|---|---|---|---|
| t3_dense | 100.00±0.00 | 100.00±0.00 | 0.826±0.000 |
| t3_filter | 33.33±2.72 | 46.67±12.47 | 1.494±0.021 |
| t3_gram | 93.33±7.20 | 96.67±2.72 | 0.917±0.011 |
| t3_gram_f01 | 100.00±0.00 | 100.00±0.00 | 0.833±0.002 |
| t3_gram_f01_zero | 51.11±15.95 | 64.44±11.33 | 1.237±0.024 |
