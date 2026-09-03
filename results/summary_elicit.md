# Elicitation: recovering the removed 5 by finetuning

Best recall on 5s the attacker reaches while the model stays useful (acc on non-5 digits >= 95%); best over 2 learning rates; mean±sd over 3 seeds.

| shipped model | k=10 | k=50 | k=200 |
|---|---|---|---|
| dense, all data (capability present) | 98.5±0.2 | 98.5±0.4 | 98.8±0.4 |
| data-filtered (never saw 5s) | 28.6±40.4 | 92.6±0.7 | 96.3±0.3 |
| GRAM, module-5 ablated | 17.6±24.9 | 93.0±4.5 | 97.4±0.2 |
| GRAM @3% labels, module-5 ablated | 33.6±41.2 | 94.4±0.8 | 97.0±0.1 |

Alpha sweep (mean over seeds):
| α | recall 5 | FP->5 |
|---|---|---|
| 0.0 | 0.00 | 0.00 |
| 0.02 | 0.00 | 0.00 |
| 0.05 | 0.00 | 0.00 |
| 0.1 | 0.00 | 0.00 |
| 0.2 | 0.60 | 0.00 |
| 0.3 | 19.54 | 0.00 |
| 0.5 | 61.47 | 0.02 |
| 0.7 | 81.65 | 0.19 |
| 0.85 | 88.30 | 0.62 |
| 1.0 | 92.94 | 1.59 |
