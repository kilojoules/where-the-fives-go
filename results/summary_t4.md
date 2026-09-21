# Track 4: entailment vs removal

Best-retained-mimic baselines: mod-5 = 50.1% (chance 20%), mod-13 = 9.6% (chance 7.7%).
Strict disagreement sets: |mod-5| = 100, |mod-13| = 184

## Behavioral (held-out pairs; strict set in parens; vs mimic baseline)

| arm | profile | target acc | strict-set acc | mimic baseline |
|---|---|---|---|---|
| t4_filter5 | core | 14.9±4.3 | 1.0±1.4 | 50.1 |
| t4_gram5 | full | 98.7±1.9 | 99.3±0.9 | 50.1 |
| t4_gram5 | core | 87.7±14.6 | 87.3±15.8 | 50.1 |
| t4_dense | core | 95.6±1.6 | 98.3±0.5 | 50.1 |
| t4_filter13 | core | 6.6±1.9 | 1.6±1.2 | 9.6 |
| t4_gram13 | full | 89.9±2.5 | 97.3±0.4 | 9.6 |
| t4_gram13 | core | 75.0±5.6 | 81.2±5.1 | 9.6 |
| t4_dense | core | 93.0±3.5 | 95.8±2.9 | 9.6 |

## Target probes on core-mask features of (a, target-b) inputs
(chance-normalized; probe split over a's; 'cross' = other target, trained — probe-power control)

| arm | decode | raw acc | chance-norm |
|---|---|---|---|
| t4_dense | mod-5 | 99.3±0.0 | 99.2 |
| t4_dense | mod-13 | 97.6±0.8 | 97.4 |
| t4_filter5 | mod-5 | 31.1±8.1 | 13.9 |
| t4_filter5 | mod-13 | 98.0±0.5 | 97.8 |
| t4_gram5 | mod-5 | 98.0±1.9 | 97.5 |
| t4_gram5 | mod-13 | 98.7±0.9 | 98.6 |
| t4_filter13 | mod-5 | 99.6±0.3 | 99.4 |
| t4_filter13 | mod-13 | 1.8±1.1 | -6.4 |
| t4_gram13 | mod-5 | 99.6±0.3 | 99.4 |
| t4_gram13 | mod-13 | 94.9±2.5 | 94.5 |

## Compositional leakage: decode target residue from CARRIER-task features
(carriers never equal the target; entailing = {10,15,20} for mod-5; non-entailing controls = {7,11,17,19}; both stages)

| arm | decode | carriers | stage | chance-norm acc |
|---|---|---|---|---|
| t4_filter5 | mod-5 | entailing(5) | embed | -13.9±3.9 |
| t4_filter5 | mod-5 | entailing(5) | block2 | 63.8±37.6 |
| t4_filter5 | mod-5 | non-entailing | embed | -16.1±0.6 |
| t4_filter5 | mod-5 | non-entailing | block2 | -15.0±1.0 |
| t4_filter5 | mod-13 | entailing(5) | embed | -7.9±0.2 |
| t4_filter5 | mod-13 | entailing(5) | block2 | -7.5±0.3 |
| t4_filter5 | mod-13 | non-entailing | embed | -7.4±0.5 |
| t4_filter5 | mod-13 | non-entailing | block2 | -7.3±0.1 |
| t4_filter13 | mod-5 | entailing(5) | embed | -15.4±2.4 |
| t4_filter13 | mod-5 | entailing(5) | block2 | 88.2±13.9 |
| t4_filter13 | mod-5 | non-entailing | embed | -12.5±4.5 |
| t4_filter13 | mod-5 | non-entailing | block2 | -15.8±2.0 |
| t4_filter13 | mod-13 | entailing(5) | embed | -7.5±0.6 |
| t4_filter13 | mod-13 | entailing(5) | block2 | -7.2±0.9 |
| t4_filter13 | mod-13 | non-entailing | embed | -7.6±0.3 |
| t4_filter13 | mod-13 | non-entailing | block2 | -7.7±0.3 |
| t4_dense | mod-5 | entailing(5) | embed | -5.4±3.7 |
| t4_dense | mod-5 | entailing(5) | block2 | 96.7±2.0 |
| t4_dense | mod-5 | non-entailing | embed | -12.7±1.8 |
| t4_dense | mod-5 | non-entailing | block2 | -14.2±2.1 |
| t4_dense | mod-13 | entailing(5) | embed | -7.5±0.4 |
| t4_dense | mod-13 | entailing(5) | block2 | -7.2±1.1 |
| t4_dense | mod-13 | non-entailing | embed | -7.9±0.4 |
| t4_dense | mod-13 | non-entailing | block2 | -7.5±0.6 |
