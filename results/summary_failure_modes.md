
## A1. Fashion: per-class recall of ablated core (restricted argmax)
| ship | T-shirt/top | Trouser | Pullover | Dress | Coat | Sandal | Shirt | Sneaker | Bag | Ankle boot |
| t1_gram | 84 | 96 | 76 | 76 | 80 | 67 | 31 | 75 | 89 | 92 |
| t1_gram_pas01 | 0 | 32 | 11 | 40 | 0 | 37 | 16 | 0 | 60 | 0 |
| t1_filter | 29 | 40 | 3 | 11 | 3 | 3 | 13 | 10 | 13 | 20 |

## A2. Modmath: which multiplication pairs survive in the ablated core?
| ship | a or b = 0 | a or b = 1 (not 0) | both >= 2 | overall |
| t2_gram | 100.0 | 33.3 | 26.3 | 29.5 |
| t2_filter | 0.0 | 0.0 | 2.1 | 2.0 |

## A3. LM: fact survival in ablated core under partial labeling (t3_gram_f01)
| seed | partner acc (10-way) | state acc (8-way) |
| 0 | 83 | 90 |
| 1 | 80 | 100 |
| 2 | 47 | 83 |
(state facts survive more than partner facts — the leak is structured, not uniform)
/Users/julianquick/minst_gradient_routing/train.py:75: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  ytr = torch.tensor(tr.targets).to(device)
/Users/julianquick/minst_gradient_routing/train.py:77: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  yte = torch.tensor(te.targets).to(device)

## B1. MNIST: PGD on true 5s — success = ablated model predicts 5
| ship (mask) | eps=0.03 | 0.06 | 0.12 | 0.25 |
| ck_gram_main | 0.0 | 0.0 | 0.0 | 0.3 |
| dose_k0.03 | 0.0 | 0.0 | 0.0 | 0.0 |
| ck_filter_no5 | 0.0 | 0.0 | 2.6 | 54.8 |
| ck_dense_all | 99.8 | 100.0 | 100.0 | 100.0 |

## B2. Fashion: PGD on fashion images vs localized core (p_as=0.1)
| ship | eps=0.03 | 0.06 | 0.12 | 0.25 |
| t1_gram_pas01 | 4.0 | 7.8 | 13.9 | 32.3 |
| t1_filter | 0.0 | 0.0 | 1.6 | 30.7 |
| t1_dense | 98.8 | 99.9 | 100.0 | 100.0 |

## B3. Modmath: activation-space PGD (first hidden layer) toward true product
| ship | eps/||h||: 0.05 | 0.1 | 0.25 | 0.5 |
| t2_gram | 60.8 | 77.9 | 95.0 | 95.1 |
| t2_filter | 7.8 | 24.7 | 73.4 | 95.8 |
| t2_dense | 99.0 | 100.0 | 100.0 | 100.0 |

## C. LM: partner-fact accuracy vs # in-context OTHER-element facts (core profile)
| ship | m=0 | m=1 | m=2 | m=3 | m=4 | (full profile m=0) |
| t3_dense | 100±0 | 100±0 | 100±0 | 100±0 | 100±0 | 100 |
| t3_filter | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | 0 |
| t3_gram | 0±0 | 0±0 | 2±3 | 10±3 | 8±7 | 99 |
| t3_gram_f01 | 70±17 | 93±7 | 79±19 | 86±9 | 84±12 | 100 |
| t3_gram_f01_zero | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | 13 |

FAILURE MODES DONE
