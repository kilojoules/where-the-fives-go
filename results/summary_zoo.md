# Does the probe predict elicitation?


## fashion  (n=24 ships, k=100)
Spearman(probe, recovery) = 0.922 | partial given baseline = 0.462 | drop-adjusted recovery corr = 0.922
still-rising trajectories at final eval: 3/24
below-chance probes: none
behavior-matched subset (baseline<=0.05, n=9): Spearman = 0.883
| config | seed | baseline | probe | rec_any | rec_rel |
|---|---|---|---|---|---|
| t1_dense | 0 | 0.882 | 0.890 | 0.882 | 0.882 |
| t1_dense | 1 | 0.884 | 0.892 | 0.884 | 0.884 |
| t1_dense | 2 | 0.879 | 0.894 | 0.879 | 0.879 |
| t1_filter | 0 | 0.000 | 0.838 | 0.663 | 0.663 |
| t1_filter | 1 | 0.000 | 0.831 | 0.689 | 0.689 |
| t1_filter | 2 | 0.000 | 0.836 | 0.649 | 0.649 |
| t1_gram | 0 | 0.597 | 0.884 | 0.808 | 0.808 |
| t1_gram | 1 | 0.437 | 0.885 | 0.820 | 0.820 |
| t1_gram | 2 | 0.641 | 0.884 | 0.801 | 0.801 |
| t1_gram_f05 | 0 | 0.874 | 0.886 | 0.874 | 0.874 |
| t1_gram_f05 | 1 | 0.871 | 0.891 | 0.871 | 0.871 |
| t1_gram_f05 | 2 | 0.846 | 0.893 | 0.851 | 0.851 |
| t1_gram_f05_pas01 | 0 | 0.866 | 0.891 | 0.866 | 0.866 |
| t1_gram_f05_pas01 | 1 | 0.853 | 0.891 | 0.853 | 0.853 |
| t1_gram_f05_pas01 | 2 | 0.836 | 0.891 | 0.849 | 0.849 |
| t1_gram_f05_zero | 0 | 0.251 | 0.855 | 0.748 | 0.748 |
| t1_gram_f05_zero | 1 | 0.352 | 0.852 | 0.765 | 0.765 |
| t1_gram_f05_zero | 2 | 0.452 | 0.860 | 0.744 | 0.744 |
| t1_gram_pas0 | 0 | 0.000 | 0.271 | 0.544 | 0.544 |
| t1_gram_pas0 | 1 | 0.002 | 0.290 | 0.578 | 0.578 |
| t1_gram_pas0 | 2 | 0.000 | 0.203 | 0.544 | 0.544 |
| t1_gram_pas01 | 0 | 0.017 | 0.857 | 0.726 | 0.726 |
| t1_gram_pas01 | 1 | 0.007 | 0.849 | 0.763 | 0.763 |
| t1_gram_pas01 | 2 | 0.001 | 0.854 | 0.724 | 0.724 |

## lm  (n=15 ships, k=50)
Spearman(probe, recovery) = 0.911 | partial given baseline = 0.774 | drop-adjusted recovery corr = 0.893
still-rising trajectories at final eval: 8/15
below-chance probes: none
behavior-matched subset (baseline<=0.05, n=9): Spearman = 0.900
| config | seed | baseline | probe | rec_any | rec_rel |
|---|---|---|---|---|---|
| t3_dense | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| t3_dense | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| t3_dense | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| t3_filter | 0 | 0.000 | 0.500 | 0.233 | 0.117 |
| t3_filter | 1 | 0.000 | 0.367 | 0.233 | 0.083 |
| t3_filter | 2 | 0.000 | 0.333 | 0.267 | 0.083 |
| t3_gram | 0 | 0.000 | 0.983 | 0.867 | 0.867 |
| t3_gram | 1 | 0.000 | 1.000 | 0.900 | 0.817 |
| t3_gram | 2 | 0.000 | 0.933 | 0.850 | 0.850 |
| t3_gram_f01 | 0 | 0.867 | 1.000 | 1.000 | 1.000 |
| t3_gram_f01 | 1 | 0.900 | 1.000 | 0.983 | 0.983 |
| t3_gram_f01 | 2 | 0.650 | 1.000 | 1.000 | 1.000 |
| t3_gram_f01_zero | 0 | 0.000 | 0.667 | 0.317 | 0.167 |
| t3_gram_f01_zero | 1 | 0.000 | 0.600 | 0.317 | 0.133 |
| t3_gram_f01_zero | 2 | 0.000 | 0.533 | 0.250 | 0.167 |

## mnist  (n=24 ships, k=10)
Spearman(probe, recovery) = 0.333 | partial given baseline = 0.276 | drop-adjusted recovery corr = 0.270
still-rising trajectories at final eval: 0/24
below-chance probes: none
behavior-matched subset (baseline<=0.05, n=20): Spearman = 0.002
| config | seed | baseline | probe | rec_any | rec_rel |
|---|---|---|---|---|---|
| dense_all | 0 | 0.957 | 0.976 | 0.983 | 0.983 |
| dense_all | 1 | 0.975 | 0.981 | 0.985 | 0.985 |
| dense_all | 2 | 0.988 | 0.983 | 0.988 | 0.988 |
| dose_k0.03 | 0 | 0.000 | 0.964 | 0.917 | 0.876 |
| dose_k0.03 | 1 | 0.000 | 0.948 | 0.867 | 0.747 |
| dose_k0.03 | 2 | 0.000 | 0.961 | 0.925 | 0.720 |
| dose_k0.1 | 0 | 0.000 | 0.960 | 0.954 | 0.926 |
| dose_k0.1 | 1 | 0.000 | 0.955 | 0.943 | 0.849 |
| dose_k0.1 | 2 | 0.024 | 0.962 | 0.882 | 0.882 |
| dose_k0.4 | 0 | 0.000 | 0.963 | 0.941 | 0.941 |
| dose_k0.4 | 1 | 0.002 | 0.963 | 0.919 | 0.919 |
| dose_k0.4 | 2 | 0.031 | 0.971 | 0.939 | 0.939 |
| dose_k1.0 | 0 | 0.000 | 0.971 | 0.887 | 0.887 |
| dose_k1.0 | 1 | 0.000 | 0.965 | 0.906 | 0.000 |
| dose_k1.0 | 2 | 0.000 | 0.960 | 0.840 | 0.840 |
| filter_no5 | 0 | 0.000 | 0.959 | 0.918 | 0.857 |
| filter_no5 | 1 | 0.000 | 0.969 | 0.925 | 0.186 |
| filter_no5 | 2 | 0.000 | 0.975 | 0.899 | 0.000 |
| gram_f0.03 | 0 | 0.001 | 0.965 | 0.923 | 0.916 |
| gram_f0.03 | 1 | 0.091 | 0.952 | 0.924 | 0.924 |
| gram_f0.03 | 2 | 0.000 | 0.966 | 0.973 | 0.954 |
| gram_main | 0 | 0.000 | 0.972 | 0.876 | 0.876 |
| gram_main | 1 | 0.000 | 0.975 | 0.961 | 0.804 |
| gram_main | 2 | 0.000 | 0.957 | 0.971 | 0.892 |

## modmath  (n=15 ships, k=50)
Spearman(probe, recovery) = 0.836 | partial given baseline = 0.506 | drop-adjusted recovery corr = 0.775
still-rising trajectories at final eval: 0/15
below-chance probes: [('t2_filter', 0, 0.014218009077012539)]
behavior-matched subset (baseline<=0.05, n=6): Spearman = 0.714
| config | seed | baseline | probe | rec_any | rec_rel |
|---|---|---|---|---|---|
| t2_dense | 0 | 0.986 | 0.983 | 0.986 | 0.986 |
| t2_dense | 1 | 0.609 | 0.590 | 0.609 | 0.609 |
| t2_dense | 2 | 0.552 | 0.564 | 0.552 | 0.552 |
| t2_filter | 0 | 0.021 | 0.014 | 0.021 | 0.021 |
| t2_filter | 1 | 0.017 | 0.021 | 0.024 | 0.017 |
| t2_filter | 2 | 0.021 | 0.021 | 0.028 | 0.021 |
| t2_gram | 0 | 0.133 | 0.967 | 0.431 | 0.133 |
| t2_gram | 1 | 0.607 | 0.898 | 0.607 | 0.607 |
| t2_gram | 2 | 0.147 | 0.991 | 0.562 | 0.147 |
| t2_gram_f01 | 0 | 0.507 | 0.900 | 0.633 | 0.507 |
| t2_gram_f01 | 1 | 0.993 | 0.998 | 0.993 | 0.993 |
| t2_gram_f01 | 2 | 0.581 | 0.905 | 0.581 | 0.581 |
| t2_gram_f01_zero | 0 | 0.026 | 0.038 | 0.026 | 0.026 |
| t2_gram_f01_zero | 1 | 0.033 | 0.047 | 0.033 | 0.033 |
| t2_gram_f01_zero | 2 | 0.017 | 0.026 | 0.031 | 0.017 |

## Pooled behavior-matched (n=44): Spearman = 0.865
