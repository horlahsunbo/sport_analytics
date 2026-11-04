# Goal-rating tuning (over/under 2.5)

Chosen on 2005-2014 only; `valid` = 2015-2025. Binary log loss of the raw Poisson over-2.5 probability (no calibration), all 38 leagues. Lower is better.

Baseline (always predict the tuning-period over rate): tune 0.6926, valid 0.6936

## Best setting

|     k |   regress |   k_early |   w_shots |   tune |   valid |   valid_mean_pred |   valid_actual |
|------:|----------:|----------:|----------:|-------:|--------:|------------------:|---------------:|
| 0.007 |       0.2 |         1 |         0 | 0.6852 |  0.6822 |            0.4928 |         0.4993 |


## Best per shot weight

|     k |   regress |   k_early |   w_shots |   tune |   valid |   valid_mean_pred |   valid_actual |
|------:|----------:|----------:|----------:|-------:|--------:|------------------:|---------------:|
| 0.007 |       0.2 |         1 |       0   | 0.6852 |  0.6822 |            0.4928 |         0.4993 |
| 0.01  |       0.4 |         1 |       0.3 | 0.6852 |  0.6821 |            0.4942 |         0.4993 |
| 0.01  |       0.4 |         1 |       0.5 | 0.6856 |  0.6821 |            0.4952 |         0.4993 |
| 0.01  |       0.4 |         1 |       0.7 | 0.6862 |  0.6822 |            0.4961 |         0.4993 |


## Best per season reset

|     k |   regress |   k_early |   w_shots |   tune |   valid |   valid_mean_pred |   valid_actual |
|------:|----------:|----------:|----------:|-------:|--------:|------------------:|---------------:|
| 0.007 |       0   |         1 |       0.3 | 0.6857 |  0.6823 |            0.4934 |         0.4993 |
| 0.007 |       0.2 |         1 |       0   | 0.6852 |  0.6822 |            0.4928 |         0.4993 |
| 0.01  |       0.4 |         1 |       0.3 | 0.6852 |  0.6821 |            0.4942 |         0.4993 |
| 0.01  |       0.6 |         1 |       0   | 0.6855 |  0.6829 |            0.4929 |         0.4993 |
| 0.01  |       0.8 |         1 |       0   | 0.6859 |  0.6837 |            0.493  |         0.4993 |

