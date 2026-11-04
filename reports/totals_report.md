# Over/under 2.5 goals

Walk-forward, test seasons 2012-2025: 106,697 matches with average pre-match over/under odds (22 leagues). Over rate 49.9%. Binary log loss, lower is better (always guessing 50/50 = 0.6931).

## 1. All test seasons, against average pre-match odds

|                          |   logloss |   accuracy |
|:-------------------------|----------:|-----------:|
| base rate                |    0.6938 |     0.5013 |
| raw Poisson goal ratings |    0.683  |     0.5568 |
| stats model              |    0.6819 |     0.5594 |
| hybrid model             |    0.6793 |     0.5653 |
| average pre-match odds   |    0.6784 |     0.5672 |



## 2. Against Pinnacle (46,823 matches, 2019-2025)

|                                 |   logloss |
|:--------------------------------|----------:|
| stats model                     |    0.6802 |
| hybrid model                    |    0.6774 |
| average pre-match odds          |    0.6772 |
| Pinnacle opening                |    0.677  |
| Pinnacle closing                |    0.6746 |
| Pinnacle closing (power de-vig) |    0.6745 |


Median margin: average pre-match 5.9%, Pinnacle close 3.4%.


## 3. Betting at real prices (flat stakes, 95% CI)

| model        | price          |   min_edge |   bets |   overs |     roi |   ci_low |   ci_high |
|:-------------|:---------------|-----------:|-------:|--------:|--------:|---------:|----------:|
| stats model  | Pinnacle close |       0    |  32834 |   16120 | -0.0504 |  -0.0617 |   -0.0394 |
| stats model  | Pinnacle close |       0.03 |  22921 |   11196 | -0.0541 |  -0.0688 |   -0.0411 |
| stats model  | Pinnacle close |       0.06 |  15492 |    7415 | -0.052  |  -0.0719 |   -0.0324 |
| stats model  | best close     |       0    |  41294 |   20159 | -0.0194 |  -0.0298 |   -0.0089 |
| stats model  | best close     |       0.03 |  30807 |   14967 | -0.0228 |  -0.0367 |   -0.0097 |
| stats model  | best close     |       0.06 |  21527 |   10329 | -0.0275 |  -0.0424 |   -0.012  |
| stats model  | Pinnacle open  |       0    |  29239 |   13306 | -0.0366 |  -0.0489 |   -0.0239 |
| stats model  | Pinnacle open  |       0.03 |  18427 |    8185 | -0.045  |  -0.0608 |   -0.029  |
| stats model  | Pinnacle open  |       0.06 |  11034 |    4728 | -0.0401 |  -0.0623 |   -0.0186 |
| hybrid model | Pinnacle close |       0    |  28497 |   15156 | -0.0368 |  -0.0478 |   -0.0238 |
| hybrid model | Pinnacle close |       0.03 |  16601 |    8918 | -0.0399 |  -0.0557 |   -0.0206 |
| hybrid model | Pinnacle close |       0.06 |   9083 |    4821 | -0.0462 |  -0.0707 |   -0.0241 |
| hybrid model | best close     |       0    |  38960 |   20427 | -0.0148 |  -0.0251 |   -0.0034 |
| hybrid model | best close     |       0.03 |  25319 |   13426 | -0.0169 |  -0.0311 |   -0.0041 |
| hybrid model | best close     |       0.06 |  14954 |    7928 | -0.0188 |  -0.0354 |   -0.001  |
| hybrid model | Pinnacle open  |       0    |  18704 |    9398 | -0.0146 |  -0.0284 |    0.0007 |
| hybrid model | Pinnacle open  |       0.03 |   6580 |    3058 | -0.0134 |  -0.0383 |    0.0119 |
| hybrid model | Pinnacle open  |       0.06 |   2040 |     769 | -0.0356 |  -0.0896 |    0.0155 |



## 4. Closing line value, taking Pinnacle's opening price

CLV = opening odds / fair Pinnacle closing odds - 1. Positive = beating the sharpest final price.

| model                  |   min_edge |   bets |   mean_clv |   ci_low |   ci_high |   share_beat_close |
|:-----------------------|-----------:|-------:|-----------:|---------:|----------:|-------------------:|
| stats model            |       0    |  29239 |    -0.0308 |  -0.0316 |   -0.0301 |             0.2801 |
| stats model            |       0.03 |  18427 |    -0.0302 |  -0.0311 |   -0.0293 |             0.2863 |
| stats model            |       0.06 |  11034 |    -0.0299 |  -0.0312 |   -0.0287 |             0.2924 |
| hybrid model           |       0    |  18704 |    -0.0299 |  -0.0307 |   -0.029  |             0.2748 |
| hybrid model           |       0.03 |   6580 |    -0.0284 |  -0.0303 |   -0.027  |             0.293  |
| hybrid model           |       0.06 |   2040 |    -0.0278 |  -0.031  |   -0.025  |             0.3206 |
| random side (baseline) |     nan    |  46823 |    -0.0387 | nan      |  nan      |             0.2221 |


- corr(stats model - Pinnacle open, close - open) = **0.113**

- corr(hybrid model - Pinnacle open, close - open) = **0.113**


## 5. Is Pinnacle's over/under price biased?

Over probability at close (power de-vig). z > 3 = real miscalibration. ROI = backing every over / every under in the bucket.

| over prob    |     n |   priced |   actual |       z |   ROI all overs |   ROI all unders |
|:-------------|------:|---------:|---------:|--------:|----------------:|-----------------:|
| (0.0, 0.35]  |  2204 |   0.3214 |   0.3235 |  0.2158 |         -0.0575 |          -0.0252 |
| (0.35, 0.45] | 12988 |   0.4089 |   0.4108 |  0.4569 |         -0.0407 |          -0.0301 |
| (0.45, 0.5]  |  9701 |   0.4757 |   0.4761 |  0.0764 |         -0.0341 |          -0.0309 |
| (0.5, 0.55]  |  8777 |   0.524  |   0.5267 |  0.5187 |         -0.0258 |          -0.0415 |
| (0.55, 0.6]  |  6389 |   0.5732 |   0.5729 | -0.0523 |         -0.0292 |          -0.0432 |
| (0.6, 0.7]   |  5515 |   0.6383 |   0.6461 |  1.1954 |         -0.0131 |          -0.0761 |
| (0.7, 1.0]   |  1249 |   0.7399 |   0.7398 | -0.0104 |         -0.0226 |          -0.1045 |



## 6. By league: is the market softer anywhere?

`model - close` < 0 would mean our hybrid beats Pinnacle's close in that league. ROI = hybrid bets with edge > 3% at the best closing price. With 22 leagues, about 1 will look significant by chance.

| league   | tier   |    n |   close_ll |   model_ll |   model - close |   margin |   bets |     roi |   ci_low |   ci_high |
|:---------|:-------|-----:|-----------:|-----------:|----------------:|---------:|-------:|--------:|---------:|----------:|
| SC3      | lower  |  998 |     0.6884 |     0.6865 |         -0.0019 |   0.0456 |    436 |  0.0635 |  -0.0338 |    0.1715 |
| SC2      | lower  |  971 |     0.6888 |     0.6876 |         -0.0012 |   0.0485 |    354 |  0.0405 |  -0.0615 |    0.1518 |
| E2       | lower  | 3306 |     0.6842 |     0.6848 |          0.0006 |   0.0364 |   1605 | -0.0063 |  -0.0551 |    0.0382 |
| D2       | lower  | 1903 |     0.6727 |     0.6736 |          0.0009 |   0.0338 |   1063 | -0.0111 |  -0.0661 |    0.0539 |
| E3       | lower  | 3346 |     0.6787 |     0.6797 |          0.001  |   0.0378 |   1311 | -0.0052 |  -0.0604 |    0.0541 |
| E0       | top    | 2385 |     0.6726 |     0.6741 |          0.0014 |   0.029  |   1591 |  0.0303 |  -0.0296 |    0.0823 |
| E1       | lower  | 3427 |     0.6828 |     0.6845 |          0.0017 |   0.0323 |   1789 | -0.0042 |  -0.0518 |    0.0428 |
| SP1      | top    | 2461 |     0.6678 |     0.67   |          0.0022 |   0.0298 |   1578 | -0.0153 |  -0.0718 |    0.0334 |
| F2       | lower  | 2193 |     0.6738 |     0.6763 |          0.0025 |   0.038  |   1030 | -0.0022 |  -0.0681 |    0.0644 |
| B1       | top    | 1865 |     0.6741 |     0.6768 |          0.0027 |   0.0372 |   1076 | -0.0182 |  -0.0773 |    0.0471 |
| SC0      | top    | 1336 |     0.6742 |     0.6769 |          0.0028 |   0.0334 |    823 | -0.0098 |  -0.0843 |    0.0598 |
| EC       | lower  | 2885 |     0.6828 |     0.6856 |          0.0028 |   0.0452 |   1394 | -0.0239 |  -0.0807 |    0.03   |
| SP2      | lower  | 2961 |     0.6612 |     0.6643 |          0.0032 |   0.0352 |   1278 | -0.0472 |  -0.1054 |    0.0144 |
| P1       | top    | 1986 |     0.6718 |     0.6757 |          0.0038 |   0.034  |   1165 | -0.0058 |  -0.0653 |    0.0623 |
| I2       | lower  | 2349 |     0.6838 |     0.6879 |          0.0041 |   0.0335 |   1170 | -0.0452 |  -0.1076 |    0.0208 |
| I1       | top    | 2474 |     0.6768 |     0.6809 |          0.0042 |   0.0295 |   1629 | -0.0151 |  -0.0653 |    0.0378 |
| N1       | top    | 1898 |     0.6586 |     0.6627 |          0.0042 |   0.0347 |   1172 |  0.0019 |  -0.0608 |    0.0721 |
| G1       | top    |  893 |     0.6712 |     0.6757 |          0.0046 |   0.0469 |    468 |  0.001  |  -0.1116 |    0.0972 |
| T1       | top    | 2239 |     0.6725 |     0.6775 |          0.005  |   0.0338 |   1312 | -0.0385 |  -0.0944 |    0.0208 |
| D1       | top    | 1746 |     0.65   |     0.6562 |          0.0062 |   0.03   |   1165 | -0.0628 |  -0.1306 |   -0.0008 |
| F1       | top    | 2172 |     0.6728 |     0.6793 |          0.0065 |   0.0291 |   1374 | -0.0628 |  -0.1224 |   -0.0092 |
| SC1      | lower  | 1029 |     0.6803 |     0.6872 |          0.0069 |   0.0452 |    536 | -0.0787 |  -0.163  |    0.0145 |



### Top flights vs lower divisions

| tier   |   matches |   model - close (weighted) |   median margin |
|:-------|----------:|---------------------------:|----------------:|
| lower  |     25368 |                     0.0019 |          0.0378 |
| top    |     21455 |                     0.0039 |          0.0334 |



### Betting by tier (hybrid model)

| tier   | model        | price          |   min_edge |   bets |   overs |     roi |   ci_low |   ci_high |
|:-------|:-------------|:---------------|-----------:|-------:|--------:|--------:|---------:|----------:|
| lower  | hybrid model | Pinnacle close |       0.03 |   7786 |    3948 | -0.0294 |  -0.0545 |   -0.0036 |
| lower  | hybrid model | Pinnacle close |       0.06 |   3972 |    2008 | -0.015  |  -0.0491 |    0.0195 |
| lower  | hybrid model | best close     |       0.03 |  11966 |    6322 | -0.0155 |  -0.0345 |    0.0018 |
| lower  | hybrid model | best close     |       0.06 |   6521 |    3482 | -0.005  |  -0.0338 |    0.0212 |
| top    | hybrid model | Pinnacle close |       0.03 |   8815 |    4970 | -0.0492 |  -0.07   |   -0.0273 |
| top    | hybrid model | Pinnacle close |       0.06 |   5111 |    2813 | -0.0705 |  -0.098  |   -0.0396 |
| top    | hybrid model | best close     |       0.03 |  13353 |    7104 | -0.0181 |  -0.0334 |   -0.0001 |
| top    | hybrid model | best close     |       0.06 |   8433 |    4446 | -0.0295 |  -0.0531 |   -0.0057 |

