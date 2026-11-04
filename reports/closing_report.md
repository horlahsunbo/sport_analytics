# Closing-odds evaluation

98,530 out-of-sample matches (2012-2025, 22 leagues) that have average pre-match odds, Pinnacle opening and Pinnacle closing odds.

## 1. Who predicts best? (log loss, lower is better)

|                                        |   logloss |   accuracy |
|:---------------------------------------|----------:|-----------:|
| avg pre-match odds (current benchmark) |    1.0047 |     0.5    |
| Pinnacle opening                       |    1.004  |     0.5002 |
| Pinnacle closing                       |    1.0009 |     0.5027 |
| Pinnacle closing (power de-vig)        |    1.0007 |     0.5027 |
| our stats model                        |    1.0156 |     0.4919 |
| our hybrid model                       |    1.0071 |     0.4991 |


Median bookmaker margin: avg pre-match 6.3%, Pinnacle open 3.0%, Pinnacle close 2.8%


## 2. Betting at real closing prices (flat stakes, 95% CI)

Bet the outcome with the largest model edge when edge > min_edge at that price.

| model            | price              |   min_edge |   bets |     roi |   ci_low |   ci_high |
|:-----------------|:-------------------|-----------:|-------:|--------:|---------:|----------:|
| our stats model  | Pinnacle close     |       0    |  93991 | -0.0413 |  -0.0501 |   -0.032  |
| our stats model  | Pinnacle close     |       0.03 |  83350 | -0.0435 |  -0.0546 |   -0.0323 |
| our stats model  | Pinnacle close     |       0.08 |  61987 | -0.0477 |  -0.061  |   -0.0347 |
| our stats model  | best close (2019+) |       0    |  46350 | -0.0108 |  -0.0256 |    0.0048 |
| our stats model  | best close (2019+) |       0.03 |  43469 | -0.0089 |  -0.0254 |    0.0076 |
| our stats model  | best close (2019+) |       0.08 |  33995 | -0.0154 |  -0.033  |    0.0023 |
| our hybrid model | Pinnacle close     |       0    |  91659 | -0.0263 |  -0.0364 |   -0.0164 |
| our hybrid model | Pinnacle close     |       0.03 |  76271 | -0.0243 |  -0.0358 |   -0.0134 |
| our hybrid model | Pinnacle close     |       0.08 |  48635 | -0.0216 |  -0.0351 |   -0.0067 |
| our hybrid model | best close (2019+) |       0    |  46026 |  0.0045 |  -0.0108 |    0.0174 |
| our hybrid model | best close (2019+) |       0.03 |  41269 |  0.0114 |  -0.0065 |    0.0269 |
| our hybrid model | best close (2019+) |       0.08 |  27957 |  0.0076 |  -0.0121 |    0.0272 |



## 3. Closing line value (CLV)

We take the early average price on the model's pick; CLV = early odds / fair Pinnacle closing odds - 1. Positive mean CLV = we systematically got better prices than the sharpest final price, i.e. a real edge. Average-odds margins make CLV negative by default, so the 'random picks' row is the baseline to beat.

| model                   |   min_edge |   bets |   mean_clv |   ci_low |   ci_high |   share_beat_close |
|:------------------------|-----------:|-------:|-----------:|---------:|----------:|-------------------:|
| our stats model         |       0    |  82836 |    -0.0453 |  -0.046  |   -0.0447 |             0.255  |
| our stats model         |       0.05 |  58771 |    -0.0435 |  -0.0443 |   -0.0427 |             0.2742 |
| our stats model         |       0.1  |  38719 |    -0.0418 |  -0.0429 |   -0.0409 |             0.2941 |
| our hybrid model        |       0    |  70804 |    -0.0425 |  -0.043  |   -0.0419 |             0.2403 |
| our hybrid model        |       0.05 |  37831 |    -0.0388 |  -0.0397 |   -0.0379 |             0.2649 |
| our hybrid model        |       0.1  |  18490 |    -0.0356 |  -0.0369 |   -0.0345 |             0.2869 |
| random picks (baseline) |     nan    |  98530 |    -0.0586 | nan      |  nan      |             0.1974 |



### 3b. Taking Pinnacle's opening price instead

Same test, but we bet at Pinnacle's opening odds (about 3% margin, available days before kickoff). Edge is judged against the de-vigged opening price. ROI = actual profit at the opening odds.

| model                   |   min_edge |   bets |   mean_clv |   clv_ci_low |   clv_ci_high |   share_beat_close |      roi |   roi_ci_low |   roi_ci_high |
|:------------------------|-----------:|-------:|-----------:|-------------:|--------------:|-------------------:|---------:|-------------:|--------------:|
| our stats model         |       0    |  92412 |    -0.0219 |      -0.0225 |       -0.0213 |             0.3457 |  -0.0313 |      -0.041  |       -0.0207 |
| our stats model         |       0.03 |  80346 |    -0.0206 |      -0.0213 |       -0.02   |             0.3549 |  -0.033  |      -0.0428 |       -0.0235 |
| our stats model         |       0.05 |  70837 |    -0.0196 |      -0.0203 |       -0.019  |             0.3614 |  -0.0361 |      -0.0472 |       -0.0233 |
| our stats model         |       0.1  |  49190 |    -0.0169 |      -0.0177 |       -0.016  |             0.3784 |  -0.0396 |      -0.0542 |       -0.0229 |
| our hybrid model        |       0    |  87358 |    -0.0214 |      -0.0219 |       -0.0209 |             0.3338 |  -0.0107 |      -0.02   |       -0.0014 |
| our hybrid model        |       0.03 |  67260 |    -0.019  |      -0.0196 |       -0.0184 |             0.346  |  -0.0115 |      -0.0232 |        0.0002 |
| our hybrid model        |       0.05 |  53790 |    -0.0175 |      -0.0183 |       -0.0169 |             0.3543 |  -0.0064 |      -0.0213 |        0.0052 |
| our hybrid model        |       0.1  |  28239 |    -0.0138 |      -0.0149 |       -0.0129 |             0.3759 |  -0.0057 |      -0.026  |        0.013  |
| random picks (baseline) |     nan    |  98530 |    -0.0318 |     nan      |      nan      |             0.2943 | nan      |     nan      |      nan      |


Caveat: football-data.co.uk's 'opening' odds are a snapshot taken days before kickoff, not the very first price. In practice, sharp bettors are limited quickly at Pinnacle's opening lines.


## 4. Favourite-longshot bias at Pinnacle close

Home win. gap = actual - priced; z > 3 is a real miscalibration. ROI = flat bet on every home win in the bucket at that price.

| Pinnacle prob   |     n |   priced (prop.) |   priced (power) |   actual |   z (prop.) |   z (power) |   ROI Pinnacle close |   ci_low |   ci_high |   ROI best close (2019+) |   n best |
|:----------------|------:|-----------------:|-----------------:|---------:|------------:|------------:|---------------------:|---------:|----------:|-------------------------:|---------:|
| 0-0.2           |  6154 |           0.1428 |           0.1375 |   0.1381 |     -1.0433 |      0.1378 |              -0.0763 |  -0.1384 |   -0.0151 |                  -0.0807 |     3191 |
| 0.2-0.4         | 35724 |           0.3212 |           0.3208 |   0.316  |     -2.0906 |     -1.9309 |              -0.0464 |  -0.0607 |   -0.0324 |                  -0.0235 |    17796 |
| 0.4-0.6         | 42793 |           0.4849 |           0.4895 |   0.4859 |      0.3986 |     -1.5015 |              -0.0268 |  -0.0358 |   -0.0173 |                  -0.0026 |    19405 |
| 0.6-0.7         |  8002 |           0.6437 |           0.654  |   0.6543 |      1.988  |      0.0626 |              -0.0144 |  -0.0289 |    0.0012 |                   0.0267 |     3727 |
| 0.7-0.8         |  3887 |           0.7442 |           0.7587 |   0.7785 |      4.9002 |      2.8819 |               0.0144 |  -0.0025 |    0.0328 |                   0.0238 |     1889 |
| 0.8-1           |  1970 |           0.8441 |           0.8628 |   0.8711 |      3.2973 |      1.0673 |               0.0009 |  -0.0168 |    0.0166 |                   0.0133 |      909 |



## 5. Line movement (Pinnacle open -> close, home win probability)

If the close is efficient, `actual - close` is ~0 in every bucket: the move is fully priced.

| move            |     n |   open |   close |   actual |   actual - close |       z |
|:----------------|------:|-------:|--------:|---------:|-----------------:|--------:|
| (-1.0, -0.05]   |  5232 | 0.4715 |  0.4039 |   0.3937 |          -0.0102 | -1.4989 |
| (-0.05, -0.02]  | 18121 | 0.4326 |  0.4008 |   0.3975 |          -0.0033 | -0.9129 |
| (-0.02, -0.005] | 18964 | 0.4172 |  0.4051 |   0.4109 |           0.0059 |  1.6484 |
| (-0.005, 0.005] | 14993 | 0.4224 |  0.4223 |   0.4303 |           0.0079 |  1.9648 |
| (0.005, 0.02]   | 18424 | 0.4387 |  0.4508 |   0.4526 |           0.0019 |  0.5062 |
| (0.02, 0.05]    | 17678 | 0.448  |  0.4799 |   0.4784 |          -0.0015 | -0.3896 |
| (0.05, 1.0]     |  5118 | 0.4463 |  0.5142 |   0.5092 |          -0.005  | -0.714  |


- Does our **stats** model predict the move? corr(model - Pinnacle open, close - open) = **0.106**

- Does our **hybrid** model predict the move? corr(model - Pinnacle open, close - open) = **0.094**


A positive correlation means the model anticipates where informed money pushes the line.
