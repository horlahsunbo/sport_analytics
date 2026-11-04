# In-season test: first 10 matches -> next 5

## 1. Is in-season form more stable than season-to-season form?

Correlation across team-seasons (teams present in both seasons). Columns = what we try to predict.

| metric   |   teams |   first10 -> next5 (same season) |   prev last10 -> next5 |   prev full -> next5 |   first10 -> next10 (same season) |   prev last10 -> first10 (across break) |
|:---------|--------:|---------------------------------:|-----------------------:|---------------------:|----------------------------------:|----------------------------------------:|
| pts      |   11664 |                            0.332 |                  0.204 |                0.256 |                             0.419 |                                   0.258 |
| gd       |   11664 |                            0.399 |                  0.254 |                0.319 |                             0.488 |                                   0.323 |
| sd       |    5398 |                            0.577 |                  0.372 |                0.436 |                             0.649 |                                   0.456 |
| sotd     |    5350 |                            0.497 |                  0.318 |                0.396 |                             0.585 |                                   0.409 |
| xs       |   11586 |                           -0.021 |                  0.017 |               -0.004 |                            -0.018 |                                   0.006 |

Metric key: pts = points/game, gd = goal diff, sd = shot diff, sotd = shots-on-target diff, xs = points ABOVE what the odds expected (i.e. performance the market did not price).

Fair same-size comparison of the season break: `prev last10 -> first10` (crosses the summer) vs `first10 -> next10` (both windows inside one season).


## 2. Predicting matches 11-15 (walk-forward, 23,099 matches, seasons 2008-2025)

|            |   logloss |   accuracy |
|:-----------|----------:|-----------:|
| market     |    1.0055 |     0.5012 |
| inseason10 |    1.0236 |     0.488  |
| carryover  |    1.0275 |     0.4822 |
| both       |    1.0236 |     0.4886 |
| both+mkt   |    1.0138 |     0.4961 |


### Log loss by horizon (1 = the very next match after the first 10)

|   matches ahead |   market |   inseason10 |   carryover |   both |   both+mkt |
|----------------:|---------:|-------------:|------------:|-------:|-----------:|
|               1 |   1.007  |       1.0251 |      1.0311 | 1.0262 |     1.016  |
|               2 |   1.0096 |       1.0237 |      1.0309 | 1.026  |     1.0163 |
|               3 |   1.0056 |       1.0287 |      1.029  | 1.0261 |     1.014  |
|               4 |   1.0001 |       1.0172 |      1.0201 | 1.0182 |     1.0099 |
|               5 |   1.006  |       1.0236 |      1.0279 | 1.0223 |     1.0139 |


### Betting check (flat stakes)

| model      |   min_edge |   bets |   roi_avg_odds |   roi_best_odds |
|:-----------|-----------:|-------:|---------------:|----------------:|
| inseason10 |       0.03 |  18175 |        -0.0537 |          0.0086 |
| inseason10 |       0.08 |  14142 |        -0.058  |          0.0065 |
| both       |       0.03 |  17867 |        -0.0608 |         -0.0017 |
| both       |       0.08 |  13665 |        -0.0593 |          0.002  |
| both+mkt   |       0.03 |  14934 |        -0.0549 |          0.0009 |
| both+mkt   |       0.08 |   9771 |        -0.0568 |          0.0002 |
