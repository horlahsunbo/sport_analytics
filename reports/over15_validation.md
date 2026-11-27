# Over 1.5 goals: validation

Walk-forward, test seasons 2015-2025, 83,322 matches with over/under 2.5 odds (22 leagues). Over-1.5 rate 74.5%. Binary log loss, lower is better.

There are no over 1.5 odds in our data, so the market benchmark is derived from the over/under 2.5 price and **betting returns cannot be tested**: the comparison is about probabilities only.

|                            |   logloss |   brier |
|:---------------------------|----------:|--------:|
| base rate                  |    0.5685 |  0.1903 |
| raw Poisson goal ratings   |    0.5605 |  0.1874 |
| stats model                |    0.5592 |  0.1869 |
| market (from O/U 2.5 odds) |    0.5571 |  0.1861 |
| hybrid model               |    0.5575 |  0.1862 |



## Calibration (hybrid): when it says X%, how often does it happen?

| p_hybrid    |     n |   predicted |   actual |     gap |
|:------------|------:|------------:|---------:|--------:|
| (0.0, 0.6]  |   610 |      0.5854 |   0.6164 |  0.031  |
| (0.6, 0.65] |  5477 |      0.6316 |   0.6431 |  0.0114 |
| (0.65, 0.7] | 15003 |      0.6776 |   0.676  | -0.0017 |
| (0.7, 0.75] | 25541 |      0.7267 |   0.7284 |  0.0018 |
| (0.75, 0.8] | 21545 |      0.7727 |   0.7748 |  0.0021 |
| (0.8, 0.85] | 10063 |      0.8223 |   0.8189 | -0.0034 |
| (0.85, 0.9] |  4580 |      0.8698 |   0.8755 |  0.0057 |
| (0.9, 1.0]  |   503 |      0.9097 |   0.9066 | -0.0031 |



## Most confident picks

| model   | top                |   matches |   avg predicted |   hit rate |
|:--------|:-------------------|----------:|----------------:|-----------:|
| stats   | 10% most confident |      8333 |          0.8478 |     0.851  |
| stats   | 5% most confident  |      4167 |          0.8669 |     0.874  |
| market  | 10% most confident |      8336 |          0.8552 |     0.861  |
| market  | 5% most confident  |      4168 |          0.8757 |     0.8832 |
| hybrid  | 10% most confident |      8333 |          0.8605 |     0.8622 |
| hybrid  | 5% most confident  |      4167 |          0.8783 |     0.8826 |



## International breaks: are the first matches back different?

`post_break` = the league's first round after a pause of 11+ days (excluding season start and winter breaks). `actual - predicted` > 0 means more over-1.5 results than the model expected.

|                            | post-break round             | all other matches            |
|:---------------------------|:-----------------------------|:-----------------------------|
| matches                    | 7194                         | 76128                        |
| over-1.5 rate              | 0.7583                       | 0.7434                       |
| market: actual - predicted | +0.0105 (+0.0008 to +0.0205) | +0.0016 (-0.0017 to +0.0042) |
| stats: actual - predicted  | +0.0042 (-0.0051 to +0.0131) | +0.0038 (+0.0010 to +0.0070) |
| hybrid: actual - predicted | +0.0019 (-0.0076 to +0.0116) | +0.0016 (-0.0014 to +0.0042) |
| hybrid logloss             | 0.5425                       | 0.5589                       |
| market logloss             | 0.5427                       | 0.5585                       |

(95% bootstrap CI in brackets)


Raw over-1.5 rate across all history (2000-2026, no model): 

|            |   size |   mean |
|:-----------|-------:|-------:|
| other      | 230308 | 0.738  |
| post-break |  20274 | 0.7555 |

