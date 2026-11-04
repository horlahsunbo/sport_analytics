# Soccer prediction & "when do stats fail" research

Goal: build match-outcome probabilities good enough to compete with bookmaker odds, then study
**when stats fail** and **which teams follow the stats**.

## Pipeline
```
src/download.py           raw sources -> data/raw/
src/ingest.py             canonical match table -> data/processed/matches.parquet
src/features.py           leakage-free pre-match features (own Elo, EWMA form, rest, de-vigged market)
src/model.py              walk-forward by season: market vs stats-only vs hybrid LightGBM + betting sim
src/analysis.py           failure analysis: by league / favourite strength / phase, market calibration,
                          stats-vs-market disagreement, team predictability persistence
src/analysis_controls.py  robustness: favourite-status-controlled persistence, bias profitability
src/inseason.py           in-season test: first 10 matches of a season -> predict matches 11-15
src/ratings.py            Elo engine: season reset toward league mean, early-season speed-up, shot rating
src/tune_ratings.py       tunes rating parameters on 2005-2014, validates on 2015-2025
src/compare_versions.py   paired comparison of two model versions on identical matches
src/closing.py            closing-odds tests: Pinnacle benchmark, betting at closing prices, CLV, movement
src/goal_ratings.py       dynamic Poisson attack/defence ratings for goals markets (+ tuning)
src/totals.py             over/under 2.5 goals: walk-forward model and market tests
```
Run everything: `./run_all.sh`. Reports land in `reports/`.

## Data
- 238,854 matches, 38 leagues, 2000-2026 (xgabora/Club-Football-Match-Data, derived from
  football-data.co.uk + clubelo). Average pre-match odds for 99% of matches; shots/corners for ~51%.
- football-data.co.uk (2012-13 onward): Pinnacle opening + closing odds on 94.9% of matches, average/best
  closing odds from 2019-20, Betfair Exchange closing odds for non-European leagues from 2024.
  Over/under 2.5: average/best odds from 2012-13, Pinnacle opening/closing from 2019-20 (22 European
  leagues). Asian handicap lines and odds from 2019-20 (loaded, not yet modelled).
- Data issues found and fixed (swapped day/month in 16 leagues, 19,393 matches): `reports/data_quality.md`.

## Findings so far (out-of-sample, 175,537 matches, 2010-2025)
| predictor | log loss | accuracy |
|---|---|---|
| bookmaker market (average pre-match odds) | **1.0048** | 50.1% |
| hybrid (stats + market) | 1.0073 | 50.0% |
| stats only (v2, dates fixed) | 1.0194 | 49.1% |
| stats only (v1) | 1.0221 | 48.8% |

On the 98,530 matches that also have Pinnacle odds: Pinnacle closing **1.0009**, Pinnacle opening 1.0040,
average pre-match 1.0047, our hybrid 1.0071, our stats model 1.0156.
| base rates | 1.0728 | 44.1% |

1. **Stats alone do not beat the market**, in any league or season. When the stats model disagrees with
   the market, actual outcomes track the market almost exactly.
2. **The market's real weakness is the favourite-longshot bias**: strong favourites win more often than
   priced (z = 5-7), longshots less. At *best available* odds, backing 70%+ favourites returned ~+2.5%
   (CI +1.2% to +3.7%), stable across 2010-14 / 2015-19 / 2020-25. At *average* odds it loses.
3. **"Predictable teams" are mostly just strong favourites.** Raw predictability persists year-to-year
   (r = 0.34), but after controlling for what the odds implied, persistence drops to r = 0.04 (small but
   above chance). The teams that "follow the odds" (Benfica, Porto, Olympiakos, Atletico...) are largely
   the favourite-longshot bias again.
4. Least predictable contexts: lower divisions (E3, I2, D2, F2), evenly matched games, high-draw leagues.

5. **In-season form is more stable than carry-over form, but the market already prices it**
   (`reports/inseason_report.md`). Same-size windows: points/game correlate 0.42 within a season vs 0.26
   across the summer break; shot difference 0.65 vs 0.46 (shots are the most stable signal). A model on
   the first 10 matches beats our carry-over Elo/form model on matches 11-15 (log loss 1.0236 vs 1.0275),
   but "points above market expectation" has zero persistence (r = -0.02), so in-season form gives no
   edge vs the odds. No difference between predicting 1 vs 5 matches ahead.

6. **v2: tuned season reset + shot rating + season-to-date stats** (`reports/ratings_tuning.md`,
   `reports/v2_vs_v1.md`). A *strong* reset hurts; the best is a light 10% pull toward the mean of the
   team's current league (not a global average, which erased the gap between divisions), plus 1.5x faster
   updates in each team's first 10 matches. A separate rating learned from shots-on-target share adds
   information on top. Stats-only log loss improved 1.0221 -> 1.0199 (95% CI -0.0027 to -0.0018, better in
   15 of 16 seasons), closing 13% of the gap to the market. The hybrid did not move (1.0076 -> 1.0075):
   the market already contains this information.

7. **Closing odds: no edge against the sharpest price** (`reports/closing_report.md`). Pinnacle's closing
   line is far sharper than the average odds we benchmarked against before (log loss 1.0009 vs 1.0047),
   so our real gap to the market is about 0.006, not 0.0025. Betting our model at Pinnacle closing odds
   loses 2-5%. At the *best* closing odds across bookmakers (2019+) the hybrid is about +0.5% to +1.1%,
   but every confidence interval includes zero. Closing line value is negative even when taking Pinnacle's
   opening price (-1.4% to -2.2%), though better than random picks (-3.2%): the model has a small sense of
   where the line will move (correlation 0.1), but not enough to beat the margin.
8. **The favourite-longshot bias was mostly an artefact.** With power de-vig at Pinnacle close, favourites
   are close to fairly priced (largest z = 2.9). Backing 70-80% home favourites returns +1.4% at Pinnacle
   close (CI -0.2% to +3.0%, not significant) and +2.4% at best close. The +2.5% earlier came from average
   pre-match odds, which include softer bookmakers.
9. **Closing prices are efficient.** After the line moves, results match the closing probability in every
   movement bucket (all |z| < 2).

10. **Over/under 2.5 goals: same result, no edge** (`reports/totals_report.md`, 106,697 test matches;
    46,823 with Pinnacle odds, 2019-2025). Log loss: Pinnacle close 0.6746, Pinnacle open 0.6770, average
    odds 0.6772, our hybrid 0.6774, our stats model 0.6802 (coin flip 0.6931). Pinnacle's over/under price
    is well calibrated at every level (all |z| < 1.2). Betting loses at every price tested; CLV at
    Pinnacle's opening price is -2.8% to -3.1% (random side -3.9%). Lower divisions are the least
    efficient part of the market relative to our model (gap to close 0.0019 vs 0.0039 in top flights)
    and lose less (-0.5% to -3% vs -2% to -7%), but still lose. Scottish League One/Two show positive ROI
    on about 1,000 matches each, with confidence intervals well across zero: most likely chance.
    Goal ratings: a slow update speed and a moderate 20% season reset work best; the tuning surface is
    flat, so the exact setting matters little (`reports/goal_ratings_tuning.md`).

## Caveats
- "Best odds" is the max across many bookmakers: an upper bound. Stale lines exist and sharp-winning
  accounts get limited.
- football-data.co.uk's "opening" odds are a snapshot days before kickoff, not the very first price.
- `reports/inseason_report.md` was produced with v1 features and the pre-fix dates; re-run `src/inseason.py`
  to refresh it.

## Roadmap (more data where it can actually add signal)
- [x] Faster rating reset at season start + shot-weighted ratings (v2)
- [x] Closing odds (Pinnacle) to benchmark against the sharpest price
- [ ] xG / shot-quality data (Understat, FBref) - stronger than shots/goals
- [ ] Lineups, injuries, transfers/market value (Transfermarkt) - information markets price late
- [x] Power de-vig (in closing.py)
- [ ] Probability calibration of the hybrid model
- [x] Over/under 2.5 goals (no edge; lower divisions least efficient but still negative)
- [ ] Asian handicap (data loaded from 2019-20)
- [ ] Other sports via the same ingest -> features -> walk-forward -> analysis pattern
