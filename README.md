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
```
Run everything: `./run_all.sh`. Reports land in `reports/`.

## Data (v1)
238,854 matches, 38 leagues, 2000-2026 (xgabora/Club-Football-Match-Data, derived from
football-data.co.uk + clubelo). Odds for 99% of matches; shots/corners for ~51%.

## Findings so far (out-of-sample, 175,537 matches, 2010-2025)
| predictor | log loss | accuracy |
|---|---|---|
| bookmaker market | **1.0048** | 50.1% |
| hybrid (stats + market) | 1.0075 | 50.0% |
| stats only (v2) | 1.0199 | 49.0% |
| stats only (v1) | 1.0221 | 48.8% |
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

## Caveats
- "Best odds" is the max across many bookmakers: an upper bound. Stale lines exist and sharp-winning
  accounts get limited. Treat finding #2 as a lead to validate, not a strategy.
- Odds are pre-match (not closing). Closing odds are the gold-standard benchmark.
- Proportional de-vig slightly exaggerates favourite-longshot bias; Shin/power de-vig should be tested.

## Roadmap (more data where it can actually add signal)
- [x] Faster rating reset at season start + shot-weighted ratings (v2)
- [ ] Closing odds (Pinnacle) to benchmark against the sharpest price
- [ ] xG / shot-quality data (Understat, FBref) - stronger than shots/goals
- [ ] Lineups, injuries, transfers/market value (Transfermarkt) - information markets price late
- [ ] Shin / power de-vig; probability calibration of the hybrid model
- [ ] Over/under & Asian handicap markets (less efficient than 1X2 in lower leagues)
- [ ] Other sports via the same ingest -> features -> walk-forward -> analysis pattern
