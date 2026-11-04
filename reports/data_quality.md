# Data quality log

## 1. Swapped day/month in 16 non-European leagues (fixed)

**Found:** 2026-11-04, while matching the xgabora dataset to football-data.co.uk.

**Problem:** in the xgabora `Matches.csv`, every match in ARG, AUT, BRA, CHN, DEN, FIN, IRL, JAP, MEX,
NOR, POL, ROM, RUS, SUI, SWE and USA played on day 1-12 of a month has its day and month swapped
(the source's `dd/mm/yyyy` was read as `mm/dd/yyyy` whenever that gave a valid date).
Example: Brazil, America MG 2-1 Vitoria is stored as 2018-01-05; it was played on 2018-05-01.

**Evidence:** of 19,392 such matches (day <= 12, day != month), 19,384 line up with football-data.co.uk
only after swapping; 8 matched without swapping, consistent with coincidences (same fixture, same score on
the mirrored date). The 22 European leagues are unaffected.

**Impact before the fix (v1/v2 results):**
- About 19.4k of 54.5k matches in those leagues sat at the wrong point in the timeline, so some
  "pre-match" form and ratings for them were computed in the wrong order, and rest days were wrong.
- In autumn-spring leagues among them (AUT, DEN, MEX, POL, ROM, RUS, SUI) some matches were assigned to
  the wrong season.
- The 22 European leagues (about 77% of all matches) were not affected, which is why headline results
  barely changed.

**Fix:** `src/ingest.py::fix_swapped_dates` swaps day and month for those leagues when day <= 12.
After the fix, football-data.co.uk matching rose from 140k to 160k matches and Pinnacle closing-odds
coverage since 2012-13 rose from 82.8% to 94.9%.

## 2. Team-name differences between sources (fixed)
Mexico: `Tigres UANL` / `UNAM Pumas` (football-data.co.uk) = `U.A.N.L.- Tigres` / `U.N.A.M.- Pumas`
(xgabora). Mapped in `src/ingest.py::FD_TEAM`. All other team names match exactly.

## 3. Known gaps (not fixable from these sources)
- Pinnacle closing odds start in 2012-13 and thin out in 2025-26 (39% of 2025 matches), when
  football-data.co.uk's Pinnacle feed became incomplete. Average and best closing odds are complete there.
- Average and best *closing* odds only exist from 2019-20; Betfair Exchange closing odds only for the
  non-European leagues from 2024.
- Scottish League One and Two (SC2, SC3) have Pinnacle odds for only 64% of matches.
- Shots, corners and cards exist for about half of all matches (mainly European leagues).
