"""Configurable Elo-style team ratings.

Two ideas from the in-season test (reports/inseason_report.md):
  1. Faster season reset: ratings are pulled harder toward the mean at the start of a season
     (`regress`) and move faster during each team's first `n_early` matches (`k_early`).
  2. Shot weighting: the "score" a team earns from a match blends the result with its share of
     shots on target (`w_shots`). Shot share is a more stable signal than goals or points.
     When shot data is missing for a match, the result alone is used.
Every rating returned is the PRE-match value (no leakage).
"""
import numpy as np
import pandas as pd


def compute_ratings(df: pd.DataFrame, k=20.0, hfa=60.0, regress=0.2, k_early=1.0, n_early=10,
                    w_shots=0.0, gd_mult=True, init=1450.0, regress_to="global"):
    """df must be sorted by date and have: league, home_id, away_id, season, hg, ag, hst, ast.
    regress_to: "global" pulls toward 1500 at a new season; "league" pulls toward the mean rating
    of the teams in the league the team plays in THIS season (the team list of a season is known
    before it starts, so this uses no future results).
    Returns (pre_home, pre_away, home_season_idx, away_season_idx)."""
    r, last_season, n_in_season = {}, {}, {}
    league_teams = (pd.concat([df[["league", "season", "home_id"]].rename(columns={"home_id": "t"}),
                               df[["league", "season", "away_id"]].rename(columns={"away_id": "t"})])
                    .drop_duplicates().groupby(["league", "season"]).t.apply(list).to_dict()
                    if regress_to == "league" else {})
    league_mean = {}
    n = len(df)
    pre_h = np.empty(n); pre_a = np.empty(n)
    idx_h = np.empty(n, dtype=np.int32); idx_a = np.empty(n, dtype=np.int32)
    hst = df.hst.to_numpy(dtype=float); ast = df.ast.to_numpy(dtype=float)
    cols = zip(df.home_id.to_numpy(), df.away_id.to_numpy(), df.hg.to_numpy(), df.ag.to_numpy(),
               df.season.to_numpy(), df.league.to_numpy())
    for i, (h, a, hg, ag, s, lg) in enumerate(cols):
        if regress_to == "league" and (lg, s) not in league_mean:
            # snapshot at the league-season's first match: current ratings of its teams
            known = [r[t] for t in league_teams[(lg, s)] if t in r]
            league_mean[(lg, s)] = float(np.mean(known)) if known else 1500.0
        target = league_mean.get((lg, s), 1500.0)
        for t in (h, a):
            if t not in r:
                r[t] = init if regress_to == "global" else target - 50.0
                last_season[t] = s; n_in_season[t] = 0
            elif last_season[t] != s:
                r[t] = r[t] * (1 - regress) + target * regress
                last_season[t] = s; n_in_season[t] = 0
        rh, ra = r[h], r[a]
        pre_h[i], pre_a[i] = rh, ra
        idx_h[i], idx_a[i] = n_in_season[h], n_in_season[a]
        exp_h = 1.0 / (1.0 + 10.0 ** ((ra - rh - hfa) / 400.0))
        res = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        gd = abs(hg - ag)
        mult = (1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8) if gd_mult else 1.0
        score = res
        if w_shots > 0 and hst[i] == hst[i] and ast[i] == ast[i]:  # not NaN
            tot = hst[i] + ast[i]
            share = hst[i] / tot if tot > 0 else 0.5
            score = (1 - w_shots) * res + w_shots * share
            mult = (1 - w_shots) * mult + w_shots * 1.0
        base = k * mult * (score - exp_h)
        kh = k_early if n_in_season[h] < n_early else 1.0
        ka = k_early if n_in_season[a] < n_early else 1.0
        r[h] = rh + base * kh
        r[a] = ra - base * ka
        n_in_season[h] += 1; n_in_season[a] += 1
    return pre_h, pre_a, idx_h, idx_a
