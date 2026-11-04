"""Dynamic Poisson goal ratings (attack / defence), for goals markets such as over/under 2.5.

Expected goals:  lam_home = exp(mu_L + hfa_L/2 + att_home + dfn_away)
                 lam_away = exp(mu_L - hfa_L/2 + att_away + dfn_home)
(att > 0 scores more than the league average; dfn > 0 concedes more.)
After each match every term takes a gradient step on the Poisson log-likelihood, whose gradient in
log-space is simply (observed - expected). League scoring level mu_L and home advantage hfa_L are
learned online too, so leagues and eras with different goal rates are handled automatically.

Shot weighting (w_shots): the "observed" signal can blend goals with shots on target x the league's
current goals-per-shot-on-target rate (a crude expected-goals proxy). Missing shots -> goals only.
All outputs are PRE-match values.

Run directly to tune on 2005-2014 and validate on 2015-2025 -> reports/goal_ratings_tuning.md
"""
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARAMS_FILE = ROOT / "data" / "processed" / "goal_rating_params.json"
DEFAULT = dict(k=0.04, k_league=0.002, regress=0.2, k_early=1.5, n_early=10, w_shots=0.0)


def p_over(lam_total, line=2.5):
    """P(total goals > line) for Poisson total (independent Poisson home/away goals)."""
    n = int(np.floor(line))
    term = np.exp(-lam_total); cdf = term.copy()
    for i in range(1, n + 1):
        term = term * lam_total / i; cdf = cdf + term
    return 1 - cdf


def compute_goal_ratings(df, k=0.04, k_league=0.002, regress=0.2, k_early=1.5, n_early=10, w_shots=0.0):
    """df sorted by date with: league, season, home_id, away_id, hg, ag, hst, ast.
    Returns DataFrame (index aligned to df) with pre-match lam_h, lam_a, att/dfn for both teams."""
    att, dfn, last_season, n_season = {}, {}, {}, {}
    mu, hfa, gps = {}, {}, {}  # per league: log goals/team/game, log home adv, goals per shot on target
    n = len(df)
    out = np.empty((n, 6))
    hst = df.hst.to_numpy(float); ast = df.ast.to_numpy(float)
    rows = zip(df.league.to_numpy(), df.season.to_numpy(), df.home_id.to_numpy(), df.away_id.to_numpy(),
               df.hg.to_numpy(float), df.ag.to_numpy(float))
    for i, (lg, s, h, a, hg, ag) in enumerate(rows):
        if lg not in mu:
            mu[lg], hfa[lg], gps[lg] = np.log(1.35), 0.25, 0.30
        for t, new_att, new_dfn in ((h, -0.10, 0.10), (a, -0.10, 0.10)):
            if t not in att:
                att[t], dfn[t] = new_att, new_dfn  # unseen teams are usually promoted -> weaker
                last_season[t], n_season[t] = s, 0
            elif last_season[t] != s:
                att[t] *= 1 - regress; dfn[t] *= 1 - regress
                last_season[t], n_season[t] = s, 0
        lh = np.exp(mu[lg] + hfa[lg] / 2 + att[h] + dfn[a])
        la = np.exp(mu[lg] - hfa[lg] / 2 + att[a] + dfn[h])
        out[i] = (lh, la, att[h], dfn[h], att[a], dfn[a])
        # observed signal: goals, optionally blended with shots-on-target x league conversion
        oh, oa = hg, ag
        if hst[i] == hst[i] and ast[i] == ast[i]:
            tot_sot = hst[i] + ast[i]
            if tot_sot > 0:
                gps[lg] += 0.01 * ((hg + ag) / tot_sot - gps[lg])
            if w_shots > 0:
                oh = (1 - w_shots) * hg + w_shots * hst[i] * gps[lg]
                oa = (1 - w_shots) * ag + w_shots * ast[i] * gps[lg]
        eh, ea = oh - lh, oa - la
        kh = k * (k_early if n_season[h] < n_early else 1.0)
        ka = k * (k_early if n_season[a] < n_early else 1.0)
        att[h] += kh * eh; dfn[a] += ka * eh
        att[a] += ka * ea; dfn[h] += kh * ea
        mu[lg] += k_league * (eh + ea) / 2
        hfa[lg] += k_league * (eh - ea)
        n_season[h] += 1; n_season[a] += 1
    return pd.DataFrame(out, index=df.index, columns=["lam_h", "lam_a", "att_h", "dfn_h", "att_a", "dfn_a"])


# ---------------------------------------------------------------- tuning
TUNE, VALID = (2005, 2014), (2015, 2025)
_df = None


def _load():
    global _df
    f = pd.read_parquet(ROOT / "data/processed/features.parquet",
                        columns=["league", "season", "home_id", "away_id", "hg", "ag", "hst", "ast"])
    _df = f.reset_index(drop=True)


def _bll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def _eval(params):
    if _df is None:
        _load()
    g = compute_goal_ratings(_df, **params)
    y = ((_df.hg + _df.ag) > 2.5).to_numpy(float)
    p = p_over((g.lam_h + g.lam_a).to_numpy())
    s = _df.season.to_numpy()
    out = dict(params)
    for name, m in [("tune", (s >= TUNE[0]) & (s <= TUNE[1])), ("valid", (s >= VALID[0]) & (s <= VALID[1]))]:
        out[name] = _bll(p[m], y[m])
        out[f"{name}_mean_pred"] = p[m].mean(); out[f"{name}_actual"] = y[m].mean()
    return out


def tune():
    grid = [dict(k=k, k_league=0.002, regress=rg, k_early=ke, w_shots=w)
            for k, rg, ke, w in itertools.product([0.007, 0.01, 0.015, 0.02, 0.04], [0.0, 0.2, 0.4, 0.6, 0.8], [1.0, 1.5], [0.0, 0.3, 0.5, 0.7])]
    with Pool(4, initializer=_load) as pool:
        R = pd.DataFrame(pool.map(_eval, grid))
    best = R.sort_values("tune").iloc[0]
    _load()
    y = ((_df.hg + _df.ag) > 2.5).to_numpy(float); s = _df.season.to_numpy()
    base_rate = {n: _bll(np.full(m.sum(), y[(s >= TUNE[0]) & (s <= TUNE[1])].mean()), y[m])
                 for n, m in [("tune", (s >= TUNE[0]) & (s <= TUNE[1])), ("valid", (s >= VALID[0]) & (s <= VALID[1]))]}
    cols = ["k", "regress", "k_early", "w_shots", "tune", "valid", "valid_mean_pred", "valid_actual"]
    md = ["# Goal-rating tuning (over/under 2.5)\n",
          f"Chosen on {TUNE[0]}-{TUNE[1]} only; `valid` = {VALID[0]}-{VALID[1]}. Binary log loss of the raw Poisson "
          f"over-2.5 probability (no calibration), all {_df.league.nunique()} leagues. Lower is better.\n",
          f"Baseline (always predict the tuning-period over rate): tune {base_rate['tune']:.4f}, valid {base_rate['valid']:.4f}\n",
          "## Best setting\n", R.sort_values("tune").head(1)[cols].round(4).to_markdown(index=False), "\n",
          "## Best per shot weight\n", R.loc[R.groupby("w_shots").tune.idxmin()][cols].round(4).to_markdown(index=False), "\n",
          "## Best per season reset\n", R.loc[R.groupby("regress").tune.idxmin()][cols].round(4).to_markdown(index=False), "\n"]
    (ROOT / "reports/goal_ratings_tuning.md").write_text("\n".join(md))
    params = {c: float(best[c]) for c in ["k", "k_league", "regress", "k_early", "w_shots"]}
    PARAMS_FILE.write_text(json.dumps(params, indent=2))
    print("\n".join(md)); print(params)


def load_params():
    return json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else DEFAULT


if __name__ == "__main__":
    tune()
