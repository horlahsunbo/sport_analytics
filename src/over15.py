"""Over 1.5 goals: validation + predictions for upcoming fixtures.

Part 1  Walk-forward test (train on seasons < S, test S, S = 2015..2025) of:
          market  : over 1.5 implied by the over/under 2.5 odds (Poisson total fitted to the de-vigged
                    over-2.5 price, then re-calibrated on past seasons). No 1.5 odds exist in our data.
          stats   : goal ratings + form + league scoring + break flags (no odds)
          hybrid  : stats + market
Part 2  International breaks: are post-break matches different, and does any model mis-price them?
Part 3  Train on all played matches, predict upcoming fixtures (data/raw/fixtures/*.csv).

Usage: python3 src/over15.py [YYYY-MM-DD]   (first fixture date to predict; default today)
Writes reports/over15_validation.md, predictions/<date>_over15.csv and reports/predictions_<date>_over15.md
"""
import datetime as dt
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from goal_ratings import compute_goal_ratings, load_params  # noqa: E402
from features import build as build_features  # noqa: E402
from ingest import season_of, FD_TEAM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=300,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7)
ROUNDS = 300
COUNTRY_CODE = {"Argentina": "ARG", "Austria": "AUT", "Brazil": "BRA", "China": "CHN", "Denmark": "DEN",
                "Finland": "FIN", "Ireland": "IRL", "Japan": "JAP", "Mexico": "MEX", "Norway": "NOR",
                "Poland": "POL", "Romania": "ROM", "Russia": "RUS", "Sweden": "SWE", "Switzerland": "SUI",
                "USA": "USA"}


# ---------------------------------------------------------------- helpers
def bll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def boot(x, n=500):
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def p_at_least2(lam):
    return 1 - np.exp(-lam) * (1 + lam)


def lam_from_over25(p25):
    """Total-goals Poisson mean whose P(total >= 3) equals p25 (vectorised bisection)."""
    lo, hi = np.full(len(p25), 0.05), np.full(len(p25), 9.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        f = 1 - np.exp(-mid) * (1 + mid + mid ** 2 / 2)
        lo = np.where(f < p25, mid, lo); hi = np.where(f < p25, hi, mid)
    return (lo + hi) / 2


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- features
def add_over15_features(f):
    f = f.copy()
    played = f.hg.notna()
    f["over15"] = np.where(played, ((f.hg + f.ag) >= 2).astype(float), np.nan)
    g = compute_goal_ratings(f, **load_params())
    gs = compute_goal_ratings(f, k=0.01, k_league=0.002, regress=0.4, k_early=1.0, w_shots=0.3)
    f[["lam_h", "lam_a", "att_h", "dfn_h", "att_a", "dfn_a"]] = g.to_numpy()
    f["lam_tot"] = f.lam_h + f.lam_a
    f["p15_pois"] = p_at_least2(f.lam_tot)
    f["lam_tot_s"] = gs.lam_h.to_numpy() + gs.lam_a.to_numpy()
    f["p15_pois_s"] = p_at_least2(f.lam_tot_s)
    f["abs_elo_diff"] = f.elo_diff.abs()
    # league scoring context: prior played matches only (expanding mean skips unplayed NaNs)
    f["lg_over15_rate"] = f.groupby("league").over15.transform(lambda s: s.shift(1).expanding().mean())
    f["lg_season_over15_rate"] = f.groupby(["league", "season"]).over15.transform(lambda s: s.shift(1).expanding().mean())
    f["lg_season_n"] = f.groupby(["league", "season"]).cumcount()
    # market: de-vigged over 2.5 -> Poisson total -> P(>= 2 goals)
    oo = f.odds_o25.fillna(f.get("ou_avg_o")); ou = f.odds_u25.fillna(f.get("ou_avg_u"))
    f["mkt_o25"] = (1 / oo) / (1 / oo + 1 / ou)
    f["mkt_lam"] = np.where(f.mkt_o25.notna(), lam_from_over25(f.mkt_o25.fillna(0.5).to_numpy()), np.nan)
    f["mkt_p15_raw"] = p_at_least2(f.mkt_lam)
    return f


STATS = (["lam_h", "lam_a", "lam_tot", "p15_pois", "lam_tot_s", "p15_pois_s", "att_h", "dfn_h", "att_a", "dfn_a",
          "abs_elo_diff", "lg_over15_rate", "lg_season_over15_rate", "lg_season_n", "month", "h_rest_days",
          "a_rest_days", "post_break", "post_winter_break", "break_gap_days"]
         + [f"{s}_{c}_ew{hl}" for s in "ha" for c in ["gf", "ga", "sf", "sa", "stf", "sta"] for hl in (3, 10)])
MKT = ["mkt_o25", "mkt_lam", "mkt_p15_raw", "mkt_h", "mkt_a"]


def fit(train, feats):
    X = train[feats + ["league"]].copy(); X["league"] = X.league.astype("category")
    return lgb.train(PARAMS, lgb.Dataset(X, train.over15), num_boost_round=ROUNDS), X.league.cat.categories


def predict(model_cats, df, feats):
    model, cats = model_cats
    X = df[feats + ["league"]].copy()
    X["league"] = pd.Categorical(X.league.astype(str), categories=cats)
    return model.predict(X)


def fit_market_calibration(train):
    t = train.dropna(subset=["mkt_p15_raw"])
    return LogisticRegression().fit(logit(t.mkt_p15_raw.to_numpy())[:, None], t.over15)


def market_calibrated(lr, df):
    out = np.full(len(df), np.nan); ok = df.mkt_p15_raw.notna().to_numpy()
    out[ok] = lr.predict_proba(logit(df.mkt_p15_raw.to_numpy()[ok])[:, None])[:, 1]
    return out


# ---------------------------------------------------------------- part 1 + 2: validation
def validate(f):
    hist = f[f.over15.notna()]
    out = []
    for S in range(2015, 2026):
        tr, te = hist[hist.season < S], hist[(hist.season == S) & hist.mkt_p15_raw.notna()].copy()
        trm = tr[tr.mkt_p15_raw.notna()]
        te["p_market"] = market_calibrated(fit_market_calibration(trm), te)
        te["p_stats"] = predict(fit(tr, STATS), te, STATS)
        te["p_hybrid"] = predict(fit(trm, STATS + MKT), te, STATS + MKT)
        out.append(te)
        print(f"over1.5 validation season {S}: {len(te):,}")
    v = pd.concat(out); y = v.over15.to_numpy()
    base = hist[hist.season < 2015].over15.mean()
    M = {"base rate": np.full(len(v), base), "raw Poisson goal ratings": v.p15_pois.to_numpy(),
         "stats model": v.p_stats.to_numpy(), "market (from O/U 2.5 odds)": v.p_market.to_numpy(),
         "hybrid model": v.p_hybrid.to_numpy()}
    md = ["# Over 1.5 goals: validation\n",
          f"Walk-forward, test seasons 2015-2025, {len(v):,} matches with over/under 2.5 odds "
          f"({v.league.nunique()} leagues). Over-1.5 rate {y.mean():.1%}. Binary log loss, lower is better.\n",
          "There are no over 1.5 odds in our data, so the market benchmark is derived from the over/under 2.5 "
          "price and **betting returns cannot be tested**: the comparison is about probabilities only.\n",
          pd.DataFrame({k: {"logloss": bll(p, y), "brier": float(((p - y) ** 2).mean())} for k, p in M.items()}).T
          .round(4).to_markdown(), "\n"]
    # calibration of the hybrid
    b = pd.cut(v.p_hybrid, [0, .6, .65, .7, .75, .8, .85, .9, 1])
    cal = v.groupby(b, observed=True).agg(n=("over15", "size"), predicted=("p_hybrid", "mean"), actual=("over15", "mean"))
    cal["gap"] = cal.actual - cal.predicted
    md += ["\n## Calibration (hybrid): when it says X%, how often does it happen?\n", cal.round(4).to_markdown(), "\n"]
    # top picks
    rows = []
    for name, col in (("stats", "p_stats"), ("market", "p_market"), ("hybrid", "p_hybrid")):
        for q in (0.10, 0.05):
            thr = v[col].quantile(1 - q); sel = v[v[col] >= thr]
            rows.append({"model": name, "top": f"{q:.0%} most confident", "matches": len(sel),
                         "avg predicted": sel[col].mean(), "hit rate": sel.over15.mean()})
    md += ["\n## Most confident picks\n", pd.DataFrame(rows).round(4).to_markdown(index=False), "\n"]

    # part 2: international breaks
    pb = v.post_break == 1
    md.append("\n## International breaks: are the first matches back different?\n\n"
              "`post_break` = the league's first round after a pause of 11+ days (excluding season start and "
              "winter breaks). `actual - predicted` > 0 means more over-1.5 results than the model expected.\n")
    rows = {}
    for label, m in (("post-break round", pb), ("all other matches", ~pb)):
        r = {"matches": int(m.sum()), "over-1.5 rate": round(v.over15[m].mean(), 4)}
        for name, col in (("market", "p_market"), ("stats", "p_stats"), ("hybrid", "p_hybrid")):
            d = (v.over15 - v[col])[m].to_numpy(); lo, hi = boot(d, 300)
            r[f"{name}: actual - predicted"] = f"{d.mean():+.4f} ({lo:+.4f} to {hi:+.4f})"
        r["hybrid logloss"] = round(bll(v.p_hybrid[m].to_numpy(), v.over15[m].to_numpy()), 4)
        r["market logloss"] = round(bll(v.p_market[m].to_numpy(), v.over15[m].to_numpy()), 4)
        rows[label] = r
    md += [pd.DataFrame(rows).to_markdown(), "\n(95% bootstrap CI in brackets)\n"]
    raw = hist.groupby(hist.post_break == 1).over15.agg(["size", "mean"])
    raw.index = ["other", "post-break"]
    md += ["\nRaw over-1.5 rate across all history (2000-2026, no model): \n", raw.round(4).to_markdown(), "\n"]
    (ROOT / "reports/over15_validation.md").write_text("\n".join(md))
    print("\n".join(md))
    return v


# ---------------------------------------------------------------- part 3: upcoming fixtures
def load_fixtures(start):
    fx = pd.read_csv(ROOT / "data/raw/fixtures/fixtures.csv", encoding="utf-8-sig")
    eu = pd.DataFrame({"league": fx.Div, "date": pd.to_datetime(fx.Date, dayfirst=True), "time": fx.get("Time"),
                       "home": fx.HomeTeam.str.strip(), "away": fx.AwayTeam.str.strip(),
                       "odds_h": fx.AvgH, "odds_d": fx.AvgD, "odds_a": fx.AvgA,
                       "max_h": fx.MaxH, "max_d": fx.MaxD, "max_a": fx.MaxA,
                       "odds_o25": fx["Avg>2.5"], "odds_u25": fx["Avg<2.5"],
                       "ou_max_o": fx.get("Max>2.5"), "ou_max_u": fx.get("Max<2.5")})
    nx = pd.read_csv(ROOT / "data/raw/fixtures/new_league_fixtures.csv", encoding="utf-8-sig")
    ex = pd.DataFrame({"league": nx.Country.map(COUNTRY_CODE), "date": pd.to_datetime(nx.Date, dayfirst=True),
                       "time": nx.get("Time"), "home": nx.Home.str.strip().replace(FD_TEAM),
                       "away": nx.Away.str.strip().replace(FD_TEAM),
                       "odds_h": nx.AvgH, "odds_d": nx.AvgD, "odds_a": nx.AvgA,
                       "max_h": nx.MaxH, "max_d": nx.MaxD, "max_a": nx.MaxA})
    fx = pd.concat([eu, ex], ignore_index=True).dropna(subset=["league"])
    return fx[fx.date >= pd.Timestamp(start)]


def predict_upcoming(start):
    hist = pd.read_parquet(ROOT / "data/processed/matches.parquet")
    fx = load_fixtures(start)
    known = set(zip(hist.league, hist.date, hist.home, hist.away))
    fx = fx[[k not in known for k in zip(fx.league, fx.date, fx.home, fx.away)]].copy()
    # teams with no history in that league's country would get default ratings: flag them
    fx["is_fixture"] = True
    fx["season"] = season_of(fx.league, fx.date)
    fx["match_id"] = np.arange(len(fx)) + hist.match_id.max() + 1
    allm = pd.concat([hist.assign(is_fixture=False), fx], ignore_index=True).sort_values(["date", "match_id"])
    allm = allm.reset_index(drop=True)
    f = add_over15_features(build_features(allm))
    train = f[f.over15.notna()]
    up = f[f.is_fixture == True].copy()  # noqa: E712
    m_stats = fit(train, STATS)
    trm = train[train.mkt_p15_raw.notna()]
    m_hyb = fit(trm, STATS + MKT)
    lr = fit_market_calibration(trm)
    up["p_stats"] = predict(m_stats, up, STATS)
    up["p_market"] = market_calibrated(lr, up)
    up["p_hybrid"] = np.where(up.mkt_p15_raw.notna(), predict(m_hyb, up, STATS + MKT), np.nan)
    up["p_final"] = up.p_hybrid.fillna(up.p_stats)
    up["fair_odds"] = 1 / up.p_final
    up["model_basis"] = np.where(up.p_hybrid.notna(), "stats + market", "stats only (no O/U odds)")
    seen = set(train.home_id) | set(train.away_id)
    up["new_team"] = [(h not in seen) or (a not in seen) for h, a in zip(up.home_id, up.away_id)]
    return up


def write_predictions(up, start):
    day = pd.Timestamp(start).date().isoformat()
    cols = ["date", "time", "league", "home", "away", "post_break", "break_gap_days", "lam_tot", "p_stats", "p_market",
            "p_hybrid", "p_final", "fair_odds", "model_basis", "new_team", "match_id"]
    out = up[cols].sort_values(["p_final"], ascending=False)
    (ROOT / "predictions").mkdir(exist_ok=True)
    out.to_csv(ROOT / f"predictions/{day}_over15.csv", index=False)
    show = out.assign(match=out.home + " v " + out.away, date=out.date.dt.strftime("%a %d %b"),
                      exp_goals=out.lam_tot.round(2), model=(out.p_final * 100).round(1),
                      market=(out.p_market * 100).round(1), fair_odds=out.fair_odds.round(2),
                      post_break=np.where(out.post_break == 1, "yes", ""))
    md = [f"# Over 1.5 goals predictions from {day}\n",
          f"{len(out)} fixtures. `model` = probability of 2+ goals (stats + market where over/under odds exist, "
          "otherwise stats only). `market` = the same probability implied by the average over/under 2.5 odds. "
          "`fair_odds` = 1 / model probability: a bet only has value at bookmaker odds above it. "
          "`post_break` = the league's first round after the international break.\n",
          show[["date", "league", "match", "exp_goals", "model", "market", "fair_odds", "post_break", "model_basis"]]
          .to_markdown(index=False)]
    (ROOT / f"reports/predictions_{day}_over15.md").write_text("\n".join(md))
    return out


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else dt.date.today().isoformat()
    hist_f = add_over15_features(pd.read_parquet(ROOT / "data/processed/features.parquet"))
    validate(hist_f)
    up = predict_upcoming(start)
    out = write_predictions(up, start)
    print(out.head(25).to_string())
