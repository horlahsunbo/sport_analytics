"""Walk-forward evaluation. For each test season S: train on seasons < S, predict S.
Three predictors are compared on identical matches:
  market : de-vigged bookmaker probabilities (the bar to beat)
  stats  : LightGBM on pre-match stats only (no odds)  -> how much do stats explain?
  hybrid : LightGBM on stats + market probs           -> do stats add info beyond the market?
Writes data/processed/predictions.parquet and reports/model_report.md
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "processed" / "features.parquet"
PRED = ROOT / "data" / "processed" / "predictions.parquet"
REPORT = ROOT / "reports" / "model_report.md"

STATS_FEATS = (["elo_h", "elo_a", "elo_diff", "lg_home_rate", "month", "h_rest_days", "a_rest_days",
                "h_n_prior", "a_n_prior"]
               + [f"{s}_{c}_ew{hl}" for s in ("h", "a") for c in
                  ["gf", "ga", "sf", "sa", "stf", "sta", "cf", "ca", "pts"] for hl in (3, 10)]
               + [f"d_{c}_ew{hl}" for c in ["gf", "ga", "sf", "sa", "stf", "sta", "pts"] for hl in (3, 10)])
MKT_FEATS = ["mkt_h", "mkt_d", "mkt_a", "overround"]
PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=31,
              min_child_samples=300, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, verbose=-1, seed=7)
ROUNDS = 400
FIRST_TEST, LAST_TEST = 2010, 2025


def fit_predict(tr, te, feats, league_cat=True):
    X = lambda d: d[feats + (["league"] if league_cat else [])].assign(
        **({"league": d.league.astype("category")} if league_cat else {}))
    Xtr, Xte = X(tr), X(te)
    if league_cat:
        Xte["league"] = pd.Categorical(Xte["league"], categories=Xtr["league"].cat.categories)
    m = lgb.train(PARAMS, lgb.Dataset(Xtr, tr.y), num_boost_round=ROUNDS)
    return m.predict(Xte), m


def metrics(p, y):
    p = np.clip(p, 1e-6, 1)
    onehot = np.eye(3)[y]
    return dict(n=len(y), logloss=log_loss(y, p, labels=[0, 1, 2]),
                brier=((p - onehot) ** 2).sum(1).mean(), acc=(p.argmax(1) == y).mean())


def main():
    df = pd.read_parquet(FEAT)
    df = df.dropna(subset=["mkt_h", "y"]).copy()
    df["y"] = df.y.astype(int)
    out, imp = [], None
    for S in range(FIRST_TEST, LAST_TEST + 1):
        tr, te = df[df.season < S], df[df.season == S].copy()
        if te.empty:
            continue
        te[["p_stats_h", "p_stats_d", "p_stats_a"]], m_s = fit_predict(tr, te, STATS_FEATS)
        te[["p_hyb_h", "p_hyb_d", "p_hyb_a"]], _ = fit_predict(tr, te, STATS_FEATS + MKT_FEATS)
        if S == LAST_TEST:
            imp = pd.Series(m_s.feature_importance("gain"), index=m_s.feature_name()).sort_values(ascending=False)
        out.append(te)
        print(f"season {S}: {len(te):,} test matches done")
    pr = pd.concat(out)
    keep = ["match_id", "date", "season", "league", "home", "away", "home_id", "away_id", "hg", "ag",
            "result", "y", "odds_h", "odds_d", "odds_a", "max_h", "max_d", "max_a", "overround",
            "mkt_h", "mkt_d", "mkt_a", "elo_diff", "p_stats_h", "p_stats_d", "p_stats_a",
            "p_hyb_h", "p_hyb_d", "p_hyb_a"]
    pr[keep].to_parquet(PRED, index=False)

    y = pr.y.to_numpy()
    P = {"market": pr[["mkt_h", "mkt_d", "mkt_a"]].to_numpy(),
         "stats": pr[["p_stats_h", "p_stats_d", "p_stats_a"]].to_numpy(),
         "hybrid": pr[["p_hyb_h", "p_hyb_d", "p_hyb_a"]].to_numpy(),
         "naive (base rates)": np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1))}
    overall = pd.DataFrame({k: metrics(v, y) for k, v in P.items()}).T
    by_season = pd.DataFrame({S: {k: metrics(v[(pr.season == S).to_numpy()], y[(pr.season == S).to_numpy()])["logloss"]
                                  for k, v in P.items() if k != "naive (base rates)"}
                              for S in sorted(pr.season.unique())}).T

    # betting simulation: flat 1-unit stake when model edge vs offered odds exceeds threshold
    bets = []
    odds = pr[["odds_h", "odds_d", "odds_a"]].to_numpy()
    for name in ("stats", "hybrid"):
        ev = P[name] * odds - 1
        for thr in (0.0, 0.05, 0.10):
            pick = ev.argmax(1); best = ev.max(1); mask = best > thr
            win = (pick == y)[mask]
            profit = np.where(win, odds[mask, pick[mask]] - 1, -1)
            rng = np.random.default_rng(0)
            boots = [rng.choice(profit, len(profit)).mean() for _ in range(500)] if len(profit) else [np.nan]
            bets.append(dict(model=name, min_edge=thr, bets=int(mask.sum()), roi=profit.mean() if len(profit) else np.nan,
                             roi_ci_low=np.percentile(boots, 2.5), roi_ci_high=np.percentile(boots, 97.5)))
    bets = pd.DataFrame(bets)

    REPORT.parent.mkdir(exist_ok=True)
    with open(REPORT, "w") as f:
        f.write(f"# Walk-forward model report\n\nTest seasons {FIRST_TEST}-{LAST_TEST}, {len(pr):,} matches, "
                f"{pr.league.nunique()} leagues. Lower log loss / Brier = better.\n\n")
        f.write("## Overall\n\n" + overall.round(4).to_markdown() + "\n\n")
        f.write("## Log loss by season\n\n" + by_season.round(4).to_markdown() + "\n\n")
        f.write("## Betting simulation (flat stakes at listed odds, 95% bootstrap CI)\n\n" + bets.round(4).to_markdown(index=False) + "\n\n")
        f.write("## Top 20 stats-model features (gain)\n\n" + (imp.head(20) / imp.sum()).round(4).to_frame("share").to_markdown() + "\n")
    print(REPORT.read_text())


if __name__ == "__main__":
    main()
