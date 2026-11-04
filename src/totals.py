"""Over/under 2.5 goals: walk-forward model + market tests. Writes reports/totals_report.md
and data/processed/totals_predictions.parquet.

Models (LightGBM binary, trained on seasons < S, tested on S, S = 2012..2025):
  poisson : raw dynamic Poisson goal ratings (src/goal_ratings.py), no learning on top
  stats   : goal ratings + recent goals/shots form + league scoring context (no odds)
  hybrid  : stats + average pre-match over/under odds
Benchmarks: average pre-match odds (2012+), Pinnacle opening and closing (2019+).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from goal_ratings import compute_goal_ratings, load_params, p_over  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "totals_report.md"
OUT = ROOT / "data" / "processed" / "totals_predictions.parquet"
TOP_FLIGHT = {"E0", "D1", "I1", "SP1", "F1", "N1", "P1", "B1", "T1", "G1", "SC0"}
FIRST_TEST, LAST_TEST = 2012, 2025
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=300,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7)
rng = np.random.default_rng(0)


def bll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def devig2(o_over, o_under, method="prop"):
    a, b = 1 / o_over, 1 / o_under
    if method == "prop":
        return a / (a + b)
    lo, hi = np.full(len(a), 0.5), np.full(len(a), 3.0)  # power method
    for _ in range(60):
        k = (lo + hi) / 2; s = a ** k + b ** k
        lo = np.where(s > 1, k, lo); hi = np.where(s > 1, hi, k)
    return a ** ((lo + hi) / 2)


def boot(x, n=400):
    if len(x) < 2:
        return np.nan, np.nan
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def build():
    f = pd.read_parquet(ROOT / "data/processed/features.parquet").reset_index(drop=True)  # includes ou_* odds
    f["over"] = ((f.hg + f.ag) > 2.5).astype(float)
    # goal ratings: tuned goals-only and a shot-weighted variant
    g = compute_goal_ratings(f, **load_params())
    gs = compute_goal_ratings(f, k=0.01, k_league=0.002, regress=0.4, k_early=1.0, w_shots=0.3)
    f[["lam_h", "lam_a", "att_h", "dfn_h", "att_a", "dfn_a"]] = g.to_numpy()
    f["lam_tot"] = f.lam_h + f.lam_a
    f["p_pois"] = p_over(f.lam_tot.to_numpy())
    f["lam_tot_s"] = gs.lam_h.to_numpy() + gs.lam_a.to_numpy()
    f["p_pois_s"] = p_over(f.lam_tot_s.to_numpy())
    # league scoring context from prior matches only
    f["lg_over_rate"] = f.groupby("league").over.transform(lambda s: s.shift(1).expanding().mean())
    f["lg_season_over_rate"] = f.groupby(["league", "season"]).over.transform(lambda s: s.shift(1).expanding().mean())
    f["lg_season_n"] = f.groupby(["league", "season"]).cumcount()
    # market: average pre-match over/under (xgabora avg, else football-data average)
    oo = f.odds_o25.fillna(f.ou_avg_o); ou = f.odds_u25.fillna(f.ou_avg_u)
    f["mkt_over"] = devig2(oo, ou); f["mkt_ou_margin"] = 1 / oo + 1 / ou - 1
    f["avg_o"], f["avg_u"] = oo, ou
    return f


STATS = (["lam_h", "lam_a", "lam_tot", "p_pois", "lam_tot_s", "p_pois_s", "att_h", "dfn_h", "att_a", "dfn_a",
          "lg_over_rate", "lg_season_over_rate", "lg_season_n", "month", "h_rest_days", "a_rest_days"]
         + [f"{s}_{c}_ew{hl}" for s in "ha" for c in ["gf", "ga", "sf", "sa", "stf", "sta"] for hl in (3, 10)])
MKT = ["mkt_over", "mkt_ou_margin"]


def walk_forward(f):
    data = f.dropna(subset=["mkt_over"]).copy()
    data["league"] = data.league.astype("category")
    out = []
    for S in range(FIRST_TEST, LAST_TEST + 1):
        tr, te = data[data.season < S], data[data.season == S].copy()
        if te.empty:
            continue
        for name, feats in (("stats", STATS + ["league"]), ("hybrid", STATS + MKT + ["league"])):
            mdl = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr.over), num_boost_round=300)
            te[f"p_{name}"] = mdl.predict(te[feats])
        out.append(te)
        print(f"totals season {S}: {len(te):,}")
    return pd.concat(out)


def bet_table(P, y, prices, models, thrs=(0.0, 0.03, 0.06)):
    rows = []
    for model in models:
        p = P[model]
        for pname, (oo, ou) in prices.items():
            for thr in thrs:
                ev_o, ev_u = p * oo - 1, (1 - p) * ou - 1
                ev_o = np.where(np.isnan(ev_o), -np.inf, ev_o); ev_u = np.where(np.isnan(ev_u), -np.inf, ev_u)
                take_o = (ev_o >= ev_u) & (ev_o > thr); take_u = (ev_u > ev_o) & (ev_u > thr)
                prof = np.concatenate([np.where(y[take_o] == 1, oo[take_o] - 1, -1.0),
                                       np.where(y[take_u] == 0, ou[take_u] - 1, -1.0)])
                lo, hi = boot(prof)
                rows.append(dict(model=model, price=pname, min_edge=thr, bets=len(prof),
                                 overs=int(take_o.sum()), roi=prof.mean() if len(prof) else np.nan, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def main():
    f = build()
    pr = walk_forward(f)
    keep = ["match_id", "date", "season", "league", "home", "away", "hg", "ag", "over", "p_pois", "p_stats", "p_hybrid",
            "mkt_over", "avg_o", "avg_u"] + [c for c in pr.columns if c.startswith("ou_")]
    pr[keep].to_parquet(OUT, index=False)
    y = pr.over.to_numpy()
    md = ["# Over/under 2.5 goals\n",
          f"Walk-forward, test seasons {FIRST_TEST}-{LAST_TEST}: {len(pr):,} matches with average pre-match "
          f"over/under odds ({pr.league.nunique()} leagues). Over rate {y.mean():.1%}. Binary log loss, lower is better "
          "(always guessing 50/50 = 0.6931).\n"]

    # A. all seasons vs average odds
    P = {"base rate": np.full(len(pr), f.loc[f.season < FIRST_TEST, "over"].mean()),
         "raw Poisson goal ratings": pr.p_pois.to_numpy(), "stats model": pr.p_stats.to_numpy(),
         "hybrid model": pr.p_hybrid.to_numpy(), "average pre-match odds": pr.mkt_over.to_numpy()}
    t = pd.DataFrame({k: {"logloss": bll(v, y), "accuracy": ((v > .5) == y).mean()} for k, v in P.items()}).T
    md += ["## 1. All test seasons, against average pre-match odds\n", t.round(4).to_markdown(), "\n"]

    # B. 2019+ with Pinnacle
    pin = pr.dropna(subset=["ou_ps_open_o", "ou_ps_open_u", "ou_ps_close_o", "ou_ps_close_u"]).copy()
    yp = pin.over.to_numpy()
    pin["pin_open"] = devig2(pin.ou_ps_open_o, pin.ou_ps_open_u)
    pin["pin_close"] = devig2(pin.ou_ps_close_o, pin.ou_ps_close_u)
    pin["pin_close_pow"] = devig2(pin.ou_ps_close_o.to_numpy(), pin.ou_ps_close_u.to_numpy(), "power")
    PP = {"stats model": pin.p_stats.to_numpy(), "hybrid model": pin.p_hybrid.to_numpy(),
          "average pre-match odds": pin.mkt_over.to_numpy(), "Pinnacle opening": pin.pin_open.to_numpy(),
          "Pinnacle closing": pin.pin_close.to_numpy(), "Pinnacle closing (power de-vig)": pin.pin_close_pow.to_numpy()}
    t = pd.DataFrame({k: {"logloss": bll(v, yp)} for k, v in PP.items()}).T
    md += [f"\n## 2. Against Pinnacle ({len(pin):,} matches, {pin.season.min()}-{pin.season.max()})\n",
           t.round(4).to_markdown(), "\n",
           f"Median margin: average pre-match {pin.mkt_ou_margin.median() if 'mkt_ou_margin' in pin else np.nan:.1%}, "
           f"Pinnacle close {(1 / pin.ou_ps_close_o + 1 / pin.ou_ps_close_u - 1).median():.1%}.\n"]

    # C. betting
    prices = {"Pinnacle close": (pin.ou_ps_close_o.to_numpy(), pin.ou_ps_close_u.to_numpy()),
              "best close": (pin.ou_max_close_o.to_numpy(), pin.ou_max_close_u.to_numpy()),
              "Pinnacle open": (pin.ou_ps_open_o.to_numpy(), pin.ou_ps_open_u.to_numpy())}
    md += ["\n## 3. Betting at real prices (flat stakes, 95% CI)\n",
           bet_table(PP, yp, prices, ["stats model", "hybrid model"]).round(4).to_markdown(index=False), "\n"]

    # D. CLV taking Pinnacle open
    fair_o, fair_u = 1 / pin.pin_close.to_numpy(), 1 / (1 - pin.pin_close.to_numpy())
    oo, ou = pin.ou_ps_open_o.to_numpy(), pin.ou_ps_open_u.to_numpy()
    rows = []
    for model in ("stats model", "hybrid model"):
        p = PP[model]
        for thr in (0.0, 0.03, 0.06):
            ev_o, ev_u = p * oo - 1, (1 - p) * ou - 1
            to = (ev_o >= ev_u) & (ev_o > thr); tu = (ev_u > ev_o) & (ev_u > thr)
            clv = np.concatenate([oo[to] / fair_o[to] - 1, ou[tu] / fair_u[tu] - 1])
            lo, hi = boot(clv)
            rows.append(dict(model=model, min_edge=thr, bets=len(clv), mean_clv=clv.mean(), ci_low=lo, ci_high=hi,
                             share_beat_close=(clv > 0).mean()))
    side = rng.integers(0, 2, len(pin)).astype(bool)
    clv_r = np.where(side, oo / fair_o - 1, ou / fair_u - 1)
    rows.append(dict(model="random side (baseline)", min_edge=np.nan, bets=len(pin), mean_clv=clv_r.mean(),
                     share_beat_close=(clv_r > 0).mean()))
    md += ["\n## 4. Closing line value, taking Pinnacle's opening price\n",
           "CLV = opening odds / fair Pinnacle closing odds - 1. Positive = beating the sharpest final price.\n",
           pd.DataFrame(rows).round(4).to_markdown(index=False), "\n"]
    move = pin.pin_close.to_numpy() - pin.pin_open.to_numpy()
    for model in ("stats model", "hybrid model"):
        md.append(f"- corr({model} - Pinnacle open, close - open) = **{np.corrcoef(PP[model] - pin.pin_open, move)[0, 1]:.3f}**\n")

    # E. market calibration
    md.append("\n## 5. Is Pinnacle's over/under price biased?\n\nOver probability at close (power de-vig). "
              "z > 3 = real miscalibration. ROI = backing every over / every under in the bucket.\n")
    b = pd.cut(pin.pin_close_pow, [0, .35, .45, .5, .55, .6, .7, 1])
    rows = []
    for k_, g_ in pin.assign(b=b).groupby("b", observed=True):
        pp_ = g_.pin_close_pow.mean(); act = g_.over.mean(); n = len(g_)
        ro = np.where(g_.over == 1, g_.ou_ps_close_o - 1, -1).mean()
        ru = np.where(g_.over == 0, g_.ou_ps_close_u - 1, -1).mean()
        rows.append({"over prob": str(k_), "n": n, "priced": pp_, "actual": act,
                     "z": (act - pp_) / np.sqrt(pp_ * (1 - pp_) / n), "ROI all overs": ro, "ROI all unders": ru})
    md += [pd.DataFrame(rows).round(4).to_markdown(index=False), "\n"]

    # F. by league / tier: where is the closing price least efficient?
    md.append("\n## 6. By league: is the market softer anywhere?\n\n`model - close` < 0 would mean our hybrid beats "
              "Pinnacle's close in that league. ROI = hybrid bets with edge > 3% at the best closing price. With "
              "22 leagues, about 1 will look significant by chance.\n")
    rows = []
    for lg, g_ in pin.groupby("league"):
        yy = g_.over.to_numpy()
        p = g_.p_hybrid.to_numpy(); oo_, ou_ = g_.ou_max_close_o.to_numpy(), g_.ou_max_close_u.to_numpy()
        ev_o, ev_u = p * oo_ - 1, (1 - p) * ou_ - 1
        to = (ev_o >= ev_u) & (ev_o > .03); tu = (ev_u > ev_o) & (ev_u > .03)
        prof = np.concatenate([np.where(yy[to] == 1, oo_[to] - 1, -1.0), np.where(yy[tu] == 0, ou_[tu] - 1, -1.0)])
        lo, hi = boot(prof, 300)
        rows.append(dict(league=lg, tier="top" if lg in TOP_FLIGHT else "lower", n=len(g_),
                         close_ll=bll(g_.pin_close.to_numpy(), yy), model_ll=bll(p, yy),
                         **{"model - close": bll(p, yy) - bll(g_.pin_close.to_numpy(), yy)},
                         margin=(1 / g_.ou_ps_close_o + 1 / g_.ou_ps_close_u - 1).median(),
                         bets=len(prof), roi=prof.mean() if len(prof) else np.nan, ci_low=lo, ci_high=hi))
    L = pd.DataFrame(rows).sort_values("model - close")
    md += [L.round(4).to_markdown(index=False), "\n"]
    tier = L.groupby("tier").apply(lambda d: pd.Series({"matches": d.n.sum(),
                                                         "model - close (weighted)": np.average(d["model - close"], weights=d.n),
                                                         "median margin": d.margin.median()}), include_groups=False)
    md += ["\n### Top flights vs lower divisions\n", tier.round(4).to_markdown(), "\n"]
    # pre-stated hypothesis: lower divisions are less efficient -> bet by tier (not per league: too many tests)
    pin["tier"] = np.where(pin.league.isin(TOP_FLIGHT), "top", "lower")
    rows = []
    for tr_, g_ in pin.groupby("tier"):
        yy = g_.over.to_numpy()
        prices_t = {"Pinnacle close": (g_.ou_ps_close_o.to_numpy(), g_.ou_ps_close_u.to_numpy()),
                    "best close": (g_.ou_max_close_o.to_numpy(), g_.ou_max_close_u.to_numpy())}
        bt = bet_table({"hybrid model": g_.p_hybrid.to_numpy()}, yy, prices_t, ["hybrid model"], thrs=(0.03, 0.06))
        bt.insert(0, "tier", tr_)
        rows.append(bt)
    md += ["\n### Betting by tier (hybrid model)\n", pd.concat(rows).round(4).to_markdown(index=False), "\n"]
    REPORT.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
