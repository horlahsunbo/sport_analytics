"""Frozen-data over/under backtest for one weekend, then grading against the real scores.

The model only sees matches played BEFORE --freeze (as if predicting on that morning), predicts every
match from --start to --end, and is graded once results exist. Same freeze = same predictions (fixed
seeds), so predictions can be produced before or after the games without leaking anything.

  python3 src/ou_backtest.py predict --line 2.5 --freeze 2026-11-20 --start 2026-11-20 --end 2026-11-22
  python3 src/ou_backtest.py grade   --line 2.5 --freeze 2026-11-20 --start 2026-11-20 --end 2026-11-22
  python3 src/ou_backtest.py walkforward --line 2.5      # 2015-2025 history used to check filters

Fixtures come from match rows in that window (results stripped) plus the saved fixtures files.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from features import build as build_features  # noqa: E402
from ingest import season_of  # noqa: E402
from over15 import add_over15_features, load_fixtures, STATS as STATS15, MKT, PARAMS, ROUNDS, bll, boot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOP_FLIGHT = {"E0", "D1", "I1", "SP1", "F1", "N1", "P1", "B1", "T1", "G1", "SC0"}
# everything that is only known after kick-off: removed from window rows before predicting
POST_MATCH = ["hg", "ag", "result", "hthg", "htag", "hs", "as_", "hst", "ast", "hc", "ac", "hf", "af", "hy", "ay",
              "hr", "ar", "hxg", "axg"]
CLOSING_PREFIX = ("ps_close", "avg_close", "max_close", "bfe_close", "b365_close", "ou_ps_close", "ou_avg_close",
                  "ou_max_close", "ah_ps_close", "ah_avg_close", "ah_max_close", "ah_line_close")


def p_over_line(lam, line):
    n = int(np.floor(line)); term = np.exp(-lam); cdf = term.copy()
    for i in range(1, n + 1):
        term = term * lam / i; cdf = cdf + term
    return 1 - cdf


def add_line_features(f, line):
    f = add_over15_features(f)
    tag = str(line).replace(".", "")
    played = f.hg.notna()
    f["target"] = np.where(played, ((f.hg + f.ag) > line).astype(float), np.nan)
    f["p_line_pois"] = p_over_line(f.lam_tot, line)
    f["p_line_pois_s"] = p_over_line(f.lam_tot_s, line)
    f["lg_line_rate"] = f.groupby("league").target.transform(lambda s: s.shift(1).expanding().mean())
    f["lg_season_line_rate"] = f.groupby(["league", "season"]).target.transform(lambda s: s.shift(1).expanding().mean())
    # market probability for this line
    if line == 2.5:
        f["p_market_raw"] = f.mkt_o25
    else:
        f["p_market_raw"] = p_over_line(f.mkt_lam, line)
    f.attrs["tag"] = tag
    return f


def feats_for(line):
    base = [c for c in STATS15 if c not in ("p15_pois", "p15_pois_s", "lg_over15_rate", "lg_season_over15_rate")]
    stats = base + ["p_line_pois", "p_line_pois_s", "lg_line_rate", "lg_season_line_rate"]
    return stats, stats + MKT


def fit(train, feats):
    X = train[feats + ["league"]].copy(); X["league"] = X.league.astype("category")
    return lgb.train(PARAMS, lgb.Dataset(X, train.target), num_boost_round=ROUNDS), X.league.cat.categories


def pred(mc, df, feats):
    m, cats = mc
    X = df[feats + ["league"]].copy(); X["league"] = pd.Categorical(X.league.astype(str), categories=cats)
    return m.predict(X)


# ---------------------------------------------------------------- predict
def predict(line, freeze, start, end):
    allm = pd.read_parquet(ROOT / "data/processed/matches.parquet")
    hist = allm[allm.date < pd.Timestamp(freeze)].copy()
    win_rows = allm[(allm.date >= pd.Timestamp(start)) & (allm.date <= pd.Timestamp(end))].copy()
    win_rows = win_rows.drop(columns=[c for c in win_rows.columns if c.startswith(CLOSING_PREFIX)])
    win_rows[POST_MATCH] = np.nan
    fx = load_fixtures(start)
    fx = fx[fx.date <= pd.Timestamp(end)]
    have = set(zip(win_rows.league, win_rows.date, win_rows.home, win_rows.away))
    fx = fx[[k not in have for k in zip(fx.league, fx.date, fx.home, fx.away)]].copy()
    fx["season"] = season_of(fx.league, fx.date)
    win = pd.concat([win_rows, fx], ignore_index=True)
    win["is_fixture"] = True
    win = win.drop(columns=["match_id"], errors="ignore")
    win["match_id"] = np.arange(len(win)) + hist.match_id.max() + 1
    data = pd.concat([hist.assign(is_fixture=False), win], ignore_index=True).sort_values(["date", "match_id"])
    f = add_line_features(build_features(data.reset_index(drop=True)), line)
    stats, hyb = feats_for(line)
    train = f[f.target.notna()]
    up = f[f.is_fixture == True].copy()  # noqa: E712
    up["p_stats"] = pred(fit(train, stats), up, stats)
    trm = train[train.p_market_raw.notna()]
    has_mkt = up.p_market_raw.notna()
    up["p_hybrid"] = np.nan
    if has_mkt.any():
        up.loc[has_mkt, "p_hybrid"] = pred(fit(trm, hyb), up[has_mkt], hyb)
    up["p_market"] = up.p_market_raw
    up["p_model"] = up.p_hybrid.fillna(up.p_stats)
    up["basis"] = np.where(has_mkt, "stats + market", "stats only")
    up["odds_over"] = up.odds_o25 if line == 2.5 else np.nan
    up["odds_under"] = up.odds_u25 if line == 2.5 else np.nan
    up["odds_over_max"] = up.ou_max_o if line == 2.5 else np.nan
    up["odds_under_max"] = up.ou_max_u if line == 2.5 else np.nan
    cols = ["date", "time", "league", "home", "away", "post_break", "lam_tot", "p_stats", "p_hybrid", "p_market",
            "p_model", "basis", "odds_over", "odds_under", "odds_over_max", "odds_under_max"]
    out = up[cols].sort_values(["date", "league", "home"])
    path = ROOT / f"predictions/{start}_{end}_over{str(line).replace('.', '')}_frozen{freeze}.csv"
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    print(f"{len(out)} matches predicted (freeze {freeze}) -> {path.relative_to(ROOT)}")
    return out, path


# ---------------------------------------------------------------- walk-forward history (for filters)
def walkforward(line):
    f = add_line_features(pd.read_parquet(ROOT / "data/processed/features.parquet"), line)
    stats, hyb = feats_for(line)
    hist = f[f.target.notna()]
    out = []
    for S in range(2015, 2026):
        tr, te = hist[hist.season < S], hist[(hist.season == S) & hist.p_market_raw.notna()].copy()
        te["p_stats"] = pred(fit(tr, stats), te, stats)
        te["p_hybrid"] = pred(fit(tr[tr.p_market_raw.notna()], hyb), te, hyb)
        out.append(te); print(f"walk-forward {line} season {S}: {len(te):,}")
    w = pd.concat(out)
    w["p_market"] = w.p_market_raw; w["p_model"] = w.p_hybrid
    w["odds_over"], w["odds_under"] = w.odds_o25.fillna(w.ou_avg_o), w.odds_u25.fillna(w.ou_avg_u)
    w["odds_over_max"], w["odds_under_max"] = w.ou_max_o, w.ou_max_u
    keep = ["date", "season", "league", "home", "away", "target", "post_break", "p_stats", "p_hybrid", "p_market",
            "p_model", "odds_over", "odds_under", "odds_over_max", "odds_under_max"]
    w[keep].to_parquet(ROOT / f"data/processed/ou{str(line).replace('.', '')}_walkforward.parquet", index=False)
    return w[keep]


# ---------------------------------------------------------------- filters
def picks(df, thr_over=0.5):
    """Each match -> a pick (over if p_model >= 0.5 else under), its probability, market prob, odds, hit."""
    d = df.copy()
    d["pick"] = np.where(d.p_model >= thr_over, "over", "under")
    d["p_pick"] = np.where(d.pick == "over", d.p_model, 1 - d.p_model)
    d["mkt_pick"] = np.where(d.pick == "over", d.p_market, 1 - d.p_market)
    d["odds_pick"] = np.where(d.pick == "over", d.odds_over, d.odds_under)
    d["hit"] = np.where(d.pick == "over", d.target == 1, d.target == 0).astype(float)
    d["profit"] = np.where(d.hit == 1, d.odds_pick - 1, -1.0)
    d["odds_pick_max"] = np.where(d.pick == "over", d.odds_over_max, d.odds_under_max)
    d["profit_max"] = np.where(d.odds_pick_max.isna(), np.nan, np.where(d.hit == 1, d.odds_pick_max - 1, -1.0))
    d["tier"] = np.where(d.league.isin(TOP_FLIGHT), "top flight", "lower/other")
    return d


FILTERS = {
    "all matches": lambda d: d.index == d.index,
    "model >= 60% on its side": lambda d: d.p_pick >= 0.60,
    "model >= 65% on its side": lambda d: d.p_pick >= 0.65,
    "model and market agree, model more confident": lambda d: (d.mkt_pick > 0.5) & (d.p_pick > d.mkt_pick),
    "model disagrees with market (>5 pts)": lambda d: (d.p_pick - d.mkt_pick) > 0.05,
    "over picks only": lambda d: d.pick == "over",
    "under picks only": lambda d: d.pick == "under",
    "post-break round": lambda d: d.post_break == 1,
    "top flights": lambda d: d.tier == "top flight",
    "lower divisions": lambda d: d.tier != "top flight",
}


def filter_table(d):
    rows = []
    for name, fn in FILTERS.items():
        s = d[fn(d) & d.mkt_pick.notna()]
        if len(s) == 0:
            rows.append({"filter": name, "matches": 0}); continue
        r = {"filter": name, "matches": len(s), "hit rate": s.hit.mean(), "model expected": s.p_pick.mean(),
             "market expected": s.mkt_pick.mean(), "ROI at avg odds": s.profit.mean()}
        if len(s) >= 30:
            lo, hi = boot(s.profit.to_numpy(), 300); r["ROI 95% CI"] = f"{lo:+.3f} to {hi:+.3f}"
        pm = s.profit_max.dropna().to_numpy()
        r["ROI at best odds"] = pm.mean() if len(pm) else np.nan
        if len(pm) >= 30:
            lo, hi = boot(pm, 300); r["best-odds 95% CI"] = f"{lo:+.3f} to {hi:+.3f}"
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- grade
def grade(line, freeze, start, end):
    tag = str(line).replace(".", "")
    path = ROOT / f"predictions/{start}_{end}_over{tag}_frozen{freeze}.csv"
    if not path.exists():
        predict(line, freeze, start, end)
    p = pd.read_csv(path, parse_dates=["date"])
    res = pd.read_parquet(ROOT / "data/processed/matches.parquet")
    res = res[(res.date >= pd.Timestamp(start)) & (res.date <= pd.Timestamp(end))]
    from apifootball import load_results as api_results
    api = api_results()
    api = api[(api.date >= pd.Timestamp(start)) & (api.date <= pd.Timestamp(end))]
    known = set(zip(res.league, res.date, res.home, res.away))
    api = api[[k not in known for k in zip(api.league, api.date, api.home, api.away)]]
    res = pd.concat([res.assign(result_source="football-data"), api.assign(result_source="api-football")], ignore_index=True)
    if len(api):
        print(f"using {len(api)} API-Football results not yet in football-data.co.uk")
    fd_res = res[res.result_source == "football-data"]
    have_res = set(zip(fd_res.league, fd_res.date, fd_res.home, fd_res.away))
    missing_pred = [k for k in have_res if k not in set(zip(p.league, p.date, p.home, p.away))]
    if missing_pred and any(src == "football-data" for src in res.result_source):  # e.g. Friday games -> re-predict
        print(f"{len(missing_pred)} played matches were not in the prediction file; re-predicting from frozen data")
        p, path = predict(line, freeze, start, end)
        p["date"] = pd.to_datetime(p.date)
    rc = ["league", "date", "home", "away", "hg", "ag", "ou_avg_close_o", "ou_avg_close_u", "result_source"]
    g = p.merge(res[rc], on=["league", "date", "home", "away"], how="left")
    done = g.hg.notna()
    md = [f"# Over {line} goals: frozen backtest {start} to {end}\n",
          f"Model trained only on matches before **{freeze}**. {len(p)} matches predicted, **{int(done.sum())} graded** "
          f"({int((~done).sum())} without a result in the data yet).\n"]
    g = g[done].copy()
    if g.empty:
        (ROOT / f"reports/backtest_{start}_{end}_over{tag}.md").write_text("\n".join(md)); print("\n".join(md)); return
    g["total"] = g.hg + g.ag
    g["target"] = (g.total > line).astype(float)
    y = g.target.to_numpy()
    mk = g.p_market.notna().to_numpy()
    g["p_close"] = (1 / g.ou_avg_close_o) / (1 / g.ou_avg_close_o + 1 / g.ou_avg_close_u) if line == 2.5 else np.nan
    cl = mk & g.p_close.notna().to_numpy()
    rows = [{"predictor": "model", "matches": len(g), "logloss": bll(g.p_model.to_numpy(), y),
             "accuracy": ((g.p_model >= .5) == (y == 1)).mean()}]
    for nm, col, m in (("model", "p_model", mk), ("market (pre-match)", "p_market", mk), ("model", "p_model", cl),
                       ("market (closing)", "p_close", cl)):
        if m.any():
            rows.append({"predictor": f"{nm} [{'with odds' if m is mk else 'with closing odds'}]", "matches": int(m.sum()),
                         "logloss": bll(g[col].to_numpy()[m], y[m]), "accuracy": ((g[col].to_numpy()[m] >= .5) == (y[m] == 1)).mean()})
    md += [f"Over {line} happened in **{y.mean():.1%}** of graded matches (model expected {g.p_model.mean():.1%}).\n",
           "## Scores (log loss: lower is better; accuracy = right side of 50%)\n",
           pd.DataFrame(rows).round(4).to_markdown(index=False), "\n"]
    b = pd.cut(g.p_model, [0, .35, .45, .55, .65, 1])
    cal = g.groupby(b, observed=True).agg(matches=("target", "size"), predicted=("p_model", "mean"), actual=("target", "mean"))
    md += ["\n## Calibration\n", cal.round(3).to_markdown(), "\n"]
    d = picks(g)
    lg = d.groupby("league").agg(matches=("hit", "size"), hit_rate=("hit", "mean"), model_expected=("p_pick", "mean"),
                                 over_rate=("target", "mean")).sort_values("hit_rate", ascending=False)
    md += ["\n## By league (pick = side the model favours)\n", lg.round(3).to_markdown(), "\n"]
    # filters: weekend vs 2015-2025 history
    wf_path = ROOT / f"data/processed/ou{tag}_walkforward.parquet"
    wf = pd.read_parquet(wf_path) if wf_path.exists() else walkforward(line)
    fw, fh = filter_table(d), filter_table(picks(wf))
    md += ["\n## Filters: this weekend vs 2015-2025 history\n",
           "A filter is only worth keeping if it holds on the history: one weekend is far too small to tune on "
           f"(with ~{len(g)} matches, a hit rate has a margin of error of about +-{100 / np.sqrt(max(len(g), 1)):.0f} points). "
           "`market expected` = how often the market thought the pick would land; a filter beats the market only "
           "if `hit rate` > `market expected` on the history.\n",
           "### This weekend\n", fw.round(3).to_markdown(index=False), "\n",
           f"### History ({len(wf):,} matches, 2015-2025, same models trained walk-forward)\n",
           fh.round(3).to_markdown(index=False), "\n"]
    d["result"] = d.hg.astype(int).astype(str) + "-" + d.ag.astype(int).astype(str)
    show = d.sort_values("p_model", ascending=False)
    show = show.assign(date=show.date.dt.strftime("%a %d"), match=show.home + " v " + show.away,
                       model=(show.p_model * 100).round(1), market=(show.p_market * 100).round(1),
                       outcome=np.where(show.target == 1, f"OVER", "under"), correct=np.where(show.hit == 1, "yes", "NO"))
    md += ["\n## Every match\n", show[["date", "league", "match", "model", "market", "pick", "result", "outcome", "correct"]]
           .to_markdown(index=False), "\n"]
    out = ROOT / f"reports/backtest_{start}_{end}_over{tag}.md"
    out.write_text("\n".join(md)); print("\n".join(md[:12]))
    d.to_csv(ROOT / f"predictions/{start}_{end}_over{tag}_graded.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["predict", "grade", "walkforward"])
    ap.add_argument("--line", type=float, default=2.5)
    ap.add_argument("--freeze", default="2026-11-20"); ap.add_argument("--start", default="2026-11-20")
    ap.add_argument("--end", default="2026-11-22")
    a = ap.parse_args()
    if a.cmd == "predict":
        predict(a.line, a.freeze, a.start, a.end)
    elif a.cmd == "walkforward":
        walkforward(a.line)
    else:
        grade(a.line, a.freeze, a.start, a.end)
