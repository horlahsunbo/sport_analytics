"""In-season hypothesis test.
Idea: summarise each team from its FIRST 10 matches of the season, then predict its NEXT 5
(matches 11-15). Claim to test: within-season variability < season-to-season variability, so
current-season info should beat information carried over from last season.

Part 1 (stability): team-level correlations, same-size windows.
Part 2 (prediction): walk-forward models on matches 11-15, all scored on identical matches:
   inseason10  first-10 summaries only (the pure idea)
   carryover   Elo + EWMA form carried across seasons (our existing stats model)
   both        inseason10 + carryover
   market      de-vigged odds
   both+mkt    everything + odds  (can in-season info beat the market?)
Writes reports/inseason_report.md
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "inseason_report.md"
N_FIRST, N_NEXT = 10, 5
METRICS = ["pts", "gd", "gf", "ga", "sd", "sotd", "xs"]  # xs = points above market expectation

df = pd.read_parquet(ROOT / "data/processed/features.parquet")
df = df.dropna(subset=["y"]).copy(); df["y"] = df.y.astype(int)

# ---- long team-match table with in-season match index k (0 = first match of season)
parts = []
for side in ("home", "away"):
    h = side == "home"
    pw, pl = (df.mkt_h, df.mkt_a) if h else (df.mkt_a, df.mkt_h)
    gf, ga = (df.hg, df.ag) if h else (df.ag, df.hg)
    sf, sa = (df.hs, df.as_) if h else (df.as_, df.hs)
    tf, ta = (df.hst, df.ast) if h else (df.ast, df.hst)
    pts = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))
    parts.append(pd.DataFrame({"match_id": df.match_id, "date": df.date, "season": df.season, "league": df.league,
                               "team": df[f"{side}_id"], "side": side, "pts": pts, "gd": gf - ga, "gf": gf, "ga": ga,
                               "sd": sf - sa, "sotd": tf - ta, "xs": pts - (3 * pw + df.mkt_d)}))
L = pd.concat(parts).sort_values(["team", "season", "date", "match_id"]).reset_index(drop=True)
L["k"] = L.groupby(["team", "season"]).cumcount()
L["n_season"] = L.groupby(["team", "season"]).k.transform("size")


def window(lo, hi, name):
    w = L[(L.k >= lo) & (L.k < hi)].groupby(["team", "season"])[METRICS].mean()
    return w.add_suffix(f"_{name}")


first = window(0, N_FIRST, "f10")
nxt = window(N_FIRST, N_FIRST + N_NEXT, "n5")
nxt10 = window(N_FIRST, N_FIRST + 10, "n10")
last10 = (L[L.k >= L.n_season - 10].groupby(["team", "season"])[METRICS].mean().add_suffix("_prevlast10"))
full = L.groupby(["team", "season"])[METRICS].mean().add_suffix("_prevfull")
# previous season -> shift season label forward so it joins on the current season
shift = lambda t: t.rename(index=lambda s: s + 1, level="season")
T = first.join([nxt, nxt10, shift(last10), shift(full)], how="inner")
md = [f"# In-season test: first {N_FIRST} matches -> next {N_NEXT}\n"]

# ================= Part 1: stability
md.append("## 1. Is in-season form more stable than season-to-season form?\n\n"
          "Correlation across team-seasons (teams present in both seasons). Columns = what we try to predict.\n")
rows = []
for m in ["pts", "gd", "sd", "sotd", "xs"]:
    sub = T.dropna(subset=[f"{m}_f10", f"{m}_n5", f"{m}_prevlast10"])
    rows.append({"metric": m, "teams": len(sub),
                 "first10 -> next5 (same season)": sub[f"{m}_f10"].corr(sub[f"{m}_n5"]),
                 "prev last10 -> next5": sub[f"{m}_prevlast10"].corr(sub[f"{m}_n5"]),
                 "prev full -> next5": sub[f"{m}_prevfull"].corr(sub[f"{m}_n5"]),
                 "first10 -> next10 (same season)": sub[f"{m}_f10"].corr(sub[f"{m}_n10"]),
                 "prev last10 -> first10 (across break)": sub[f"{m}_prevlast10"].corr(sub[f"{m}_f10"])})
md.append(pd.DataFrame(rows).round(3).to_markdown(index=False) + "\n\n"
          "Metric key: pts = points/game, gd = goal diff, sd = shot diff, sotd = shots-on-target diff, "
          "xs = points ABOVE what the odds expected (i.e. performance the market did not price).\n\n"
          "Fair same-size comparison of the season break: `prev last10 -> first10` (crosses the summer) "
          "vs `first10 -> next10` (both windows inside one season).\n")

# ================= Part 2: prediction of matches 11-15
idx = L.pivot_table(index="match_id", columns="side", values="k")
ok = idx[(idx.home >= N_FIRST) & (idx.home < N_FIRST + N_NEXT) & (idx.away >= N_FIRST) & (idx.away < N_FIRST + N_NEXT)].index
M = df[df.match_id.isin(ok)].copy()
M["horizon"] = idx.loc[M.match_id, ["home", "away"]].max(axis=1).values - N_FIRST + 1  # 1..5 matches ahead
fs = first.reset_index()
for side in ("home", "away"):
    M = M.merge(fs.rename(columns={"team": f"{side}_id", **{c: f"{side[0]}_{c}" for c in first.columns}}),
                on=[f"{side}_id", "season"], how="left")
F10 = [f"{s}_{c}" for s in "ha" for c in first.columns]
for c in first.columns:
    M[f"d_{c}"] = M[f"h_{c}"] - M[f"a_{c}"]
F10 += [f"d_{c}" for c in first.columns] + ["lg_home_rate"]
from model import STATS_FEATS, MKT_FEATS  # noqa: E402
uniq = lambda l: list(dict.fromkeys(l))
SETS = {"inseason10": F10, "carryover": STATS_FEATS, "both": uniq(F10 + STATS_FEATS), "both+mkt": uniq(F10 + STATS_FEATS + MKT_FEATS)}
PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=15, min_child_samples=200,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7)
M = M.dropna(subset=["mkt_h"])
out = []
for S in range(2008, 2026):
    tr, te = M[M.season < S], M[M.season == S].copy()
    if te.empty or len(tr) < 3000:
        continue
    for name, feats in SETS.items():
        mdl = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr.y), num_boost_round=250)
        te[[f"{name}_h", f"{name}_d", f"{name}_a"]] = mdl.predict(te[feats])
    out.append(te)
P = pd.concat(out)
P[["market_h", "market_d", "market_a"]] = P[["mkt_h", "mkt_d", "mkt_a"]].to_numpy()
names = ["market", "inseason10", "carryover", "both", "both+mkt"]
y = P.y.to_numpy()
ll = lambda n, m=slice(None): log_loss(y[m], np.clip(P[[f"{n}_h", f"{n}_d", f"{n}_a"]].to_numpy()[m], 1e-6, 1), labels=[0, 1, 2])
acc = lambda n: (P[[f"{n}_h", f"{n}_d", f"{n}_a"]].to_numpy().argmax(1) == y).mean()
res = pd.DataFrame({n: {"logloss": ll(n), "accuracy": acc(n)} for n in names}).T
md.append(f"\n## 2. Predicting matches 11-15 (walk-forward, {len(P):,} matches, seasons "
          f"{P.season.min()}-{P.season.max()})\n\n" + res.round(4).to_markdown() + "\n")
hz = pd.DataFrame({n: {h: ll(n, (P.horizon == h).to_numpy()) for h in range(1, 6)} for n in names})
hz.index.name = "matches ahead"
md.append("\n### Log loss by horizon (1 = the very next match after the first 10)\n\n" + hz.round(4).to_markdown() + "\n")

# betting: does in-season info find value vs the offered odds?
odds = P[["odds_h", "odds_d", "odds_a"]].to_numpy(); best = P[["max_h", "max_d", "max_a"]].to_numpy()
rows = []
for n in ["inseason10", "both", "both+mkt"]:
    p = P[[f"{n}_h", f"{n}_d", f"{n}_a"]].to_numpy()
    for thr in (0.03, 0.08):
        ev = p * odds - 1; pick = ev.argmax(1); mask = ev.max(1) > thr
        win = pick[mask] == y[mask]
        r_avg = np.where(win, odds[mask, pick[mask]] - 1, -1)
        r_best = np.where(win, best[mask, pick[mask]] - 1, -1)
        rows.append(dict(model=n, min_edge=thr, bets=int(mask.sum()), roi_avg_odds=r_avg.mean(), roi_best_odds=np.nanmean(r_best)))
md.append("\n### Betting check (flat stakes)\n\n" + pd.DataFrame(rows).round(4).to_markdown(index=False) + "\n")
REPORT.write_text("\n".join(md))
print(REPORT.read_text())
