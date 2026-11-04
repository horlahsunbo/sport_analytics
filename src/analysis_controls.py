"""Robustness checks for analysis.py findings.
A) Team 'follows stats' metric controlled for favourite status: excess surprise = actual - expected
   (expected = entropy of market distribution). Persistence of THAT is the real trait test.
B) Favourite-longshot bias: does it survive as profit at real prices (avg and best odds)?
Appends to reports/analysis_report.md
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data/processed/predictions.parquet")
REPORT = ROOT / "reports/analysis_report.md"
P = df[["mkt_h", "mkt_d", "mkt_a"]].to_numpy()
y = df.y.to_numpy()
df["s_mkt"] = -np.log(np.clip(P[np.arange(len(df)), y], 1e-6, 1))
df["exp_s"] = -(P * np.log(np.clip(P, 1e-9, 1))).sum(1)
df["excess"] = df.s_mkt - df.exp_s  # >0: more chaotic than the odds implied
md = ["\n## 5. Controlled 'follows the stats' test (excess surprise vs what the odds implied)\n\n"
      "excess = actual surprise - expected surprise (entropy of market probs). This removes the "
      "'big favourites look predictable' effect. Negative = team's results follow the odds more than priced.\n"]

long = pd.concat([df[["season", "home_id", "excess"]].rename(columns={"home_id": "team"}),
                  df[["season", "away_id", "excess"]].rename(columns={"away_id": "team"})])
ts = long.groupby(["team", "season"]).agg(n=("excess", "size"), ex=("excess", "mean")).reset_index()
ts = ts[ts.n >= 20]
pair = ts.merge(ts.assign(season=ts.season - 1).rename(columns={"ex": "ex_next", "n": "n2"}), on=["team", "season"])
r = pair[["ex", "ex_next"]].corr().iloc[0, 1]
rng = np.random.default_rng(0)
null = [np.corrcoef(pair.ex, rng.permutation(pair.ex_next.values))[0, 1] for _ in range(500)]
md.append(f"- Pairs: {len(pair):,}. Year-over-year correlation of excess surprise: **{r:.3f}** "
          f"(null 95%: {np.percentile(null, 2.5):.3f} to {np.percentile(null, 97.5):.3f})\n")
pair["q"] = pair.groupby("season").ex.transform(lambda s: pd.qcut(s, 5, labels=False))
qt = pair.groupby("q").agg(teams=("team", "size"), this_season=("ex", "mean"), next_season=("ex_next", "mean"))
qt.index = ["Q1 followed odds most", "Q2", "Q3", "Q4", "Q5 most chaotic"]
md.append("\n" + qt.round(4).to_markdown() + "\n")
agg = ts.groupby("team").agg(seasons=("season", "size"), mean_ex=("ex", "mean"), sd=("ex", "std"))
agg = agg[agg.seasons >= 8]
agg["t"] = agg.mean_ex / (agg.sd / np.sqrt(agg.seasons))
md.append("\n### Teams that follow the odds more than priced (controlled, >= 8 seasons)\n\n"
          + agg.sort_values("t").head(12).round(4).to_markdown()
          + "\n\n### Teams more chaotic than priced (controlled)\n\n"
          + agg.sort_values("t", ascending=False).head(12).round(4).to_markdown() + "\n")
md.append(f"\nWith {len(agg)} teams tested, ~{len(agg)*0.05:.0f} would show |t|>2 by chance alone; "
          f"observed |t|>2: {(agg.t.abs() > 2).sum()}.\n")

# B) Favourite-longshot: flat-stake ROI by market-prob bucket, at average and best odds
md.append("\n## 6. Is the favourite-longshot bias profitable at real prices?\n\n"
          "Flat 1-unit bet on every selection in the bucket. 'avg' = typical bookmaker odds, 'best' = best available "
          "(line shopping). 95% bootstrap CI on best-odds ROI.\n")
rows = []
for side, k in (("home", 0), ("draw", 1), ("away", 2)):
    o_avg = df[["odds_h", "odds_d", "odds_a"]].to_numpy()[:, k]
    o_max = df[["max_h", "max_d", "max_a"]].to_numpy()[:, k]
    win = (y == k)
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .7), (.7, .8), (.8, 1)]:
        m = (P[:, k] > lo) & (P[:, k] <= hi) & ~np.isnan(o_max)
        if m.sum() < 200:
            continue
        pa = np.where(win[m], o_avg[m] - 1, -1); pm = np.where(win[m], o_max[m] - 1, -1)
        b = [rng.choice(pm, len(pm)).mean() for _ in range(300)]
        rows.append(dict(side=side, prob=f"{lo}-{hi}", bets=int(m.sum()), roi_avg=pa.mean(), roi_best=pm.mean(),
                         ci_low=np.percentile(b, 2.5), ci_high=np.percentile(b, 97.5)))
md.append(pd.DataFrame(rows).round(4).to_markdown(index=False) + "\n")
# era check on the most promising cell
m = (P[:, 0] > .6) & ~np.isnan(df.max_h.to_numpy())
era = df[m].assign(p=np.where(y[m] == 0, df.max_h[m] - 1, -1)).groupby(pd.cut(df.season[m], [2009, 2014, 2019, 2025])).p.agg(["size", "mean"])
md.append("\nHome favourites (p>0.6) at best odds, by era:\n\n" + era.round(4).to_markdown() + "\n")
with open(REPORT, "a") as f:
    f.write("".join(md))
print("".join(md))
