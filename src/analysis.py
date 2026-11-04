"""Secondary analysis: WHEN do stats fail, and WHICH teams follow stats?
Reads walk-forward predictions (out-of-sample only). Writes reports/analysis_report.md
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "processed" / "predictions.parquet"
REPORT = ROOT / "reports" / "analysis_report.md"


def surprise(df, prefix):
    p = df[[f"{prefix}_h", f"{prefix}_d", f"{prefix}_a"]].to_numpy()
    return -np.log(np.clip(p[np.arange(len(df)), df.y.to_numpy()], 1e-6, 1))


def calib_table(df, pcol, outcome, bins):
    b = pd.cut(df[pcol], bins)
    t = df.groupby(b, observed=True).agg(n=(pcol, "size"), predicted=(pcol, "mean"), actual=(outcome, "mean"))
    t["gap"] = t.actual - t.predicted
    t["z"] = t.gap / np.sqrt(t.predicted * (1 - t.predicted) / t.n)
    return t


def main():
    df = pd.read_parquet(PRED)
    df["s_mkt"] = surprise(df, "mkt")
    df["s_stats"] = surprise(df, "p_stats")
    df["fav_p"] = df[["mkt_h", "mkt_d", "mkt_a"]].max(axis=1)
    df["fav_won"] = (df[["mkt_h", "mkt_d", "mkt_a"]].to_numpy().argmax(1) == df.y).astype(float)
    df["home_won"] = (df.y == 0).astype(float)
    df["draw"] = (df.y == 1).astype(float)
    df["away_won"] = (df.y == 2).astype(float)
    md = []

    # 1. Where are outcomes least predictable? (surprise = log loss per match)
    md.append("## 1. When do stats fail? Mean surprise (log loss) by context\n\n"
              "Higher = less predictable. 'stats_gap' = how much worse the stats model is than the market there.\n")
    df["fav_bucket"] = pd.cut(df.fav_p, [0, .4, .5, .6, .7, .8, 1])
    df["phase"] = pd.cut(df.date.dt.month, [0, 2, 5, 7, 9, 12], labels=["Jan-Feb", "Mar-May", "Jun-Jul", "Aug-Sep", "Oct-Dec"])
    for col, label in [("league", "League"), ("fav_bucket", "Favourite strength (market prob)"), ("phase", "Season phase")]:
        t = df.groupby(col, observed=True).agg(n=("y", "size"), market=("s_mkt", "mean"), stats=("s_stats", "mean"),
                                               draw_rate=("draw", "mean"), fav_win=("fav_won", "mean"))
        t["stats_gap"] = t.stats - t.market
        t = t.sort_values("market", ascending=False) if col == "league" else t
        md.append(f"### {label}\n\n" + t.round(4).to_markdown() + "\n")

    # 2. Is the market itself miscalibrated anywhere? (real, exploitable structure)
    md.append("## 2. Market calibration: where are the bookmakers' probabilities off?\n\n"
              "|z| > 3 is a serious, non-random miscalibration. Positive gap = outcome happens more than priced.\n")
    bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    for pcol, out, name in [("mkt_h", "home_won", "Home win"), ("mkt_d", "draw", "Draw"), ("mkt_a", "away_won", "Away win")]:
        md.append(f"### {name}\n\n" + calib_table(df, pcol, out, bins).round(4).to_markdown() + "\n")

    # 3. When stats and market disagree, who is right?
    md.append("## 3. Stats vs market disagreement: who is right?\n\n"
              "disagreement = stats-model home prob - market home prob. If 'actual' tracks 'market', the market wins.\n")
    df["disagree"] = df.p_stats_h - df.mkt_h
    t = df.groupby(pd.cut(df.disagree, [-1, -.15, -.08, -.03, .03, .08, .15, 1]), observed=True).agg(
        n=("y", "size"), stats_p=("p_stats_h", "mean"), market_p=("mkt_h", "mean"), actual=("home_won", "mean"))
    md.append(t.round(4).to_markdown() + "\n")

    # 4. Which teams follow stats -- and is that a persistent trait or noise?
    md.append("## 4. Which teams follow the stats? Persistence test\n\n"
              "Per team-season predictability = mean market surprise in that team's matches minus the league-season "
              "average (negative = more predictable than peers). If it's a real trait, it should correlate with the "
              "same team's value next season.\n")
    long = pd.concat([df[["season", "league", "home_id", "s_mkt", "s_stats"]].rename(columns={"home_id": "team"}),
                      df[["season", "league", "away_id", "s_mkt", "s_stats"]].rename(columns={"away_id": "team"})])
    long["rel_mkt"] = long.s_mkt - long.groupby(["league", "season"]).s_mkt.transform("mean")
    long["rel_stats"] = long.s_stats - long.groupby(["league", "season"]).s_stats.transform("mean")
    ts = long.groupby(["team", "season"]).agg(n=("rel_mkt", "size"), rel_mkt=("rel_mkt", "mean"),
                                              rel_stats=("rel_stats", "mean")).reset_index()
    ts = ts[ts.n >= 20]
    nxt = ts.assign(season=ts.season - 1).rename(columns={"rel_mkt": "next_rel_mkt", "rel_stats": "next_rel_stats", "n": "n_next"})
    pair = ts.merge(nxt, on=["team", "season"])
    r_mkt = pair[["rel_mkt", "next_rel_mkt"]].corr().iloc[0, 1]
    r_stats = pair[["rel_stats", "next_rel_stats"]].corr().iloc[0, 1]
    # null: shuffle within season to get chance correlation
    rng = np.random.default_rng(0)
    null = [np.corrcoef(pair.rel_mkt, pair.groupby("season").next_rel_mkt.transform(lambda s: rng.permutation(s.values)))[0, 1]
            for _ in range(200)]
    md.append(f"- Team-season pairs: {len(pair):,}\n- Year-over-year correlation (market surprise): **{r_mkt:.3f}** "
              f"(shuffled null 95% range: {np.percentile(null, 2.5):.3f} to {np.percentile(null, 97.5):.3f})\n"
              f"- Year-over-year correlation (stats-model surprise): **{r_stats:.3f}**\n\n")
    # quintile view: do last season's most-predictable teams stay predictable?
    pair["q"] = pair.groupby("season").rel_mkt.transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    qt = pair.groupby("q").agg(teams=("team", "size"), this_season=("rel_mkt", "mean"), next_season=("next_rel_mkt", "mean"))
    qt.index = ["Q1 most predictable", "Q2", "Q3", "Q4", "Q5 least predictable"][:len(qt)]
    md.append(qt.round(4).to_markdown() + "\n\n")
    # most consistently predictable teams (need several seasons)
    agg = ts.groupby("team").agg(seasons=("season", "size"), mean_rel=("rel_mkt", "mean"), sd=("rel_mkt", "std"))
    agg = agg[agg.seasons >= 8]
    agg["t"] = agg.mean_rel / (agg.sd / np.sqrt(agg.seasons))
    md.append("### Most consistently predictable teams (>= 8 seasons, sorted by t-stat)\n\n"
              + agg.sort_values("t").head(15).round(4).to_markdown() + "\n\n"
              "### Most consistently unpredictable teams\n\n"
              + agg.sort_values("t", ascending=False).head(15).round(4).to_markdown() + "\n")

    REPORT.write_text("# Secondary analysis: when do stats fail?\n\nOut-of-sample walk-forward predictions, "
                      f"{len(df):,} matches, seasons {df.season.min()}-{df.season.max()}.\n\n" + "\n".join(md))
    print(REPORT.read_text())


if __name__ == "__main__":
    main()
