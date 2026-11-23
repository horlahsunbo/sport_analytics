"""Grade a saved over-1.5 prediction file against actual results.

Run after the matches: python3 src/download.py footballdata && python3 src/ingest.py
then:                  python3 src/grade_predictions.py predictions/2026-11-21_over15.csv
Writes reports/grade_<file stem>.md
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def bll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main(path):
    pred = pd.read_csv(path, parse_dates=["date"])
    res = pd.read_parquet(ROOT / "data/processed/matches.parquet", columns=["league", "date", "home", "away", "hg", "ag"])
    sys.path.insert(0, str(Path(__file__).parent))
    from apifootball import load_results as api_results
    api = api_results()[["league", "date", "home", "away", "hg", "ag"]]
    known = set(zip(res.league, res.date, res.home, res.away))
    res = pd.concat([res, api[[k not in known for k in zip(api.league, api.date, api.home, api.away)]]], ignore_index=True)
    g = pred.merge(res, on=["league", "date", "home", "away"], how="left")
    done = g.hg.notna()
    g = g[done].copy()
    g["over15"] = ((g.hg + g.ag) >= 2).astype(float)
    y = g.over15.to_numpy()
    md = [f"# Grading {Path(path).name}\n", f"{len(g)} of {len(pred)} predicted matches have results "
          f"({(~done).sum()} not found yet: postponed, not yet played, or not yet in the data).\n"]
    if len(g) == 0:
        print("\n".join(md)); return
    md.append(f"Over 1.5 happened in **{y.mean():.1%}** of matches; the model's average prediction was "
              f"{g.p_final.mean():.1%}.\n")
    rows = {"model (p_final)": g.p_final.to_numpy()}
    mk = g.p_market.notna().to_numpy()
    summ = [{"predictor": "model", "matches": len(g), "logloss": bll(g.p_final.to_numpy(), y),
             "brier": float(((g.p_final - y) ** 2).mean())}]
    if mk.any():
        summ += [{"predictor": "model (matches with market)", "matches": int(mk.sum()),
                  "logloss": bll(g.p_final.to_numpy()[mk], y[mk]), "brier": float(((g.p_final - y) ** 2)[mk].mean())},
                 {"predictor": "market (same matches)", "matches": int(mk.sum()),
                  "logloss": bll(g.p_market.to_numpy()[mk], y[mk]), "brier": float(((g.p_market - y) ** 2)[mk].mean())}]
    md += ["## Scores (lower is better)\n", pd.DataFrame(summ).round(4).to_markdown(index=False), "\n"]
    b = pd.cut(g.p_final, [0, .7, .75, .8, .85, .9, 1])
    cal = g.groupby(b, observed=True).agg(matches=("over15", "size"), predicted=("p_final", "mean"), hit_rate=("over15", "mean"))
    md += ["\n## Calibration: did X% picks land X% of the time?\n", cal.round(3).to_markdown(), "\n"]
    for n in (5, 10, 20):
        top = g.nlargest(n, "p_final")
        md.append(f"- Top {n} picks: {int(top.over15.sum())}/{n} landed (expected {top.p_final.sum():.1f}); "
                  f"all {n} together as one accumulator: {'WON' if top.over15.all() else 'lost'} "
                  f"(model chance {top.p_final.prod():.1%})\n")
    misses = g[g.over15 == 0].sort_values("p_final", ascending=False).head(10)
    md += ["\n## Biggest misses\n", misses[["date", "league", "home", "away", "hg", "ag", "p_final"]]
           .assign(date=misses.date.dt.date).round(3).to_markdown(index=False), "\n"]
    out = ROOT / f"reports/grade_{Path(path).stem}.md"
    out.write_text("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1])
