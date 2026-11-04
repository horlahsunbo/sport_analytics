"""Paired comparison of two prediction files on identical matches (default: v1 vs current).
Writes reports/v2_vs_v1.md"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
old = pd.read_parquet(ROOT / "data/processed/predictions_v1.parquet")
new = pd.read_parquet(ROOT / "data/processed/predictions.parquet")
m = old.merge(new, on="match_id", suffixes=("_v1", "_v2"))
y = m.y_v1.to_numpy()


def nll(cols):
    p = np.clip(m[cols].to_numpy(), 1e-6, 1)
    return -np.log(p[np.arange(len(m)), y])


rows, per = [], {}
for name, c in [("market", "mkt"), ("stats", "p_stats"), ("hybrid", "p_hyb")]:
    for v in ("v1", "v2"):
        per[(name, v)] = nll([f"{c}_{o}_{v}" for o in "hda"])
for name in ("stats", "hybrid"):
    d = per[(name, "v2")] - per[(name, "v1")]
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, len(d)).mean() for _ in range(1000)]
    rows.append(dict(model=name, v1=per[(name, "v1")].mean(), v2=per[(name, "v2")].mean(),
                     change=d.mean(), ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5),
                     market=per[("market", "v1")].mean(),
                     gap_closed=1 - (per[(name, "v2")].mean() - per[("market", "v1")].mean())
                     / (per[(name, "v1")].mean() - per[("market", "v1")].mean())))
summary = pd.DataFrame(rows)
m["early"] = np.where(m.date_v1.dt.month.isin([7, 8, 9]) | (m.league_v1.isin(["BRA", "CHN", "FIN", "IRL", "JAP", "NOR", "SWE", "USA"]) & m.date_v1.dt.month.isin([2, 3, 4])), "season start", "rest of season")
seg = pd.DataFrame({f"{n}_{v}": per[(n, v)] for n in ("market", "stats", "hybrid") for v in ("v1", "v2") if not (n == "market" and v == "v2")})
seg["phase"] = m.early.values
by_phase = seg.groupby("phase").mean()
seg["season"] = m.season_v1.values
by_season = seg.groupby("season")[["stats_v1", "stats_v2", "hybrid_v1", "hybrid_v2", "market_v1"]].mean()
md = ["# v2 (tuned season reset + shot rating + season-to-date stats) vs v1\n",
      f"Paired on {len(m):,} identical out-of-sample matches. Log loss, lower is better. "
      "`change` < 0 and CI entirely below 0 = a real improvement. `gap_closed` = share of the gap to the market removed.\n",
      summary.round(4).to_markdown(index=False), "\n## By season phase\n", by_phase.round(4).to_markdown(),
      "\n## By season\n", by_season.round(4).to_markdown()]
(ROOT / "reports/v2_vs_v1.md").write_text("\n".join(md))
print("\n".join(md))
