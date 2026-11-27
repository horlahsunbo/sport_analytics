"""Accumulator ("acca") planner and historical replay for over 1.5 / over 2.5 slips.

Replays, for every weekend 2015-2025, a slip built from the model's most confident picks (models trained
walk-forward, so every pick is out-of-sample), and records how often it would have won and what it paid.
Then builds this weekend's candidate slips from the latest prediction files.

Over 2.5 uses real average bookmaker odds. Our data has no over 1.5 odds, so those are ESTIMATED from the
market probability plus a bookmaker margin (MARGIN_15); check them against your bookmaker before trusting.
Only matches with market odds are used (model-only leagues are weaker, see reports/backtest_*).

  python3 src/acca.py --o15 predictions/2026-11-27_over15.csv \
                      --o25 predictions/2026-11-27_2026-11-30_over25_frozen2026-11-27.csv --target 10
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MARGIN_15 = 0.06  # assumed bookmaker margin on over 1.5 prices (typical 2-way margin 4-8%)


def est_odds_15(p_market):
    return 1 / (p_market * (1 + MARGIN_15))


def replay(wf, legs, odds_col):
    """wf: walk-forward predictions with target, p_model, odds. One slip per Fri-Mon weekend."""
    d = wf.dropna(subset=["p_model", odds_col]).copy()
    d = d[pd.to_datetime(d.date).dt.dayofweek.isin([4, 5, 6, 0])]
    wk = pd.to_datetime(d.date) - pd.to_timedelta((pd.to_datetime(d.date).dt.dayofweek - 4) % 7, unit="D")
    d["weekend"] = wk.dt.normalize()
    rows = []
    for w, g in d.groupby("weekend"):
        if len(g) < legs:
            continue
        s = g.nlargest(legs, "p_model")
        rows.append({"weekend": w, "odds": s[odds_col].prod(), "p_model": s.p_model.prod(),
                     "won": bool((s.target == 1).all()), "legs_hit": int(s.target.sum())})
    r = pd.DataFrame(rows)
    r["profit"] = np.where(r.won, r.odds - 1, -1.0)
    return r


def summary(r, name):
    return {"slip": name, "weekends": len(r), "median total odds": round(r.odds.median(), 2),
            "won": int(r.won.sum()), "win rate": f"{r.won.mean():.1%}", "model said": f"{r.p_model.mean():.1%}",
            "avg legs landed": round(r.legs_hit.mean(), 1), "return per 1 staked": f"{r.profit.mean():+.0%}"}


def slip(df, legs, odds_col, prob_col="p_model"):
    s = df.dropna(subset=[odds_col, prob_col]).nlargest(legs, prob_col)
    return s, s[odds_col].prod(), s[prob_col].prod()


def legs_to_target(df, odds_col, target, prob_col="p_model"):
    s = df.dropna(subset=[odds_col, prob_col]).sort_values(prob_col, ascending=False)
    cum = s[odds_col].cumprod().to_numpy()
    n = int(np.argmax(cum >= target)) + 1 if (cum >= target).any() else None
    return n, (s.head(n)[prob_col].prod() if n else None)


def best_slip(df, odds_col, target, legs, pool=24, prob_col="p_model"):
    """Highest model win chance among slips of exactly `legs` picks whose total odds reach `target`
    (exhaustive search over the `pool` most confident picks)."""
    from itertools import combinations
    c = df.dropna(subset=[odds_col, prob_col]).nlargest(pool, prob_col).reset_index(drop=True)
    lp, lo = np.log(c[prob_col].to_numpy()), np.log(c[odds_col].to_numpy())
    best, best_lp = None, -np.inf
    for combo in combinations(range(len(c)), legs):
        idx = list(combo)
        if lo[idx].sum() >= np.log(target) and lp[idx].sum() > best_lp:
            best, best_lp = idx, lp[idx].sum()
    if best is None:
        return None, None, None
    s = c.loc[best]
    return s, s[odds_col].prod(), s[prob_col].prod()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--o15", required=True); ap.add_argument("--o25", required=True)
    ap.add_argument("--target", type=float, default=10.0)
    ap.add_argument("--legs15", type=int, default=15); ap.add_argument("--legs25", type=int, default=10)
    a = ap.parse_args()
    md = [f"# Accumulator planner: {a.legs15} x over 1.5 vs {a.legs25} x over 2.5 (target odds {a.target:g})\n",
          f"Over 1.5 odds are estimates (market probability + {MARGIN_15:.0%} margin); over 2.5 odds are real "
          "average bookmaker odds. Only matches with bookmaker odds are used.\n"]

    # ---- history
    w25 = pd.read_parquet(ROOT / "data/processed/ou25_walkforward.parquet")
    w15 = pd.read_parquet(ROOT / "data/processed/ou15_walkforward.parquet")
    w15["odds_est"] = est_odds_15(w15.p_market)
    hist = []
    for legs in sorted({5, 8, a.legs15, 20}):
        hist.append(summary(replay(w15, legs, "odds_est"), f"{legs} x over 1.5 (estimated odds)"))
    for legs in sorted({3, 5, a.legs25, 7}):
        hist.append(summary(replay(w25, legs, "odds_over"), f"{legs} x over 2.5"))
    md += ["## How each slip would have done every weekend, 2015-2025\n",
           "Each weekend: take the model's most confident picks (out-of-sample), all legs must win.\n",
           pd.DataFrame(hist).to_markdown(index=False), "\n"]

    # ---- this weekend
    o15 = pd.read_csv(ROOT / a.o15); o25 = pd.read_csv(ROOT / a.o25)
    o15 = o15[o15.p_market.notna()].copy(); o15["p_model"] = o15.p_final
    o15["odds_est"] = est_odds_15(o15.p_market); o15["fair_odds"] = 1 / o15.p_model
    o25 = o25[o25.p_market.notna()].copy(); o25["fair_odds"] = 1 / o25.p_model
    s15, od15, p15 = slip(o15, a.legs15, "odds_est")
    s25, od25, p25 = slip(o25, a.legs25, "odds_over")
    n15, pn15 = legs_to_target(o15, "odds_est", a.target)
    n25, pn25 = legs_to_target(o25, "odds_over", a.target)
    cmp_rows = [
        {"slip": f"{a.legs15} x over 1.5", "total odds (est.)": round(od15, 2), "model win chance": f"{p15:.1%}",
         "expected return": f"{p15 * od15 - 1:+.0%}"},
        {"slip": f"{a.legs25} x over 2.5", "total odds (avg odds)": round(od25, 2), "model win chance": f"{p25:.1%}",
         "expected return": f"{p25 * od25 - 1:+.0%}"},
        {"slip": f"fewest over 1.5 legs to reach {a.target:g}", "legs": n15,
         "model win chance": f"{pn15:.1%}" if pn15 else "target not reachable"},
        {"slip": f"fewest over 2.5 legs to reach {a.target:g}", "legs": n25,
         "model win chance": f"{pn25:.1%}" if pn25 else "target not reachable"},
    ]
    for legs in (6, 7, 8):
        b, bo, bp = best_slip(o25, "odds_over", a.target, legs)
        cmp_rows.append({"slip": f"best {legs} x over 2.5 reaching {a.target:g}", "total odds (avg odds)": round(bo, 2) if bo else None,
                         "model win chance": f"{bp:.1%}" if bp else "not reachable",
                         "expected return": f"{bp * bo - 1:+.0%}" if bp else None})
        if legs == 7 and b is not None:
            best7 = b
    md += ["\n## This weekend\n", pd.DataFrame(cmp_rows).to_markdown(index=False), "\n"]
    if "best7" in locals():
        t = best7.assign(match=best7.home + " v " + best7.away, model=(best7.p_model * 100).round(1),
                         market=(best7.p_market * 100).round(1), fair=(1 / best7.p_model).round(2), bookmaker=best7.odds_over.round(2))
        md += [f"\n### Best 7 x over 2.5 reaching {a.target:g}\n",
               t[["date", "time", "league", "match", "model", "market", "fair", "bookmaker"]].to_markdown(index=False), "\n"]
    for name, s, oc in ((f"{a.legs15} x over 1.5", s15, "odds_est"), (f"{a.legs25} x over 2.5", s25, "odds_over")):
        t = s.assign(match=s.home + " v " + s.away, model=(s.p_model * 100).round(1),
                     market=(s.p_market * 100).round(1), fair=s.fair_odds.round(2), bookmaker=s[oc].round(2))
        md += [f"\n### {name}\n", t[["date", "time", "league", "match", "model", "market", "fair", "bookmaker"]]
               .to_markdown(index=False), "\n"]
    out = ROOT / f"reports/acca_{Path(a.o25).stem.split('_')[0]}.md"
    out.write_text("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main()
