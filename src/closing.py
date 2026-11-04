"""Closing-odds evaluation (needs football-data.co.uk enrichment from ingest.py).

The closing price, especially Pinnacle's, is the sharpest public estimate of a match. Questions:
  1. Benchmarks: how good are average pre-match odds, Pinnacle opening and Pinnacle closing?
  2. Can our models beat the closing price (log loss)?
  3. Betting at REAL closing prices: does any model edge survive?
  4. Closing line value (CLV): when we take the early price, is it better than where the line closes?
     Positive CLV is the standard evidence of a genuine edge; it needs far fewer bets than ROI.
  5. Favourite-longshot bias re-tested at Pinnacle close, with two de-vig methods.
  6. Line movement: is there information left after the close? Can our model predict the move?
Writes reports/closing_report.md
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "closing_report.md"
rng = np.random.default_rng(0)


def devig_prop(o):
    inv = 1 / o
    return inv / inv.sum(1, keepdims=True)


def devig_power(o, iters=60):
    """Power method: p_i = (1/o_i)^k with k chosen so probabilities sum to 1. Removes margin
    more from longshots than favourites (proportional spreads it evenly)."""
    inv = 1 / o
    lo, hi = np.full(len(o), 0.5), np.full(len(o), 3.0)
    for _ in range(iters):
        k = (lo + hi) / 2
        s = (inv ** k[:, None]).sum(1)
        lo = np.where(s > 1, k, lo); hi = np.where(s > 1, hi, k)
    return inv ** ((lo + hi) / 2)[:, None]


def ll(p, y):
    return log_loss(y, np.clip(p, 1e-6, 1), labels=[0, 1, 2])


def boot_ci(x, n=1000):
    if len(x) == 0:
        return np.nan, np.nan
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    pr = pd.read_parquet(ROOT / "data/processed/predictions.parquet")
    mt = pd.read_parquet(ROOT / "data/processed/matches.parquet")
    oc = [c for c in mt.columns if c.startswith(("ps_", "avg_close", "max_close", "bfe_close", "b365_close"))]
    df = pr.merge(mt[["match_id"] + oc], on="match_id", how="left")
    PO, PC = ["ps_open_h", "ps_open_d", "ps_open_a"], ["ps_close_h", "ps_close_d", "ps_close_a"]
    df = df.dropna(subset=PC + PO + ["odds_h", "odds_d", "odds_a"]).reset_index(drop=True)
    y = df.y.to_numpy()
    p_close = devig_prop(df[PC].to_numpy()); p_open = devig_prop(df[PO].to_numpy())
    P = {"avg pre-match odds (current benchmark)": df[["mkt_h", "mkt_d", "mkt_a"]].to_numpy(),
         "Pinnacle opening": p_open,
         "Pinnacle closing": p_close,
         "Pinnacle closing (power de-vig)": devig_power(df[PC].to_numpy()),
         "our stats model": df[["p_stats_h", "p_stats_d", "p_stats_a"]].to_numpy(),
         "our hybrid model": df[["p_hyb_h", "p_hyb_d", "p_hyb_a"]].to_numpy()}
    md = ["# Closing-odds evaluation\n",
          f"{len(df):,} out-of-sample matches ({df.season.min()}-{df.season.max()}, {df.league.nunique()} leagues) "
          "that have average pre-match odds, Pinnacle opening and Pinnacle closing odds.\n"]

    # 1-2. benchmarks
    t = pd.DataFrame({k: {"logloss": ll(v, y), "accuracy": (v.argmax(1) == y).mean()} for k, v in P.items()}).T
    md += ["## 1. Who predicts best? (log loss, lower is better)\n", t.round(4).to_markdown(), "\n"]
    margin = pd.DataFrame({"avg pre-match": (1 / df[["odds_h", "odds_d", "odds_a"]]).sum(1) - 1,
                           "Pinnacle open": (1 / df[PO]).sum(1) - 1, "Pinnacle close": (1 / df[PC]).sum(1) - 1}).median()
    md.append("Median bookmaker margin: " + ", ".join(f"{k} {v:.1%}" for k, v in margin.items()) + "\n")

    # 3. betting at real closing prices
    rows = []
    prices = {"Pinnacle close": df[PC].to_numpy(),
              "best close (2019+)": df[["max_close_h", "max_close_d", "max_close_a"]].to_numpy()}
    for model in ("our stats model", "our hybrid model"):
        p = P[model]
        for pname, o in prices.items():
            for thr in (0.0, 0.03, 0.08):
                ev = p * o - 1
                ev = np.where(np.isnan(ev), -np.inf, ev)
                pick = ev.argmax(1); mask = ev.max(1) > thr
                prof = np.where(pick[mask] == y[mask], o[mask, pick[mask]] - 1, -1.0)
                lo, hi = boot_ci(prof, 400)
                rows.append(dict(model=model, price=pname, min_edge=thr, bets=int(mask.sum()),
                                 roi=prof.mean() if len(prof) else np.nan, ci_low=lo, ci_high=hi))
    md += ["\n## 2. Betting at real closing prices (flat stakes, 95% CI)\n",
           "Bet the outcome with the largest model edge when edge > min_edge at that price.\n",
           pd.DataFrame(rows).round(4).to_markdown(index=False), "\n"]

    # 4. CLV: take the early (avg pre-match) price on the model's pick, compare with fair Pinnacle close
    o_early = df[["odds_h", "odds_d", "odds_a"]].to_numpy()
    md.append("\n## 3. Closing line value (CLV)\n\nWe take the early average price on the model's pick; CLV = early odds / "
              "fair Pinnacle closing odds - 1. Positive mean CLV = we systematically got better prices than the "
              "sharpest final price, i.e. a real edge. Average-odds margins make CLV negative by default, so the "
              "'random picks' row is the baseline to beat.\n")
    rows = []
    fair_close = 1 / p_close
    for model in ("our stats model", "our hybrid model"):
        p = P[model]
        for thr in (0.0, 0.05, 0.10):
            ev = p * o_early - 1; pick = ev.argmax(1); mask = ev.max(1) > thr
            clv = o_early[mask, pick[mask]] / fair_close[mask, pick[mask]] - 1
            lo, hi = boot_ci(clv, 400)
            rows.append(dict(model=model, min_edge=thr, bets=int(mask.sum()), mean_clv=clv.mean(), ci_low=lo, ci_high=hi,
                             share_beat_close=(clv > 0).mean()))
    rp = rng.integers(0, 3, len(df))
    clv_r = o_early[np.arange(len(df)), rp] / fair_close[np.arange(len(df)), rp] - 1
    rows.append(dict(model="random picks (baseline)", min_edge=np.nan, bets=len(df), mean_clv=clv_r.mean(),
                     ci_low=np.nan, ci_high=np.nan, share_beat_close=(clv_r > 0).mean()))
    md += [pd.DataFrame(rows).round(4).to_markdown(index=False), "\n"]

    # 5. favourite-longshot bias at Pinnacle close
    md.append("\n## 4. Favourite-longshot bias at Pinnacle close\n\nHome win. gap = actual - priced; z > 3 is a real "
              "miscalibration. ROI = flat bet on every home win in the bucket at that price.\n")
    win = (y == 0).astype(float)
    pp, pw = p_close[:, 0], devig_power(df[PC].to_numpy())[:, 0]
    oc_h, mx_h = df.ps_close_h.to_numpy(), df.max_close_h.to_numpy()
    rows = []
    for lo_, hi_ in [(0, .2), (.2, .4), (.4, .6), (.6, .7), (.7, .8), (.8, 1)]:
        m = (pp > lo_) & (pp <= hi_)
        n = m.sum(); act = win[m].mean()
        r_pin = np.where(win[m] == 1, oc_h[m] - 1, -1.0)
        mm = m & ~np.isnan(mx_h)
        r_max = np.where(win[mm] == 1, mx_h[mm] - 1, -1.0)
        lo_ci, hi_ci = boot_ci(r_pin, 400)
        rows.append({"Pinnacle prob": f"{lo_}-{hi_}", "n": int(n), "priced (prop.)": pp[m].mean(),
                     "priced (power)": pw[m].mean(), "actual": act,
                     "z (prop.)": (act - pp[m].mean()) / np.sqrt(pp[m].mean() * (1 - pp[m].mean()) / n),
                     "z (power)": (act - pw[m].mean()) / np.sqrt(pw[m].mean() * (1 - pw[m].mean()) / n),
                     "ROI Pinnacle close": r_pin.mean(), "ci_low": lo_ci, "ci_high": hi_ci,
                     "ROI best close (2019+)": r_max.mean() if len(r_max) else np.nan, "n best": int(mm.sum())})
    md += [pd.DataFrame(rows).round(4).to_markdown(index=False), "\n"]

    # 6. movement
    move = p_close[:, 0] - p_open[:, 0]
    md.append("\n## 5. Line movement (Pinnacle open -> close, home win probability)\n\n"
              "If the close is efficient, `actual - close` is ~0 in every bucket: the move is fully priced.\n")
    b = pd.cut(move, [-1, -.05, -.02, -.005, .005, .02, .05, 1])
    t = pd.DataFrame({"move": b, "open": p_open[:, 0], "close": p_close[:, 0], "actual": win}).groupby("move", observed=True).agg(
        n=("actual", "size"), open=("open", "mean"), close=("close", "mean"), actual=("actual", "mean"))
    t["actual - close"] = t.actual - t.close
    t["z"] = t["actual - close"] / np.sqrt(t.close * (1 - t.close) / t.n)
    md += [t.round(4).to_markdown(), "\n"]
    for model, key in (("stats", "p_stats_h"), ("hybrid", "p_hyb_h")):
        disagree = df[key].to_numpy() - p_open[:, 0]
        c = np.corrcoef(disagree, move)[0, 1]
        md.append(f"- Does our **{model}** model predict the move? corr(model - Pinnacle open, close - open) = **{c:.3f}**\n")
    md.append("\nA positive correlation means the model anticipates where informed money pushes the line.\n")

    REPORT.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
