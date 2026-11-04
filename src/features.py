"""Leakage-free pre-match features. Every feature for a match uses only information
available strictly before kickoff (stats are shifted by one match per team).
Output: data/processed/features.parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed" / "matches.parquet"
OUT = ROOT / "data" / "processed" / "features.parquet"

COUNTRY = {  # team identity = (country, name) so promoted/relegated teams keep history
    "E0": "ENG", "E1": "ENG", "E2": "ENG", "E3": "ENG", "EC": "ENG",
    "SC0": "SCO", "SC1": "SCO", "SC2": "SCO", "SC3": "SCO",
    "D1": "GER", "D2": "GER", "SP1": "ESP", "SP2": "ESP", "I1": "ITA", "I2": "ITA",
    "F1": "FRA", "F2": "FRA", "N1": "NED", "B1": "BEL", "P1": "POR", "T1": "TUR", "G1": "GRE",
}


# ---------------------------------------------------------------- Elo
def elo_ratings(df, k=20.0, hfa=60.0, regress=0.2):
    """Goal-difference-weighted Elo, regressed toward 1500 at each new season.
    Returns pre-match ratings (home, away)."""
    r, last_season = {}, {}
    pre_h = np.empty(len(df)); pre_a = np.empty(len(df))
    for i, (h, a, hg, ag, s) in enumerate(zip(df.home_id, df.away_id, df.hg, df.ag, df.season)):
        for t in (h, a):
            if t not in r:
                r[t] = 1450.0  # new teams are usually promoted -> below average
            elif last_season[t] != s:
                r[t] = r[t] * (1 - regress) + 1500 * regress
            last_season[t] = s
        rh, ra = r[h], r[a]
        pre_h[i], pre_a[i] = rh, ra
        exp_h = 1 / (1 + 10 ** ((ra - rh - hfa) / 400))
        res = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        gd = abs(hg - ag)
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        delta = k * mult * (res - exp_h)
        r[h] = rh + delta; r[a] = ra - delta
    return pre_h, pre_a


# ---------------------------------------------------------------- rolling team form
STATS = {  # long-format column: (home col, away col)
    "gf": ("hg", "ag"), "ga": ("ag", "hg"),
    "sf": ("hs", "as_"), "sa": ("as_", "hs"),
    "stf": ("hst", "ast"), "sta": ("ast", "hst"),
    "cf": ("hc", "ac"), "ca": ("ac", "hc"),
}


def team_form(df, halflives=(3, 10)):
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        part = pd.DataFrame({"match_id": df.match_id, "date": df.date, "team": df[f"{side}_id"],
                             "is_home": side == "home"})
        for name, (hc, ac) in STATS.items():
            part[name] = df[hc] if side == "home" else df[ac]
        part["pts"] = np.where(df.hg == df.ag, 1, np.where((df.hg > df.ag) == (side == "home"), 3, 0))
        rows.append(part)
    long = pd.concat(rows).sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    g = long.groupby("team", sort=False)
    out = long[["match_id", "team", "is_home"]].copy()
    cols = list(STATS) + ["pts"]
    shifted = g[cols].shift(1)  # strictly prior matches
    for hl in halflives:
        ew = shifted.groupby(long["team"]).transform(lambda s: s.ewm(halflife=hl, ignore_na=True).mean())
        for c in cols:
            out[f"{c}_ew{hl}"] = ew[c].values
    out["n_prior"] = g.cumcount().values
    out["rest_days"] = g["date"].diff().dt.days.values
    feats = [c for c in out.columns if c not in ("match_id", "team", "is_home")]
    h = out[out.is_home].set_index("match_id")[feats].add_prefix("h_")
    a = out[~out.is_home].set_index("match_id")[feats].add_prefix("a_")
    return h.join(a)


# ---------------------------------------------------------------- market
def market_probs(df):
    inv = 1 / df[["odds_h", "odds_d", "odds_a"]].to_numpy()
    over = inv.sum(1)
    p = inv / over[:, None]  # proportional de-vig
    return pd.DataFrame({"mkt_h": p[:, 0], "mkt_d": p[:, 1], "mkt_a": p[:, 2], "overround": over},
                        index=df.index)


def main():
    df = pd.read_parquet(IN)
    ctry = df.league.map(COUNTRY).fillna(df.league)
    df["home_id"] = ctry + ":" + df.home
    df["away_id"] = ctry + ":" + df.away
    df["elo_h"], df["elo_a"] = elo_ratings(df)
    df["elo_diff"] = df.elo_h - df.elo_a
    df = df.join(market_probs(df))
    form = team_form(df)
    df = df.merge(form, left_on="match_id", right_index=True, how="left")
    for c in ["gf", "ga", "sf", "sa", "stf", "sta", "pts"]:
        for hl in (3, 10):
            df[f"d_{c}_ew{hl}"] = df[f"h_{c}_ew{hl}"] - df[f"a_{c}_ew{hl}"]
    # league home-advantage prior: expanding home-win rate, prior matches only
    df["is_hw"] = (df.result == "H").astype(float)
    df["lg_home_rate"] = df.groupby("league")["is_hw"].transform(lambda s: s.shift(1).expanding().mean())
    df["month"] = df.date.dt.month
    df["y"] = df.result.map({"H": 0, "D": 1, "A": 2})
    df.to_parquet(OUT, index=False)
    print(f"wrote {len(df):,} rows x {df.shape[1]} cols -> {OUT}")


if __name__ == "__main__":
    main()
