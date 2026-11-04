"""Leakage-free pre-match features. Every feature for a match uses only information
available strictly before kickoff (stats are shifted by one match per team).
Output: data/processed/features.parquet
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ratings import compute_ratings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed" / "matches.parquet"
OUT = ROOT / "data" / "processed" / "features.parquet"
PARAMS = ROOT / "data" / "processed" / "rating_params.json"  # written by tune_ratings.py
# Fallback = tuned values from reports/ratings_tuning.md
DEFAULT_PARAMS = {
    "goal": {"k": 15.0, "regress": 0.1, "k_early": 1.5, "regress_to": "league"},
    "shot": {"w_shots": 1.0, "gd_mult": False, "hfa": 40.0, "k": 60.0, "regress": 0.2, "k_early": 1.0,
             "regress_to": "league"},
}

COUNTRY = {  # team identity = (country, name) so promoted/relegated teams keep history
    "E0": "ENG", "E1": "ENG", "E2": "ENG", "E3": "ENG", "EC": "ENG",
    "SC0": "SCO", "SC1": "SCO", "SC2": "SCO", "SC3": "SCO",
    "D1": "GER", "D2": "GER", "SP1": "ESP", "SP2": "ESP", "I1": "ITA", "I2": "ITA",
    "F1": "FRA", "F2": "FRA", "N1": "NED", "B1": "BEL", "P1": "POR", "T1": "TUR", "G1": "GRE",
}


# ---------------------------------------------------------------- rolling team form
STATS = {  # long-format column: (home col, away col)
    "gf": ("hg", "ag"), "ga": ("ag", "hg"),
    "sf": ("hs", "as_"), "sa": ("as_", "hs"),
    "stf": ("hst", "ast"), "sta": ("ast", "hst"),
    "cf": ("hc", "ac"), "ca": ("ac", "hc"),
}


STD_COLS = ["pts", "gd", "sd", "sotd"]
SEASON_OF = {}


def team_form(df, halflives=(3, 10)):
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        part = pd.DataFrame({"match_id": df.match_id, "date": df.date, "team": df[f"{side}_id"],
                             "is_home": side == "home"})
        for name, (hc, ac) in STATS.items():
            part[name] = df[hc] if side == "home" else df[ac]
        part["gd"] = part["gf"] - part["ga"]
        part["sd"] = part["sf"] - part["sa"]
        part["sotd"] = part["stf"] - part["sta"]
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
    # season-to-date averages: reset every season, strictly prior matches only
    long["season"] = long["match_id"].map(SEASON_OF)
    gs = long.groupby(["team", "season"], sort=False)
    keys = [long["team"], long["season"]]
    vals, present = long[STD_COLS].fillna(0), long[STD_COLS].notna().astype(int)
    prior_sum = vals.groupby(keys).cumsum() - vals
    prior_cnt = present.groupby(keys).cumsum() - present
    std = prior_sum.where(prior_cnt > 0) / prior_cnt.where(prior_cnt > 0)
    for c in STD_COLS:
        out[f"std_{c}"] = std[c].values
    out["std_n"] = gs.cumcount().values
    out["n_shot_prior"] = (long["sd"].notna().groupby(long["team"]).cumsum() - long["sd"].notna()).values
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
    params = json.loads(PARAMS.read_text()) if PARAMS.exists() else DEFAULT_PARAMS
    df["elo_h"], df["elo_a"], _, _ = compute_ratings(df, **params["goal"])
    df["elo_diff"] = df.elo_h - df.elo_a
    df["selo_h"], df["selo_a"], _, _ = compute_ratings(df, **params["shot"])
    SEASON_OF.update(dict(zip(df.match_id, df.season)))
    df = df.join(market_probs(df))
    form = team_form(df)
    df = df.merge(form, left_on="match_id", right_index=True, how="left")
    # shot rating is only meaningful once a team has shot data
    for side in ("h", "a"):
        df.loc[df[f"{side}_n_shot_prior"] < 5, f"selo_{side}"] = np.nan
    df["selo_diff"] = df.selo_h - df.selo_a
    for c in ["std_pts", "std_gd", "std_sd", "std_sotd"]:
        df[f"d_{c}"] = df[f"h_{c}"] - df[f"a_{c}"]
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
