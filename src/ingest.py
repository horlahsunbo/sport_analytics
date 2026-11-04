"""Ingest raw sources into one clean, canonical match table (data/processed/matches.parquet).

Each source gets a loader returning the canonical schema. To add a source
(e.g. football-data.co.uk, Understat xG, Pinnacle closing odds), write a new
loader and append it in SOURCES.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "matches.parquet"

RENAME = {
    "Division": "league", "MatchDate": "date", "HomeTeam": "home", "AwayTeam": "away",
    "FTHome": "hg", "FTAway": "ag", "FTResult": "result", "HTHome": "hthg", "HTAway": "htag",
    "HomeShots": "hs", "AwayShots": "as_", "HomeTarget": "hst", "AwayTarget": "ast",
    "HomeCorners": "hc", "AwayCorners": "ac", "HomeFouls": "hf", "AwayFouls": "af",
    "HomeYellow": "hy", "AwayYellow": "ay", "HomeRed": "hr", "AwayRed": "ar",
    "OddHome": "odds_h", "OddDraw": "odds_d", "OddAway": "odds_a",
    "MaxHome": "max_h", "MaxDraw": "max_d", "MaxAway": "max_a",
    "Over25": "odds_o25", "Under25": "odds_u25",
    "HomeElo": "clubelo_h", "AwayElo": "clubelo_a",
}


def load_xgabora() -> pd.DataFrame:
    df = pd.read_csv(RAW / "xgabora" / "Matches.csv", low_memory=False)
    df = df.rename(columns=RENAME)[list(RENAME.values())]
    df["date"] = pd.to_datetime(df["date"])
    df["source"] = "xgabora"
    return df


SOURCES = [load_xgabora]


CALENDAR_LEAGUES = {"BRA", "CHN", "FIN", "IRL", "JAP", "NOR", "SWE", "USA"}


def season_of(league: pd.Series, d: pd.Series) -> np.ndarray:
    """Season label. Autumn-spring leagues: Jul-Jun labelled by starting year.
    Calendar-year leagues (Feb-Nov): the year. Argentina switched to calendar year in 2021
    (and ran a one-off calendar season in 2015)."""
    jul = np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1)
    cal = league.isin(CALENDAR_LEAGUES) | ((league == "ARG") & ((d.dt.year >= 2021) | (d.dt.year == 2015)))
    return np.where(cal, d.dt.year, jul)


def main():
    df = pd.concat([f() for f in SOURCES], ignore_index=True)
    df = df.dropna(subset=["hg", "ag", "result"])
    df = df[df["result"].isin(["H", "D", "A"])]
    df = df.drop_duplicates(subset=["league", "date", "home", "away"], keep="first")
    df["season"] = season_of(df["league"], df["date"])
    # sanity: odds must be > 1
    for c in ["odds_h", "odds_d", "odds_a", "max_h", "max_d", "max_a", "odds_o25", "odds_u25"]:
        df.loc[df[c] <= 1.0, c] = np.nan
    df = df.sort_values(["date", "league", "home"]).reset_index(drop=True)
    df["match_id"] = np.arange(len(df))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {len(df):,} matches, {df.league.nunique()} leagues, "
          f"{df.date.min().date()}..{df.date.max().date()} -> {OUT}")


if __name__ == "__main__":
    main()
