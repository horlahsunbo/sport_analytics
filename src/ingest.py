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


# Leagues the xgabora dataset added from football-data.co.uk's "new" files. In these, every date with
# day <= 12 has day and month swapped (dd/mm read as mm/dd): verified against football-data.co.uk,
# 19,384 of 19,392 such matches only line up after swapping. See reports/data_quality.md.
XGABORA_SWAPPED_DATE_LEAGUES = {"ARG", "AUT", "BRA", "CHN", "DEN", "FIN", "IRL", "JAP", "MEX", "NOR", "POL",
                                "ROM", "RUS", "SUI", "SWE", "USA"}


def fix_swapped_dates(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"]
    swap = df["league"].isin(XGABORA_SWAPPED_DATE_LEAGUES) & (d.dt.day <= 12) & (d.dt.day != d.dt.month)
    fixed = pd.to_datetime(dict(year=d[swap].dt.year, month=d[swap].dt.day, day=d[swap].dt.month))
    df.loc[swap, "date"] = fixed
    print(f"fixed swapped day/month on {int(swap.sum()):,} matches in {df.loc[swap, 'league'].nunique()} leagues")
    return df


def load_xgabora() -> pd.DataFrame:
    df = pd.read_csv(RAW / "xgabora" / "Matches.csv", low_memory=False)
    df = df.rename(columns=RENAME)[list(RENAME.values())]
    df["date"] = pd.to_datetime(df["date"])
    df = fix_swapped_dates(df)
    df["source"] = "xgabora"
    return df


SOURCES = [load_xgabora]

# ---------------------------------------------------------------- football-data.co.uk enrichment
FD_LEAGUE = {"DNK": "DEN", "JPN": "JAP", "ROU": "ROM", "SWZ": "SUI"}  # their code -> ours
# Over/under 2.5 goals: canonical name -> candidate football-data columns (first one present wins).
# Older files (2012-13..2018-19) only have Betbrain average/max ("BbAv"/"BbMx"); Pinnacle O/U from 2019-20.
FD_OU = {
    "ou_avg": ["Avg", "BbAv"], "ou_max": ["Max", "BbMx"], "ou_ps_open": ["P"], "ou_ps_close": ["PC"],
    "ou_avg_close": ["AvgC"], "ou_max_close": ["MaxC"],
}
# Asian handicap (2019-20 onward): line (home handicap) + home/away odds, opening and closing.
FD_AH = {"ah_line": "AHh", "ah_line_close": "AHCh", "ah_ps_open_h": "PAHH", "ah_ps_open_a": "PAHA",
         "ah_ps_close_h": "PCAHH", "ah_ps_close_a": "PCAHA", "ah_avg_close_h": "AvgCAHH",
         "ah_avg_close_a": "AvgCAHA", "ah_max_close_h": "MaxCAHH", "ah_max_close_a": "MaxCAHA"}
FD_TEAM = {"Tigres UANL": "U.A.N.L.- Tigres", "UNAM Pumas": "U.N.A.M.- Pumas"}  # their name -> ours
FD_ODDS = {  # canonical name: football-data column (H/D/A suffix appended)
    "ps_open": "PS", "ps_close": "PSC", "avg_close": "AvgC", "max_close": "MaxC",
    "bfe_close": "BFEC", "b365_close": "B365C",
}


def _read_fd(path):
    try:
        d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        d = pd.read_csv(path, encoding="latin-1", low_memory=False)
    d.columns = [c.replace("\ufeff", "").replace("ï»¿", "").strip() for c in d.columns]
    return d


def load_footballdata() -> pd.DataFrame:
    """Odds-only table keyed by (league, date, home, away). Used to ENRICH existing matches."""
    frames = []
    for f in sorted((RAW / "footballdata").glob("*/*.csv")):
        d = _read_fd(f)
        if "HomeTeam" in d:  # main-league files
            d = d.rename(columns={"Div": "league", "HomeTeam": "home", "AwayTeam": "away", "FTHG": "hg", "FTAG": "ag"})
        else:  # extra-league files
            d = d.rename(columns={"Home": "home", "Away": "away", "HG": "hg", "AG": "ag"})
            d["league"] = FD_LEAGUE.get(f.stem, f.stem)
        d = d.dropna(subset=["home", "away", "Date"])
        out = pd.DataFrame({"league": d["league"], "home": d["home"].str.strip().replace(FD_TEAM),
                            "away": d["away"].str.strip().replace(FD_TEAM),
                            "date": pd.to_datetime(d["Date"], dayfirst=True, format="mixed"),
                            "fd_hg": pd.to_numeric(d["hg"], errors="coerce"),
                            "fd_ag": pd.to_numeric(d["ag"], errors="coerce")})
        for name, pre in FD_ODDS.items():
            for o, suf in zip("hda", "HDA"):
                col = pre + suf
                out[f"{name}_{o}"] = pd.to_numeric(d[col], errors="coerce") if col in d else np.nan
        for name, prefixes in FD_OU.items():
            for o, sign in (("o", ">"), ("u", "<")):
                col = next((p + sign + "2.5" for p in prefixes if p + sign + "2.5" in d), None)
                out[f"{name}_{o}"] = pd.to_numeric(d[col], errors="coerce") if col else np.nan
        for name, col in FD_AH.items():
            out[name] = pd.to_numeric(d[col], errors="coerce") if col in d else np.nan
        frames.append(out)
    fd = pd.concat(frames, ignore_index=True)
    odds_cols = [c for c in fd.columns if c[-2:] in ("_h", "_d", "_a", "_o", "_u") and not c.startswith("ah_line")]
    fd[odds_cols] = fd[odds_cols].where(fd[odds_cols] > 1.0)
    return fd.drop_duplicates(subset=["league", "date", "home", "away"])


def enrich_footballdata(df: pd.DataFrame) -> pd.DataFrame:
    """Attach football-data.co.uk odds. Exact (league, date, home, away) match first; then the same
    fixture within +-2 days (kick-off times near midnight / timezone differences), checked by score."""
    if not (RAW / "footballdata").exists():
        print("football-data.co.uk files not found - skipping closing-odds enrichment")
        return df
    fd = load_footballdata()
    keys = ["league", "date", "home", "away"]
    m = df.merge(fd, on=keys, how="left", indicator=True)
    hit = m["_merge"] == "both"
    m = m.drop(columns="_merge")
    # fallback: fuzzy date
    miss = m.loc[~hit, ["match_id", "league", "date", "home", "away", "hg", "ag"]]
    cand = miss.merge(fd, on=["league", "home", "away"], suffixes=("", "_fd"))
    cand = cand[((cand.date - cand.date_fd).abs() <= pd.Timedelta(days=2)) & (cand.hg == cand.fd_hg) & (cand.ag == cand.fd_ag)]
    cand = cand.drop_duplicates("match_id").set_index("match_id")
    fill = [c for c in fd.columns if c not in keys]
    m = m.set_index("match_id")
    m.loc[cand.index, fill] = cand[fill]
    m = m.reset_index()
    matched = m["fd_hg"].notna()
    # sanity: scores must agree
    bad = matched & ((m.fd_hg != m.hg) | (m.fd_ag != m.ag))
    m.loc[bad, fill] = np.nan
    since = m.season >= 2012
    print(f"football-data enrichment: {int(hit.sum()):,} exact + {len(cand):,} date-shifted matches; "
          f"{int(bad.sum())} rejected for score mismatch; "
          f"Pinnacle closing odds on {m.loc[since, 'ps_close_h'].notna().mean():.1%} of matches since 2012-13")
    return m.drop(columns=["fd_hg", "fd_ag"])


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
    df = enrich_footballdata(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {len(df):,} matches, {df.league.nunique()} leagues, "
          f"{df.date.min().date()}..{df.date.max().date()} -> {OUT}")


if __name__ == "__main__":
    main()
