"""Fetch raw sources into data/raw/. Raw files are not committed; re-run this to rebuild.

Sources
  xgabora          238k matches 2000-2026 with average pre-match odds, shots, Elo (GitHub mirror)
  football-data    football-data.co.uk: Pinnacle opening + closing odds, average/best closing odds,
                   Betfair Exchange closing odds. Closing odds exist from 2012-13 onwards.
Usage: python3 src/download.py [xgabora|footballdata ...]   (default: all)
"""
import sys
import time
from pathlib import Path
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
XGABORA = {
    "xgabora/Matches.csv": "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv",
    "xgabora/EloRatings.csv": "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/EloRatings.csv",
}
FD_MAIN = ["E0", "E1", "E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3", "D1", "D2", "I1", "I2", "SP1", "SP2",
           "F1", "F2", "N1", "B1", "P1", "T1", "G1"]
FD_EXTRA = ["ARG", "AUT", "BRA", "CHN", "DNK", "FIN", "IRL", "JPN", "MEX", "NOR", "POL", "ROU", "RUS", "SWE",
            "SWZ", "USA"]
FD_FIRST_SEASON = 2012  # first season with Pinnacle closing odds
FD_BASE = "https://www.football-data.co.uk"


def fetch(url, dest, session, skip_existing=False):
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        return "cached"
    r = session.get(url, timeout=60)
    if r.status_code == 404:
        return "missing"
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    time.sleep(0.5)  # be polite to a free, volunteer-run site
    return f"{len(r.content) / 1e3:.0f} kB"


def download_xgabora(s):
    for rel, url in XGABORA.items():
        print(rel, fetch(url, RAW / rel, s))


def download_footballdata(s, last_season=2025):
    # past seasons never change -> cache them; always refresh the current season
    for start in range(FD_FIRST_SEASON, last_season + 1):
        code = f"{start % 100:02d}{(start + 1) % 100:02d}"
        for lg in FD_MAIN:
            st = fetch(f"{FD_BASE}/mmz4281/{code}/{lg}.csv", RAW / "footballdata" / code / f"{lg}.csv", s,
                       skip_existing=start < last_season)
            if st not in ("cached",):
                print(f"footballdata {code}/{lg}: {st}")
    for lg in FD_EXTRA:  # one file per league covering all seasons
        print(f"footballdata new/{lg}:", fetch(f"{FD_BASE}/new/{lg}.csv", RAW / "footballdata" / "new" / f"{lg}.csv", s))


if __name__ == "__main__":
    which = sys.argv[1:] or ["xgabora", "footballdata"]
    with requests.Session() as s:
        s.headers["User-Agent"] = "sport-analytics-research/1.0"
        if "xgabora" in which:
            download_xgabora(s)
        if "footballdata" in which:
            download_footballdata(s)
