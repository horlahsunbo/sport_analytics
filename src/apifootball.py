"""API-Football (v3.football.api-sports.io) results loader.

The key is injected by the environment's network proxy (saved as a network secret), so no key appears
in code. Free plan: 100 requests/day and only dates from yesterday to tomorrow, so results must be
pulled within a day of the match (football-data.co.uk remains the bulk historical source).

  python3 src/apifootball.py fetch 2026-11-22        # one request per date ("yesterday" also accepted)
                                                     # raw JSON -> data/raw/apifootball/ (not committed)
                                                     # our 38 leagues -> data/results/apifootball/<date>.csv (committed)
  python3 src/apifootball.py match                   # map cached results onto our prediction files

Team names differ from football-data.co.uk, so each predicted match is paired with the API fixture in the
same league, within +-1 day, whose home and away names are most similar. Low-similarity pairs are
listed in reports/apifootball_matching.md for review. Scores are 90-minute scores (score.fulltime).
"""
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "apifootball"
OUT = ROOT / "data" / "processed" / "results_apifootball.parquet"
COMPACT = ROOT / "data" / "results" / "apifootball"  # small, committed: survives container resets
BASE = "https://v3.football.api-sports.io"
# API league id -> (our code, expected country, expected league name) - verified against the API response
LEAGUES = {
    39: ("E0", "England", "Premier League"), 40: ("E1", "England", "Championship"), 41: ("E2", "England", "League One"),
    42: ("E3", "England", "League Two"), 43: ("EC", "England", "National League"),
    179: ("SC0", "Scotland", "Premiership"), 180: ("SC1", "Scotland", "Championship"),
    183: ("SC2", "Scotland", "League One"), 184: ("SC3", "Scotland", "League Two"),
    78: ("D1", "Germany", "Bundesliga"), 79: ("D2", "Germany", "2. Bundesliga"),
    135: ("I1", "Italy", "Serie A"), 136: ("I2", "Italy", "Serie B"),
    140: ("SP1", "Spain", "La Liga"), 141: ("SP2", "Spain", "Segunda División"),
    61: ("F1", "France", "Ligue 1"), 62: ("F2", "France", "Ligue 2"), 88: ("N1", "Netherlands", "Eredivisie"),
    144: ("B1", "Belgium", "Jupiler Pro League"), 94: ("P1", "Portugal", "Primeira Liga"),
    203: ("T1", "Turkey", "Süper Lig"), 197: ("G1", "Greece", "Super League 1"),
    128: ("ARG", "Argentina", "Liga Profesional Argentina"), 218: ("AUT", "Austria", "Bundesliga"),
    71: ("BRA", "Brazil", "Serie A"), 169: ("CHN", "China", "Super League"), 119: ("DEN", "Denmark", "Superliga"),
    244: ("FIN", "Finland", "Veikkausliiga"), 357: ("IRL", "Ireland", "Premier Division"), 98: ("JAP", "Japan", "J1 League"),
    262: ("MEX", "Mexico", "Liga MX"), 103: ("NOR", "Norway", "Eliteserien"), 106: ("POL", "Poland", "Ekstraklasa"),
    283: ("ROM", "Romania", "Liga I"), 235: ("RUS", "Russia", "Premier League"), 113: ("SWE", "Sweden", "Allsvenskan"),
    207: ("SUI", "Switzerland", "Super League"), 253: ("USA", "USA", "Major League Soccer"),
}
FINISHED = {"FT", "AET", "PEN"}
STOP = {"fc", "cf", "afc", "sc", "ac", "as", "ss", "us", "sv", "club", "de", "cd", "ud", "rc", "rcd", "sd", "fk", "sk",
        "if", "bk", "ff", "cfc", "calcio", "1", "the", "town", "city", "united", "utd"}
ALIASES = {"man": "manchester", "nott'm": "nottingham", "nottm": "nottingham", "wolves": "wolverhampton",
           "ath": "athletic", "sp": "sporting", "ein": "eintracht", "m'gladbach": "monchengladbach",
           "inter": "internazionale", "psv eindhoven": "psv", "sheffield weds": "sheffield wednesday"}


def fetch(date):
    path = RAW / f"fixtures_{date}.json"
    r = requests.get(f"{BASE}/fixtures", params={"date": date}, timeout=60)
    r.raise_for_status()
    d = r.json()
    if d.get("errors"):
        raise SystemExit(f"API-Football error for {date}: {d['errors']}")
    RAW.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d))
    comp = parse(d["response"])
    COMPACT.mkdir(parents=True, exist_ok=True)
    comp.to_csv(COMPACT / f"{date}.csv", index=False)
    fin = comp.status.isin(FINISHED).sum() if len(comp) else 0
    print(f"{date}: {d.get('results')} fixtures worldwide; {len(comp)} in our leagues ({fin} finished) "
          f"-> {(COMPACT / f'{date}.csv').relative_to(ROOT)}")


def parse(response):
    rows = []
    for x in response:
        lid = x["league"]["id"]
        if lid not in LEAGUES:
            continue
        code, country, name = LEAGUES[lid]
        if x["league"]["country"] != country:  # id/name drift guard
            continue
        ft = x["score"]["fulltime"]
        rows.append({"league": code, "api_date": pd.Timestamp(x["fixture"]["date"]).tz_convert("Europe/London").tz_localize(None),
                     "api_home": x["teams"]["home"]["name"], "api_away": x["teams"]["away"]["name"],
                     "status": x["fixture"]["status"]["short"], "hg": ft["home"], "ag": ft["away"],
                     "api_fixture_id": x["fixture"]["id"]})
    return pd.DataFrame(rows)


def api_results():
    """All saved results: compact committed CSVs plus any raw JSON (later fetches of a date win)."""
    frames = [pd.read_csv(f, parse_dates=["api_date"]) for f in sorted(COMPACT.glob("*.csv"))]
    frames += [parse(json.loads(f.read_text())["response"]) for f in sorted(RAW.glob("fixtures_*.json"))]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=["league", "api_date", "api_home", "api_away", "status", "hg", "ag", "api_fixture_id"])
    allr = pd.concat(frames, ignore_index=True)
    allr["fin"] = allr.status.isin(FINISHED)
    return allr.sort_values("fin").drop_duplicates("api_fixture_id", keep="last").drop(columns="fin")


def norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    for a, b in ALIASES.items():
        s = re.sub(rf"(?<![a-z]){re.escape(a)}(?![a-z])", b, s)
    toks = [t for t in re.split(r"[^a-z0-9']+", s) if t and t not in STOP]
    return " ".join(toks) or s


def sim(a, b):
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    ta, tb = set(na.split()), set(nb.split())
    overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
    contain = 1.0 if (na in nb or nb in na) else 0.0
    return max(SequenceMatcher(None, na, nb).ratio(), overlap, contain)


def match(expected):
    """expected: DataFrame with league, date, home, away (our names). Returns matched results in our names."""
    api = api_results()
    out, report = [], []
    used = set()
    for (lg, date), grp in expected.groupby(["league", "date"]):
        cand = api[(api.league == lg) & ((api.api_date.dt.normalize() - pd.Timestamp(date)).abs() <= pd.Timedelta(days=1))]
        if cand.empty:
            continue
        for _, e in grp.iterrows():
            sc = cand.apply(lambda c: (sim(e.home, c.api_home) + sim(e.away, c.api_away)) / 2, axis=1)
            j = sc.idxmax(); best = sc.max()
            c = cand.loc[j]
            same_day = c.api_date.normalize() == pd.Timestamp(date)
            report.append({"league": lg, "date": pd.Timestamp(date).date(), "ours": f"{e.home} v {e.away}",
                           "api": f"{c.api_home} v {c.api_away}", "similarity": round(best, 2), "status": c.status,
                           "same_day": same_day})
            if best >= 0.6 and c.api_fixture_id not in used:
                used.add(c.api_fixture_id)
                if c.status in FINISHED:
                    out.append({"league": lg, "date": pd.Timestamp(date), "home": e.home, "away": e.away,
                                "hg": float(c.hg), "ag": float(c.ag), "source": "apifootball", "similarity": best})
    rep = pd.DataFrame(report).sort_values("similarity") if report else pd.DataFrame()
    return pd.DataFrame(out), rep


def load_results():
    return pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["league", "date", "home", "away", "hg", "ag"])


def match_prediction_files():
    exp = pd.concat([pd.read_csv(p, usecols=["league", "date", "home", "away"], parse_dates=["date"])
                     for p in (ROOT / "predictions").glob("*.csv") if "graded" not in p.name]).drop_duplicates()
    res, rep = match(exp)
    res.to_parquet(OUT, index=False)
    weak = rep[(rep.similarity < 0.8) & rep.same_day] if len(rep) else rep
    acc = rep[rep.similarity >= 0.6] if len(rep) else rep
    md = ["# API-Football matching\n", f"{len(acc)} predicted matches paired with an API fixture (similarity >= 0.6); "
          f"{len(res)} of them finished (90-minute score used). Pairs from other days below 0.6 are simply different "
          "games and are ignored.\n",
          "## Same-day pairs below 0.8 similarity (check these)\n",
          weak.drop(columns="same_day").to_markdown(index=False) if len(weak) else "none", "\n",
          "## All accepted pairs\n", acc.drop(columns="same_day").sort_values(["league", "date"]).to_markdown(index=False) if len(acc) else "none", "\n"]
    (ROOT / "reports/apifootball_matching.md").write_text("\n".join(md))
    print("\n".join(md[:5]))


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        import datetime as dt
        for d in sys.argv[2:]:
            if d == "yesterday":
                d = (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)).isoformat()
            fetch(d)
    elif sys.argv[1] == "match":
        match_prediction_files()
