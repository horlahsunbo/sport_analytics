"""Fetch raw sources into data/raw/. Raw files are not committed; re-run this to rebuild."""
from pathlib import Path
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SOURCES = {
    "xgabora/Matches.csv": "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv",
    "xgabora/EloRatings.csv": "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/EloRatings.csv",
}

for rel, url in SOURCES.items():
    dest = RAW / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=300); r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"{rel}: {len(r.content)/1e6:.1f} MB")
