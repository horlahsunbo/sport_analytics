#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 src/download.py
python3 src/ingest.py
python3 src/features.py
python3 src/model.py            # ~8 min: 16 walk-forward seasons x 2 models
python3 src/analysis.py
python3 src/analysis_controls.py
python3 src/inseason.py         # ~10 min: in-season first-10 -> next-5 test
