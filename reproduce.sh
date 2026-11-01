#!/bin/bash
# Full pipeline from scratch on an Apple-silicon Mac. data/ already holds every intermediate file,
# so to recompute only the tables and figures, run the analysis steps at the end.
set -e
cd "$(dirname "$0")"
python harness/memguard.py &   # optional safety net: kills runaway experiment processes if memory runs low
GUARD=$!
trap "kill $GUARD 2>/dev/null" EXIT
python scripts/00_validate_harness.py   # every reference solution must pass its own tests (about 20 s)
python scripts/01_generate.py 16        # 16 samples per problem from Qwen2.5-Coder-1.5B-Instruct (about 25 min)
python scripts/02_execute.py            # run every unique sample on the hidden tests in a sandbox (about 10 min)
python scripts/03_judge.py A           # judge A scores, only needed for the judge comparison table (about 1.5 h)
# analysis
python scripts/04_analyze.py
