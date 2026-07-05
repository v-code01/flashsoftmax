#!/usr/bin/env bash
# Regenerate the flashsoftmax study end to end. Self-contained: the benchmark generates its own scores
# with a deterministic PRNG, so no model or data files are needed.
# Usage: ./reproduce.sh [REPS]   (default 400)
set -euo pipefail
cd "$(dirname "$0")"
. .venv/bin/activate
REPS="${1:-400}"
make >/dev/null
bin/bench "$REPS" > results/bench.jsonl
python tools/analyze.py
python tools/verify.py
echo "regenerated; see bench_results/frontier.md"
