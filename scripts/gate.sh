#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
fail() { echo "GATE FAIL: $*" >&2; exit 1; }
[ -d .venv ] && . .venv/bin/activate || { python3 -m venv .venv; . .venv/bin/activate; pip install -q -r requirements.txt; }
echo "== 1/7 C++ build (zero warnings) =="; make clean >/dev/null; make >/dev/null 2>build.err || { cat build.err; fail build; }; [ -s build.err ] && { cat build.err; fail "warnings"; }; rm -f build.err; echo "   ok"
echo "== 2/7 C++ tests (normalize, reference, online==two-pass, naive overflow) =="; make test >/dev/null || fail "c++ tests"; echo "   ok"
echo "== 3/7 ruff =="; ruff check tools || fail ruff; echo "   ok"
echo "== 4/7 mypy --strict =="; mypy --strict tools/analyze.py tools/verify.py || fail mypy; echo "   ok"
echo "== 5/7 pure-ASCII =="; bad=$(LC_ALL=C grep -rlP '[^\x00-\x7F]' src tests tools scripts README.md claims.toml REVIEW.md PREREG.md bench_results 2>/dev/null || true); [ -z "$bad" ] || { echo "$bad"; fail ascii; }; echo "   ok"
echo "== 6/7 no environment leak =="; leak=$(grep -rniE 'on this (box|machine)|the user.s (server|machine|box)|pre-existing server|do not touch|8080|8081|8082|8083|8084|8090|8091|8092|gemma|jane street|hrt|quant firm|high.frequency' src tests tools README.md claims.toml REVIEW.md PREREG.md reproduce.sh 2>/dev/null || true); [ -z "$leak" ] || { echo "$leak"; fail "leak"; }; echo "   ok"
echo "== 7/7 independent verify =="; python tools/verify.py || fail verify; echo "   ok"
echo "ALL GATES PASS"
