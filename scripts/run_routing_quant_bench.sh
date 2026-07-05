#!/usr/bin/env bash
#
# Routing-lane quantization gate — one command (CRYS, June 2026).
#
# Step 0 of docs/VECTOR_STORE_RESEARCH.md §6: does binary-quant + float
# rescore on the 10k routing lane pick the SAME crystal as exact-float
# routing? Runs scripts/bench_routing_quantization.py over a real bank and
# prints top-1 agreement + recall@k per oversample.
#
# Unlike the code-descriptions A/B, this needs NO encoder — the projection P
# is rebuilt model-free from its seed — so the dev-only clean venv (.venv-clean)
# is enough. No gtr-t5-base, no Qdrant, no model load.
#
#   bash scripts/run_routing_quant_bench.sh [DB_PATH]
#
# DB_PATH defaults to the repo's crystal_cache.db; an absolute path or a full
# SQLite URL also works. Env overrides:
#   CUSTOMER=CRYS-local   bank owner to route over
#   MAX_QUERIES=2000              cap on prompt-derived queries
#   OVERSAMPLE=2,4,8,16           pool multipliers (pool = 10 * m)
#   VERBOSE=1                     print a few example disagreements
# Extra args after DB_PATH pass straight through to the Python harness.
#
set -euo pipefail

# Repo in Windows-form (C:/...) so native-Windows Python is happy under Git
# Bash — pwd -W prints C:/... instead of MINGW's /c/...; plain pwd elsewhere.
_winpath() { cd "$1" && { pwd -W 2>/dev/null || pwd; }; }
REPO="$(_winpath "$(dirname "${BASH_SOURCE[0]}")/..")"
BENCH="$REPO/scripts/bench_routing_quantization.py"
CUSTOMER="${CUSTOMER:-CRYS-local}"

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# DB path: first positional arg, else the repo default.
DB_ARG="${1:-$REPO/crystal_cache.db}"
[ "$#" -gt 0 ] && shift || true   # remaining args pass through to the harness
case "$DB_ARG" in
  *://*) DB_DISPLAY="$DB_ARG" ;;                  # full URL — can't stat it
  *)     [ -f "$DB_ARG" ] || die "DB not found: $DB_ARG (pass a path or set one)";
         DB_DISPLAY="$DB_ARG" ;;
esac

say "Preflight"
python -c "import crystal_cache" 2>/dev/null \
  || die "crystal_cache not importable — activate the project venv (pip install -e .)."
python -c "import numpy" 2>/dev/null \
  || die "numpy missing in this venv."
[ -f "$BENCH" ] || die "harness not found: $BENCH"
echo "  ok — crystal_cache + numpy present (no encoder needed)"
echo "  bank    : $DB_DISPLAY"
echo "  customer: $CUSTOMER"

ARGS=( --db "$DB_ARG" --customer "$CUSTOMER"
       --max-queries "${MAX_QUERIES:-2000}"
       --oversample "${OVERSAMPLE:-2,4,8,16}" )
[ "${VERBOSE:-0}" = "1" ] && ARGS+=( --verbose )
ARGS+=( "$@" )   # passthrough

say "Routing quantization gate"
python "$BENCH" "${ARGS[@]}"
