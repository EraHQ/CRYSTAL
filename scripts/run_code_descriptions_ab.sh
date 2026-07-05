#!/usr/bin/env bash
#
# Code-descriptions A/B — one command (CRYS, June 2026).
#
# Builds two banks from the SAME source subtree and compares retrieval:
#   baseline  (CC_ENABLE_CODE_DESCRIPTIONS=0)  — code-encoded
#   treatment (CC_ENABLE_CODE_DESCRIPTIONS=1)  — description-encoded
# then runs scripts/eval_code_descriptions.py over the functional query set
# and prints recall@k / MRR side by side.
#
# Run from an activated venv that HAS the embeddings extra (gtr-t5-base).
# The dev-only clean venv cannot run this — it has no encoder. Use .venv
# (or a clean venv built with the embeddings extra).
#
#   bash scripts/run_code_descriptions_ab.sh [SUBTREE]
#
# SUBTREE is relative to the repo root (default: src/crystal_cache); an
# absolute path also works. Env overrides:
#   SKIP_INGEST=1   reuse existing banks, just re-run the comparison (fast)
#   POOL=80         candidate pool for the eval (default: settings value)
#   AB_DIR=/path    where the two scratch DBs live (default: repo/.ab_descriptions)
#
# Cost/time: each phase is its own process, so gtr-t5-base loads three times
# (~1 min each); the treatment ingest makes ~one Haiku call per code file.
#
set -euo pipefail

# Repo in Windows-form (C:/...) so native-Windows Python is happy even under
# Git Bash — pwd -W prints C:/... instead of MINGW's /c/...; plain pwd elsewhere.
_winpath() { cd "$1" && { pwd -W 2>/dev/null || pwd; }; }
REPO="$(_winpath "$(dirname "${BASH_SOURCE[0]}")/..")"
COD="$REPO/CRYS"
EVAL="$REPO/scripts/eval_code_descriptions.py"
CUSTOMER="CRYS-local"

# Subtree to ingest (arg or default), resolved against the repo unless absolute.
SUBTREE_ARG="${1:-src/crystal_cache}"
case "$SUBTREE_ARG" in
  /*|?:/*) TARGET="$SUBTREE_ARG" ;;        # absolute: /unix or C:/windows
  *)       TARGET="$REPO/$SUBTREE_ARG" ;;  # relative to repo root
esac

AB_DIR="${AB_DIR:-$REPO/.ab_descriptions}"
OFF_DB="$AB_DIR/cc_off.db"
ON_DB="$AB_DIR/cc_on.db"
POOL="${POOL:-}"

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

say "Preflight"
python -c "import crystal_cache" 2>/dev/null \
  || die "crystal_cache not importable — activate the project venv (pip install -e .)."
python -c "import sentence_transformers" 2>/dev/null \
  || die "sentence-transformers missing — this venv has no embeddings extra (gtr). Use .venv, not the dev-only clean venv."
( cd "$COD" && python -c "import crys, crystal_code" ) 2>/dev/null \
  || die "the crys CLI isn't importable from $COD."
[ -d "$TARGET" ] || die "subtree not found: $TARGET"
[ -f "$EVAL" ]   || die "eval script not found: $EVAL"
echo "  ok — encoder present, CLI present"
echo "  ingest target: $TARGET"
echo "  scratch dir  : $AB_DIR"

mkdir -p "$AB_DIR"

if [ "${SKIP_INGEST:-0}" = "1" ]; then
  say "Skipping ingest (SKIP_INGEST=1) — reusing existing banks"
  [ -f "$OFF_DB" ] || die "no baseline bank at $OFF_DB — run once without SKIP_INGEST first."
  [ -f "$ON_DB" ]  || die "no treatment bank at $ON_DB — run once without SKIP_INGEST first."
else
  # Fresh banks each run. The file text is identical across the two ingests,
  # so the ONLY difference is the flag — but the content-hash skip would
  # suppress a re-encode inside an existing bank. Separate DBs + a clean slate
  # guarantee each bank reflects its own flag.
  rm -f "$OFF_DB" "$OFF_DB"-wal "$OFF_DB"-shm "$ON_DB" "$ON_DB"-wal "$ON_DB"-shm

  say "Baseline ingest  (descriptions OFF)  ->  cc_off.db"
  ( cd "$COD" && CC_ENABLE_CODE_DESCRIPTIONS=0 \
      python -m crys "$TARGET" --ingest --db "$OFF_DB" --customer "$CUSTOMER" )

  say "Treatment ingest (descriptions ON)   ->  cc_on.db"
  echo "  (one Haiku call per code file — the cost line below reflects it)"
  ( cd "$COD" && CC_ENABLE_CODE_DESCRIPTIONS=1 \
      python -m crys "$TARGET" --ingest --db "$ON_DB" --customer "$CUSTOMER" )
fi

say "A/B comparison"
EVAL_ARGS=( --baseline-db "$OFF_DB" --treatment-db "$ON_DB"
            --baseline-customer "$CUSTOMER" --treatment-customer "$CUSTOMER" --verbose )
[ -n "$POOL" ] && EVAL_ARGS+=( --pool "$POOL" )
( cd "$COD" && python "$EVAL" "${EVAL_ARGS[@]}" )

say "Done"
echo "  banks kept:"
echo "    baseline  (code) : $OFF_DB"
echo "    treatment (desc) : $ON_DB"
echo "  re-run just the comparison (no re-ingest), e.g. a wider pool:"
echo "    SKIP_INGEST=1 POOL=80 bash scripts/run_code_descriptions_ab.sh"
echo "  clean up when done:  rm -rf \"$AB_DIR\""
