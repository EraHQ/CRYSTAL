#!/usr/bin/env bash
# Create a PRISTINE venv whose only CC_* flags are the ones you set inline
# on the launch command — immune to anything CC_* exported in your shell
# rc (e.g. CC_AGENT_COMPACTION left on for live agent runs, which leaks
# into pytest and flips "off by default" tests).
#
# How it stays clean: the venv's activate script gets a hook appended that
# unsets every CC_* variable on activation, so whatever ~/.bashrc (or any
# shell export) leaks in is wiped the moment you activate. .env still
# supplies the API keys — pydantic-settings reads the .env file directly,
# independent of the shell — so credentials survive the scrub. The ONLY
# CC_* flags in effect are the ones you put on the command itself:
#   CC_TEXT_ENCODER=semantic python -m crys "$WORK_REPO" --db "$WORK_REPO/crystal_cache.db"
#   CC_ENABLE_HYBRID_RANK=1   python scripts/eval_hybrid_rank.py --db "$WORK_REPO/crystal_cache.db"
#
# Usage (from the repo root, in Git Bash / MINGW64):
#   bash scripts/make_clean_venv.sh                  # .venv-clean, full extras
#   bash scripts/make_clean_venv.sh .venv-test       # custom name
#   bash scripts/make_clean_venv.sh .venv-test dev   # lean: pytest only, no torch download
#
# Then:
#   source .venv-clean/Scripts/activate
#   echo "CC_AGENT_COMPACTION=[$CC_AGENT_COMPACTION]"   # -> empty
#   pytest -q
#
# (.venv-* is already gitignored, so this venv won't be committed.)
set -euo pipefail

VENV_DIR="${1:-.venv-clean}"
# embeddings = gtr-t5-base (server + eval harness); agent = CRYS MCP;
# dev = pytest/ruff/mypy. Override with arg 2, e.g. "dev" for a fast
# pytest-only venv that skips the sentence-transformers/torch download.
EXTRAS="${2:-embeddings,agent,dev}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "pyproject.toml" ]; then
  echo "error: run from the crystal-cache-v2 repo root (no pyproject.toml found)" >&2
  exit 1
fi

echo "Creating venv at $VENV_DIR ..."
python -m venv "$VENV_DIR"

# Resolve activate + python for both Windows (Scripts/) and POSIX (bin/).
if [ -f "$VENV_DIR/Scripts/activate" ]; then
  ACTIVATE="$VENV_DIR/Scripts/activate"
  VPY="$VENV_DIR/Scripts/python.exe"
else
  ACTIVATE="$VENV_DIR/bin/activate"
  VPY="$VENV_DIR/bin/python"
fi

echo "Installing crystal-cache (editable) with extras: [$EXTRAS] ..."
"$VPY" -m pip install --upgrade pip >/dev/null
"$VPY" -m pip install -e ".[$EXTRAS]"

# Append the CC_* scrub hook to activate — idempotent via a sentinel so
# re-running this script never duplicates it. Single-quoted heredoc so the
# $(...) and $_cc_var stay literal in the written file (evaluated at
# activation time in your shell, not now).
MARKER="# >>> crystal-cache clean-flags hook >>>"
if grep -qF "$MARKER" "$ACTIVATE"; then
  echo "Scrub hook already present in $ACTIVATE (left as-is)."
else
  cat >> "$ACTIVATE" <<'HOOK'

# >>> crystal-cache clean-flags hook >>>
# Pristine baseline: drop every CC_* variable inherited from the shell
# (e.g. CC_AGENT_COMPACTION exported in ~/.bashrc for live runs) so the
# ONLY flags active are the ones set inline on the launch command. .env
# still provides API keys (pydantic-settings reads the file, not the
# shell), so credentials survive the scrub.
if command -v compgen >/dev/null 2>&1; then
  for _cc_var in $(compgen -v | grep '^CC_' || true); do
    unset "$_cc_var"
  done
  unset _cc_var
fi
# <<< crystal-cache clean-flags hook <<<
HOOK
  echo "Added the CC_* scrub hook to $ACTIVATE."
fi

echo
echo "Done."
echo "  source $ACTIVATE"
echo "  echo \"CC_AGENT_COMPACTION=[\$CC_AGENT_COMPACTION]\"   # should print []"
echo "  pytest -q"
