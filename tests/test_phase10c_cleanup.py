"""Phase 10C / Phase 11 tests — cleanup phase: doc + prose + wiring.

Per P0.89: 1 test verifying the env-var deprecation alias behavior
(CU-24 / P0.88). Other Phase 10C changes are doc/prose edits or
trivial config additions that don't warrant pytest backing.

**Phase 11 (P0.99) update**: P0.98 retired the deprecation alias.
The test's assertions are updated to reflect the new behavior:
the old name `CC_METACOGNITION_INTERVAL_SECONDS` is now silently
ignored. The "old name set" case now asserts the function falls
through to the default rather than honoring the old value. No
new tests; just a contract update on the existing one.

The single test verifies that `_resolve_cognition_poll_interval`:
- Returns the new env var value when CC_COGNITION_WORKER_INTERVAL_SECONDS
  is set (regardless of old var).
- Returns the DEFAULT (600) when only CC_METACOGNITION_INTERVAL_SECONDS
  is set, since the alias was retired in Phase 11.
- Returns the default (600) when neither is set.
"""
from __future__ import annotations

import pytest

from crystal_cache.workers.cognition import _resolve_cognition_poll_interval


# ---------------------------------------------------------------------------
# CL1 — env var resolution after Phase 11 (P0.98) alias retirement
# ---------------------------------------------------------------------------

def test_cl1_env_var_after_alias_retirement(monkeypatch):
    """Phase 10C P0.88 / CU-24 introduced the rename + alias.
    Phase 11 P0.98 retired the alias.

    Post-Phase-11 behavior: only `CC_COGNITION_WORKER_INTERVAL_SECONDS`
    is honored. The old name is silently ignored.
    """
    # --- Both unset → default 600 ---
    monkeypatch.delenv("CC_COGNITION_WORKER_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("CC_METACOGNITION_INTERVAL_SECONDS", raising=False)
    assert _resolve_cognition_poll_interval() == 600

    # --- Only OLD set → IGNORED (returns default 600 post-Phase-11) ---
    # Pre-Phase-11 this returned the old value with a deprecation warning;
    # P0.98 retired the alias. Operators who didn't migrate now fall
    # through to the default.
    monkeypatch.delenv("CC_COGNITION_WORKER_INTERVAL_SECONDS", raising=False)
    monkeypatch.setenv("CC_METACOGNITION_INTERVAL_SECONDS", "120")
    assert _resolve_cognition_poll_interval() == 600

    # --- Only new set → use new value ---
    monkeypatch.delenv("CC_METACOGNITION_INTERVAL_SECONDS", raising=False)
    monkeypatch.setenv("CC_COGNITION_WORKER_INTERVAL_SECONDS", "300")
    assert _resolve_cognition_poll_interval() == 300

    # --- Both set → new wins (old is ignored, no precedence path) ---
    monkeypatch.setenv("CC_METACOGNITION_INTERVAL_SECONDS", "999")
    monkeypatch.setenv("CC_COGNITION_WORKER_INTERVAL_SECONDS", "60")
    assert _resolve_cognition_poll_interval() == 60
