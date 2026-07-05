"""Load-aware idle gate (workers/idle.py, 2026-07-02 — BACKLOG §3 remainder).

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

import pytest

from crystal_cache.workers import idle


@pytest.fixture(autouse=True)
def _fresh_idle_state():
    idle.reset_for_tests()
    yield
    idle.reset_for_tests()


def test_never_stamped_is_quiet():
    """A process that never served a request is quiet — the worker-only
    compose shape keeps its pre-gate behavior (gate inert, no regression)."""
    assert idle.seconds_since_last_request() == float("inf")
    assert idle.is_quiet(30) is True


def test_recent_request_is_not_quiet():
    idle.note_request()
    assert idle.seconds_since_last_request() < 1.0
    assert idle.is_quiet(30) is False


def test_quiet_after_enough_time(monkeypatch):
    idle.note_request()
    real = idle.time.monotonic()
    monkeypatch.setattr(idle.time, "monotonic", lambda: real + 31.0)
    assert idle.is_quiet(30) is True


def test_zero_disables_the_gate():
    idle.note_request()
    assert idle.is_quiet(0) is True
    assert idle.is_quiet(-5) is True
