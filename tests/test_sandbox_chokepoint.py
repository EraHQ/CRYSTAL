"""E1c Phase 0 (2026-07-03) — the sandbox execution chokepoint.

Phase 0 introduces one seam (sandbox_run) that every model-influenced
execution path routes through, with a passthrough backend (no isolation
yet). These tests lock the seam's contract so Phase 1 (the bubblewrap
backend) is a change BEHIND the seam, not a change TO it:
  - the shell/no-shell execution-model split (E1b) is honored;
  - profiles are accepted and plumbed;
  - timeout / empty-command / output shape are stable;
  - the async wrapper matches the sync one.

Imports the CRYS module directly (CRYS has no pytest suite of its
own; sandbox_run is pure/importable).

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

_CA = Path(__file__).resolve().parents[1] / "CRYS"
if str(_CA) not in sys.path:
    sys.path.insert(0, str(_CA))

from crystal_code.sandbox import (  # noqa: E402
    SandboxProfile,
    SandboxResult,
    sandbox_run,
    sandbox_run_async,
    scrubbed_env,
)


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


# --- execution-model split (E1b preserved through the seam) ----------------

def test_shell_features_allows_chaining():
    r = sandbox_run("echo one && echo two", _tmp(), 10,
                    allow_shell_features=True)
    assert r.exit_code == 0
    assert "one" in r.output and "two" in r.output


def test_no_shell_makes_operators_inert():
    r = sandbox_run("echo hello; echo INJECTED", _tmp(), 10,
                    allow_shell_features=False)
    assert r.exit_code == 0
    # The ; and second echo are literal args, not a second command.
    assert "hello; echo INJECTED" in r.output


def test_no_shell_empty_command_is_error():
    r = sandbox_run("   ", _tmp(), 10, allow_shell_features=False)
    assert r.is_error


# --- result contract --------------------------------------------------------

def test_result_is_sandboxresult_with_fields():
    r = sandbox_run("echo hi", _tmp(), 10, allow_shell_features=True)
    assert isinstance(r, SandboxResult)
    assert r.exit_code == 0
    assert r.timed_out is False
    assert r.is_error is False
    # Backend depends on the host: bubblewrap when usable, else passthrough.
    assert r.backend in ("passthrough", "bubblewrap")


def test_nonzero_exit_is_error():
    r = sandbox_run("exit 3", _tmp(), 10, allow_shell_features=True)
    assert r.exit_code == 3
    assert r.is_error is True


def test_timeout_is_flagged():
    r = sandbox_run("sleep 5", _tmp(), 1, allow_shell_features=True)
    assert r.timed_out is True
    assert r.is_error is True


def test_output_is_truncated_to_tail():
    # Emit more than the cap; the head should be dropped, tail kept.
    r = sandbox_run(
        "python3 -c \"print('X'*9000)\"", _tmp(), 10,
        allow_shell_features=True,
    )
    assert "truncated" in r.output
    assert r.output.rstrip().endswith("X")


# --- profiles are accepted and plumbed (all passthrough at Phase 0) --------

@pytest.mark.parametrize("profile", list(SandboxProfile))
def test_all_profiles_accepted(profile):
    r = sandbox_run("echo p", _tmp(), 10, allow_shell_features=True,
                    profile=profile)
    assert r.exit_code == 0


# --- env scrubbing ----------------------------------------------------------

def test_secrets_are_scrubbed_from_child(monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "supersecret")
    monkeypatch.setenv("HARMLESS_VAR", "ok")
    env = scrubbed_env()
    assert "MY_API_KEY" not in env
    assert env.get("HARMLESS_VAR") == "ok"


def test_explicit_env_overrides_default():
    r = sandbox_run(
        "python3 -c \"import os; print(os.environ.get('FOO','none'))\"",
        _tmp(), 10, allow_shell_features=True, env={"FOO": "bar", "PATH": __import__("os").environ.get("PATH", "")},
    )
    assert "bar" in r.output


# --- async wrapper matches sync --------------------------------------------

async def test_async_wrapper_matches_sync():
    r = await sandbox_run_async("echo async", _tmp(), 10,
                                allow_shell_features=True)
    assert r.exit_code == 0
    assert "async" in r.output
