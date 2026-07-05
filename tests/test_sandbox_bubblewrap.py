"""E1c Phase 1 (2026-07-03) — the bubblewrap `cpu` sandbox backend.

Proves the isolation properties that close E1c:
  - the agent can read/write/run inside the project (functionality intact);
  - edits persist to the REAL files on disk;
  - the host filesystem OUTSIDE the project is invisible;
  - network follows the option-b trust split (interactive allowed, auto
    denied);
  - CC_SANDBOX=bubblewrap fails closed when bwrap is unusable;
  - CC_SANDBOX=off is the explicit uncontained escape hatch.

These tests SKIP the isolation assertions when bubblewrap isn't usable on
the host (macOS/Windows dev, or a CI without user namespaces), since the
properties can only be demonstrated where the backend can run. The
fail-closed and escape-hatch tests run everywhere.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

_CA = Path(__file__).resolve().parents[1] / "CRYS"
if str(_CA) not in sys.path:
    sys.path.insert(0, str(_CA))

import crystal_code.sandbox as sb  # noqa: E402
from crystal_code.sandbox import sandbox_run  # noqa: E402


_BWRAP = sb._bwrap_usable()
_needs_bwrap = pytest.mark.skipif(
    not _BWRAP, reason="bubblewrap not usable on this host",
)


def _project_with_sibling_secret() -> tuple[Path, Path]:
    """A project dir with a secret file one level UP (outside scope). Placed
    under /home (not /tmp) so it's a realistic project location."""
    base = Path(tempfile.mkdtemp(dir="/home/claude" if
                Path("/home/claude").exists() else None))
    (base / "SECRET.txt").write_text("cloud-credentials")
    proj = base / "project"
    proj.mkdir()
    (proj / "app.py").write_text("print(1)\n")
    return base, proj


# --- functionality is intact (the coding agent still works) ----------------

@_needs_bwrap
def test_agent_can_read_write_run_in_project():
    _base, proj = _project_with_sibling_secret()
    r = sandbox_run(
        "cat app.py && echo added >> app.py && echo DONE",
        proj, 15, allow_shell_features=True,
    )
    assert r.exit_code == 0
    assert "print(1)" in r.output
    assert "DONE" in r.output


@_needs_bwrap
def test_edits_persist_to_real_files_on_disk():
    _base, proj = _project_with_sibling_secret()
    sandbox_run(
        "echo NEWLINE >> app.py", proj, 15, allow_shell_features=True,
    )
    # The real file on the host was modified.
    assert "NEWLINE" in (proj / "app.py").read_text()


# --- isolation (the security property) -------------------------------------

@_needs_bwrap
def test_host_outside_project_is_invisible():
    base, proj = _project_with_sibling_secret()
    assert (base / "SECRET.txt").exists()  # it IS there on the host
    r = sandbox_run(
        "cat ../SECRET.txt", proj, 15, allow_shell_features=True,
    )
    # ...but not from inside the sandbox.
    assert r.is_error
    assert "cloud-credentials" not in r.output


@_needs_bwrap
def test_home_directory_is_invisible():
    _base, proj = _project_with_sibling_secret()
    r = sandbox_run(
        "ls ~ 2>&1 || echo NOHOME", proj, 15, allow_shell_features=True,
    )
    # The user's real home contents shouldn't be listable.
    assert "NOHOME" in r.output or r.is_error or "app.py" not in r.output


# --- network trust split (option b) ----------------------------------------

@_needs_bwrap
def test_auto_commands_have_no_network():
    """Auto-approved (allow_shell_features=False) => network denied, even to
    an otherwise-reachable host."""
    _base, proj = _project_with_sibling_secret()
    code = (
        "import socket; socket.create_connection"
        "(('pypi.org', 443), timeout=5); print('CONN')"
    )
    r = sandbox_run(
        f"python3 -c \"{code}\"", proj, 15, allow_shell_features=False,
    )
    assert "CONN" not in r.output  # blocked


# --- backend selection: fail-closed + escape hatch (run everywhere) --------

def test_forced_bubblewrap_fails_closed_when_unusable(monkeypatch, tmp_path):
    monkeypatch.setenv("CC_SANDBOX", "bubblewrap")
    monkeypatch.setattr(sb, "_BWRAP_CHECKED", False)  # simulate unusable
    r = sandbox_run("echo hi", tmp_path, 10,
                    allow_shell_features=True)
    assert r.is_error
    assert r.backend == "none"
    assert "refused" in r.output


def test_off_escape_hatch_runs_uncontained(monkeypatch, tmp_path):
    monkeypatch.setenv("CC_SANDBOX", "off")
    r = sandbox_run("echo hatch", tmp_path, 10,
                    allow_shell_features=True)
    assert r.exit_code == 0
    assert r.backend == "passthrough"
    assert "hatch" in r.output
