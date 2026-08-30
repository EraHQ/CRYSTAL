"""CC_AGENT_DISABLED_TOOLS (2026-08-30): tools a process must never offer.

The 20-question LongMemEval pass at concurrency 4 was disqualified by one
row whose answer turn called web_search for a real-world date. web_fetch
has no gate at all and web_search is offered even when unconfigured (the
compose default configures searxng anyway). The registry's
`list_for_context` is the one place every consumer (agent loop, cognition
dispatcher, system-prompt tool list) gets its tool set, so the operator's
disabled list is applied there. Empty by default: a deployment that does
not set it is unchanged.

Pinned here:
  1. Default (empty) hides nothing.
  2. Names in the list are excluded in every context; other tools stay;
     parsing is case- and whitespace-insensitive.
  3. The `available` hook still applies alongside the list.
"""
from __future__ import annotations

import pytest

from crystal_cache.agent.tool_registry import Tool, ToolRegistry
from crystal_cache.config import Settings


async def _impl(**_kw):
    return {}


def _tool(name: str, contexts=("agent", "cognition"), available=None) -> Tool:
    return Tool(
        name=name, description=name, contexts=frozenset(contexts),
        parameters_schema={"type": "object", "properties": {}}, impl=_impl,
        available=available,
    )


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    for t in (_tool("web_search"), _tool("web_fetch"), _tool("crystal_recall"),
              _tool("source_lookup", contexts=("cognition",)),
              _tool("gated", available=lambda: False)):
        r.register(t)
    return r


def _names(r: ToolRegistry, context: str) -> list[str]:
    return [t.name for t in r.list_for_context(context)]


def test_default_hides_nothing(monkeypatch):
    monkeypatch.setattr("crystal_cache.config.get_settings", lambda: Settings(agent_disabled_tools=""))
    r = _registry()
    assert _names(r, "agent") == ["crystal_recall", "web_fetch", "web_search"]
    assert _names(r, "cognition") == ["crystal_recall", "source_lookup", "web_fetch", "web_search"]


def test_disabled_list_applies_in_every_context(monkeypatch):
    monkeypatch.setattr(
        "crystal_cache.config.get_settings",
        lambda: Settings(agent_disabled_tools=" Web_Search, web_fetch ,"),
    )
    r = _registry()
    assert _names(r, "agent") == ["crystal_recall"]
    assert _names(r, "cognition") == ["crystal_recall", "source_lookup"]
    assert r.get("web_search") is not None          # registered, just never offered


def test_available_hook_still_applies(monkeypatch):
    monkeypatch.setattr("crystal_cache.config.get_settings", lambda: Settings(agent_disabled_tools="web_fetch"))
    r = _registry()
    names = _names(r, "agent")
    assert "gated" not in names and "web_fetch" not in names and "web_search" in names


def test_knob_default_is_empty():
    assert Settings.model_fields["agent_disabled_tools"].default == ""
