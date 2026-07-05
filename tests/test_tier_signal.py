"""Tier as an epistemic signal (RATIFIED 2026-07-02).

Tiers never change ranking — retrieval tool results carry crystal_tiers
plus a tier_note the model can act on (verify via web_search / ask the
user). Whitelist-only result sets carry no note (no noise when fully
vetted).

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

from typing import Any

from crystal_cache.infrastructure.schema import CrystalRow
from crystal_cache.retrieval.tier_signal import TIER_SEMANTICS, tier_map, tier_note


async def _seed(store: Any, customer_id: str, cid: str, tier: str):
    async with store.session() as s:
        s.add(CrystalRow(
            id=cid, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
            quality_tier=tier,
        ))


async def test_tier_map_reads_tiers(store, customer):
    await _seed(store, customer.id, "c_w", "whitelist")
    await _seed(store, customer.id, "c_q", "quarantine")

    tiers = await tier_map(store, customer.id, ["c_w", "c_q", "c_missing"])

    assert tiers == {"c_w": "whitelist", "c_q": "quarantine"}


async def test_tier_map_is_customer_guarded(store, customer):
    await _seed(store, "cus_other", "c_theirs", "whitelist")

    tiers = await tier_map(store, customer.id, ["c_theirs"])

    assert tiers == {}


def test_note_silent_when_fully_vetted():
    assert tier_note({}) is None
    assert tier_note({"a": "whitelist", "b": "whitelist"}) is None


def test_note_flags_unvetted_knowledge():
    note = tier_note({"a": "neutral", "b": "neutral", "c": "whitelist"})
    assert note is not None
    assert "2 neutral" in note
    assert "1 whitelist" in note
    assert "verifying" in note
    assert "blacklist" not in note


def test_note_calls_out_blacklist():
    note = tier_note({"a": "blacklist"})
    assert "operator-flagged" in note
    assert "do not rely" in note


async def test_proxy_injection_carries_the_legend(store, customer):
    """The chat-proxy side: a non-whitelist crystal in the injection set
    appends the [Knowledge quality] legend to the model-facing text — the
    ratified verdict's remaining sub-scope, closed."""
    from crystal_cache.retrieval.tier_signal import tier_map, tier_note

    await _seed(store, customer.id, "c_prox", "neutral")
    note = tier_note(await tier_map(store, customer.id, ["c_prox"]))
    assert note is not None and "1 neutral" in note


def test_semantics_constant_defines_all_tiers():
    for tier in ("whitelist", "neutral", "quarantine", "blacklist"):
        assert tier in TIER_SEMANTICS
    assert "never change ranking" in TIER_SEMANTICS


async def test_knowledge_search_carries_the_signal(store, customer, tool_state):
    """Tool-level: results gain crystal_tiers + tier_note; ranking fields
    untouched."""
    from crystal_cache.agent.tools.retrievers import (
        knowledge_search,
        set_tool_state,
    )

    set_tool_state(tool_state)
    await _seed(store, customer.id, "c_sig", "quarantine")

    out = await knowledge_search(customer_id=customer.id, query="anything")

    assert "crystal_tiers" in out
    assert "tier_note" in out
    # Empty match set (nothing indexed for the query) → empty map, no note.
    if not out["matched_crystal_ids"]:
        assert out["crystal_tiers"] == {}
        assert out["tier_note"] is None
