"""C1 — assumption injection framing (ratified 2026-08-07, Q1=C).

The retrieval-time annotation surface: the batched store read
(`list_assumption_annotations`), the single-source renderer
(`tier_signal.assumption_note`), and the agent lane's fail-safe
application in `_apply_tier_signal`. The proxy lane appends the same
renderer's output; its wiring is exercised transitively through the
renderer + read tests (the pipeline block is a verbatim sibling of the
tier-note block pinned here at the read/render layer).

R14 note: verified by `pytest`; the renderer additionally ran 14
standalone container checks at authoring time (2026-08-07, /tmp/work3
rig).
"""
from __future__ import annotations

from typing import Any

import pytest

from crystal_cache.infrastructure.metadata_store_assumption_ext import (
    parse_assumption_tags,
)
from crystal_cache.infrastructure.schema import CrystalRow
from crystal_cache.retrieval.tier_signal import assumption_note


async def _seed_crystal(store, crystal_id, customer_id, *, summary=None):
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
            summary_text=summary,
        ))


async def _seed_assumption(store, customer, encoder, *, gap_id=None):
    """Two parents + one assumption via the production write path."""
    await _seed_crystal(store, "cr_a", customer.id,
                        summary="Deploys go through Cloud Run")
    await _seed_crystal(store, "cr_b", customer.id,
                        summary="Deploy failures spike on Fridays")
    result = await store.create_assumption_crystal(
        customer.id,
        statement="Friday deploys through Cloud Run carry elevated risk",
        subject="Deploy risk windows",
        parent_a_id="cr_a",
        parent_b_id="cr_b",
        confidence=0.82,
        encoder=encoder,
        gap_id=gap_id,
    )
    return result["crystal_id"]


# ---------------------------------------------------------------------------
# Store read — list_assumption_annotations
# ---------------------------------------------------------------------------

async def test_annotation_read_returns_assumption_shape(
    store, customer, semantic_encoder_stub,
):
    asm_id = await _seed_assumption(store, customer, semantic_encoder_stub)

    # Mixed id list: the assumption, a plain parent, and a miss — only
    # the assumption annotates; non-assumption crystals are invisible.
    annotations = await store.list_assumption_annotations(
        customer.id, [asm_id, "cr_a", "cr_missing"],
    )
    assert list(annotations) == [asm_id]
    info = annotations[asm_id]
    assert info["quality_tier"] == "quarantine"
    assert info["confidence"] == pytest.approx(0.82)
    assert info["invalidated_parents"] == []
    parent_summaries = {p["summary_text"] for p in info["parents"]}
    assert parent_summaries == {
        "Deploys go through Cloud Run",
        "Deploy failures spike on Fridays",
    }


async def test_annotation_read_empty_and_non_assumption(
    store, customer, semantic_encoder_stub,
):
    assert await store.list_assumption_annotations(customer.id, []) == {}
    await _seed_crystal(store, "cr_plain", customer.id, summary="plain")
    assert await store.list_assumption_annotations(
        customer.id, ["cr_plain"],
    ) == {}


async def test_annotation_read_tenant_guarded(
    store, customer, semantic_encoder_stub,
):
    asm_id = await _seed_assumption(store, customer, semantic_encoder_stub)
    foreign = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="sk-other",
    )
    # A foreign tenant asking about this id sees nothing.
    assert await store.list_assumption_annotations(
        foreign.id, [asm_id],
    ) == {}


async def test_annotation_read_dead_parent_from_tags(
    store, customer, semantic_encoder_stub,
):
    """Capture-at-delete (slice 4) blacklists + tags; the annotation
    read must surface the dead parent from the AUDIT TAGS (its edge is
    gone) and keep the survivor as a live parent."""
    asm_id = await _seed_assumption(store, customer, semantic_encoder_stub)
    await store.delete_crystal("cr_a", customer.id)

    annotations = await store.list_assumption_annotations(
        customer.id, [asm_id],
    )
    info = annotations[asm_id]
    assert info["quality_tier"] == "blacklist"
    assert info["invalidated_parents"] == ["cr_a"]
    assert [p["id"] for p in info["parents"]] == ["cr_b"]


# ---------------------------------------------------------------------------
# Renderer — tier_signal.assumption_note (single source, both lanes)
# ---------------------------------------------------------------------------

def test_assumption_note_silent_when_empty():
    assert assumption_note({}) is None


def test_assumption_note_names_inference_confidence_and_parents():
    note = assumption_note({
        "asm_1": {
            "quality_tier": "quarantine",
            "confidence": 0.82,
            "gap_id": None,
            "invalidated_parents": [],
            "parents": [
                {"id": "cr_a", "summary_text": "Parent A summary"},
                {"id": "cr_b", "summary_text": "Parent B summary"},
            ],
        },
    })
    assert "NOT" in note and "stated facts" in note
    assert "(confidence 0.82)" in note
    assert '"Parent A summary" + "Parent B summary"' in note


def test_assumption_note_invalidated_and_cap():
    annotations: dict[str, dict[str, Any]] = {
        "asm_dead": {
            "quality_tier": "blacklist", "confidence": 0.9, "gap_id": None,
            "invalidated_parents": ["cr_gone"],
            "parents": [{"id": "cr_b", "summary_text": "Survivor"}],
        },
    }
    annotations.update({
        f"asm_{i}": {
            "quality_tier": "quarantine", "confidence": 0.7, "gap_id": None,
            "invalidated_parents": [], "parents": [],
        } for i in range(4)
    })
    note = assumption_note(annotations)
    assert "INVALIDATED" in note and "do not rely" in note
    assert note.count("\n- ") == 3  # conflict_note's cap discipline
    assert "(+2 more assumption(s)" in note


def test_parse_assumption_tags_shared_parser():
    parsed = parse_assumption_tags([
        "assumption_confidence:0.75",
        "assumption_gap:gap_9",
        "assumption_invalidated:parent:cr_x",
        "unrelated_tag",
    ])
    assert parsed == {
        "confidence": 0.75,
        "gap_id": "gap_9",
        "invalidated_parents": ["cr_x"],
    }


# ---------------------------------------------------------------------------
# Agent lane — _apply_tier_signal carries the note, fail-safe
# ---------------------------------------------------------------------------

async def test_apply_tier_signal_carries_assumption_note(
    store, customer, semantic_encoder_stub,
):
    from crystal_cache.agent.tools.retrievers import _apply_tier_signal

    asm_id = await _seed_assumption(store, customer, semantic_encoder_stub)
    payload = await _apply_tier_signal(store, customer.id, {
        "matched_crystal_ids": [asm_id, "cr_a"],
        "matched_fact_ids": [],
    })
    assert asm_id in payload["assumption_crystals"]
    assert "ASSUMPTIONS" in payload["assumption_note"]
    # The plain parent contributes no annotation.
    assert "cr_a" not in payload["assumption_crystals"]


async def test_apply_tier_signal_assumption_failsafe():
    """A store whose every read raises must never break retrieval —
    all three annotation families default instead."""
    from crystal_cache.agent.tools.retrievers import _apply_tier_signal

    class _BrokenStore:
        def __getattr__(self, name: str):
            async def _boom(*a: Any, **k: Any):
                raise RuntimeError("simulated store failure")
            return _boom

    payload = await _apply_tier_signal(_BrokenStore(), "cus_x", {
        "matched_crystal_ids": ["cr_1"],
        "matched_fact_ids": ["f_1"],
    })
    assert payload["assumption_crystals"] == {}
    assert payload["assumption_note"] is None
    assert payload["crystal_tiers"] == {}
    assert payload["conflict_note"] is None
