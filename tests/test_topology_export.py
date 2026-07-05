"""Topology-exact export/import (verdict 5, ratified 2026-07-02).

The bank arrives with earned trust intact: crystal identity, tiers,
scope stamps, chains, edges, conflicts, and citations survive a round
trip verbatim, id-preserving. Import policies: tenant rewrite, collision
skip, ghost-owner clearing, unknown-field tolerance.

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

from datetime import datetime, timezone

from crystal_cache.infrastructure.schema import (
    CrystalChainRow,
    CrystalEdgeRow,
    CrystalRow,
    FactRow,
)


async def _seed_bank(store, customer_id: str) -> None:
    """Two crystals with a chain + an edge; one earned whitelist personal
    crystal with a fact — the trust-and-topology fixture."""
    now = datetime.now(timezone.utc)
    async with store.session() as s:
        s.add(CrystalRow(
            id="cx_a", customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[0.1, 0.2],
            quality_tier="whitelist", owner_operator_id="op_keep",
            group_team_id=customer_id, mode=0o600,
            summary_text="alpha",
        ))
        s.add(CrystalRow(
            id="cx_b", customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[0.3, 0.4],
            quality_tier="neutral",
        ))
        s.add(FactRow(
            id="fx_1", crystal_id="cx_a", claim_text="the claim",
            prompt_text="Src|Loc|Sub|Dom", pair_type="question_answer",
        ))
        s.add(CrystalChainRow(
            source_crystal_id="cx_a", target_crystal_id="cx_b",
        ))
        s.add(CrystalEdgeRow(
            crystal_a_id="cx_a", crystal_b_id="cx_b",
            edge_type="co_queried", weight=0.7,
            last_reinforced_at=now,
        ))


async def _wipe_bank(store) -> None:
    """Simulate the restore precondition: the original rows are gone
    (fresh deployment or post-wipe). Ids are global PKs, so importing
    while the originals exist collides BY DESIGN."""
    async with store.session() as s:
        for row_cls, pk in (
            (FactRow, "fx_1"),
            (CrystalChainRow, ("cx_a", "cx_b")),
            (CrystalEdgeRow, ("cx_a", "cx_b", "co_queried")),
            (CrystalRow, "cx_a"),
            (CrystalRow, "cx_b"),
        ):
            row = await s.get(row_cls, pk)
            if row is not None:
                await s.delete(row)


async def test_topology_round_trip_preserves_everything(store, customer):
    await _seed_bank(store, customer.id)
    # An operator that exists in the TARGET team keeps ownership exact.
    async with store.session() as s:
        from crystal_cache.infrastructure.schema import OperatorRow
        s.add(OperatorRow(
            id="op_keep", team_id="cus_target", display_name="K",
            role="operator", status="active",
            created_at=datetime.now(timezone.utc),
        ))

    payload = await store.export_bank_topology(customer.id)
    assert payload["format"] == "crystal_topology_v2"
    assert {c["id"] for c in payload["crystals"]} == {"cx_a", "cx_b"}
    assert payload["crystals"][0]["summary_vector"]  # vectors ride along
    assert len(payload["facts"]) == 1
    assert len(payload["chains"]) == 1
    assert len(payload["edges"]) == 1

    await _wipe_bank(store)  # the restore precondition
    counts = await store.import_bank_topology("cus_target", payload)
    assert counts["crystals"] == 2
    assert counts["facts"] == 1
    assert counts["chains"] == 1
    assert counts["edges"] == 1
    assert counts["skipped_collisions"] == 0
    assert counts["owners_cleared"] == 0  # op_keep exists on the target

    restored = await store.get_crystal("cx_a")
    assert restored.customer_id == "cus_target"       # tenant rewritten
    assert restored.group_team_id == "cus_target"     # group follows team
    assert restored.quality_tier == "whitelist"        # earned trust intact
    assert restored.mode == 0o600                      # scope intact
    assert restored.owner_operator_id == "op_keep"     # ownership exact
    facts = await store.list_facts_for_crystal("cx_a")
    assert facts[0].id == "fx_1"                       # id-preserving


async def test_topology_import_collisions_and_ghost_owners(store, customer):
    await _seed_bank(store, customer.id)
    payload = await store.export_bank_topology(customer.id)

    # Importing while the originals still exist: every id collides —
    # BY DESIGN (ids are global; exact copies need a fresh/wiped bank).
    counts = await store.import_bank_topology(customer.id, payload)
    assert counts["crystals"] == 0
    assert counts["skipped_collisions"] >= 2

    # After the wipe, import into a team where op_keep does NOT exist:
    # scope stays, the ghost owner is cleared and counted.
    await _wipe_bank(store)
    counts = await store.import_bank_topology("cus_fresh", payload)
    assert counts["crystals"] == 2
    assert counts["owners_cleared"] == 1
    restored = await store.get_crystal("cx_a")
    assert restored.mode == 0o600
    assert restored.owner_operator_id is None


async def test_topology_import_drops_unknown_fields(store, customer):
    await _seed_bank(store, customer.id)
    payload = await store.export_bank_topology(customer.id)
    payload["crystals"][0]["field_from_the_future"] = "v99"

    await _wipe_bank(store)
    counts = await store.import_bank_topology("cus_v99", payload)
    assert counts["crystals"] == 2
    assert counts["dropped_fields"] == 1
