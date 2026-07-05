"""Growth G1 — citation record store surface (CitationExtensionsMixin).

record_citations writes the raw per-claim record the proxy produces;
list_citations_for_query / _for_crystal read it back. The table carries no
uniqueness (G4's ledger dedupes on interaction+crystal); `grounded` gates
which citations are G4-relevant.
"""
from __future__ import annotations


async def test_record_and_list_citations_for_query(store, customer):
    log_id = "qlog_cite_test"
    ids = await store.record_citations(
        customer.id,
        query_log_id=log_id,
        citations=[
            {
                "crystal_id": "cryst_a",
                "version": "hash1",
                "handle": "1",
                "claim_span": "The director is Jane Doe.",
                "grounding_score": 0.82,
                "grounded": True,
            },
            {
                "crystal_id": "cryst_b",
                "handle": "2",
                "grounding_score": 0.10,
                "grounded": False,
            },
        ],
    )
    assert len(ids) == 2

    rows = await store.list_citations_for_query(log_id)
    assert len(rows) == 2
    by_crystal = {r["crystal_id"]: r for r in rows}
    assert by_crystal["cryst_a"]["grounded"] is True
    assert by_crystal["cryst_a"]["crystal_version"] == "hash1"
    assert by_crystal["cryst_a"]["claim_span"] == "The director is Jane Doe."
    assert by_crystal["cryst_a"]["handle"] == "1"
    # A cited-but-ungrounded span is recorded (telemetry) but not grounded,
    # and its missing version round-trips as None.
    assert by_crystal["cryst_b"]["grounded"] is False
    assert by_crystal["cryst_b"]["crystal_version"] is None


async def test_record_citations_empty_is_noop(store, customer):
    assert (
        await store.record_citations(
            customer.id, query_log_id="qlog_x", citations=[]
        )
        == []
    )
    assert await store.list_citations_for_query("qlog_x") == []


async def test_list_citations_for_crystal_grounded_filter(store, customer):
    # Two grounded + one spurious citation of the same crystal across turns.
    await store.record_citations(
        customer.id,
        query_log_id="qlog_1",
        citations=[
            {"crystal_id": "cryst_z", "grounded": True, "grounding_score": 0.7}
        ],
    )
    await store.record_citations(
        customer.id,
        query_log_id="qlog_2",
        citations=[
            {"crystal_id": "cryst_z", "grounded": True, "grounding_score": 0.6},
            {"crystal_id": "cryst_z", "grounded": False, "grounding_score": 0.1},
        ],
    )

    # grounded_only (default) — the G4-relevant set.
    grounded = await store.list_citations_for_crystal(customer.id, "cryst_z")
    assert len(grounded) == 2
    assert all(r["grounded"] for r in grounded)

    # Full telemetry view includes the spurious one.
    everything = await store.list_citations_for_crystal(
        customer.id, "cryst_z", grounded_only=False
    )
    assert len(everything) == 3
