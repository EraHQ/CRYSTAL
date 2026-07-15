"""Bank fact ops + the immutable ledger (Q1A + Q6B, ratified 2026-07-15).

History lives in ONE immutable home: fact_ledger carries the full
before/after claim text of every bank-surface mutation; the fact row
itself leaves through the existing delete_fact machinery. The mixin is
append-only by construction — no update/delete methods exist.

R14: verified by pytest.
"""
from __future__ import annotations


async def test_ledger_append_and_reads(store):
    row = await store.append_fact_ledger(
        "cus_a", "cry_1", "fact_1",
        op="supersede", actor="tenant",
        before_prompt="What CLI framework?",
        before_text="Use click for CLIs.",
        after_text="Use typer for CLIs (click underneath).",
        successor_fact_id="fact_2",
    )
    assert row["id"].startswith("fl_")
    await store.append_fact_ledger(
        "cus_a", "cry_1", "fact_3", op="retire",
        before_text="Stale claim.")
    await store.append_fact_ledger(
        "cus_b", "cry_9", "fact_9", op="retire", before_text="other tenant")

    by_crystal = await store.list_fact_ledger_for_crystal("cry_1")
    assert len(by_crystal) == 2
    assert by_crystal[0]["op"] == "retire"  # newest first
    assert by_crystal[1]["after_text"].startswith("Use typer")
    assert by_crystal[1]["successor_fact_id"] == "fact_2"

    by_customer = await store.list_fact_ledger_for_customer("cus_a")
    assert len(by_customer) == 2
    assert all(e["customer_id"] == "cus_a" for e in by_customer)


async def test_ledger_is_append_only_by_construction(store):
    """The immutability guarantee is the ABSENCE of mutation methods.
    This test fails the moment someone adds one."""
    forbidden = [
        n for n in dir(store)
        if "fact_ledger" in n and any(
            v in n for v in ("update", "delete", "set_", "edit", "remove"))
    ]
    assert forbidden == [], f"fact_ledger mutation methods exist: {forbidden}"


async def test_bank_graph_read_is_tenant_readable():
    from crystal_cache.ingress.auth import _tenant_readable
    assert _tenant_readable("GET", "/admin/api/bank/graph")
    assert _tenant_readable("GET", "/admin/api/bank/ledger")


async def test_tenant_write_allowance_covers_fact_ops():
    from crystal_cache.ingress.auth import _tenant_writable
    assert _tenant_writable("POST", "/admin/api/crystals/c1/facts/f1/supersede")
    assert _tenant_writable("POST", "/admin/api/crystals/c1/facts/f1/retire")
    # Reads and unrelated writes stay closed.
    assert not _tenant_writable("POST", "/admin/api/crystals/c1")
    assert not _tenant_writable("DELETE", "/admin/api/crystals/c1/facts/f1/retire")
