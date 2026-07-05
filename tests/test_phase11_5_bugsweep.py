"""Phase 11.5 bug-sweep tests.

Per P0.108: tests that close specific Phase 11.5 in-scope items.
Phase 11.5 is the FINAL phase of the v2 port; tests here are
bug-sweep, not feature-add.

Tests in this file:
  M1 — CU-20 reconciliation: `reconcile_total_action_items` corrects
       drift between CritiqueRow.total_action_items and the actual
       count of ActionItemRow rows pointing at the critique. (P0.101)

The smoke tests for CLI + HTTP (PRD-7) live in
`tests/test_phase11_5_smoke.py` because they have a different
testing pattern (subprocess + FastAPI TestClient).

The chat_proxy chars gaps (P8, P9) extend
`tests/test_phase9c_chat_proxy_chars.py` in place because they are
chars of the same module.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# M1 — CU-20: reconcile_total_action_items corrects drift
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m1_reconcile_total_action_items(store, customer):
    """CU-20 (P0.101) — verify the reconciliation helper.

    Setup simulates the drift scenario:
      1. Create a Critique with total_action_items=2 (the "expected"
         count at creation time, e.g. set by a writer that knew
         about 2 items).
      2. Create 3 ActionItems via `create_action_item` AFTER the
         critique. `create_action_item` does NOT update the parent
         counter (verified — see McrExtensionsMixin source). The
         counter is now stale: reads 2, actual count is 3.
      3. Call `reconcile_total_action_items(critique_id)`.
      4. Assertions:
         a. The helper returns the actual count (3).
         b. The CritiqueRow's counter is now 3.
         c. Idempotent: calling again returns 3 again, counter
            stays at 3.

    The test also covers the not-found case: calling with a bogus
    critique_id returns None without raising.
    """
    # Seed a critique with declared total_action_items=2.
    critique = await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="haiku",
        total_action_items=2,
    )
    assert critique.total_action_items == 2

    # Create 3 action items via create_action_item.
    for i in range(3):
        await store.create_action_item(
            critique_id=critique.id,
            customer_id=customer.id,
            action_type="research_task",
            content={"topic": f"topic_{i}"},
        )

    # Verify the drift exists (the counter is still 2; actual = 3).
    critique_before = await store.get_critique(critique.id)
    assert critique_before is not None
    assert critique_before.total_action_items == 2  # stale
    actual_items = await store.list_action_items_for_critique(critique.id)
    assert len(actual_items) == 3  # actual

    # Reconcile. Returns the corrected count.
    corrected = await store.reconcile_total_action_items(critique.id)
    assert corrected == 3

    # Counter is now in sync.
    critique_after = await store.get_critique(critique.id)
    assert critique_after is not None
    assert critique_after.total_action_items == 3

    # Idempotent: a second call returns the same value and doesn't
    # break the counter.
    corrected_again = await store.reconcile_total_action_items(critique.id)
    assert corrected_again == 3
    critique_final = await store.get_critique(critique.id)
    assert critique_final is not None
    assert critique_final.total_action_items == 3

    # Not-found case: bogus critique_id returns None, doesn't raise.
    result = await store.reconcile_total_action_items("nonexistent_id_abc")
    assert result is None
