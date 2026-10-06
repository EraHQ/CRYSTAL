"""2026-10-06: every source_kind written anywhere is a member of the Fact
model's closed set, and a respond thread's rows load without error.

Found by the new ERROR alert within an hour of its creation: v110 wrote
the operator's response fact with source_kind='operator_response', which
the Fact model rejects, so the assumptions worker's cycle and the
idle-phase gap discovery failed on every bank holding a respond thread,
for four days, silently. The write now uses 'operator_stated'; migration
e5b7c9d1f3a4 corrects existing rows.
"""
import inspect
import re
import typing

import pytest

from crystal_cache.models.crystal import SourceKind

ALLOWED = set(typing.get_args(SourceKind))


def test_every_source_kind_literal_in_the_package_is_allowed():
    import importlib
    import pkgutil

    import crystal_cache

    offenders = []
    for mod in pkgutil.walk_packages(crystal_cache.__path__, "crystal_cache."):
        if ".benchmarks" in mod.name or ".scripts" in mod.name:
            continue
        try:
            m = importlib.import_module(mod.name)
        except Exception:  # noqa: BLE001  (optional deps)
            continue
        try:
            src = inspect.getsource(m)
        except (OSError, TypeError):
            continue
        for value in re.findall(r"source_kind\s*=\s*[\"']([a-z_]+)[\"']", src):
            if value not in ALLOWED:
                offenders.append(f"{mod.name}: {value}")
    assert offenders == [], offenders


@pytest.mark.asyncio
async def test_a_respond_thread_loads_through_the_fact_model(store, customer, semantic_encoder_stub):
    from crystal_cache.infrastructure.schema import CrystalRow

    async with store.session() as s:
        s.add(CrystalRow(id="ev_a", customer_id=customer.id, summary_vector=[], summary_text="a"))
        s.add(CrystalRow(id="ev_b", customer_id=customer.id, summary_vector=[], summary_text="b"))
    out = await store.create_assumption_crystal(
        customer.id, statement="revised", subject="s", parent_a_id="ev_a", parent_b_id="ev_b",
        confidence=0.7, encoder=semantic_encoder_stub,
        extra_tags=["assumption_thread:root", "assumption_responds_to:root", "assumption_iteration:1"],
        operator_response="what is actually true",
    )
    # The exact loaders the assumptions worker and gap discovery use.
    facts = await store.list_facts_for_crystal(out["crystal_id"], include_deactivated=True)
    kinds = {f.source_kind for f in facts}
    assert kinds <= ALLOWED
    assert "operator_stated" in kinds
    everything = await store.list_all_facts_for_customer(customer.id)  # gated facts are excluded (RC-03)...
    assert all(f.source_kind in ALLOWED for f in everything)
    thread = await store.list_assumption_thread(customer.id, "root")
    assert any(m["operator_response"] == "what is actually true" for m in thread)
