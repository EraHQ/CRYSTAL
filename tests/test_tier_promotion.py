"""Tier promotion scan (launch-prep sweep, 2026-07-02).

Quality tiers that MOVE, on signals the bank already produces: grounded
citations promote (quarantine → neutral → whitelist, one rung per pass,
age-gated at the top rung), open conflicts demote (whitelist → neutral),
and blacklist is human-set and never touched. No model calls anywhere.

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from crystal_cache.infrastructure.schema import CitationRow
from crystal_cache.models import Crystal
from crystal_cache.scan import run_tier_promotion_scan


def _crystal(customer_id: str, cid: str, tier: str, *, age_days: int = 30) -> Crystal:
    return Crystal(
        id=cid,
        customer_id=customer_id,
        summary_vector=[],
        quality_tier=tier,  # type: ignore[arg-type]
        created_at=datetime.now(timezone.utc) - timedelta(days=age_days),
    )


async def _cite(store: Any, customer_id: str, cid: str, n: int, *, grounded: bool = True):
    await store.record_citations(
        customer_id,
        query_log_id=None,
        citations=[
            {"crystal_id": cid, "handle": f"[C{i}]", "grounded": grounded}
            for i in range(n)
        ],
    )


async def _conflict(store: Any, customer_id: str, cid: str, key: str):
    await store.create_knowledge_conflict(
        customer_id,
        fact_a_id=f"fa_{key}",
        fact_b_id=f"fb_{key}",
        claim_a="the deadline is April",
        claim_b="the deadline is May",
        pair_key=key,
        crystal_a_id=cid,
    )


async def test_neutral_promotes_to_whitelist_on_citations_age_no_conflicts(
    store, customer,
):
    await store.upsert_crystal(_crystal(customer.id, "crys_up", "neutral", age_days=30))
    await _cite(store, customer.id, "crys_up", 3)

    result = await run_tier_promotion_scan(store=store, customer_id=customer.id)

    assert result.promoted == 1
    assert (await store.get_crystal("crys_up")).quality_tier == "whitelist"


async def test_young_neutral_is_not_promoted(store, customer):
    await store.upsert_crystal(_crystal(customer.id, "crys_young", "neutral", age_days=1))
    await _cite(store, customer.id, "crys_young", 5)

    result = await run_tier_promotion_scan(store=store, customer_id=customer.id)

    assert result.promoted == 0
    assert (await store.get_crystal("crys_young")).quality_tier == "neutral"


async def test_ungrounded_citations_do_not_count(store, customer):
    await store.upsert_crystal(_crystal(customer.id, "crys_ug", "neutral", age_days=30))
    await _cite(store, customer.id, "crys_ug", 5, grounded=False)

    result = await run_tier_promotion_scan(store=store, customer_id=customer.id)

    assert result.promoted == 0
    assert (await store.get_crystal("crys_ug")).quality_tier == "neutral"


async def test_open_conflict_blocks_promotion(store, customer):
    await store.upsert_crystal(_crystal(customer.id, "crys_cf", "neutral", age_days=30))
    await _cite(store, customer.id, "crys_cf", 3)
    await _conflict(store, customer.id, "crys_cf", "pk_block")

    result = await run_tier_promotion_scan(store=store, customer_id=customer.id)

    assert result.promoted == 0
    assert (await store.get_crystal("crys_cf")).quality_tier == "neutral"


async def test_whitelist_demotes_on_open_conflict(store, customer):
    await store.upsert_crystal(_crystal(customer.id, "crys_dn", "whitelist", age_days=60))
    await _conflict(store, customer.id, "crys_dn", "pk_demote")

    result = await run_tier_promotion_scan(store=store, customer_id=customer.id)

    assert result.demoted == 1
    assert (await store.get_crystal("crys_dn")).quality_tier == "neutral"


async def test_quarantine_promotes_one_rung_on_first_grounded_citation(
    store, customer,
):
    await store.upsert_crystal(_crystal(customer.id, "crys_q", "quarantine", age_days=30))
    await _cite(store, customer.id, "crys_q", 5)

    result = await run_tier_promotion_scan(store=store, customer_id=customer.id)

    # One rung per pass: quarantine reaches neutral, NOT whitelist, even
    # with citations that would satisfy the top rung.
    assert result.promoted == 1
    assert (await store.get_crystal("crys_q")).quality_tier == "neutral"


async def test_blacklist_is_never_touched(store, customer):
    await store.upsert_crystal(_crystal(customer.id, "crys_bl", "blacklist", age_days=60))
    await _cite(store, customer.id, "crys_bl", 10)

    result = await run_tier_promotion_scan(store=store, customer_id=customer.id)

    assert result.promoted == 0
    assert result.demoted == 0
    assert (await store.get_crystal("crys_bl")).quality_tier == "blacklist"


async def test_knob_overrides_apply(store, customer):
    """Explicit thresholds override the settings knobs."""
    await store.upsert_crystal(_crystal(customer.id, "crys_k", "neutral", age_days=2))
    await _cite(store, customer.id, "crys_k", 1)

    result = await run_tier_promotion_scan(
        store=store, customer_id=customer.id, min_citations=1, min_age_days=1,
    )

    assert result.promoted == 1
    assert (await store.get_crystal("crys_k")).quality_tier == "whitelist"


# ---------------------------------------------------------------------------
# Decay (ratified 2026-07-02: 30 days) — whitelist drifts back to neutral
# when its newest grounded citation falls outside the window. Never below
# neutral; conflict demotion (tested above) takes precedence in the pass.
# ---------------------------------------------------------------------------

async def _cite_at(store: Any, customer_id: str, cid: str, age_days: int):
    """One grounded citation with a backdated created_at (direct row —
    record_citations always stamps now)."""
    async with store.session() as s:
        s.add(CitationRow(
            id=f"cit_{cid}_{age_days}",
            customer_id=customer_id,
            crystal_id=cid,
            handle="1",
            claim_span="cited claim",
            grounded=True,
            created_at=datetime.now(timezone.utc) - timedelta(days=age_days),
        ))


async def test_whitelist_decays_when_citations_go_stale(store, customer):
    await store.upsert_crystal(_crystal(customer.id, "crys_dk", "whitelist", age_days=90))
    await _cite_at(store, customer.id, "crys_dk", 40)

    result = await run_tier_promotion_scan(store=store, customer_id=customer.id)

    assert result.demoted == 1
    assert (await store.get_crystal("crys_dk")).quality_tier == "neutral"


async def test_whitelist_with_recent_citation_stays(store, customer):
    await store.upsert_crystal(_crystal(customer.id, "crys_fresh", "whitelist", age_days=90))
    await _cite_at(store, customer.id, "crys_fresh", 5)

    result = await run_tier_promotion_scan(store=store, customer_id=customer.id)

    assert result.demoted == 0
    assert (await store.get_crystal("crys_fresh")).quality_tier == "whitelist"


async def test_whitelist_with_no_citations_decays(store, customer):
    await store.upsert_crystal(_crystal(customer.id, "crys_none", "whitelist", age_days=90))

    result = await run_tier_promotion_scan(store=store, customer_id=customer.id)

    assert result.demoted == 1
    assert (await store.get_crystal("crys_none")).quality_tier == "neutral"


async def test_decay_window_override(store, customer):
    await store.upsert_crystal(_crystal(customer.id, "crys_win", "whitelist", age_days=90))
    await _cite_at(store, customer.id, "crys_win", 10)

    stays = await run_tier_promotion_scan(
        store=store, customer_id=customer.id, decay_days=30,
    )
    assert stays.demoted == 0

    decays = await run_tier_promotion_scan(
        store=store, customer_id=customer.id, decay_days=5,
    )
    assert decays.demoted == 1
    assert (await store.get_crystal("crys_win")).quality_tier == "neutral"
