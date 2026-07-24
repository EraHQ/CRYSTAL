"""Never-Idle Convergence — conflict resolution-apply (B2).

The curation gate makes the resolution verbs real. Covers each verb's effect
on the bank: superseded/blacklisted deactivate the losing fact (grating→0),
blacklisted also records the wrong claim, qualified keeps both, dismissed is a
no-op — plus the validation errors, blacklist idempotency, and the endpoint
(400 / 404 / apply).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from crystal_cache.endpoints.admin import (
    ResolveConflictRequest,
    admin_resolve_conflict,
)
from crystal_cache.infrastructure.schema import (
    BlacklistedReflectionRow,
    CrystalRow,
    FactRow,
)

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_NOW = datetime(2026, 6, 16, tzinfo=timezone.utc)


class _FakeRequest:
    """Request stub for direct endpoint calls: no tenant pin, so the
    endpoint falls through to its customer_id param (default '' → no
    tenancy filter — the pre-pin behavior these tests were written
    against)."""

    class _State:
        tenant_pin = None

    state = _State()


_REQ = _FakeRequest()


async def _seed_conflict(store, customer_id, *, claim_a, claim_b):
    """A crystal with two facts (grating 1.0) and an open conflict over them."""
    async with store.session() as s:
        s.add(CrystalRow(
            id="cX", customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
        ))
        for fid, claim in (("fa", claim_a), ("fb", claim_b)):
            s.add(FactRow(
                id=fid, crystal_id="cX", pair_type="question_answer",
                prompt_text="", claim_text=claim, source_kind="model_reasoning",
                vector=[], grating_strength=1.0, created_at=_T0,
            ))
    return await store.create_knowledge_conflict(
        customer_id, fact_a_id="fa", fact_b_id="fb",
        claim_a=claim_a, claim_b=claim_b, pair_key="pk-res",
        crystal_a_id="cX", crystal_b_id="cX", subject="Rate",
    )


async def _grating(store, fact_id):
    async with store.session() as s:
        f = await s.get(FactRow, fact_id)
        return f.grating_strength if f is not None else None


# --- store: per-verb effects ---

async def test_superseded_deactivates_loser_only(store, customer):
    c = await _seed_conflict(store, customer.id, claim_a="rate is 120", claim_b="rate is 95")
    updated = await store.apply_conflict_resolution(
        c.id, resolution="superseded", loser="a", resolved_at=_NOW,
    )
    assert updated.status == "resolved"
    assert updated.resolution == "superseded"
    assert await _grating(store, "fa") == 0.0   # loser deactivated
    assert await _grating(store, "fb") == 1.0   # winner untouched


async def test_blacklisted_deactivates_and_records_claim(store, customer):
    c = await _seed_conflict(store, customer.id, claim_a="correct", claim_b="a wrong claim")
    updated = await store.apply_conflict_resolution(
        c.id, resolution="blacklisted", loser="b", resolved_at=_NOW,
    )
    assert updated.resolution == "blacklisted"
    assert await _grating(store, "fb") == 0.0
    rhash = hashlib.sha256("a wrong claim".encode("utf-8")).hexdigest()[:64]
    assert await store.is_reflection_blacklisted(customer.id, rhash) is True


async def test_qualified_keeps_both_active(store, customer):
    c = await _seed_conflict(store, customer.id, claim_a="true on weekdays", claim_b="true on weekends")
    updated = await store.apply_conflict_resolution(
        c.id, resolution="qualified", resolved_at=_NOW,
    )
    assert updated.status == "resolved"
    assert updated.resolution == "qualified"
    assert await _grating(store, "fa") == 1.0
    assert await _grating(store, "fb") == 1.0


async def test_dismissed_has_no_fact_effect(store, customer):
    c = await _seed_conflict(store, customer.id, claim_a="x", claim_b="y")
    updated = await store.apply_conflict_resolution(
        c.id, resolution="dismissed", resolved_at=_NOW,
    )
    assert updated.status == "dismissed"
    assert updated.resolution == "dismissed"
    assert await _grating(store, "fa") == 1.0
    assert await _grating(store, "fb") == 1.0


# --- store: validation + idempotency ---

async def test_superseded_requires_loser(store, customer):
    c = await _seed_conflict(store, customer.id, claim_a="x", claim_b="y")
    with pytest.raises(ValueError):
        await store.apply_conflict_resolution(
            c.id, resolution="superseded", resolved_at=_NOW,
        )


async def test_unknown_resolution_raises(store, customer):
    c = await _seed_conflict(store, customer.id, claim_a="x", claim_b="y")
    with pytest.raises(ValueError):
        await store.apply_conflict_resolution(
            c.id, resolution="bogus", resolved_at=_NOW,
        )


async def test_apply_missing_conflict_returns_none(store, customer):
    assert await store.apply_conflict_resolution(
        "kc_nope", resolution="qualified", resolved_at=_NOW,
    ) is None


async def test_blacklist_apply_is_idempotent(store, customer):
    c = await _seed_conflict(store, customer.id, claim_a="ok", claim_b="wrong")
    await store.apply_conflict_resolution(c.id, resolution="blacklisted", loser="b", resolved_at=_NOW)
    # A re-click resolves again without duplicating the blacklist row.
    await store.apply_conflict_resolution(c.id, resolution="blacklisted", loser="b", resolved_at=_NOW)
    rhash = hashlib.sha256("wrong".encode("utf-8")).hexdigest()[:64]
    async with store.session() as s:
        rows = (await s.execute(
            select(BlacklistedReflectionRow)
            .where(BlacklistedReflectionRow.customer_id == customer.id)
            .where(BlacklistedReflectionRow.reflection_hash == rhash)
        )).scalars().all()
    assert len(rows) == 1


# --- endpoint ---

async def test_resolve_endpoint_applies(store, customer):
    c = await _seed_conflict(store, customer.id, claim_a="rate 120", claim_b="rate 95")
    resp = await admin_resolve_conflict(
        request=_REQ,
        conflict_id=c.id,
        body=ResolveConflictRequest(resolution="superseded", loser="a"),
        store=store,
    )
    assert resp["conflict"]["resolution"] == "superseded"
    assert await _grating(store, "fa") == 0.0


async def test_resolve_endpoint_bad_resolution_is_400(store, customer):
    c = await _seed_conflict(store, customer.id, claim_a="x", claim_b="y")
    with pytest.raises(HTTPException) as ei:
        await admin_resolve_conflict(
            request=_REQ,
            conflict_id=c.id,
            body=ResolveConflictRequest(resolution="bogus"),
            store=store,
        )
    assert ei.value.status_code == 400


async def test_resolve_endpoint_missing_is_404(store, customer):
    with pytest.raises(HTTPException) as ei:
        await admin_resolve_conflict(
            request=_REQ,
            conflict_id="kc_nope",
            body=ResolveConflictRequest(resolution="qualified"),
            store=store,
        )
    assert ei.value.status_code == 404


async def test_resolve_endpoint_foreign_tenant_is_404(store, customer):
    """Tenancy pin (2026-07-23): a real conflict id under the WRONG
    customer_id resolves nothing and reads exactly like a missing one
    — never an existence oracle, never a cross-tenant write."""
    c = await _seed_conflict(store, customer.id, claim_a="x", claim_b="y")
    with pytest.raises(HTTPException) as ei:
        await admin_resolve_conflict(
            request=_REQ,
            conflict_id=c.id,
            body=ResolveConflictRequest(resolution="dismissed"),
            store=store,
            customer_id="cus_someone_else",
        )
    assert ei.value.status_code == 404
    # And with the RIGHT customer_id it applies.
    resp = await admin_resolve_conflict(
        request=_REQ,
        conflict_id=c.id,
        body=ResolveConflictRequest(resolution="dismissed"),
        store=store,
        customer_id=customer.id,
    )
    assert resp["conflict"]["status"] == "dismissed"


def test_tenant_write_allowlist_covers_curation_paths():
    """2026-07-23: the resolve buttons 401'd in production because
    these shapes weren't in the tenant-write allowlist. Pinned so the
    console's curation writes stay open — and unlisted writes stay
    closed."""
    from crystal_cache.ingress.auth import _tenant_writable

    assert _tenant_writable("POST", "/admin/api/conflicts/kc_x/resolve")
    assert _tenant_writable("POST", "/admin/api/conflicts/scan")
    assert _tenant_writable("POST", "/admin/api/source-schemas/ss_1/approve")
    assert _tenant_writable("POST", "/admin/api/source-schemas/ss_1/reject")
    assert _tenant_writable("PUT", "/admin/api/source-schemas/ss_1/mapping")
    assert _tenant_writable("POST", "/admin/api/source-schemas/preview")
    # Unlisted admin writes remain closed to tenant consoles.
    assert not _tenant_writable("POST", "/admin/api/conflicts/kc_x")
    assert not _tenant_writable("POST", "/admin/api/customers")
