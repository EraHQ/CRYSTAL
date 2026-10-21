"""Console Refine PR 7 (docs/CONSOLE_UX_AUDIT.md 5.5): what the Conflicts
page needs from the server.

- The list carries each claim's learned date, source and active flag.
- /reopen is the Undo of a resolution: the conflict is open again, a
  deactivated loser is back in recall, a blacklist mark written by the
  conflict is gone; foreign ids are 404 and an open conflict cannot be
  reopened.
- A scan leaves a conflict_scan_completed event the header reads back.

PR 7b (2026-10-16): every CONTRADICTS verdict gets a second small call
that writes `reasoning` ("why it looks like a contradiction") and a
`recommendation` {resolution, loser, reason}; a bad answer leaves the
conflict written but unexplained; POST /explain fills an old row; a
duplicate cannot be explained; a foreign id is 404.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from crystal_cache.infrastructure.schema import BlacklistedReflectionRow, CrystalRow, FactRow

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_T1 = datetime(2026, 3, 1, tzinfo=timezone.utc)

try:
    from tests.fakes import SeamFake
except ModuleNotFoundError:
    from fakes import SeamFake

_GOOD = (
    '{"why": "A rate of 120 and a rate of 95 cannot both be the current rate.", '
    '"resolution": "superseded", "loser": "a", "reason": "Claim B was stated later by you."}'
)


class _ExplainingFake(SeamFake):
    """Verdict word for the discriminator prompt, `explanation` for the
    explain prompt (the schema sent tells them apart — both calls carry
    one since 2026-10-21; None answers the explain prompt with a bare
    verdict word, the bad-answer lane)."""

    def __init__(self, explanation=_GOOD):
        self.explanation = explanation
        self.calls = 0
        self.systems: list = []

    def is_ready(self) -> bool:
        return True

    def complete(self, **kwargs):
        from crystal_cache.scan.contradiction import EXPLAIN_SCHEMA

        self.calls += 1
        self.systems.append(kwargs.get("system"))
        if kwargs.get("json_schema") is EXPLAIN_SCHEMA and self.explanation is not None:
            return self.explanation
        return "CONTRADICTS"


async def _seed_pair(store, customer_id: str, tag: str, key_a: str, key_b: str):
    async with store.session() as s:
        s.add(CrystalRow(id=f"cA{tag}", customer_id=customer_id, crystal_type="customer:legacy",
                         summary_vector=[], source_path=f"docs/{tag}-a.md"))
        s.add(CrystalRow(id=f"cB{tag}", customer_id=customer_id, crystal_type="customer:legacy",
                         summary_vector=[], source_path=f"docs/{tag}-b.md"))
        s.add(FactRow(id=f"fa{tag}", crystal_id=f"cA{tag}", pair_type="question_answer", prompt_text=key_a,
                      claim_text=f"the {tag} is 120", source_kind="document_extraction", vector=[], created_at=_T0))
        s.add(FactRow(id=f"fb{tag}", crystal_id=f"cB{tag}", pair_type="question_answer", prompt_text=key_b,
                      claim_text=f"the {tag} is 95", source_kind="operator_stated", vector=[], created_at=_T1))


def _app(store, encoder, vector_store, fact_vector_store):
    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app
    return _build_app(store, encoder, vector_store, fact_vector_store)


def _client(app):
    from httpx import ASGITransport, AsyncClient

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


async def _seed(store, customer_id: str):
    async with store.session() as s:
        s.add(CrystalRow(id="cR", customer_id=customer_id, crystal_type="customer:legacy",
                         summary_vector=[], source_path="docs/rates.md"))
        s.add(FactRow(id="f_old", crystal_id="cR", pair_type="question_answer", prompt_text="Rate|Old",
                      claim_text="rate is 120", source_kind="document_extraction", vector=[], grating_strength=1.0, created_at=_T0))
        s.add(FactRow(id="f_new", crystal_id="cR", pair_type="question_answer", prompt_text="Rate|New",
                      claim_text="rate is 95", source_kind="operator_stated", vector=[], grating_strength=1.0, created_at=_T1))
    return await store.create_knowledge_conflict(
        customer_id, fact_a_id="f_old", fact_b_id="f_new", claim_a="rate is 120", claim_b="rate is 95",
        pair_key="pk-refine", crystal_a_id="cR", crystal_b_id="cR", subject="Rate",
    )


@pytest.mark.asyncio
async def test_list_carries_claim_details_and_reopen_undoes_a_blacklist(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    c = await _seed(store, customer.id)
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    async with _client(app) as cl:
        r = await cl.get("/admin/api/conflicts", params={"customer_id": customer.id})
        assert r.status_code == 200, r.text[:300]
        row = next(x for x in r.json()["conflicts"] if x["id"] == c.id)
        assert row["facts"]["a"]["learned_at"].startswith("2026-01-01")
        assert row["facts"]["b"]["learned_at"].startswith("2026-03-01")
        assert row["facts"]["a"]["source_kind"] == "document_extraction"
        assert row["facts"]["a"]["source_path"] == "docs/rates.md"
        assert row["facts"]["a"]["active"] is True and row["facts"]["b"]["active"] is True
        # Mark A wrong: it leaves recall and its claim is blacklisted.
        r = await cl.post(f"/admin/api/conflicts/{c.id}/resolve", params={"customer_id": customer.id},
                          json={"resolution": "blacklisted", "loser": "a"})
        assert r.status_code == 200, r.text[:300]
        r = await cl.get("/admin/api/conflicts", params={"customer_id": customer.id, "status": "resolved"})
        row = next(x for x in r.json()["conflicts"] if x["id"] == c.id)
        assert row["facts"]["a"]["active"] is False and row["facts"]["b"]["active"] is True
        async with store.session() as s:
            marks = (await s.execute(select(BlacklistedReflectionRow).where(
                BlacklistedReflectionRow.customer_id == customer.id))).scalars().all()
        assert len(marks) == 1
        # A foreign tenant cannot reopen it; the owner can, and everything comes back.
        other = await store.create_customer(provider="anthropic", model_id="claude-sonnet-4-5-20250929", api_key_ref="sk-other")
        r = await cl.post(f"/admin/api/conflicts/{c.id}/reopen", params={"customer_id": other.id})
        assert r.status_code == 404
        r = await cl.post(f"/admin/api/conflicts/{c.id}/reopen", params={"customer_id": customer.id})
        assert r.status_code == 200, r.text[:300]
        assert r.json()["conflict"]["status"] == "open" and r.json()["conflict"]["resolution"] is None
        r = await cl.get("/admin/api/conflicts", params={"customer_id": customer.id})
        row = next(x for x in r.json()["conflicts"] if x["id"] == c.id)
        assert row["facts"]["a"]["active"] is True
        async with store.session() as s:
            marks = (await s.execute(select(BlacklistedReflectionRow).where(
                BlacklistedReflectionRow.customer_id == customer.id))).scalars().all()
        assert marks == []
        # Reopening an open conflict is refused.
        r = await cl.post(f"/admin/api/conflicts/{c.id}/reopen", params={"customer_id": customer.id})
        assert r.status_code == 404
        # The activity feed filters by event type and recorded the reopen.
        r = await cl.get("/admin/api/curation/activity", params={"customer_id": customer.id, "event_type": "conflict_reopened"})
        assert r.status_code == 200
        assert [e["event_type"] for e in r.json()["events"]] == ["conflict_reopened"]


@pytest.mark.asyncio
async def test_dismiss_then_reopen_and_the_scan_record(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch,
):
    c = await _seed(store, customer.id)
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    async with _client(app) as cl:
        r = await cl.post(f"/admin/api/conflicts/{c.id}/resolve", params={"customer_id": customer.id},
                          json={"resolution": "dismissed"})
        assert r.status_code == 200
        r = await cl.get("/admin/api/conflicts", params={"customer_id": customer.id, "status": "dismissed"})
        assert [x["id"] for x in r.json()["conflicts"]] == [c.id]
        r = await cl.post(f"/admin/api/conflicts/{c.id}/reopen", params={"customer_id": customer.id})
        assert r.status_code == 200 and r.json()["conflict"]["status"] == "open"

        # The scan route records what it did (the scanner itself is faked:
        # no model in the suite).
        import dataclasses

        from crystal_cache.endpoints import admin as admin_mod
        from crystal_cache.scan.contradiction import ScanResult

        async def _fake_scan(*, store, customer_id, max_candidate_pairs, max_discriminator_calls):
            return ScanResult(customer_id=customer_id, facts_scanned=2354, candidate_pairs=900,
                              pairs_evaluated=812, skipped_existing=0, conflicts_found=1, budget_exhausted=False)

        class _Ready:
            def is_ready(self):
                return True

        monkeypatch.setattr(admin_mod, "scan_for_contradictions", _fake_scan)
        monkeypatch.setattr(admin_mod, "get_llm_client", lambda: _Ready())
        r = await cl.post("/admin/api/conflicts/scan", params={"customer_id": customer.id})
        assert r.status_code == 200, r.text[:300]
        assert r.json()["scan"]["facts_scanned"] == 2354
        r = await cl.get("/admin/api/curation/activity", params={"customer_id": customer.id, "event_type": "conflict_scan_completed", "limit": 1})
        ev = r.json()["events"][0]
        assert ev["payload"]["facts_scanned"] == 2354 and ev["payload"]["conflicts_found"] == 1
        assert dataclasses.fields(ScanResult)  # the payload is the dataclass, nothing hand-copied


def test_parse_explanation_is_strict_about_the_loser():
    from crystal_cache.scan.contradiction import parse_explanation

    good = parse_explanation(_GOOD)
    assert good == {
        "why": "A rate of 120 and a rate of 95 cannot both be the current rate.",
        "resolution": "superseded", "loser": "a", "reason": "Claim B was stated later by you.",
    }
    # superseded / blacklisted need a side; qualified / dismissed never carry one.
    assert parse_explanation('{"why": "w", "resolution": "superseded", "loser": "none", "reason": "r"}') is None
    assert parse_explanation('{"why": "w", "resolution": "qualified", "loser": "a", "reason": "r"}')["loser"] is None
    assert parse_explanation('{"why": "", "resolution": "qualified", "loser": "none", "reason": "r"}') is None
    assert parse_explanation('{"why": "w", "resolution": "delete", "loser": "a", "reason": "r"}') is None
    assert parse_explanation("CONTRADICTS") is None
    assert parse_explanation(None) is None


@pytest.mark.asyncio
async def test_scan_explains_each_contradiction_and_survives_a_bad_answer(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    from crystal_cache.scan.contradiction import scan_for_contradictions

    await _seed_pair(store, customer.id, "rate", "Assumptions|Rate card", "Ops|Rate card|Policy|Section 3")
    fake = _ExplainingFake()
    result = await scan_for_contradictions(store=store, slm_client=fake, customer_id=customer.id)
    assert result.conflicts_found >= 1
    assert result.explained == result.conflicts_found
    # One verdict call per pair plus one explanation per found conflict.
    assert fake.calls == result.pairs_evaluated + result.conflicts_found
    conflicts = await store.list_knowledge_conflicts(customer.id, status="open", limit=10)
    c = next(x for x in conflicts if x.fact_a_id in ("farate", "fbrate"))
    assert c.reasoning == "A rate of 120 and a rate of 95 cannot both be the current rate."
    assert c.recommendation == {"resolution": "superseded", "loser": "a", "reason": "Claim B was stated later by you."}
    # The list route carries both, and the scan payload counts them.
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    async with _client(app) as cl:
        r = await cl.get("/admin/api/conflicts", params={"customer_id": customer.id})
        row = next(x for x in r.json()["conflicts"] if x["id"] == c.id)
        assert row["reasoning"] == c.reasoning and row["recommendation"]["loser"] == "a"
    import dataclasses
    assert dataclasses.asdict(result)["explained"] == result.explained

    # A second pair, with a model that answers the explain prompt with a
    # bare verdict word: the conflict is still written, just unexplained.
    await _seed_pair(store, customer.id, "fee", "Assumptions|Fee schedule", "Ops|Fee schedule|Policy|Section 3")
    result = await scan_for_contradictions(store=store, slm_client=_ExplainingFake(explanation=None), customer_id=customer.id)
    assert result.conflicts_found >= 1 and result.explained == 0
    conflicts = await store.list_knowledge_conflicts(customer.id, status="open", limit=10)
    c2 = next(x for x in conflicts if x.fact_a_id in ("fafee", "fbfee"))
    assert c2.reasoning is None and c2.recommendation is None


@pytest.mark.asyncio
async def test_explain_route_fills_an_old_conflict(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch,
):
    from crystal_cache.endpoints import admin as admin_mod

    c = await _seed(store, customer.id)
    assert c.reasoning is None
    dup = await store.create_knowledge_conflict(
        customer.id, fact_a_id="f_old", fact_b_id="f_new", claim_a="rate is 120", claim_b="rate is 120",
        pair_key="pk-dup", detector="dedup_scan",
    )
    fake = _ExplainingFake()
    monkeypatch.setattr(admin_mod, "get_llm_client", lambda: fake)
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    async with _client(app) as cl:
        other = await store.create_customer(provider="anthropic", model_id="claude-sonnet-4-5-20250929", api_key_ref="sk-other")
        r = await cl.post(f"/admin/api/conflicts/{c.id}/explain", params={"customer_id": other.id})
        assert r.status_code == 404
        r = await cl.post("/admin/api/conflicts/kc_missing/explain", params={"customer_id": customer.id})
        assert r.status_code == 404
        # 2026-10-21: a duplicate is explained too, with the dedup prompt.
        r = await cl.post(f"/admin/api/conflicts/{dup.id}/explain", params={"customer_id": customer.id})
        assert r.status_code == 200, r.text[:300]
        assert fake.calls == 1
        from crystal_cache.scan.dedup import EXPLAIN_DUP_SYSTEM
        assert fake.systems[-1] == EXPLAIN_DUP_SYSTEM
        assert r.json()["conflict"]["recommendation"]["resolution"] == "superseded"
        r = await cl.post(f"/admin/api/conflicts/{c.id}/explain", params={"customer_id": customer.id})
        assert r.status_code == 200, r.text[:300]
        body = r.json()["conflict"]
        assert body["reasoning"].startswith("A rate of 120") and body["recommendation"] == {
            "resolution": "superseded", "loser": "a", "reason": "Claim B was stated later by you."}
        assert fake.calls == 2
        from crystal_cache.scan.contradiction import EXPLAIN_SYSTEM
        assert fake.systems[-1] == EXPLAIN_SYSTEM
        r = await cl.get("/admin/api/conflicts", params={"customer_id": customer.id})
        row = next(x for x in r.json()["conflicts"] if x["id"] == c.id)
        assert row["reasoning"] == body["reasoning"] and row["facts"]["a"]["active"] is True
        # An unusable answer stores nothing and says so.
        fresh = await store.create_knowledge_conflict(
            customer.id, fact_a_id="f_old", fact_b_id="f_new", claim_a="rate is 120", claim_b="rate is 95",
            pair_key="pk-fresh",
        )
        monkeypatch.setattr(admin_mod, "get_llm_client", lambda: _ExplainingFake(explanation=None))
        r = await cl.post(f"/admin/api/conflicts/{fresh.id}/explain", params={"customer_id": customer.id})
        assert r.status_code == 502
        r = await cl.get("/admin/api/conflicts", params={"customer_id": customer.id})
        row = next(x for x in r.json()["conflicts"] if x["id"] == fresh.id)
        assert row["reasoning"] is None and row["recommendation"] is None
