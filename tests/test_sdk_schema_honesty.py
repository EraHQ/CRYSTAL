"""Phase 1.3 — API honesty pins (Q1=B, Q2=B, Q4=A, ratified 2026-08-25).

The contract this file enforces: for every SDK route with a
response_model, the declared schema IS the wire shape — validated here by
driving the real routes over ASGI and strict-validating the raw JSON
against the schema models (all of which carry extra='forbid', so an
undeclared key on the wire OR an invented field in the model fails).
/openapi.json stops being fiction because generation and reality are
pinned to the same source.

Also pinned:
  - Q1=B: a pipeline exception is HTTP 500 with a clean detail — never
    the old 200-with-empty-body that made an outage indistinguishable
    from an empty bank. And the inverse: an EMPTY bank really is an
    honest 200 no_match end-to-end.
  - Q2=B: `k` is wired (top_k=body.k) and its schema default is the
    pipeline's real 10.

App pattern follows test_endpoint_smoke: httpx ASGITransport, sdk router
mounted alone, conftest store + stub encoder, app.state wired by hand.

asyncio_mode=auto.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from crystal_cache.ingress.schema import (
    BankStatsResponse,
    CrystalDetailResponse,
    CrystalListResponse,
    QueryLogResponse,
    RetrieveResponse,
)


@pytest.fixture
def sdk_app(store, semantic_encoder_stub, vector_store, fact_vector_store):
    from crystal_cache.endpoints import sdk
    from crystal_cache.infrastructure.metadata_store import (
        get_metadata_store,
        set_metadata_store,
    )
    from crystal_cache.infrastructure.vector_index import InMemoryVectorIndex

    class _MessagesEncoder:
        """conftest's semantic stub + the `encode_messages` the real
        pipeline's first step calls (windowed multi-turn encode). Local
        adapter — delegates everything else to the stub, so the shared
        fixture stays untouched."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def encode_messages(self, messages, window=3, **_kw):
            texts = [
                m.get("content", "")
                for m in messages
                if m.get("role") == "user" and isinstance(m.get("content"), str)
            ][-window:]
            return self._inner.encode(" ".join(texts) or " ")

    app = FastAPI()
    app.include_router(sdk.router)

    async def _get_test_store():
        return store

    app.dependency_overrides[get_metadata_store] = _get_test_store
    set_metadata_store(store)

    app.state.metadata_store = store
    app.state.prompt_encoder = _MessagesEncoder(semantic_encoder_stub)
    app.state.vector_store = vector_store
    app.state.fact_vector_store = fact_vector_store
    app.state.vector_index = InMemoryVectorIndex(
        fact_store=fact_vector_store,
        vector_store=vector_store,
        metadata_store=store,
    )
    return app


def _client(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


def _auth(customer) -> dict[str, str]:
    return {"Authorization": f"Bearer {customer.api_key}"}


async def _seed_pair(store, customer, encoder, vector_store, key, value):
    return await store.add_pair_for_customer(
        customer_id=customer.id,
        prompt_text=key,
        answer_text=value,
        pair_type="question_answer",
        encoder=encoder,
        vector_store=vector_store,
        vector_index=None,
        crystal_type="customer:legacy",
        source_kind="model_reasoning",
        answer_value=None,
    )


# ---------------------------------------------------------------------------
# Q1=B — failure is a 500; empty is an honest 200
# ---------------------------------------------------------------------------

async def test_retrieve_pipeline_failure_is_500(sdk_app, customer, monkeypatch):
    async def _boom(**kwargs):
        raise RuntimeError("db fell over")

    import crystal_cache.retrieval as retrieval_pkg
    monkeypatch.setattr(retrieval_pkg, "retrieve_and_inject", _boom)

    async with _client(sdk_app) as client:
        r = await client.post(
            "/v1/retrieve", json={"query": "anything"}, headers=_auth(customer),
        )
    assert r.status_code == 500
    assert r.json()["detail"] == "retrieval pipeline failure"


async def test_retrieve_empty_bank_is_honest_200(sdk_app, customer):
    """The inverse pin: with failures now 5xx, an empty 200 MEANS an empty
    bank — proven end-to-end through the real pipeline over a real (empty)
    index."""
    async with _client(sdk_app) as client:
        r = await client.post(
            "/v1/retrieve", json={"query": "anything"}, headers=_auth(customer),
        )
    assert r.status_code == 200
    body = r.json()
    RetrieveResponse.model_validate(body)  # strict: extra='forbid'
    assert body["routing"] == "no_match"
    assert body["matched_crystal_ids"] == []
    assert body["matched_crystals"] == []
    assert body["cache_hit"] is False


# ---------------------------------------------------------------------------
# Q2=B — k is wired, default 10
# ---------------------------------------------------------------------------

def _fake_outcome() -> SimpleNamespace:
    return SimpleNamespace(
        cache_hit_response=None,
        injected_text=None,
        top_score=0.0,
        routing_decision=None,
        matched_crystal_ids=[],
        citation_manifest=None,
    )


async def test_retrieve_k_reaches_the_pipeline(sdk_app, customer, monkeypatch):
    captured: dict = {}

    async def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_outcome()

    import crystal_cache.retrieval as retrieval_pkg
    monkeypatch.setattr(retrieval_pkg, "retrieve_and_inject", _capture)

    async with _client(sdk_app) as client:
        r = await client.post(
            "/v1/retrieve",
            json={"query": "q", "k": 3},
            headers=_auth(customer),
        )
        assert r.status_code == 200
        assert captured["top_k"] == 3

        r = await client.post(
            "/v1/retrieve", json={"query": "q"}, headers=_auth(customer),
        )
        assert r.status_code == 200
        assert captured["top_k"] == 10  # the schema default = the real default


async def test_retrieve_k_bounds_still_validated(sdk_app, customer):
    async with _client(sdk_app) as client:
        r = await client.post(
            "/v1/retrieve",
            json={"query": "q", "k": 0},
            headers=_auth(customer),
        )
        assert r.status_code == 422
        r = await client.post(
            "/v1/retrieve",
            json={"query": "q", "k": 21},
            headers=_auth(customer),
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Q4=A — schema == wire, strict, for every declared response_model
# ---------------------------------------------------------------------------

async def test_stats_schema_is_the_wire_shape(
    sdk_app, store, customer, semantic_encoder_stub, vector_store,
):
    await _seed_pair(
        store, customer, semantic_encoder_stub, vector_store,
        "Honesty|Stats", "stats value",
    )
    async with _client(sdk_app) as client:
        r = await client.get("/v1/stats", headers=_auth(customer))
    assert r.status_code == 200
    body = r.json()
    model = BankStatsResponse.model_validate(body)  # strict both ways
    assert model.crystal_count >= 1
    assert model.fact_count >= 1


async def test_crystal_list_schema_is_the_wire_shape(
    sdk_app, store, customer, semantic_encoder_stub, vector_store,
):
    await _seed_pair(
        store, customer, semantic_encoder_stub, vector_store,
        "Honesty|List", "list value",
    )
    async with _client(sdk_app) as client:
        for path in ("/v1/crystals", "/v1/crystals-list"):
            r = await client.get(path, headers=_auth(customer))
            assert r.status_code == 200
            body = r.json()
            model = CrystalListResponse.model_validate(body)
            # The previously-undeclared envelope keys are now schema truth.
            assert "offset" in body and "limit" in body
            assert model.total >= 1
            assert model.crystals[0].id


async def test_crystal_detail_schema_is_the_wire_shape(
    sdk_app, store, customer, semantic_encoder_stub, vector_store,
):
    crystal, fact = await _seed_pair(
        store, customer, semantic_encoder_stub, vector_store,
        "Honesty|Detail", "detail value",
    )
    async with _client(sdk_app) as client:
        r = await client.get(
            f"/v1/crystals/{crystal.id}", headers=_auth(customer),
        )
    assert r.status_code == 200
    body = r.json()
    model = CrystalDetailResponse.model_validate(body)
    # The wire has ALWAYS been nested; the schema now says so.
    assert "crystal" in body
    assert model.crystal.id == crystal.id
    assert model.facts and model.facts[0].id == fact.id


async def test_query_logs_schema_is_the_wire_shape(sdk_app, store, customer):
    from crystal_cache.models import QueryLog

    await store.write_query_log(QueryLog(
        id="ql_honesty_1",
        customer_id=customer.id,
        query_text="what is honesty",
        match_type="none",
        injection_method="none",
        latency_ms=12,
        sequence_id="seq_h1",
        turn_index=0,
        prompt_tokens=10,
        completion_tokens=5,
    ))
    async with _client(sdk_app) as client:
        r = await client.get("/v1/query_logs", headers=_auth(customer))
    assert r.status_code == 200
    body = r.json()
    model = QueryLogResponse.model_validate(body)
    assert model.total >= 1
    item = body["query_logs"][0]
    # The fields the old model omitted are wire truth...
    for key in ("latency_ms", "sequence_id", "turn_index",
                "cache_read_tokens", "cache_creation_tokens"):
        assert key in item
    # ...and the fields it invented are gone from schema AND wire.
    assert "top_score" not in item
    assert "cache_hit" not in item


async def test_openapi_generates_from_the_pinned_schemas(sdk_app):
    """The concrete 1.3 harm was 'blocks any GPT Action built from
    /openapi.json'. Generation itself must succeed with the rewritten
    models, and the retrieve/stats/detail schemas must be present."""
    spec = sdk_app.openapi()
    schemas = spec["components"]["schemas"]
    for name in (
        "RetrieveResponse", "BankStatsResponse",
        "CrystalDetailResponse", "CrystalListResponse", "QueryLogResponse",
        "MatchedCrystal",
    ):
        assert name in schemas
    # The detail schema documents the nested reality.
    assert "crystal" in schemas["CrystalDetailResponse"]["properties"]
    # And the differentiator payload is documented (Phase 1.2, Q3=B).
    assert "matched_crystals" in schemas["RetrieveResponse"]["properties"]


# ---------------------------------------------------------------------------
# Phase 1.2 (Q3=B) — the differentiator payload on /v1/retrieve
# ---------------------------------------------------------------------------

async def test_matched_crystals_tier_conflict_citation(
    sdk_app, store, customer, semantic_encoder_stub, vector_store,
):
    """End-to-end through the REAL pipeline: a matched crystal arrives
    with its tier, its open conflict's other side, and the citation
    manifest entry — structured, not buried in injection prose."""
    crystal, fact = await _seed_pair(
        store, customer, semantic_encoder_stub, vector_store,
        "Diff|Alpha", "alpha secret content",
    )
    # Steer routing: give the crystal a routing vector that exactly
    # matches the query's encoding (PERFECT decision) and a summary_text
    # so the legacy (non-bind) injection path has content to inject —
    # citation manifests only exist when an injection was built.
    crystal.routing_vector = semantic_encoder_stub.encode(
        "alpha secret content"
    ).tolist()
    crystal.summary_text = "alpha secret content"
    await store.upsert_crystal(crystal)
    vector_store.invalidate(customer.id)  # the spawn's bond search cached

    await store.set_crystal_quality_tier(crystal.id, customer.id, "quarantine")
    await store.create_knowledge_conflict(
        customer.id,
        fact_a_id=fact.id,
        fact_b_id="f_counterpart",
        claim_a="alpha secret content",
        claim_b="the counterpart claim text",
        pair_key="pk_diff_1",
        crystal_a_id=crystal.id,
        crystal_b_id=None,
        subject="Alpha",
    )

    async with _client(sdk_app) as client:
        r = await client.post(
            "/v1/retrieve",
            json={"query": "alpha secret content"},
            headers=_auth(customer),
        )
    assert r.status_code == 200
    body = r.json()
    model = RetrieveResponse.model_validate(body)  # strict

    assert model.matched_crystal_ids[0] == crystal.id
    # One record per matched id, same order.
    assert [m.id for m in model.matched_crystals] == model.matched_crystal_ids

    entry = model.matched_crystals[0]
    assert entry.tier == "quarantine"
    assert entry.is_assumption is False
    assert entry.conflicts, "the open conflict must surface"
    assert entry.conflicts[0].counterpart_claim == "the counterpart claim text"
    assert entry.conflicts[0].subject == "Alpha"
    # Citations default ON for this endpoint: the primary injected
    # crystal carries its manifest entry.
    assert entry.citation is not None
    assert entry.citation.handle == "1"
    assert entry.citation.crystal_id == crystal.id
    assert entry.citation.origin == "model_reasoning"


async def test_matched_crystals_assumption_flag(
    store, customer, semantic_encoder_stub, vector_store,
):
    """Builder-grain against the real store: an assumption crystal in the
    matched set is flagged, with its birth tier (quarantine) riding the
    same tier read; an ordinary crystal in the same set is not.
    (Builder-grain because routing an assumption end-to-end requires the
    promotion machinery — assumptions are born recall-gated and live in
    their own crystal_type; the flag mapping is what THIS slice adds.)"""
    from crystal_cache.endpoints.sdk import _matched_crystal_payload

    parent_a, _ = await _seed_pair(
        store, customer, semantic_encoder_stub, vector_store,
        "Parent|A", "parent a content",
    )
    parent_b, _ = await _seed_pair(
        store, customer, semantic_encoder_stub, vector_store,
        "Parent|B", "parent b content",
    )
    created = await store.create_assumption_crystal(
        customer.id,
        statement="a bridging inference between A and B",
        subject="Bridge",
        parent_a_id=parent_a.id,
        parent_b_id=parent_b.id,
        confidence=0.7,
        encoder=semantic_encoder_stub,
    )
    outcome = SimpleNamespace(
        matched_crystal_ids=[created["crystal_id"], parent_a.id],
        citation_manifest=None,
    )
    payload = await _matched_crystal_payload(store, customer.id, outcome)
    assert payload[0].id == created["crystal_id"]
    assert payload[0].is_assumption is True
    assert payload[0].tier == "quarantine"  # the ratified birth tier
    assert payload[1].id == parent_a.id
    assert payload[1].is_assumption is False
