"""Phase 1.4 Gate 2 — operator threading through the retrieval surface.

Covers, in layer order:
  1. `readable_facts` (infrastructure/acl_read_filter.py): the None
     passthrough is query-free (Q2=A), and the filter drops a
     teammate's 0o600 facts while keeping team-mode ones.
  2. Router forwarding: Content/Knowledge/Depth routers pass `operator`
     through to `VectorIndex.search_facts` (the backends already
     implement the filter; the routers' job is not to drop the kwarg).
  3. Navigation + key_scan enumeration filtering end-to-end against the
     real store: op_b no longer sees op_a's private key names or
     content previews; the system lane (None) stays unfiltered.
  4. The tool layer reads the request-context operator (Q1=A) and hands
     it to the router — proven by driving `content_search` with the
     contextvar set and asserting the index saw the operator.

asyncio_mode=auto.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from crystal_cache.agent.principal import (
    reset_current_operator,
    set_current_operator,
)
from crystal_cache.infrastructure.acl_read_filter import readable_facts
from crystal_cache.models import Crystal


# ---------------------------------------------------------------------------
# Seed helpers (the test_permissions pattern: explicit POSIX stamps)
# ---------------------------------------------------------------------------

async def _seed(store, encoder, *, crystal_id, customer_id, owner, mode, key, answer):
    await store.upsert_crystal(Crystal(
        id=crystal_id,
        customer_id=customer_id,
        summary_vector=[],
        owner_operator_id=owner,
        group_team_id=customer_id,
        mode=mode,
        crystal_type="customer:legacy",
    ))
    await store.add_pair_to_crystal(
        crystal_id=crystal_id,
        prompt_text=key,
        answer_text=answer,
        encoder=encoder,
    )


# ---------------------------------------------------------------------------
# 1. readable_facts
# ---------------------------------------------------------------------------

async def test_readable_facts_none_operator_is_query_free():
    class _ExplodingStore:
        def __getattr__(self, name):  # pragma: no cover - must not be reached
            raise AssertionError(f"store touched ({name}) on the None path")

    facts = [SimpleNamespace(crystal_id="c1"), SimpleNamespace(crystal_id="c2")]
    out = await readable_facts(_ExplodingStore(), None, facts)
    assert out == facts  # unchanged, and the store was never consulted


async def test_readable_facts_filters_private(store, customer, semantic_encoder_stub):
    op_a, _ = await store.create_operator(team_id=customer.id, display_name="A")
    op_b, _ = await store.create_operator(team_id=customer.id, display_name="B")
    await _seed(
        store, semantic_encoder_stub, crystal_id="crys_rf_priv",
        customer_id=customer.id, owner=op_a.id, mode=0o600,
        key="Secret|Alpha|Detail", answer="alpha secret",
    )
    await _seed(
        store, semantic_encoder_stub, crystal_id="crys_rf_team",
        customer_id=customer.id, owner=op_a.id, mode=0o640,
        key="Shared|Beta|Detail", answer="beta shared",
    )
    facts = await store.list_all_facts_for_customer(customer.id)
    assert {f.crystal_id for f in facts} >= {"crys_rf_priv", "crys_rf_team"}

    as_b = await readable_facts(store, op_b, facts)
    ids_b = {f.crystal_id for f in as_b}
    assert "crys_rf_priv" not in ids_b
    assert "crys_rf_team" in ids_b

    as_a = await readable_facts(store, op_a, facts)
    assert "crys_rf_priv" in {f.crystal_id for f in as_a}


# ---------------------------------------------------------------------------
# 2. Router forwarding — the kwarg reaches the index
# ---------------------------------------------------------------------------

class _CapturingIndex:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def search_facts(self, **kwargs):
        self.calls.append(kwargs)
        return []


_OP = SimpleNamespace(id="op_fwd", role="operator", team_id="cus_x")
_QV = np.asarray([1.0, 0.0], dtype=np.float32)


async def test_content_router_forwards_operator(store):
    from crystal_cache.retrieval.v3_routers import ContentRouter

    idx = _CapturingIndex()
    await ContentRouter(vector_index=idx, metadata_store=store).search(
        customer_id="cus_x", query_vector=_QV, operator=_OP,
    )
    assert idx.calls and all(c["operator"] is _OP for c in idx.calls)


async def test_knowledge_router_forwards_operator(store):
    from crystal_cache.retrieval.v3_routers import KnowledgeRouter

    idx = _CapturingIndex()
    await KnowledgeRouter(vector_index=idx, metadata_store=store).search(
        customer_id="cus_x", query_vector=_QV, operator=_OP,
    )
    assert idx.calls and all(c["operator"] is _OP for c in idx.calls)


async def test_depth_router_forwards_operator_on_all_channels(store):
    from crystal_cache.retrieval.v3_depth import DepthRouter

    idx = _CapturingIndex()
    await DepthRouter(vector_index=idx, metadata_store=store).search(
        customer_id="cus_x", query_vector=_QV, operator=_OP,
    )
    # Three channels: relationship/entity, content_chunk, question_answer.
    assert len(idx.calls) == 3
    assert all(c["operator"] is _OP for c in idx.calls)


# ---------------------------------------------------------------------------
# 3. Enumeration paths end-to-end (navigation + key_scan)
# ---------------------------------------------------------------------------

async def test_navigation_hides_private_keys_from_teammate(
    store, customer, semantic_encoder_stub, fact_vector_store,
):
    from crystal_cache.retrieval.v3_navigation import NavigationRouter

    op_a, _ = await store.create_operator(team_id=customer.id, display_name="A")
    op_b, _ = await store.create_operator(team_id=customer.id, display_name="B")
    await _seed(
        store, semantic_encoder_stub, crystal_id="crys_nav_priv",
        customer_id=customer.id, owner=op_a.id, mode=0o600,
        key="Secret|AlphaProject|Plan", answer="alpha plan",
    )
    await _seed(
        store, semantic_encoder_stub, crystal_id="crys_nav_team",
        customer_id=customer.id, owner=op_a.id, mode=0o640,
        key="Shared|BetaProject|Plan", answer="beta plan",
    )
    router = NavigationRouter(fact_store=fact_vector_store, metadata_store=store)

    # Teammate: the private crystal's key names never enter the overview.
    as_b = await router.search(customer_id=customer.id, operator=op_b)
    assert "AlphaProject" not in (as_b.injection_text or "")
    assert "BetaProject" in (as_b.injection_text or "")

    # System lane (None): unfiltered — today's behavior exactly (Q2=A).
    unfiltered = await router.search(customer_id=customer.id)
    assert "AlphaProject" in (unfiltered.injection_text or "")
    assert "BetaProject" in (unfiltered.injection_text or "")

    # Owner: sees their own private keys.
    as_a = await router.search(customer_id=customer.id, operator=op_a)
    assert "AlphaProject" in (as_a.injection_text or "")


async def test_key_scan_hides_private_facts_from_teammate(
    store, customer, semantic_encoder_stub,
):
    from crystal_cache.agent.tools.retrievers import key_scan, set_tool_state

    op_a, _ = await store.create_operator(team_id=customer.id, display_name="A")
    op_b, _ = await store.create_operator(team_id=customer.id, display_name="B")
    await _seed(
        store, semantic_encoder_stub, crystal_id="crys_ks_priv",
        customer_id=customer.id, owner=op_a.id, mode=0o600,
        key="Vault|SecretThing|Note", answer="the secret content",
    )
    await _seed(
        store, semantic_encoder_stub, crystal_id="crys_ks_team",
        customer_id=customer.id, owner=op_a.id, mode=0o640,
        key="Vault|SharedThing|Note", answer="the shared content",
    )
    set_tool_state({"store": store})

    # Teammate via the request context: private fact dropped.
    token = set_current_operator(op_b)
    try:
        out_b = await key_scan(customer.id, key_prefix="Vault")
    finally:
        reset_current_operator(token)
    assert "crys_ks_priv" not in out_b["matched_crystal_ids"]
    assert "crys_ks_team" in out_b["matched_crystal_ids"]
    assert "SecretThing" not in out_b["content_text"]

    # System lane: unfiltered (Q2=A).
    out_none = await key_scan(customer.id, key_prefix="Vault")
    assert "crys_ks_priv" in out_none["matched_crystal_ids"]

    # Owner: sees their own.
    token = set_current_operator(op_a)
    try:
        out_a = await key_scan(customer.id, key_prefix="Vault")
    finally:
        reset_current_operator(token)
    assert "crys_ks_priv" in out_a["matched_crystal_ids"]


# ---------------------------------------------------------------------------
# 4. Tool layer reads the request context (Q1=A)
# ---------------------------------------------------------------------------

async def test_content_search_tool_passes_context_operator(
    store, semantic_encoder_stub,
):
    from crystal_cache.agent.tools.retrievers import (
        content_search, set_tool_state,
    )

    idx = _CapturingIndex()
    set_tool_state({
        "store": store,
        "vector_index": idx,
        "encoder": semantic_encoder_stub,
    })

    token = set_current_operator(_OP)
    try:
        await content_search("cus_x", query="anything")
    finally:
        reset_current_operator(token)
    assert idx.calls and all(c["operator"] is _OP for c in idx.calls)

    # And without the context: None flows through (system lane).
    idx.calls.clear()
    await content_search("cus_x", query="anything")
    assert idx.calls and all(c["operator"] is None for c in idx.calls)


# ---------------------------------------------------------------------------
# 5. End-to-end: the full real stack under one contextvar (a7, gate 6)
# ---------------------------------------------------------------------------

async def test_end_to_end_agent_tool_operator_scoping(
    store, customer, semantic_encoder_stub, fact_vector_store, vector_store,
):
    """No fakes anywhere: knowledge_search → KnowledgeRouter →
    InMemoryVectorIndex → FactVectorStore's real can_read filter, with the
    acting operator carried ONLY by the request contextvar. The
    door→contextvar linkage is pinned in test_operator_principal; this
    pins contextvar→filtered-results.

    Assertions are marker-token based (`in str(out)`) deliberately: for a
    security property, NOTHING about the private crystal — id or content
    — may appear anywhere in the tool output, regardless of the response
    dict's shape. Each positive control queries the text that maximally
    matches the asserted crystal, so no router score threshold can make
    the assertion flaky."""
    from crystal_cache.agent.tools.retrievers import (
        knowledge_search, set_tool_state,
    )
    from crystal_cache.infrastructure.vector_index import InMemoryVectorIndex

    op_a, _ = await store.create_operator(team_id=customer.id, display_name="A")
    op_b, _ = await store.create_operator(team_id=customer.id, display_name="B")
    await _seed(
        store, semantic_encoder_stub, crystal_id="crys_e2e_priv",
        customer_id=customer.id, owner=op_a.id, mode=0o600,
        key="Secret|E2E|Fact", answer="SECRETMARKER alpha content",
    )
    await _seed(
        store, semantic_encoder_stub, crystal_id="crys_e2e_team",
        customer_id=customer.id, owner=op_a.id, mode=0o640,
        key="Shared|E2E|Fact", answer="SHAREDMARKER beta content",
    )
    set_tool_state({
        "store": store,
        "encoder": semantic_encoder_stub,
        "vector_index": InMemoryVectorIndex(
            fact_store=fact_vector_store,
            vector_store=vector_store,
            metadata_store=store,
        ),
    })

    # Teammate hunting the private fact by its own content: the fact must
    # not surface — not its id, not its text — while the team fact remains
    # reachable in the same context (positive control on its own content).
    token = set_current_operator(op_b)
    try:
        hunt_priv = await knowledge_search(
            customer.id, query="SECRETMARKER alpha content",
        )
        hunt_team = await knowledge_search(
            customer.id, query="SHAREDMARKER beta content",
        )
    finally:
        reset_current_operator(token)
    assert "SECRETMARKER" not in str(hunt_priv)
    assert "crys_e2e_priv" not in str(hunt_priv)
    assert "SHAREDMARKER" in str(hunt_team)

    # Owner: their own private fact comes back through the same stack.
    token = set_current_operator(op_a)
    try:
        as_owner = await knowledge_search(
            customer.id, query="SECRETMARKER alpha content",
        )
    finally:
        reset_current_operator(token)
    assert "SECRETMARKER" in str(as_owner)

    # System lane (no context): unfiltered — the ratified Q2=A contract.
    as_system = await knowledge_search(
        customer.id, query="SECRETMARKER alpha content",
    )
    assert "SECRETMARKER" in str(as_system)
