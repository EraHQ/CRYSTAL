"""Parity: QdrantVectorIndex.search_facts reproduces FactVectorStore.search.

The fact_768 lane's first Qdrant slice (Step 2). Given the SAME facts in both
backends, the Qdrant-backed index must return identical results to the
in-memory matrix store across pair_type filters, the general-bank merge (with
GENERAL_TIE_BREAK), with_keys passthrough, and k-truncation. Runs on embedded
Qdrant (qdrant-client local mode — no service), so it is self-contained in the
clean dev venv.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import AsyncQdrantClient  # noqa: E402

from crystal_cache.infrastructure.fact_vector_store import FactVectorStore  # noqa: E402
from crystal_cache.infrastructure.qdrant_vector_index import QdrantVectorIndex  # noqa: E402

_DIM = 16


class _FakeFact:
    def __init__(self, id, crystal_id, pair_type, prompt_text, vector):
        self.id = id
        self.crystal_id = crystal_id
        self.pair_type = pair_type
        self.prompt_text = prompt_text
        self.vector = vector


class _FakeStore:
    """Minimal MetadataStore read-surface both backends consume. operator=None
    paths never touch get_crystal / list_acls_for_crystal."""

    def __init__(self, customer_facts, general_facts, subs):
        self._c = customer_facts
        self._g = general_facts
        self._s = subs

    async def list_all_facts_for_customer(self, customer_id):
        return self._c.get(customer_id, [])

    async def list_all_facts_general(self, crystal_type):
        return self._g.get(crystal_type, [])

    async def get_customer_general_types(self, customer_id):
        return self._s.get(customer_id, [])

    async def get_crystal(self, crystal_id):
        return None

    async def list_acls_for_crystal(self, crystal_id):
        return []


def _facts(seed, n, prefix, crystals=2):
    rng = np.random.default_rng(seed)
    pair_types = ["content_chunk", "question_answer"]
    return [
        _FakeFact(
            f"{prefix}{i}", f"{prefix}cr{i % crystals}",
            pair_types[i % len(pair_types)], f"{prefix} prompt {i}",
            rng.standard_normal(_DIM).tolist(),
        )
        for i in range(n)
    ]


def _assert_parity(expected, got, tol=1e-5):
    assert len(expected) == len(got), f"length {len(expected)} != {len(got)}"
    for i, (x, y) in enumerate(zip(expected, got)):
        assert x[:3] == y[:3], f"row {i} key {x[:3]} != {y[:3]}"
        assert abs(x[3] - y[3]) <= tol, f"row {i} score {x[3]} != {y[3]}"
        assert (len(x) > 4) == (len(y) > 4), f"row {i} arity mismatch"
        if len(x) > 4:
            assert x[4] == y[4], f"row {i} prompt_text {x[4]!r} != {y[4]!r}"


_CUST = _facts(7, 6, "f")
_GEN = _facts(8, 4, "g")
_QUERIES = [np.random.default_rng(100 + i).standard_normal(_DIM) for i in range(3)]


@pytest.fixture
def client():
    return AsyncQdrantClient(location=":memory:")


@pytest.mark.parametrize("subscribed", [False, True])
@pytest.mark.parametrize(
    "pair_types", [None, ["content_chunk"], ["question_answer", "content_chunk"]]
)
@pytest.mark.parametrize("with_keys", [False, True])
@pytest.mark.parametrize("k", [10, 3])
async def test_search_facts_matches_fact_vector_store(
    client, subscribed, pair_types, with_keys, k
):
    subs = {"cusA": ["python"]} if subscribed else {}
    store = _FakeStore({"cusA": _CUST}, {"python": _GEN}, subs)
    fvs = FactVectorStore(store)
    qvi = QdrantVectorIndex(client=client, metadata_store=store)
    for q in _QUERIES:
        expected = await fvs.search(
            "cusA", q, pair_types=pair_types, k=k, with_keys=with_keys
        )
        got = await qvi.search_facts(
            customer_id="cusA", query_vector=q,
            pair_types=pair_types, k=k, with_keys=with_keys,
        )
        _assert_parity(expected, got)


async def test_invalidate_reloads_from_db(client):
    store = _FakeStore({"cusA": _CUST}, {}, {})
    qvi = QdrantVectorIndex(client=client, metadata_store=store)
    q = _QUERIES[0]
    before = await qvi.search_facts(customer_id="cusA", query_vector=q, k=10)
    assert len(before) == 6
    store._c["cusA"] = _CUST[:3]  # facts change in the DB (source of truth)
    qvi.invalidate("cusA")
    after = await qvi.search_facts(customer_id="cusA", query_vector=q, k=10)
    assert len(after) == 3


async def test_empty_customer_no_subs_returns_empty(client):
    store = _FakeStore({}, {}, {})
    qvi = QdrantVectorIndex(client=client, metadata_store=store)
    got = await qvi.search_facts(customer_id="cusA", query_vector=_QUERIES[0], k=10)
    assert got == []


def test_build_vector_index_selects_backend():
    """The factory returns the configured backend (Step 2 2a-iv). Stores are
    only stashed at construction, so None is fine for a type-selection check."""
    from crystal_cache.infrastructure.vector_index import (
        InMemoryVectorIndex,
        build_vector_index,
    )

    mem = build_vector_index(
        backend="memory", fact_store=None, vector_store=None, metadata_store=None,
    )
    assert isinstance(mem, InMemoryVectorIndex)

    qd = build_vector_index(
        backend="qdrant", fact_store=None, vector_store=None,
        metadata_store=None, qdrant_location=":memory:",
    )
    assert isinstance(qd, QdrantVectorIndex)

    with pytest.raises(ValueError):
        build_vector_index(
            backend="bogus", fact_store=None, vector_store=None, metadata_store=None,
        )
