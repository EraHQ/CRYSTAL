"""Contract tests for the VectorIndex seam (Step 1 — 768 lane first).

These prove `InMemoryVectorIndex` is a faithful adapter: it forwards each call
verbatim to the wrapped store and returns the store's result unchanged, so
introducing the seam changes no retrieval behavior. End-to-end parity on real
banks is then covered by the existing retrieval tests once the call sites cut
over (Step 1b). Fakes keep this encoder-free and DB-free (runs in .venv-clean).

asyncio_mode=auto (pyproject) — async tests need no marker.
"""
from __future__ import annotations

import numpy as np

from crystal_cache.infrastructure.vector_index import (
    InMemoryVectorIndex,
    VectorIndex,
)


class _FakeFactStore:
    """Stands in for FactVectorStore; records the last call, returns a sentinel."""

    def __init__(self, result):
        self.result = result
        self.last = None

    async def search(self, customer_id, query_vector, *, pair_types=None, k=10,
                     operator=None, with_keys=False):
        self.last = dict(customer_id=customer_id, query_vector=query_vector,
                         pair_types=pair_types, k=k, operator=operator,
                         with_keys=with_keys)
        return self.result


class _FakeVectorStore:
    """Stands in for VectorStore (routing lane)."""

    def __init__(self, result):
        self.result = result
        self.last = None

    async def search(self, customer_id, query_vector, k=5, *, crystal_type,
                     general_crystal_types=None, operator=None):
        self.last = dict(customer_id=customer_id, query_vector=query_vector, k=k,
                         crystal_type=crystal_type,
                         general_crystal_types=general_crystal_types,
                         operator=operator)
        return self.result


class _Crystal:
    def __init__(self, customer_id, summary_vector):
        self.customer_id = customer_id
        self.summary_vector = summary_vector


class _FakeMeta:
    def __init__(self, crystals):
        self._crystals = crystals

    async def get_crystal(self, crystal_id):
        return self._crystals.get(crystal_id)


def _index(*, facts=None, routing=None, crystals=None):
    fs = _FakeFactStore(facts if facts is not None else [])
    vs = _FakeVectorStore(routing if routing is not None else [])
    meta = _FakeMeta(crystals or {})
    return InMemoryVectorIndex(fact_store=fs, vector_store=vs, metadata_store=meta), fs, vs, meta


def test_satisfies_protocol():
    idx, *_ = _index()
    assert isinstance(idx, VectorIndex)


async def test_search_facts_forwards_and_returns_verbatim():
    sentinel = [("f1", "c1", "content_chunk", 0.91),
                ("f2", "c2", "entity_attribute", 0.4)]
    idx, fs, _, _ = _index(facts=sentinel)
    qv = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    op = object()  # opaque Operator stand-in

    out = await idx.search_facts(
        customer_id="cus_x", query_vector=qv,
        pair_types=["content_chunk"], k=7, operator=op, with_keys=True,
    )

    assert out is sentinel  # returned unchanged
    assert fs.last["customer_id"] == "cus_x"
    assert fs.last["query_vector"] is qv  # not copied/mutated
    assert fs.last["pair_types"] == ["content_chunk"]
    assert fs.last["k"] == 7
    assert fs.last["operator"] is op
    assert fs.last["with_keys"] is True


async def test_search_facts_defaults_match_store_defaults():
    idx, fs, _, _ = _index(facts=[])
    await idx.search_facts(customer_id="cus_x", query_vector=np.zeros(3, dtype=np.float32))
    assert fs.last["pair_types"] is None
    assert fs.last["k"] == 10          # FactVectorStore.search default
    assert fs.last["operator"] is None
    assert fs.last["with_keys"] is False


async def test_search_routing_forwards_and_returns_verbatim():
    sentinel = [("c1", 0.88), ("c2", 0.55)]
    idx, _, vs, _ = _index(routing=sentinel)
    qv = np.array([0.5, 0.5], dtype=np.float32)

    out = await idx.search_routing(
        customer_id="cus_y", query_vector=qv, k=3,
        crystal_type="general:legacy", general_crystal_types=["g1", "g2"],
    )

    assert out is sentinel
    assert vs.last["customer_id"] == "cus_y"
    assert vs.last["query_vector"] is qv
    assert vs.last["k"] == 3
    assert vs.last["crystal_type"] == "general:legacy"
    assert vs.last["general_crystal_types"] == ["g1", "g2"]
    assert vs.last["operator"] is None


async def test_get_summary_vector_returns_copy_for_same_customer():
    vec = [0.1, 0.2, 0.3, 0.4]
    idx, _, _, _ = _index(crystals={"c1": _Crystal("cus_x", vec)})
    out = await idx.get_summary_vector(customer_id="cus_x", crystal_id="c1")
    assert out == vec
    assert out is not vec  # a copy, so callers can't mutate the cached row


async def test_get_summary_vector_general_crystal_passes():
    vec = [1.0, 2.0]
    idx, _, _, _ = _index(crystals={"g": _Crystal(None, vec)})  # customer_id None = general
    out = await idx.get_summary_vector(customer_id="cus_x", crystal_id="g")
    assert out == vec


async def test_get_summary_vector_none_cases():
    idx, _, _, _ = _index(crystals={
        "empty": _Crystal("cus_x", []),
        "other": _Crystal("cus_other", [0.1, 0.2]),
    })
    # missing crystal
    assert await idx.get_summary_vector(customer_id="cus_x", crystal_id="nope") is None
    # present but empty summary_vector
    assert await idx.get_summary_vector(customer_id="cus_x", crystal_id="empty") is None
    # cross-tenant crystal is refused
    assert await idx.get_summary_vector(customer_id="cus_x", crystal_id="other") is None
