"""Parity: QdrantVectorIndex.search_routing reproduces VectorStore.search.

The routing_10k lane's Qdrant slice (Step 2b). Given the SAME crystals in both
backends, the Qdrant-backed routing index (a binary-quantized collection + a
float rescore) must return identical (crystal_id, cosine) results to the
in-memory VectorStore across crystal_type scoping, the general-bank merge
(RAW cosine, customer-precedence by crystal_id — NO GENERAL_TIE_BREAK factor,
unlike the fact lane), and k-truncation.

At test scale (a handful of crystals, well under Qdrant's ~20k indexing
threshold) the engine does an exact brute-force scan with a float rescore, so
parity is deterministic; the binary-quant approximation only engages at scale,
which the routing-quantization + live-latency benchmarks cover (recall@1 =
1.000 at 10k). Runs on embedded Qdrant (qdrant-client local mode — no service),
so it is self-contained in the clean dev venv.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import AsyncQdrantClient  # noqa: E402

from crystal_cache.infrastructure.qdrant_vector_index import QdrantVectorIndex  # noqa: E402
from crystal_cache.infrastructure.vector_store import VectorStore  # noqa: E402

_DIM = 64


class _FakeCrystal:
    def __init__(self, id, customer_id, crystal_type, routing_vector):
        self.id = id
        self.customer_id = customer_id
        self.crystal_type = crystal_type
        self.routing_vector = routing_vector
        self.summary_vector = []  # routing search never reads this


class _FakeStore:
    """Minimal MetadataStore read-surface both backends consume.

    VectorStore loads the type-scoped subset via list_crystals_for_customer_
    and_type; the Qdrant index loads the whole customer via list_crystals_for_
    customer and filters by crystal_type in the query — both must agree.
    operator=None paths never touch get_crystal / list_acls_for_crystal.
    """

    def __init__(self, customer_crystals, general_crystals):
        self._c = customer_crystals  # customer_id -> [crystals]
        self._g = general_crystals   # crystal_type -> [crystals]

    async def list_crystals_for_customer(
        self, customer_id, *, include_recall_gated=True
    ):
        rows = self._c.get(customer_id, [])
        if not include_recall_gated:
            rows = [c for c in rows if not getattr(c, "recall_gated", False)]
        return rows

    async def list_crystals_for_customer_and_type(
        self, customer_id, crystal_type, *, include_recall_gated=True
    ):
        rows = [
            c for c in self._c.get(customer_id, [])
            if c.crystal_type == crystal_type
        ]
        if not include_recall_gated:
            rows = [c for c in rows if not getattr(c, "recall_gated", False)]
        return rows

    async def list_general_crystals(
        self, crystal_type, *, include_recall_gated=True
    ):
        rows = self._g.get(crystal_type, [])
        if not include_recall_gated:
            rows = [c for c in rows if not getattr(c, "recall_gated", False)]
        return rows

    async def get_crystal(self, crystal_id):
        return None

    async def list_acls_for_crystal(self, crystal_id):
        return []


def _crystals(seed, n, prefix, customer_id, types):
    rng = np.random.default_rng(seed)
    return [
        _FakeCrystal(
            f"{prefix}{i}", customer_id, types[i % len(types)],
            rng.standard_normal(_DIM).tolist(),
        )
        for i in range(n)
    ]


def _assert_parity(expected, got, tol=1e-5):
    assert len(expected) == len(got), f"length {len(expected)} != {len(got)}"
    for i, (x, y) in enumerate(zip(expected, got)):
        assert x[0] == y[0], f"row {i} crystal_id {x[0]} != {y[0]}"
        assert abs(x[1] - y[1]) <= tol, f"row {i} score {x[1]} != {y[1]}"


_TYPES = ["customer:legacy", "customer:medical"]
_CUST = _crystals(7, 8, "cr", "cusA", _TYPES)
_GEN = _crystals(8, 5, "gen", None, ["general:python"])
_QUERIES = [np.random.default_rng(200 + i).standard_normal(_DIM) for i in range(3)]


@pytest.fixture
def client():
    return AsyncQdrantClient(location=":memory:")


@pytest.mark.parametrize("crystal_type", _TYPES)
@pytest.mark.parametrize("general_crystal_types", [None, ["general:python"]])
@pytest.mark.parametrize("k", [5, 2])
async def test_search_routing_matches_vector_store(
    client, crystal_type, general_crystal_types, k
):
    store = _FakeStore({"cusA": _CUST}, {"general:python": _GEN})
    vs = VectorStore(store)
    qvi = QdrantVectorIndex(client=client, metadata_store=store)
    for q in _QUERIES:
        expected = await vs.search(
            "cusA", q, k,
            crystal_type=crystal_type,
            general_crystal_types=general_crystal_types,
        )
        got = await qvi.search_routing(
            customer_id="cusA", query_vector=q, k=k,
            crystal_type=crystal_type,
            general_crystal_types=general_crystal_types,
        )
        _assert_parity(expected, got)


async def test_routing_invalidate_reloads_from_db(client):
    store = _FakeStore({"cusA": _CUST}, {})
    qvi = QdrantVectorIndex(client=client, metadata_store=store)
    q = _QUERIES[0]
    legacy = [c for c in _CUST if c.crystal_type == "customer:legacy"]
    before = await qvi.search_routing(
        customer_id="cusA", query_vector=q, k=10, crystal_type="customer:legacy"
    )
    assert len(before) == len(legacy)
    store._c["cusA"] = _CUST[:2]  # crystals change in the DB (source of truth)
    qvi.invalidate("cusA")
    after = await qvi.search_routing(
        customer_id="cusA", query_vector=q, k=10, crystal_type="customer:legacy"
    )
    expected_after = [c for c in _CUST[:2] if c.crystal_type == "customer:legacy"]
    assert len(after) == len(expected_after)


async def test_routing_general_only_when_subscribed(client):
    """general_crystal_types is an explicit arg: omitting it must drop the
    general bank from the merge entirely (mirrors VectorStore.search)."""
    store = _FakeStore({"cusA": _CUST}, {"general:python": _GEN})
    qvi = QdrantVectorIndex(client=client, metadata_store=store)
    q = _QUERIES[0]
    without = await qvi.search_routing(
        customer_id="cusA", query_vector=q, k=50, crystal_type="customer:legacy"
    )
    with_gen = await qvi.search_routing(
        customer_id="cusA", query_vector=q, k=50, crystal_type="customer:legacy",
        general_crystal_types=["general:python"],
    )
    assert all(not cid.startswith("gen") for cid, _ in without)
    assert any(cid.startswith("gen") for cid, _ in with_gen)
    assert len(with_gen) == len(without) + len(_GEN)


async def test_routing_empty_customer_returns_empty(client):
    store = _FakeStore({}, {})
    qvi = QdrantVectorIndex(client=client, metadata_store=store)
    got = await qvi.search_routing(
        customer_id="cusA", query_vector=_QUERIES[0], k=5,
        crystal_type="customer:legacy",
    )
    assert got == []
