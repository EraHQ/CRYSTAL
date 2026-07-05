"""Parity: SqliteVecIndex reproduces FactVectorStore.search / VectorStore.search.

The sqlite-vec self-host backend (Step 2c). Given the SAME crystals + facts,
the sqlite-vec-backed index must return identical results to the in-memory
stores across pair_type filters, the general-bank merge (fact lane:
GENERAL_TIE_BREAK; routing lane: RAW cosine + customer-precedence), with_keys
passthrough, k-truncation, and lazy rebuild on invalidate.

WHY A REAL FILE-BACKED STORE (unlike the qdrant parity tests' _FakeStore)
-------------------------------------------------------------------------
The qdrant backend mirrors from the Python loaders, so its tests feed a fake
store of Python lists. The sqlite-vec FACT lane instead rebuilds its vec0 index
with an in-SQLite INSERT...SELECT over the real `facts`/`crystals` tables, and
the ROUTING lane scans `crystals` live via vec_distance_cosine — both read
actual SQL, so the test seeds a real MetadataStore. It is a temp-FILE DB (not
the conftest `:memory:` store) because the sqlite-vec extension is loaded on
each pooled connection in aiosqlite's worker thread, which a single-connection
in-memory StaticPool doesn't exercise the same way; a file DB matches a real
self-host deployment and lets the vec0 shadow tables persist across connections.

TOLERANCE
---------
Scores are compared with np.isclose(rtol=1e-3, atol=1e-5) rather than the
qdrant tests' flat 1e-5: the routing lane is 10000-d float32, whose
reduction-order differs between NumPy's BLAS dot and sqlite-vec's scalar
vec_distance_cosine. An in-container harness measured the actual gap at
abs<=2e-6 / rel<=2e-6 (fact lane abs<=6e-8), so this bound has ~3 orders of
margin while still catching a missing query-norm (~100x off) or the 0.995
general weight (0.5% off, > rtol).
"""
from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy import delete, update

pytest.importorskip("sqlite_vec")

from crystal_cache.config import Settings  # noqa: E402
from crystal_cache.infrastructure import MetadataStore, VectorStore  # noqa: E402
from crystal_cache.infrastructure.fact_vector_store import FactVectorStore  # noqa: E402
from crystal_cache.infrastructure.schema import FactRow  # noqa: E402
from crystal_cache.infrastructure.sqlite_vec_index import SqliteVecIndex  # noqa: E402
from crystal_cache.models import Crystal  # noqa: E402

FACT_DIM = 768
ROUTING_DIM = 10_000

# Customer banks: 2 of customer:legacy, 1 of customer:medical. General bank:
# 2 of general:python. crystal_id prefixes ('crL'/'crM'/'gp') let the
# subscription tests assert which bank a result came from.
_CUST_CRYSTALS = [
    ("crL0", "customer:legacy"),
    ("crL1", "customer:legacy"),
    ("crM0", "customer:medical"),
]
_GEN_CRYSTALS = [
    ("gp0", "general:python"),
    ("gp1", "general:python"),
]
_PAIR_TYPES = ["content_chunk", "question_answer"]

# Queries are random and distinct from the stored vectors (separate seeds).
_QUERIES_F = [np.random.default_rng(300 + i).standard_normal(FACT_DIM) for i in range(3)]
_QUERIES_R = [np.random.default_rng(400 + i).standard_normal(ROUTING_DIM) for i in range(3)]


async def _seed(store: MetadataStore, customer_id: str) -> None:
    rng = np.random.default_rng(12345)

    def rvec() -> list[float]:
        return rng.standard_normal(ROUTING_DIM).astype(np.float32).tolist()

    def fvec() -> list[float]:
        return rng.standard_normal(FACT_DIM).astype(np.float32).tolist()

    all_crystals = (
        [(cid, customer_id, ct) for cid, ct in _CUST_CRYSTALS]
        + [(cid, None, ct) for cid, ct in _GEN_CRYSTALS]
    )
    for cid, owner, ctype in all_crystals:
        await store.upsert_crystal(Crystal(
            id=cid,
            customer_id=owner,
            crystal_type=ctype,
            summary_vector=rvec(),
            routing_vector=rvec(),
        ))

    # Two facts per crystal, alternating pair_type.
    async with store.session() as session:
        i = 0
        for cid, _owner, _ctype in all_crystals:
            for _ in range(2):
                session.add(FactRow(
                    id=f"f{i}",
                    crystal_id=cid,
                    claim_text=f"claim {i}",
                    pair_type=_PAIR_TYPES[i % 2],
                    prompt_text=f"prompt {i}",
                    vector=fvec(),
                ))
                i += 1


@pytest_asyncio.fixture
async def seeded(tmp_path):
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path.as_posix()}/parity.db"
    )
    store = MetadataStore(settings_override=settings)
    await store.init()
    await store._seed_legacy_crystal_types_for_tests()
    customer = await store.create_customer(
        provider="anthropic", model_id="claude-x", api_key_ref="ref"
    )
    await _seed(store, customer.id)
    try:
        yield store, customer.id
    finally:
        await store.dispose()


def _assert_parity_facts(expected, got, rtol=1e-3, atol=1e-5):
    assert len(expected) == len(got), f"length {len(expected)} != {len(got)}"
    for i, (x, y) in enumerate(zip(expected, got)):
        assert x[:3] == y[:3], f"row {i} key {x[:3]} != {y[:3]}"
        assert np.isclose(x[3], y[3], rtol=rtol, atol=atol), (
            f"row {i} score {x[3]} != {y[3]}"
        )
        assert (len(x) > 4) == (len(y) > 4), f"row {i} arity mismatch"
        if len(x) > 4:
            assert x[4] == y[4], f"row {i} prompt_text {x[4]!r} != {y[4]!r}"


def _assert_parity_routing(expected, got, rtol=1e-3, atol=1e-5):
    assert len(expected) == len(got), f"length {len(expected)} != {len(got)}"
    for i, (x, y) in enumerate(zip(expected, got)):
        assert x[0] == y[0], f"row {i} crystal_id {x[0]} != {y[0]}"
        assert np.isclose(x[1], y[1], rtol=rtol, atol=atol), (
            f"row {i} score {x[1]} != {y[1]}"
        )


# --- fact lane -------------------------------------------------------------

@pytest.mark.parametrize("subscribed", [False, True])
@pytest.mark.parametrize(
    "pair_types", [None, ["content_chunk"], ["question_answer", "content_chunk"]]
)
@pytest.mark.parametrize("with_keys", [False, True])
@pytest.mark.parametrize("k", [10, 3])
async def test_search_facts_matches_fact_vector_store(
    seeded, subscribed, pair_types, with_keys, k
):
    store, cid = seeded
    await store.set_customer_general_types(
        cid, ["general:python"] if subscribed else []
    )
    fvs = FactVectorStore(store=store)
    svec = SqliteVecIndex(metadata_store=store)
    for q in _QUERIES_F:
        expected = await fvs.search(
            cid, q, pair_types=pair_types, k=k, with_keys=with_keys
        )
        got = await svec.search_facts(
            customer_id=cid, query_vector=q,
            pair_types=pair_types, k=k, with_keys=with_keys,
        )
        _assert_parity_facts(expected, got)


async def test_facts_invalidate_reloads_from_db(seeded):
    store, cid = seeded
    await store.set_customer_general_types(cid, [])
    fvs = FactVectorStore(store=store)
    svec = SqliteVecIndex(metadata_store=store)
    q = _QUERIES_F[0]
    before_e = await fvs.search(cid, q, k=50)
    before_g = await svec.search_facts(customer_id=cid, query_vector=q, k=50)
    _assert_parity_facts(before_e, before_g)
    n_before = len(before_g)

    # Facts change in the DB (the source of truth): drop one crystal's facts.
    async with store.session() as session:
        await session.execute(delete(FactRow).where(FactRow.crystal_id == "crL0"))

    fvs.invalidate(cid)
    svec.invalidate(cid)
    after_e = await fvs.search(cid, q, k=50)
    after_g = await svec.search_facts(customer_id=cid, query_vector=q, k=50)
    _assert_parity_facts(after_e, after_g)
    assert len(after_g) < n_before


# --- routing lane ----------------------------------------------------------

@pytest.mark.parametrize("crystal_type", ["customer:legacy", "customer:medical"])
@pytest.mark.parametrize("general_crystal_types", [None, ["general:python"]])
@pytest.mark.parametrize("k", [5, 2])
async def test_search_routing_matches_vector_store(
    seeded, crystal_type, general_crystal_types, k
):
    store, cid = seeded
    vs = VectorStore(store=store)
    svec = SqliteVecIndex(metadata_store=store)
    for q in _QUERIES_R:
        expected = await vs.search(
            cid, q, k,
            crystal_type=crystal_type,
            general_crystal_types=general_crystal_types,
        )
        got = await svec.search_routing(
            customer_id=cid, query_vector=q, k=k,
            crystal_type=crystal_type,
            general_crystal_types=general_crystal_types,
        )
        _assert_parity_routing(expected, got)


async def test_routing_invalidate_reloads_from_db(seeded):
    store, cid = seeded
    vs = VectorStore(store=store)
    svec = SqliteVecIndex(metadata_store=store)
    q = _QUERIES_R[0]
    before_e = await vs.search(cid, q, 50, crystal_type="customer:legacy")
    before_g = await svec.search_routing(
        customer_id=cid, query_vector=q, k=50, crystal_type="customer:legacy"
    )
    _assert_parity_routing(before_e, before_g)
    n_before = len(before_g)

    # A crystal's routing_vector is cleared in the DB → it leaves the bank.
    await _set_routing_vector_none(store, "crL0")

    vs.invalidate(cid)
    svec.invalidate(cid)
    after_e = await vs.search(cid, q, 50, crystal_type="customer:legacy")
    after_g = await svec.search_routing(
        customer_id=cid, query_vector=q, k=50, crystal_type="customer:legacy"
    )
    _assert_parity_routing(after_e, after_g)
    assert len(after_g) < n_before


async def test_routing_general_only_when_subscribed(seeded):
    store, cid = seeded
    svec = SqliteVecIndex(metadata_store=store)
    q = _QUERIES_R[0]
    without = await svec.search_routing(
        customer_id=cid, query_vector=q, k=50, crystal_type="customer:legacy"
    )
    with_gen = await svec.search_routing(
        customer_id=cid, query_vector=q, k=50, crystal_type="customer:legacy",
        general_crystal_types=["general:python"],
    )
    assert all(not c.startswith("gp") for c, _ in without)
    assert any(c.startswith("gp") for c, _ in with_gen)
    # customer:legacy = crL0, crL1 (2); general:python = gp0, gp1 (2)
    assert len(with_gen) == len(without) + len(_GEN_CRYSTALS)


# --- shared edge cases + factory -------------------------------------------

async def test_empty_customer_returns_empty(seeded):
    store, _ = seeded
    svec = SqliteVecIndex(metadata_store=store)
    assert await svec.search_facts(
        customer_id="cus_nobody", query_vector=_QUERIES_F[0], k=10
    ) == []
    assert await svec.search_routing(
        customer_id="cus_nobody", query_vector=_QUERIES_R[0], k=5,
        crystal_type="customer:legacy",
    ) == []


def test_build_vector_index_selects_sqlite_vec_and_guards_postgres():
    """sqlite_vec backend is selected for a SQLite store and refused for a
    non-SQLite (Postgres) one. Construction stashes only the store, so a
    lightweight namespace with the right dialect name is enough."""
    import types

    from crystal_cache.infrastructure.sqlite_vec_index import SqliteVecIndex
    from crystal_cache.infrastructure.vector_index import build_vector_index

    sqlite_store = types.SimpleNamespace(
        engine=types.SimpleNamespace(dialect=types.SimpleNamespace(name="sqlite"))
    )
    idx = build_vector_index(
        backend="sqlite_vec", fact_store=None, vector_store=None,
        metadata_store=sqlite_store,
    )
    assert isinstance(idx, SqliteVecIndex)

    pg_store = types.SimpleNamespace(
        engine=types.SimpleNamespace(dialect=types.SimpleNamespace(name="postgresql"))
    )
    with pytest.raises(ValueError):
        build_vector_index(
            backend="sqlite_vec", fact_store=None, vector_store=None,
            metadata_store=pg_store,
        )


# --- helpers ---------------------------------------------------------------

async def _set_routing_vector_none(store: MetadataStore, crystal_id: str) -> None:
    from crystal_cache.infrastructure.schema import CrystalRow
    async with store.session() as session:
        await session.execute(
            update(CrystalRow)
            .where(CrystalRow.id == crystal_id)
            .values(routing_vector=None)
        )
