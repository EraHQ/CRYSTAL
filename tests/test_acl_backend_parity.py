"""Phase 1.4 Gate 4 — the ACL contract on all three vector backends.

The permission filter is deliberately duplicated per backend (the
parity-bar note in qdrant_vector_index.py), and until this file NONE of
that duplication was tested off the in-memory backend — the qdrant
parity fakes state outright that operator paths never touch
`get_crystal`/`list_acls_for_crystal`, which is exactly how S2-87 (the
missing `operator_group_ids`, group grants silently fail-closed on
qdrant only) survived. This suite runs ONE seeded ACL scenario against
each backend's fact AND routing lane:

  crystal      mode    grant                owner  member  plain  None
  cap_grant    0o600   group grant (P3)     ✓      ✓*      ✗      ✓
  cap_priv     0o600   —                    ✓      ✗       ✗      ✓
  cap_team     0o640   —                    ✓      ✓       ✓      ✓

  (* = the S2-87 regression pin: pre-fix, qdrant hid cap_grant from the
     group member on both lanes while memory and sqlite_vec showed it.)

Backends: in-memory (conftest store), Qdrant (:memory: client mirroring
the same conftest store; importorskip qdrant_client), sqlite_vec
(temp-file MetadataStore per the test_sqlite_vec_parity pattern;
importorskip sqlite_vec). Dims match the proven parity values (768 fact
/ 10k routing) so no backend-internal assumption is disturbed.

asyncio_mode=auto.
"""
from __future__ import annotations

import numpy as np
import pytest

from crystal_cache.models import Crystal
from crystal_cache.models.crystal_type import CrystalAcl
from crystal_cache.infrastructure.schema import FactRow

FACT_DIM = 768
ROUTING_DIM = 10_000

_QF = np.random.default_rng(901).standard_normal(FACT_DIM).astype(np.float32)
_QR = np.random.default_rng(902).standard_normal(ROUTING_DIM).astype(np.float32)

_SPEC = [
    ("cap_grant", 0o600),
    ("cap_priv", 0o600),
    ("cap_team", 0o640),
]
_ALL = {cid for cid, _ in _SPEC}


async def _seed_acl_scenario(store, customer_id):
    """Three crystals under one owner + a P3 group grant on cap_grant.
    Returns (op_owner, op_member, op_plain)."""
    rng = np.random.default_rng(777)
    op_owner, _ = await store.create_operator(
        team_id=customer_id, display_name="Owner",
    )
    op_member, _ = await store.create_operator(
        team_id=customer_id, display_name="Member",
    )
    op_plain, _ = await store.create_operator(
        team_id=customer_id, display_name="Plain",
    )
    group = await store.create_group(customer_id, "acl-parity")
    await store.add_group_member(group["id"], op_member.id, customer_id)

    for cid, mode in _SPEC:
        await store.upsert_crystal(Crystal(
            id=cid,
            customer_id=customer_id,
            crystal_type="customer:legacy",
            summary_vector=[],
            routing_vector=rng.standard_normal(ROUTING_DIM)
            .astype(np.float32).tolist(),
            owner_operator_id=op_owner.id,
            group_team_id=customer_id,
            mode=mode,
        ))
    async with store.session() as session:
        for i, (cid, _mode) in enumerate(_SPEC):
            session.add(FactRow(
                id=f"fap_{i}",
                crystal_id=cid,
                claim_text=f"claim {cid}",
                pair_type="question_answer",
                prompt_text=f"Key|{cid}",
                vector=rng.standard_normal(FACT_DIM)
                .astype(np.float32).tolist(),
            ))
    await store.add_acl(CrystalAcl(
        crystal_id="cap_grant", principal_type="group",
        principal_id=group["id"], grant="read",
    ))
    return op_owner, op_member, op_plain


async def _check_contract(search, ops, *, extract):
    """The visibility table above, against one lane's `search(operator)`.
    `extract` maps the lane's result rows to a set of crystal ids."""
    op_owner, op_member, op_plain = ops

    ids = extract(await search(op_member))
    assert "cap_grant" in ids, "S2-87: group grant must open the read"
    assert "cap_team" in ids
    assert "cap_priv" not in ids

    ids = extract(await search(op_plain))
    assert "cap_grant" not in ids
    assert "cap_priv" not in ids
    assert "cap_team" in ids

    ids = extract(await search(op_owner))
    assert _ALL <= ids

    ids = extract(await search(None))  # system lane: unfiltered (Q2=A)
    assert _ALL <= ids


def _fact_ids(rows) -> set:
    return {r[1] for r in rows}


def _routing_ids(rows) -> set:
    return {cid for cid, _ in rows}


# ---------------------------------------------------------------------------
# In-memory backend (FactVectorStore + VectorStore)
# ---------------------------------------------------------------------------

async def test_memory_fact_lane_acl_contract(
    store, customer, fact_vector_store,
):
    ops = await _seed_acl_scenario(store, customer.id)

    async def search(operator):
        return await fact_vector_store.search(
            customer.id, _QF, k=10, operator=operator,
        )

    await _check_contract(search, ops, extract=_fact_ids)


async def test_memory_routing_lane_acl_contract(
    store, customer, vector_store,
):
    ops = await _seed_acl_scenario(store, customer.id)

    async def search(operator):
        return await vector_store.search(
            customer_id=customer.id, query_vector=_QR, k=10,
            crystal_type="customer:legacy", operator=operator,
        )

    await _check_contract(search, ops, extract=_routing_ids)


# ---------------------------------------------------------------------------
# Qdrant backend (:memory: client, mirroring the same conftest store)
# ---------------------------------------------------------------------------

async def test_qdrant_fact_lane_acl_contract_incl_group_grants(
    store, customer,
):
    pytest.importorskip("qdrant_client")
    from qdrant_client import AsyncQdrantClient
    from crystal_cache.infrastructure.qdrant_vector_index import (
        QdrantVectorIndex,
    )

    ops = await _seed_acl_scenario(store, customer.id)
    index = QdrantVectorIndex(
        client=AsyncQdrantClient(location=":memory:"),
        metadata_store=store,
    )

    async def search(operator):
        return await index.search_facts(
            customer_id=customer.id, query_vector=_QF, k=10,
            operator=operator,
        )

    await _check_contract(search, ops, extract=_fact_ids)


async def test_qdrant_routing_lane_acl_contract_incl_group_grants(
    store, customer,
):
    pytest.importorskip("qdrant_client")
    from qdrant_client import AsyncQdrantClient
    from crystal_cache.infrastructure.qdrant_vector_index import (
        QdrantVectorIndex,
    )

    ops = await _seed_acl_scenario(store, customer.id)
    index = QdrantVectorIndex(
        client=AsyncQdrantClient(location=":memory:"),
        metadata_store=store,
    )

    async def search(operator):
        return await index.search_routing(
            customer_id=customer.id, query_vector=_QR, k=10,
            crystal_type="customer:legacy", operator=operator,
        )

    await _check_contract(search, ops, extract=_routing_ids)


# ---------------------------------------------------------------------------
# sqlite_vec backend (temp-file store, the test_sqlite_vec_parity pattern)
# ---------------------------------------------------------------------------

async def _sqlite_vec_setup(tmp_path):
    from crystal_cache.config import Settings
    from crystal_cache.infrastructure import MetadataStore
    from crystal_cache.infrastructure.sqlite_vec_index import SqliteVecIndex

    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path.as_posix()}/aclparity.db"
    )
    store = MetadataStore(settings_override=settings)
    await store.init()
    await store._seed_legacy_crystal_types_for_tests()
    customer = await store.create_customer(
        provider="anthropic", model_id="claude-x", api_key_ref="ref",
    )
    return store, customer, SqliteVecIndex(metadata_store=store)


async def test_sqlite_vec_fact_lane_acl_contract(tmp_path):
    pytest.importorskip("sqlite_vec")
    store, customer, index = await _sqlite_vec_setup(tmp_path)
    try:
        ops = await _seed_acl_scenario(store, customer.id)

        async def search(operator):
            return await index.search_facts(
                customer_id=customer.id, query_vector=_QF, k=10,
                operator=operator,
            )

        await _check_contract(search, ops, extract=_fact_ids)
    finally:
        await store.dispose()


async def test_sqlite_vec_routing_lane_acl_contract(tmp_path):
    pytest.importorskip("sqlite_vec")
    store, customer, index = await _sqlite_vec_setup(tmp_path)
    try:
        ops = await _seed_acl_scenario(store, customer.id)

        async def search(operator):
            return await index.search_routing(
                customer_id=customer.id, query_vector=_QR, k=10,
                crystal_type="customer:legacy", operator=operator,
            )

        await _check_contract(search, ops, extract=_routing_ids)
    finally:
        await store.dispose()
