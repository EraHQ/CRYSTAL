"""RC-05 (2026-10-04): forgetting is always possible, and walls fail closed.

Forget tools called the write wall, so a customer over the fact cap or
past their trial could not delete to get back under; per-item forget
left the uploaded document's text in document_uploads; and a store error
inside a wall returned None, admitting the write. Now: no wall on
memory_forget, forget, or the wipe half of memory_import; the upload text
is blanked when the last crystal from it is forgotten; and a wall that
cannot be evaluated refuses with code admission_unavailable.
"""
from datetime import datetime, timedelta, timezone

import pytest

from crystal_cache.agent import mcp_server
from crystal_cache.infrastructure.schema import CrystalRow, DocumentUploadRow, FactRow


def _tool(name):
    fn = getattr(mcp_server, name)
    return getattr(fn, "fn", fn)


async def _seed(store, cid, n=3, source_uri="upload://doc-1"):
    async with store.session() as s:
        s.add(DocumentUploadRow(id="up_1", customer_id=cid, label="doc", text="THE TEXT",
                                status="crystallized", source_uri=source_uri))
        for i in range(n):
            s.add(CrystalRow(id=f"cr_{i}", customer_id=cid, summary_vector=[],
                             summary_text=f"c{i}", source_uri=source_uri))
        await s.flush()
        for i in range(n):
            s.add(FactRow(id=f"f_{i}", crystal_id=f"cr_{i}", claim_text="x",
                          prompt_text="k", pair_type="question_answer", vector=[1.0, 0.0]))


@pytest.fixture
def mcp_state(store, monkeypatch, semantic_encoder_stub):
    from crystal_cache.infrastructure.fact_vector_store import FactVectorStore
    from crystal_cache.infrastructure.vector_store import VectorStore

    state = {
        "store": store,
        "vector_store": VectorStore(store),
        "fact_vector_store": FactVectorStore(store),
        "encoder": semantic_encoder_stub,
        "vector_index": None,
    }
    monkeypatch.setattr(mcp_server, "_get_state", lambda: state)
    return state


@pytest.mark.asyncio
async def test_forget_works_past_the_cap_and_past_the_trial(store, customer, mcp_state, monkeypatch):
    cid = customer.id
    await _seed(store, cid)
    await store.set_customer_subscription(cid, "free", None)
    await store.set_customer_inference_mode(cid, "managed")
    # Over the cap and trial expired: writes are walled...
    monkeypatch.setattr(store, "count_billable_facts", _const(10**6))
    async with store.session() as s:
        from crystal_cache.infrastructure.schema import CustomerRow

        row = await s.get(CustomerRow, cid)
        row.trial_expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        await s.commit()
    token = mcp_server._current_customer_id.set(cid)
    try:
        assert (await mcp_server._write_admission_block()) is not None  # walled
        # ...but forgetting is not.
        out = await _tool("memory_forget")(crystal_id="cr_0")
        assert out["deleted"] is True
        out = await _tool("forget")("cr_1")
        assert out["retired"] is True
        out = await _tool("memory_forget")(fact_id="f_2")
        assert out["deleted"] is True
    finally:
        mcp_server._current_customer_id.reset(token)
    assert await store.get_crystal("cr_0") is None
    assert await store.get_crystal("cr_1") is None


@pytest.mark.asyncio
async def test_upload_text_goes_with_the_last_crystal_from_it(store, customer, mcp_state):
    cid = customer.id
    await _seed(store, cid, n=2)
    token = mcp_server._current_customer_id.set(cid)
    try:
        await _tool("memory_forget")(crystal_id="cr_0")
        async with store.session() as s:
            up = await s.get(DocumentUploadRow, "up_1")
            assert up.text == "THE TEXT"  # one crystal from it remains
        await _tool("forget")("cr_1")
        async with store.session() as s:
            up = await s.get(DocumentUploadRow, "up_1")
            assert up is not None  # the record of the upload stays
            assert up.text == ""  # the content does not
            assert up.status == "forgotten"
    finally:
        mcp_server._current_customer_id.reset(token)


@pytest.mark.asyncio
async def test_a_wall_that_cannot_be_evaluated_refuses_the_write(store, customer, mcp_state, monkeypatch):
    cid = customer.id
    await store.set_customer_subscription(cid, "free", None)

    async def _boom(*a, **k):
        raise RuntimeError("db hiccup")

    monkeypatch.setattr(store, "count_billable_facts", _boom)
    token = mcp_server._current_customer_id.set(cid)
    try:
        denied = await mcp_server._write_admission_block()
        assert denied is not None
        assert denied["code"] == "admission_unavailable"
        out = await _tool("memory_store")(key="k", value="v")
        assert out.get("code") == "admission_unavailable"
    finally:
        mcp_server._current_customer_id.reset(token)


@pytest.mark.asyncio
async def test_import_wipe_half_runs_even_when_the_import_is_walled(store, customer, mcp_state, monkeypatch):
    cid = customer.id
    await _seed(store, cid, n=2)
    await store.set_customer_subscription(cid, "free", None)
    monkeypatch.setattr(store, "count_billable_facts", _const(10**6))  # walled
    # PR-2 (B3-2): wipe is admin-only, so act as the Default Admin exactly
    # as the MCP auth middleware does for Key A.
    from crystal_cache.agent.mcp_server import reset_current_operator, set_current_operator

    admin = await store.ensure_default_admin(cid)
    token = mcp_server._current_customer_id.set(cid)
    tok_o = set_current_operator(admin)
    try:
        out = await _tool("memory_import")(records=[{"key": "a", "value": "b"}], wipe=True)
    finally:
        reset_current_operator(tok_o)
        mcp_server._current_customer_id.reset(token)
    assert out["code"] == "memory_full"
    assert out["wiped"] == 2
    assert await store.get_crystal("cr_0") is None and await store.get_crystal("cr_1") is None


def _const(v):
    async def _f(*a, **k):
        return v
    return _f


@pytest.mark.asyncio
async def test_viewers_still_cannot_forget_or_wipe(store, customer, mcp_state):
    """Removing the PLAN walls from deletes must not remove the TENANCY
    rule: a viewer-role operator may read, never mutate (Q4=C)."""
    from types import SimpleNamespace

    from crystal_cache.agent.mcp_server import reset_current_operator, set_current_operator

    cid = customer.id
    await _seed(store, cid, n=1)
    tok_c = mcp_server._current_customer_id.set(cid)
    tok_o = set_current_operator(SimpleNamespace(id="op_v", role="viewer", team_id=cid))
    try:
        assert (await _tool("memory_forget")(crystal_id="cr_0"))["code"] == "viewer_forbidden"
        assert (await _tool("forget")("cr_0"))["code"] == "viewer_forbidden"
        out = await _tool("memory_import")(records=[], wipe=True)
        assert out["code"] == "viewer_forbidden"
    finally:
        reset_current_operator(tok_o)
        mcp_server._current_customer_id.reset(tok_c)
    assert await store.get_crystal("cr_0") is not None  # nothing was wiped
