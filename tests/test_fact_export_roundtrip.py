"""The portable import format (Q44=A as modified, 2026-10-08; supersedes
the RC-07 round-trip pin of 2026-10-05).

One published record (ingress/import_schema.ImportRecord, served at
GET /v1/import/schema, documented in docs/IMPORT_FORMAT.md). Both
importers validate every record before any write; both exporters emit
records that validate. A record carries no origin, gate or ownership:
an imported fact is direct, ungated, counted, owned by the importer at
the record's scope. The portable export carries foreground memory only;
system-derived memory (assumptions, background curation) travels in the
exact-restore topology format, where it comes back as it was.

Pins drive the real HTTP routes and the real MCP tools over the real
store; key_is_path keeps keys verbatim, so no model call is made.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from crystal_cache.endpoints import sdk as sdk_mod
from crystal_cache.ingress import import_schema as S
from crystal_cache.ingress.schema import ImportRequest


async def _seed(store, customer, encoder, vector_store):
    await store.add_pair_for_customer(
        customer_id=customer.id, prompt_text="Billing|terms", answer_text="Net 30",
        encoder=encoder, vector_store=vector_store, origin="direct",
    )
    await store.add_pair_for_customer(
        customer_id=customer.id, prompt_text="Billing|derived", answer_text="Invoices monthly",
        encoder=encoder, vector_store=vector_store, origin="background_worker",
        source_kind="agent_inferred",
    )
    gated = await store.add_pair_for_customer(
        customer_id=customer.id, prompt_text="Assumptions|due", answer_text="Due on the 31st",
        encoder=encoder, vector_store=vector_store, origin="assumptions", recall_gated=True,
    )
    return gated


def _req(encoder=None, vector_store=None, fact_vector_store=None):
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()), headers={})
    req.app.state.prompt_encoder = encoder
    req.app.state.vector_store = vector_store
    req.app.state.fact_vector_store = fact_vector_store
    req.app.state.vector_index = None
    return req


async def _principal(store, customer):
    return customer, await store.ensure_default_admin(customer.id)


@pytest.mark.asyncio
async def test_http_export_carries_foreground_memory_only_and_validates(
    store, customer, semantic_encoder_stub, vector_store
):
    await _seed(store, customer, semantic_encoder_stub, vector_store)
    export = await sdk_mod.sdk_export(await _principal(store, customer), store)
    assert export.record_count == 1
    (rec,) = export.data
    assert rec["key"] == "Billing|terms" and rec["key_is_path"] is True
    assert set(rec) == {"key", "value", "key_is_path", "pair_type", "source_kind",
                        "answer_value", "crystal_type", "scope"}
    jsonschema.validate({"records": export.data}, S.json_schema())


@pytest.mark.asyncio
async def test_http_import_makes_owned_direct_counted_facts(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store
):
    await _seed(store, customer, semantic_encoder_stub, vector_store)
    export = await sdk_mod.sdk_export(await _principal(store, customer), store)

    other = await store.create_customer(provider="anthropic", model_id="m", api_key_ref="")
    other_cust, other_admin = await _principal(store, other)
    out = await sdk_mod.sdk_import(
        ImportRequest(records=export.data),
        _req(semantic_encoder_stub, vector_store, fact_vector_store),
        (other_cust, other_admin), store,
    )
    assert out.records_processed == 1 and out.errors == 0
    assert await store.count_billable_facts(other.id) == 1
    (c,) = await store.list_crystals_for_customer(other.id)
    assert c.origin == "direct" and c.recall_gated is False
    assert c.owner_operator_id == other_admin.id
    assert c.mode == 0o640  # the seed was unowned (team), so the record said team


@pytest.mark.asyncio
async def test_http_import_refuses_origin_gate_and_ownership_fields(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store
):
    from fastapi import HTTPException

    bad = [
        {"key": "k", "value": "v", "origin": "background_worker"},
        {"key": "k", "value": "v", "recall_gated": False},
        {"key": "k", "value": "v", "owner_operator_id": "op_x"},
        {"key": "k", "value": "v", "mode": 384},
        {"key": "k", "value": "v", "crystal_type": "general:python"},
        {"key": "k", "value": "v", "source_kind": "made_up"},
    ]
    with pytest.raises(HTTPException) as e:
        await sdk_mod.sdk_import(
            ImportRequest(records=bad),
            _req(semantic_encoder_stub, vector_store, fact_vector_store),
            await _principal(store, customer), store,
        )
    assert e.value.status_code == 422
    fields = sorted((err["index"], err["field"]) for err in e.value.detail["errors"])
    assert fields == [(0, "origin"), (1, "recall_gated"), (2, "owner_operator_id"),
                      (3, "mode"), (4, "crystal_type"), (5, "source_kind")]
    assert await store.list_crystals_for_customer(customer.id) == []


@pytest.mark.asyncio
async def test_import_batch_is_judged_against_the_cap_as_a_whole(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch
):
    """B3-7: a batch cannot pass a single pre-check at cap - 1 and land
    past the cap. The door sees bank + batch."""
    from fastapi import HTTPException

    from crystal_cache.control import admission

    cust, admin = await _principal(store, customer)
    cust.subscription_tier = "free"
    cust.inference_mode = "byok"
    tier = admission.resolve_tier("free")
    cap_blocked_at = int(tier.fact_cap * admission.GRACE_FACTOR)

    async def _count(_cid):
        return cap_blocked_at - 1  # one short of the wall

    monkeypatch.setattr(store, "count_billable_facts", _count)
    records = [{"key": f"k{i}", "value": "v", "key_is_path": True} for i in range(5)]
    with pytest.raises(HTTPException) as e:
        await sdk_mod.sdk_import(
            ImportRequest(records=records),
            _req(semantic_encoder_stub, vector_store, fact_vector_store),
            (cust, admin), store,
        )
    assert e.value.status_code == 402
    assert e.value.headers.get("X-Plan-Wall") == "memory_full"


@pytest.mark.asyncio
async def test_wipe_is_admin_only_on_http_and_mcp(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch
):
    from fastapi import HTTPException

    from crystal_cache.agent import mcp_server

    member, _key = await store.create_operator(customer.id, display_name="m", role="operator")
    with pytest.raises(HTTPException) as e:
        await sdk_mod.sdk_import(
            ImportRequest(records=[{"key": "k", "value": "v"}], wipe=True),
            _req(semantic_encoder_stub, vector_store, fact_vector_store),
            (customer, member), store,
        )
    assert e.value.status_code == 403 and e.value.detail["error"]["code"] == "admin_required"

    monkeypatch.setattr(mcp_server, "_get_state", lambda: {
        "store": store, "encoder": semantic_encoder_stub, "vector_store": vector_store,
    })
    fn = getattr(mcp_server.memory_import, "fn", mcp_server.memory_import)
    token = mcp_server._current_customer_id.set(customer.id)
    op_token = mcp_server.set_current_operator(member)
    try:
        out = await fn(records=[{"key": "k", "value": "v"}], wipe=True)
    finally:
        mcp_server.reset_current_operator(op_token)
        mcp_server._current_customer_id.reset(token)
    assert out.get("code") == "admin_required", out


@pytest.mark.asyncio
async def test_mcp_export_and_import_speak_the_same_record(
    store, customer, semantic_encoder_stub, vector_store, monkeypatch
):
    from crystal_cache.agent import mcp_server

    await _seed(store, customer, semantic_encoder_stub, vector_store)
    monkeypatch.setattr(mcp_server, "_get_state", lambda: {
        "store": store, "encoder": semantic_encoder_stub, "vector_store": vector_store,
    })
    export_fn = getattr(mcp_server.memory_export, "fn", mcp_server.memory_export)
    import_fn = getattr(mcp_server.memory_import, "fn", mcp_server.memory_import)
    token = mcp_server._current_customer_id.set(customer.id)
    try:
        out = await export_fn(limit=10_000)
        assert out["limit"] == mcp_server.MEMORY_IMPORT_MAX_RECORDS
        assert out["total_records"] == 1 and out["record_count"] == 1
        jsonschema.validate({"records": out["data"]}, S.json_schema())
        assert "origin" not in out["data"][0] and "recall_gated" not in out["data"][0]

        refused = await import_fn(records=[{"key": "k", "value": "v", "origin": "assumptions"}])
        assert refused["code"] == "invalid_record"
        assert refused["errors"] == [{"index": 0, "field": "origin",
                                      "message": "Extra inputs are not permitted"}]
        assert len(await store.list_crystals_for_customer(customer.id)) == 3  # nothing written

        async def _open(**kw):
            return None

        monkeypatch.setattr(mcp_server, "_write_admission_block", _open)
        ok = await import_fn(records=[{"key": "Ops|oncall", "value": "Alice", "key_is_path": True,
                                       "scope": "team"}])
        assert ok["records_processed"] == 1 and ok["errors"] == 0
    finally:
        mcp_server._current_customer_id.reset(token)
    crystals = await store.list_crystals_for_customer(customer.id)
    assert len(crystals) == 4
    new = []
    for c in crystals:
        facts = await store.list_facts_for_crystal(c.id)
        if any(f.prompt_text == "Ops|oncall" for f in facts):
            new.append(c)
    assert len(new) == 1 and new[0].origin == "direct" and new[0].mode == 0o640


@pytest.mark.asyncio
async def test_topology_export_still_carries_system_memory_as_it_was(
    store, customer, semantic_encoder_stub, vector_store
):
    """The exact-restore format keeps origin and the gate: the portable
    format's exclusions lose nothing a backup needs."""
    await _seed(store, customer, semantic_encoder_stub, vector_store)
    payload = await store.export_bank_topology(customer.id)
    assert {c["origin"] for c in payload["crystals"]} == {"direct", "background_worker", "assumptions"}
    # The restore is id-preserving and skips primary-key collisions, so
    # (as on launch eve) the bank is erased before the restore lands.
    for c in await store.list_crystals_for_customer(customer.id):
        await store.delete_crystal(c.id, customer.id, vector_store=vector_store)
    other = await store.create_customer(provider="anthropic", model_id="m", api_key_ref="")
    counts = await store.import_bank_topology(other.id, payload)
    assert counts["crystals"] == 3, counts
    gated = [c for c in await store.list_crystals_for_customer(other.id) if c.recall_gated]
    assert len(gated) == 1 and gated[0].origin == "assumptions"


@pytest.mark.asyncio
async def test_schema_route_and_doc_match_the_module(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store
):
    from httpx import ASGITransport, AsyncClient

    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app
    app = _build_app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/import/schema")  # no auth: it is a contract
    assert r.status_code == 200
    served = r.json()
    assert served == S.json_schema()
    assert served["$defs"]["ImportRecord"]["additionalProperties"] is False
    assert set(served["$defs"]["ImportRecord"]["properties"]) == {
        "key", "value", "key_is_path", "pair_type", "source_kind",
        "answer_value", "crystal_type", "scope",
    }

    doc = Path(__file__).resolve().parents[1] / "docs" / "IMPORT_FORMAT.md"
    text = doc.read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(.*?)\n```", text, flags=re.S)
    published = [json.loads(b) for b in blocks if '"$schema"' in b]
    assert published and published[-1] == served, "docs/IMPORT_FORMAT.md is out of date"


@pytest.mark.asyncio
async def test_topology_restore_refuses_rows_the_bank_cannot_trust(
    store, customer, semantic_encoder_stub, vector_store
):
    """B3-4 / B3-5: a wrong-width or non-finite HDC vector, an unknown
    tier, origin or mode, or a general:* crystal refuses the restore
    before anything is written."""
    import base64
    import copy

    import numpy as np

    from crystal_cache.infrastructure.metadata_store import TopologyRowInvalid

    await _seed(store, customer, semantic_encoder_stub, vector_store)
    payload = await store.export_bank_topology(customer.id)
    for c in await store.list_crystals_for_customer(customer.id):
        await store.delete_crystal(c.id, customer.id, vector_store=vector_store)

    def packed(n, value=1.0):
        return "f32:" + base64.b64encode(np.full(n, value, dtype="<f4").tobytes()).decode()

    def tampered(mutate):
        p = copy.deepcopy(payload)
        mutate(p)
        return p

    cases = {
        "short hdc vector": lambda p: p["crystals"][0].__setitem__("summary_vector", packed(5)),
        "nan hdc vector": lambda p: p["crystals"][0].__setitem__("summary_vector", packed(10_000, float("nan"))),
        "not base64": lambda p: p["crystals"][0].__setitem__("summary_vector", "f32:@@@"),
        "nan fact vector": lambda p: p["facts"][0].__setitem__("vector", [float("inf")] * 768),
        "unknown tier": lambda p: p["crystals"][0].__setitem__("quality_tier", "whitelisted!"),
        "unknown origin": lambda p: p["crystals"][0].__setitem__("origin", "cognition"),
        "unknown mode": lambda p: p["crystals"][0].__setitem__("mode", 0o777),
        "general crystal": lambda p: p["crystals"][0].__setitem__("crystal_type", "general:python"),
        "bad fact kind": lambda p: p["facts"][0].__setitem__("source_kind", "trust_me"),
        "bad grating": lambda p: p["facts"][0].__setitem__("grating_strength", 7),
    }
    other = await store.create_customer(provider="anthropic", model_id="m", api_key_ref="")
    for name, mutate in cases.items():
        with pytest.raises(TopologyRowInvalid) as e:
            await store.import_bank_topology(other.id, tampered(mutate))
        assert e.value.public_message, name
        assert await store.list_crystals_for_customer(other.id) == [], name


@pytest.mark.asyncio
async def test_deactivated_facts_stay_out_and_scope_and_provenance_round_trip(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch
):
    """RC-07 on the real lanes (Lockdown PR-8): a fact the curation gate
    deactivated (grating 0) is in neither exporter's output; a personal
    record comes back personal (0o600, owned by the importer), a team
    record team (0o640); the fact's provenance (source_kind) survives;
    and the importer can recall what it imported."""
    import numpy as np
    from sqlalchemy import update

    from crystal_cache.agent import mcp_server
    from crystal_cache.infrastructure.schema import FactRow

    admin = await store.ensure_default_admin(customer.id)
    await store.add_pair_for_customer(
        customer_id=customer.id, prompt_text="Prefs|editor", answer_text="vim",
        encoder=semantic_encoder_stub, vector_store=vector_store, origin="direct",
        source_kind="operator_stated", owner_operator_id=admin.id, mode=0o600,
    )
    await store.add_pair_for_customer(
        customer_id=customer.id, prompt_text="Team|oncall", answer_text="Alice",
        encoder=semantic_encoder_stub, vector_store=vector_store, origin="direct",
        source_kind="web_search_result", owner_operator_id=admin.id, mode=0o640,
    )
    _c, dead = await store.add_pair_for_customer(
        customer_id=customer.id, prompt_text="Old|belief", answer_text="the sky is green",
        encoder=semantic_encoder_stub, vector_store=vector_store, origin="direct",
    )
    async with store.session() as s:
        await s.execute(update(FactRow).where(FactRow.id == dead.id).values(grating_strength=0.0))

    # Both exporters: the deactivated fact is out; scope and provenance are in.
    export = await sdk_mod.sdk_export((customer, admin), store)
    by_key = {r["key"]: r for r in export.data}
    assert set(by_key) == {"Prefs|editor", "Team|oncall"}
    assert by_key["Prefs|editor"]["scope"] == "personal"
    assert by_key["Prefs|editor"]["source_kind"] == "operator_stated"
    assert by_key["Team|oncall"]["scope"] == "team"
    assert by_key["Team|oncall"]["source_kind"] == "web_search_result"

    monkeypatch.setattr(mcp_server, "_get_state", lambda: {
        "store": store, "encoder": semantic_encoder_stub, "vector_store": vector_store,
        "fact_vector_store": fact_vector_store, "vector_index": None,
    })
    export_fn = getattr(mcp_server.memory_export, "fn", mcp_server.memory_export)
    token = mcp_server._current_customer_id.set(customer.id)
    try:
        mcp_out = await export_fn(limit=100)
    finally:
        mcp_server._current_customer_id.reset(token)
    assert {r["key"] for r in mcp_out["data"]} == {"Prefs|editor", "Team|oncall"}
    assert sorted(mcp_out["data"], key=lambda r: r["key"]) == sorted(export.data, key=lambda r: r["key"])

    # Import into another workspace through the MCP tool, as that
    # workspace's admin, through the real door: scope and provenance
    # restored, owner is the importer; the deactivated fact never travels.
    other = await store.create_customer(provider="anthropic", model_id="m", api_key_ref="")
    other_admin = await store.ensure_default_admin(other.id)
    import_fn = getattr(mcp_server.memory_import, "fn", mcp_server.memory_import)
    token = mcp_server._current_customer_id.set(other.id)
    tok_op = mcp_server.set_current_operator(other_admin)
    try:
        out = await import_fn(records=mcp_out["data"])
    finally:
        mcp_server.reset_current_operator(tok_op)
        mcp_server._current_customer_id.reset(token)
    assert out.get("code") is None, out
    assert out["records_processed"] == 2 and out["errors"] == 0
    restored = {}
    for c in await store.list_crystals_for_customer(other.id):
        for f in await store.list_facts_for_crystal(c.id):
            restored[f.prompt_text] = (c, f)
    assert set(restored) == {"Prefs|editor", "Team|oncall"}
    c_personal, f_personal = restored["Prefs|editor"]
    c_team, f_team = restored["Team|oncall"]
    assert c_personal.mode == 0o600 and c_personal.owner_operator_id == other_admin.id
    assert c_team.mode == 0o640 and c_team.owner_operator_id == other_admin.id
    assert f_personal.source_kind == "operator_stated" and f_team.source_kind == "web_search_result"
    assert c_personal.origin == c_team.origin == "direct"

    # Recall on the importer's bank, as the importer: both facts come back
    # for their own keys, and nothing from the old belief.
    for key, fact in (("Prefs|editor", f_personal), ("Team|oncall", f_team)):
        q = np.asarray(semantic_encoder_stub.encode_native(key), dtype=np.float32)
        hits = await fact_vector_store.search(other.id, q, k=5, operator=other_admin)
        assert fact.id in {h[0] for h in hits}, key
    q = np.asarray(semantic_encoder_stub.encode_native("Old|belief"), dtype=np.float32)
    assert dead.id not in {h[0] for h in await fact_vector_store.search(other.id, q, k=5)}
