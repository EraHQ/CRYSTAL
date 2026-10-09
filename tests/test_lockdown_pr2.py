"""Lockdown PR-2 (2026-10-08, AUDIT_LAUNCH_VERIFY): nothing a request can
do runs up the platform bill.

- Q52 / B1-3: the agent's llm_invoke tool runs the managed model list and
  the output clamp, and bounds its arguments.
- Q53 / B1-4 / B1-27 / B1-52: git watches never use a platform token;
  repo, branch, source_name, token, cadence and globs are closed shapes;
  file paths are URL-quoted; the worker names token-less watches at start.
- Fresh sweep class 5: push-queue approve pays the fact cap.
- B2-12: a sync cycle ingests at most 200 files, re-checks capacity every
  20, remembers what landed under an unfinished head, never pays twice.
- RC-12 residue: the due query floors the cadence at 15 minutes.
- B1-8 / B1-9 / B5-4: chat completions cap n and max_tokens, and a
  managed turn with no max_tokens gets the tier ceiling.
- Fresh sweep class 2: the cognition precondition verifier is ledgered.

(The import schema, record cap and admin-only wipe are pinned in
tests/test_fact_export_roundtrip.py and tests/test_export_import.py.)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# llm_invoke
# ---------------------------------------------------------------------------

class _Upstream:
    def __init__(self):
        self.calls = []

    async def complete(self, *, messages, model, temperature=None, max_tokens=None, **kw):
        self.calls.append({"model": model, "temperature": temperature, "max_tokens": max_tokens})
        return SimpleNamespace(assistant_text="ok", prompt_tokens=10, completion_tokens=5,
                               openai_format={"choices": [{"finish_reason": "stop"}]})


@pytest.fixture
def llm_tool(store, customer, monkeypatch):
    from crystal_cache.agent.tools import llm as llm_mod
    from crystal_cache.execution import upstream_client

    up = _Upstream()

    async def _factory(cust, st):
        return up

    monkeypatch.setattr(upstream_client, "get_upstream_client", _factory)
    # llm.py binds `_get_state` at import (from .retrievers import _get_state),
    # so the module's own name is the one to patch.
    monkeypatch.setattr(llm_mod, "_get_state", lambda: {"store": store})
    recorded = []

    async def _record(**kw):
        recorded.append(kw)

    from crystal_cache.cost import emit
    monkeypatch.setattr(emit, "record_model_call", _record)
    return llm_mod.llm_invoke, up, recorded


@pytest.mark.asyncio
async def test_llm_invoke_on_a_managed_plan_runs_the_model_list_and_the_clamp(
    store, customer, llm_tool, monkeypatch,
):
    from crystal_cache.control import admission

    fn, up, recorded = llm_tool
    await store.set_customer_subscription(customer.id, "free", None)
    cust = await store.get_customer_by_id(customer.id)
    cust.inference_mode = "managed"

    async def _get(_cid):
        return cust

    monkeypatch.setattr(store, "get_customer_by_id", _get)
    allowed = admission.allowed_models_for("free")
    ceiling = admission.resolve_tier("free").max_output_tokens

    out = await fn(customer.id, messages=[{"role": "user", "content": "hi"}],
                   model="claude-opus-4-1-20250805")
    assert out.get("code") == "model_not_in_plan", out
    assert up.calls == []

    out = await fn(customer.id, messages=[{"role": "user", "content": "hi"}],
                   model=allowed[0], max_tokens=100_000)
    assert "error" not in out, out
    assert up.calls[-1]["max_tokens"] == ceiling  # clamped
    out = await fn(customer.id, messages=[{"role": "user", "content": "hi"}], model=allowed[0])
    assert up.calls[-1]["max_tokens"] == ceiling  # omitted = the ceiling, not the upstream default
    assert recorded and recorded[-1]["origin"] == "agent_llm_invoke"
    assert recorded[-1]["billing"] == "managed"


@pytest.mark.asyncio
async def test_llm_invoke_bounds_its_arguments(store, customer, llm_tool):
    fn, up, _ = llm_tool
    too_many = [{"role": "user", "content": "x"}] * 51
    assert (await fn(customer.id, messages=too_many))["code"] == "invalid_argument"
    long = [{"role": "user", "content": "x" * 100_001}]
    assert (await fn(customer.id, messages=long))["code"] == "invalid_argument"
    bad_role = [{"role": "tool", "content": "x"}]
    assert (await fn(customer.id, messages=bad_role))["code"] == "invalid_argument"
    ok = [{"role": "user", "content": "x"}]
    assert (await fn(customer.id, messages=ok, temperature=1.5))["code"] == "invalid_argument"
    assert (await fn(customer.id, messages=ok, max_tokens=0))["code"] == "invalid_argument"
    assert (await fn(customer.id, messages=ok, model="m" * 129))["code"] == "invalid_argument"
    assert up.calls == []
    # byok keeps its own model and cap
    out = await fn(customer.id, messages=ok, model="anything-goes", max_tokens=64_000)
    assert "error" not in out and up.calls[-1] == {"model": "anything-goes", "temperature": None,
                                                    "max_tokens": 64_000}


# ---------------------------------------------------------------------------
# Git watches
# ---------------------------------------------------------------------------

class _Git:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def __call__(self, url, token):
        self.calls.append((url, token))
        for key, value in self.responses.items():
            if key in url:
                return value
        raise AssertionError(f"unexpected URL: {url}")


def _watch(config, last_state=None):
    return SimpleNamespace(config=config, last_state=last_state, source_name="r",
                           customer_id="cus_x", encrypted_token=None)


@pytest.mark.asyncio
async def test_git_watch_without_a_token_is_refused_and_never_uses_the_environment(monkeypatch):
    from crystal_cache.ingestion.git_handler import GitSourceHandler

    monkeypatch.setenv("CC_GITHUB_TOKEN", "ghp_platform_secret")
    fake = _Git({"/branches/master": {"commit": {"sha": "aaa"}}})
    h = GitSourceHandler(http_get=fake)
    with pytest.raises(ValueError, match="no access token"):
        await h.check(_watch({"repo": "EraHQ/x"}), None)
    with pytest.raises(ValueError, match="no access token"):
        await h.fetch(_watch({"repo": "EraHQ/x"}, {"head": "aaa"}), "a.md", None)
    assert fake.calls == []
    await h.check(_watch({"repo": "EraHQ/x"}, {"head": "aaa"}), "ghp_watch")
    assert fake.calls == [("https://api.github.com/repos/EraHQ/x/branches/master", "ghp_watch")]


@pytest.mark.asyncio
async def test_git_repo_branch_and_path_cannot_leave_the_repository():
    from crystal_cache.ingestion.git_handler import GitSourceHandler, validate_git_config

    for bad in ("EraHQ/../other", "EraHQ/x?per_page=1", "Era HQ/x", "EraHQ/x/..",
                "EraHQ/" + "a" * 101):
        with pytest.raises(ValueError):
            validate_git_config({"repo": bad})
    for bad in ("../main", "main/../x", "main?x", "ma in", "/main", "a//b"):
        with pytest.raises(ValueError):
            validate_git_config({"repo": "EraHQ/x", "branch": bad})
    assert validate_git_config({"repo": "https://github.com/EraHQ/x.git", "branch": "release/1.2"}) == {
        "repo": "EraHQ/x", "branch": "release/1.2",
    }
    with pytest.raises(ValueError):
        validate_git_config({"repo": "EraHQ/x", "include": ["*.py"] * 51})
    with pytest.raises(ValueError):
        validate_git_config({"repo": "EraHQ/x", "exclude": ["x" * 257]})

    import base64
    fake = _Git({"/contents/": {"content": base64.b64encode(b"x").decode()}})
    h = GitSourceHandler(http_get=fake)
    await h.fetch(_watch({"repo": "EraHQ/x"}, {"head": "abc1234"}), "docs/my file#1.md", "t")
    assert fake.calls[-1][0] == "https://api.github.com/repos/EraHQ/x/contents/docs/my%20file%231.md?ref=abc1234"
    with pytest.raises(ValueError):
        await h.fetch(_watch({"repo": "EraHQ/x"}, {"head": "abc1234"}), "../etc/passwd", "t")
    with pytest.raises(ValueError):
        await h.fetch(_watch({"repo": "EraHQ/x"}, {"head": "abc?x"}), "a.md", "t")


class _Req:
    def __init__(self, body, customer_id):
        self._body = body
        self.state = SimpleNamespace(tenant_pin=customer_id)

    async def json(self):
        return self._body


@pytest.mark.asyncio
async def test_watch_create_route_refuses_bad_shapes_and_requires_a_git_token(store, customer):
    from fastapi import HTTPException

    from crystal_cache.endpoints.admin import admin_create_watch

    async def create(body):
        return await admin_create_watch(_Req(body, customer.id), store, customer.id)

    good = {"scheme": "git", "source_name": "crystal", "token": "ghp_x",
            "config": {"repo": "EraHQ/x", "branch": "main"}, "cadence_minutes": 15}
    for mutate, field in (
        (lambda b: b.update(source_name="bad/name"), "source_name"),
        (lambda b: b.update(source_name="x" * 129), "source_name"),
        (lambda b: b.update(source_name="a%b"), "source_name"),
        (lambda b: b.update(scheme="svn"), "scheme"),
        (lambda b: b.update(token=""), "token"),
        (lambda b: b.update(token="t" * 513), "token"),
        (lambda b: b.update(cadence_minutes=20_000), "cadence"),
        (lambda b: b.update(config={"repo": "EraHQ/../y"}), "repo"),
        (lambda b: b.update(config={"repo": "EraHQ/x", "branch": "../main"}), "branch"),
        (lambda b: b.update(config="not an object"), "config"),
    ):
        body = {**good, "config": dict(good["config"])}
        mutate(body)
        with pytest.raises(HTTPException) as e:
            await create(body)
        assert e.value.status_code == 422, field
    r = await create(dict(good, config=dict(good["config"]), cadence_minutes=1))
    import json
    w = json.loads(r.body)
    assert w["cadence_minutes"] == 15  # floored
    assert w["config"] == {"repo": "EraHQ/x", "branch": "main"}


@pytest.mark.asyncio
async def test_worker_names_git_watches_without_a_token_at_start(store, customer):
    from crystal_cache.workers.source_sync import report_watches_without_token

    silent = await store.create_source_watch(
        customer.id, scheme="git", source_name="no-token", config={"repo": "a/b"},
        cadence_minutes=15, encrypted_token=None,
    )
    await store.create_source_watch(
        customer.id, scheme="git", source_name="has-token", config={"repo": "a/c"},
        cadence_minutes=15, encrypted_token="enc",
    )
    missing = await report_watches_without_token(store)
    assert [w.id for w in missing] == [silent.id]


@pytest.mark.asyncio
async def test_due_query_floors_the_cadence_at_fifteen_minutes(store, customer):
    now = datetime.now(timezone.utc)
    w = await store.create_source_watch(
        customer.id, scheme="git", source_name="fast", config={}, cadence_minutes=15,
    )
    from crystal_cache.infrastructure.schema import SourceWatchRow
    async with store.session() as s:
        (await s.get(SourceWatchRow, w.id)).cadence_minutes = 1  # a pre-RC-12 row
    await store.update_source_watch_state(w.id, customer.id, last_state={"head": "h"}, checked_at=now)
    assert w.id not in {x.id for x in await store.list_source_watches_due(now + timedelta(minutes=5))}
    assert w.id in {x.id for x in await store.list_source_watches_due(now + timedelta(minutes=16))}


# ---------------------------------------------------------------------------
# Source sync: per-cycle cap, capacity re-check, no double pay
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sync_cycle_caps_at_200_files_and_resumes_without_paying_twice(store, customer, monkeypatch):
    from crystal_cache.ingestion.source_handlers import ChangeSet, SourceEnvelope, register_handler
    from crystal_cache.workers import source_sync

    files = [f"f{i:04d}.md" for i in range(450)]

    class _Handler:
        scheme = "testcap"

        async def check(self, watch, token):
            if (watch.last_state or {}).get("head") == "h1":
                return None
            return ChangeSet(new_state={"head": "h1"}, changed=list(files))

        async def fetch(self, watch, path, token):
            return SourceEnvelope(payload_bytes=b"x", mime_type="text/plain",
                                  source_uri=f"repo://r/{path}", label=path)

    register_handler(_Handler())
    ingested: list[str] = []

    async def _ingest(store_, encoder, vs, fvs, llm, watch, envelope):
        ingested.append(envelope.label)

    monkeypatch.setattr(source_sync, "_ingest_envelope", _ingest)
    w = await store.create_source_watch(
        customer.id, scheme="testcap", source_name="r", config={},
        cadence_minutes=15, review_mode="auto", encrypted_token=None,
    )

    async def cycle():
        cur = await store.get_source_watch(w.id, customer.id)
        return await source_sync.sync_one_watch(store, None, None, None, None, cur)

    out = await cycle()
    assert out["ingested"] == 200 and out["deferred"] == 250
    cur = await store.get_source_watch(w.id, customer.id)
    assert (cur.last_state or {}).get("head") is None  # not advanced
    assert len((cur.last_state or {})[source_sync.DONE_PATHS_KEY]["paths"]) == 200
    assert "250 file(s) remaining" in (cur.last_error or "")

    out = await cycle()
    assert out["ingested"] == 200 and out["deferred"] == 50
    out = await cycle()
    assert out["ingested"] == 50 and "deferred" not in out
    cur = await store.get_source_watch(w.id, customer.id)
    assert (cur.last_state or {}).get("head") == "h1"
    assert source_sync.DONE_PATHS_KEY not in (cur.last_state or {})
    assert sorted(ingested) == files and len(ingested) == 450  # every file once

    assert (await cycle()).get("unchanged") is True


@pytest.mark.asyncio
async def test_sync_cycle_stops_at_the_capacity_recheck(store, customer, monkeypatch):
    from crystal_cache.ingestion.source_handlers import ChangeSet, SourceEnvelope, register_handler
    from crystal_cache.workers import source_sync

    files = [f"g{i:03d}.md" for i in range(100)]

    class _Handler:
        scheme = "testbudget"

        async def check(self, watch, token):
            if (watch.last_state or {}).get("head") == "h1":
                return None
            return ChangeSet(new_state={"head": "h1"}, changed=list(files))

        async def fetch(self, watch, path, token):
            return SourceEnvelope(payload_bytes=b"x", mime_type="text/plain",
                                  source_uri=f"repo://r/{path}", label=path)

    register_handler(_Handler())
    ingested: list[str] = []

    async def _ingest(store_, encoder, vs, fvs, llm, watch, envelope):
        ingested.append(envelope.label)

    monkeypatch.setattr(source_sync, "_ingest_envelope", _ingest)
    checks = {"n": 0}

    async def _out_of_capacity(store_, cid):
        checks["n"] += 1
        return checks["n"] >= 2  # the second re-check (after 40 files) says stop

    monkeypatch.setattr(source_sync, "_tenant_out_of_capacity", _out_of_capacity)
    w = await store.create_source_watch(
        customer.id, scheme="testbudget", source_name="r", config={},
        cadence_minutes=15, review_mode="auto", encrypted_token=None,
    )
    cur = await store.get_source_watch(w.id, customer.id)
    out = await source_sync.sync_one_watch(store, None, None, None, None, cur)
    assert out["ingested"] == 40 and out["deferred"] == 60
    assert len(ingested) == 40


# ---------------------------------------------------------------------------
# Push-queue approve pays the fact cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_push_queue_approve_is_walled_at_the_fact_cap(store, customer, semantic_encoder_stub, vector_store, monkeypatch):
    from fastapi import HTTPException

    from crystal_cache.endpoints import admin as admin_mod

    item = await store.create_push_review_item(customer.id, key="k", value="v", confidence=0.7)
    await store.set_customer_subscription(customer.id, "free", None)

    async def _count(_cid):
        return 10**6

    monkeypatch.setattr(store, "count_billable_facts", _count)
    req = SimpleNamespace(state=SimpleNamespace(tenant_pin=customer.id),
                          app=SimpleNamespace(state=SimpleNamespace(
                              prompt_encoder=semantic_encoder_stub, vector_store=vector_store, vector_index=None)))
    from crystal_cache.endpoints.admin import ApprovePushRequest
    with pytest.raises(HTTPException) as e:
        await admin_mod.approve_review_item(item.id, ApprovePushRequest(), req, store, customer.id)
    assert e.value.status_code == 402 and e.value.headers.get("X-Plan-Wall") == "memory_full"
    assert (await store.get_push_review_item(item.id, customer.id)).status == "pending"


# ---------------------------------------------------------------------------
# Chat completions: n, max_tokens, managed default
# ---------------------------------------------------------------------------

def test_chat_request_caps_n_and_max_tokens():
    from pydantic import ValidationError

    from crystal_cache.ingress.schema import ChatCompletionRequest, ChatMessage

    msgs = [ChatMessage(role="user", content="hi")]
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=msgs, n=2)
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=msgs, max_tokens=128_001)
    ChatCompletionRequest(model="m", messages=msgs, n=1, max_tokens=128_000)


@pytest.mark.asyncio
async def test_managed_chat_turn_without_max_tokens_gets_the_tier_ceiling(customer, store, monkeypatch):
    try:
        from tests import test_phase9c_chat_proxy_chars as h
    except ModuleNotFoundError:
        import test_phase9c_chat_proxy_chars as h
    from crystal_cache.control import admission
    from crystal_cache.endpoints.chat_proxy import chat_completions

    await store.set_customer_subscription(customer.id, "free", None)
    cust = await store.get_customer_by_id(customer.id)
    cust.inference_mode = "managed"
    model = admission.allowed_models_for("free")[0]
    body = await h._make_request_body(user_text="What is 2+2?", model=model)
    assert body.max_tokens is None
    h._patch_retrieve_and_inject(monkeypatch, outcome=h._empty_outcome(
        [m.model_dump(exclude_none=True) for m in body.messages]))
    fake = h._FakeUpstreamClient()
    fake.script(h._make_upstream_response(assistant_text="4"))
    h._patch_upstream_client(monkeypatch, client=fake)
    await chat_completions(body=body, request=h._FakeRequest(), customer=cust, store=store)
    assert fake.calls[0]["max_tokens"] == admission.resolve_tier("free").max_output_tokens


# ---------------------------------------------------------------------------
# Cognition precondition verifier is ledgered
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_precondition_verifier_records_a_ledger_row(store, customer, monkeypatch):
    from crystal_cache.workers import cognition as w
    from crystal_cache.llm import reset_llm_client, set_llm_client

    class _Seam:
        def complete_detailed(self, **kw):
            return SimpleNamespace(text='{"match": true, "reason": "yes"}', model="claude-haiku-4-5",
                                   input_tokens=11, output_tokens=3,
                                   cache_creation_tokens=None, cache_read_tokens=None)

    set_llm_client(_Seam())
    try:
        ok, reason, usage = w._verify_candidate_against_context(
            SimpleNamespace(id="d", text="price list", filename="p.xlsx", label="p"), "the supplier price list",
        )
    finally:
        reset_llm_client()
    assert ok is True and usage.model == "claude-haiku-4-5"

    recorded = []

    async def _record(**kw):
        recorded.append(kw)

    from crystal_cache.cost import emit
    monkeypatch.setattr(emit, "record_model_call", _record)
    monkeypatch.setattr(w, "_verify_candidate_against_context",
                        lambda d, c: (True, "", SimpleNamespace(model="m", input_tokens=1, output_tokens=1,
                                                               cache_creation_tokens=None, cache_read_tokens=None)))
    w._precondition_verdicts.clear()
    from crystal_cache.infrastructure.schema import DocumentUploadRow
    doc = await store.create_document_upload(customer.id, label="price_list.xlsx", text="rows")
    async with store.session() as s:
        (await s.get(DocumentUploadRow, doc.id)).status = "crystallized"
    met, _note = await w._precondition_met(
        store, customer.id, {"kind": "document", "match": "price list", "context": "the supplier price list"},
        task_id="t1",
    )
    assert met is True
    assert recorded and recorded[0]["origin"] == "precondition_verify" and recorded[0]["customer_id"] == customer.id
