"""Connected apps and the Beta list (2026-10-24).

Anthony: "I say we build it. Make a beta list for private invites to
features like this. Everything but keep Gmail private." Q108 A: what the
apps show is kept 7 days; only what passes a connection's rules becomes
memory ("Only make crystals for x, y, or z"). Nango holds the tokens;
connectors/nango.py is the only code that talks to it, and every test
here drives it through an httpx MockTransport.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import HTTPException

from crystal_cache import beta
from crystal_cache.config import get_settings
from crystal_cache.connectors import catalog, nango, service
from crystal_cache.workers import app_sync

SIGNING = "whsec_test_signing_key"


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    beta.reset_for_tests()
    app_sync.reset_for_tests()
    s = get_settings()
    monkeypatch.setattr(s, "nango_secret_key", "nango-secret-test")
    monkeypatch.setattr(s, "nango_webhook_signing_key", SIGNING)
    monkeypatch.setattr(s, "nango_base_url", "https://nango.test")
    # Q118 A: the first five apps are configured in this test Nango.
    service.reset_for_tests()
    service._configured.update(keys={a.integration for a in catalog.APPS[:5]}, at=time.monotonic())
    yield
    service.reset_for_tests()
    nango.set_transport(None)


class _Nango:
    """A scripted Nango: records requests, answers by path."""

    def __init__(self, routes=None):
        self.calls = []
        self.routes = routes or {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.calls.append({"method": request.method, "path": request.url.path, "params": dict(request.url.params),
                           "headers": dict(request.headers), "json": body})
        for (method, prefix), answer in self.routes.items():
            if request.method == method and request.url.path.startswith(prefix):
                status, payload = answer(request, body) if callable(answer) else answer
                if isinstance(payload, (bytes, bytearray)):
                    return httpx.Response(status, content=payload)
                return httpx.Response(status, json=payload)
        return httpx.Response(404, json={"error": "no route"})

    def install(self):
        nango.set_transport(httpx.MockTransport(self))
        return self


def _sign(raw: bytes) -> str:
    return hmac.new(SIGNING.encode(), raw, hashlib.sha256).hexdigest()


# --- the beta list -------------------------------------------------------------

@pytest.mark.asyncio
async def test_gmail_is_invite_only_and_an_email_invite_binds_at_sign_in(store, customer):
    cid = customer.id
    features = await beta.features_for(store, cid)
    assert features == {"connections": True, "goals": True, "gmail": False}
    await store.create_beta_invite("gmail", email="Dana@Kerning.test", invited_by="platform admin")
    assert (await beta.features_for(store, cid, ["dana@kerning.test"]))["gmail"] is True
    assert (await beta.features_for(store, cid))["gmail"] is False  # by email only until bound
    assert await store.bind_beta_invites(cid, ["dana@kerning.test"]) == 1
    assert (await beta.features_for(store, cid))["gmail"] is True
    # A closed feature is closed even to the invited; an open one is open to all.
    await store.set_beta_feature_status("gmail", "closed")
    assert (await beta.features_for(store, cid))["gmail"] is False
    # Invites are idempotent per target.
    a = await store.create_beta_invite("gmail", customer_id=cid)
    b = await store.create_beta_invite("gmail", customer_id=cid)
    assert a["id"] == b["id"]


@pytest.mark.asyncio
async def test_the_catalog_says_who_may_connect_what(store, customer):
    view = await service.catalog_view(store, customer.id)
    assert view["configured"] is True
    by = {a["integration"]: a for a in view["apps"]}
    assert set(by) == {"google-drive", "google-calendar", "notion", "slack", "google-mail"}
    assert by["google-mail"]["available"] is False and by["google-mail"]["beta"] == "gmail"
    assert by["slack"]["available"] is True and by["slack"]["acts"] == [{"key": "post_message", "label": "Post a message"}]
    assert by["google-drive"]["acts"] == []


@pytest.mark.asyncio
async def test_a_session_is_minted_with_our_tags_and_gmail_needs_an_invite(store, customer):
    fake = _Nango({("POST", "/connect/sessions"): (201, {"data": {"token": "nst_123", "expires_at": "2026-10-24T12:00:00Z"}})}).install()
    out = await service.start_session(store, customer.id, "slack", operator_id="op_1")
    assert out["token"] == "nst_123"
    call = fake.calls[0]
    assert call["headers"]["authorization"] == "Bearer nango-secret-test"
    assert call["json"] == {"tags": {"end_user_id": customer.id, "organization_id": customer.id, "integration": "slack",
                                     "operator_id": "op_1"}, "allowed_integrations": ["slack"]}
    with pytest.raises(service.ConnectionRefused) as e:
        await service.start_session(store, customer.id, "google-mail")
    assert e.value.code == "beta_required" and e.value.status == 403
    with pytest.raises(service.ConnectionRefused) as e:
        await service.start_session(store, customer.id, "myspace")
    assert e.value.code == "unknown_app"
    await store.create_beta_invite("gmail", customer_id=customer.id)
    assert (await service.start_session(store, customer.id, "google-mail"))["token"] == "nst_123"


@pytest.mark.asyncio
async def test_without_nango_the_catalog_says_so_and_nothing_is_minted(store, customer, monkeypatch):
    monkeypatch.setattr(get_settings(), "nango_secret_key", None)
    assert (await service.catalog_view(store, customer.id))["configured"] is False
    with pytest.raises(service.ConnectionRefused) as e:
        await service.start_session(store, customer.id, "slack")
    assert e.value.code == "connections_unavailable" and e.value.status == 503


# --- the webhook -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_webhook_is_verified_before_anything_is_read_then_binds_and_marks_due(store, customer):
    from crystal_cache.endpoints.connections import nango_webhook

    class _Req:
        def __init__(self, raw, sig):
            self._raw = raw
            self.headers = {"x-nango-hmac-sha256": sig} if sig is not None else {}

        async def body(self):
            return self._raw

    auth = json.dumps({"type": "auth", "operation": "creation", "success": True, "connectionId": "nc_1",
                       "providerConfigKey": "slack", "tags": {"end_user_id": customer.id, "end_user_email": "dana@kerning.test"}}).encode()
    with pytest.raises(HTTPException) as e:
        await nango_webhook(_Req(auth, "deadbeef"), store)
    assert e.value.status_code == 401
    with pytest.raises(HTTPException):
        await nango_webhook(_Req(auth, None), store)
    assert await store.list_app_connections(customer.id) == []
    assert (await nango_webhook(_Req(auth, _sign(auth)), store)) == {"received": True}
    conns = await store.list_app_connections(customer.id)
    assert len(conns) == 1 and conns[0]["integration"] == "slack" and conns[0]["account_label"] == "dana@kerning.test"
    assert conns[0]["uses"] == {"watch": True, "remember": False, "act": False}
    # A re-authorization refreshes, never duplicates.
    await nango_webhook(_Req(auth, _sign(auth)), store)
    assert len(await store.list_app_connections(customer.id)) == 1
    await store.update_app_connection(customer.id, conns[0]["id"], sync_due=False)
    sync = json.dumps({"type": "sync", "connectionId": "nc_1", "providerConfigKey": "slack", "success": True,
                       "responseResults": {"added": 3}}).encode()
    await nango_webhook(_Req(sync, _sign(sync)), store)
    assert (await store.get_app_connection(customer.id, conns[0]["id"]))["sync_due"] is True
    # A tag naming a tenant that doesn't exist binds nothing.
    stray = json.dumps({"type": "auth", "success": True, "connectionId": "nc_2", "providerConfigKey": "slack",
                        "tags": {"end_user_id": "cus_nobody"}}).encode()
    await nango_webhook(_Req(stray, _sign(stray)), store)
    assert await store.find_app_connection_by_nango("nc_2") is None


# --- uses and rules ------------------------------------------------------------------

def test_remember_rules_are_strict_first_match_and_read_as_sentences():
    rules = service.validate_rules([
        {"decision": "deny", "field": "from", "match": "contains", "value": "noreply"},
        {"decision": "allow", "field": "subject", "match": "contains", "value": "invoice"},
        {"decision": "allow", "field": "channel", "match": "equals", "value": "C123"},
    ])
    invoice = {"author": "billing@kerning.test", "title": "Invoice 42", "snippet": "", "body": "", "meta": {}}
    bot = {"author": "noreply@stripe.test", "title": "Your invoice", "snippet": "", "body": "", "meta": {}}
    chat = {"author": "theo", "title": "", "snippet": "hi", "body": "", "meta": {"channel": "C123"}}
    other = {"author": "x", "title": "lunch", "snippet": "", "body": "", "meta": {}}
    assert service.should_remember(invoice, rules)["value"] == "invoice"
    assert service.should_remember(bot, rules) is None  # the deny above it wins
    assert service.should_remember(chat, rules)["field"] == "channel"
    assert service.should_remember(other, rules) is None  # no match: not remembered
    assert service.rule_text(rules[0]) == 'Never remember items from "noreply" (contains).'
    assert service.rule_text(rules[1]) == 'Remember items whose subject contains "invoice".'
    for bad in ([{"decision": "maybe", "field": "from", "value": "x"}], [{"decision": "allow", "field": "body", "value": "x"}],
                [{"decision": "allow", "field": "from", "value": ""}], [{"decision": "allow", "field": "from", "value": "x", "extra": 1}],
                "nope"):
        with pytest.raises(service.ConnectionRefused) as e:
            service.validate_rules(bad)
        assert e.value.code == "rules_invalid"


@pytest.mark.asyncio
async def test_uses_have_sensible_limits(store, customer):
    drive = await store.upsert_app_connection(customer.id, integration="google-drive", nango_connection_id="nc_d")
    with pytest.raises(service.ConnectionRefused) as e:
        await service.set_uses(store, customer.id, drive["id"], uses={"act": True})
    assert e.value.code == "uses_invalid"  # Drive has nothing to do
    with pytest.raises(service.ConnectionRefused):
        await service.set_uses(store, customer.id, drive["id"], uses={"watch": False, "remember": True})
    out = await service.set_uses(store, customer.id, drive["id"], uses={"remember": True},
                                 rules=[{"decision": "allow", "field": "folder", "match": "contains", "value": "Clients"}],
                                 cadence_minutes=30)
    assert out["uses"] == {"watch": True, "remember": True, "act": False} and out["watch_cadence_minutes"] == 30
    assert out["rules_text"] == ['Remember files in folder "Clients" (contains).']
    paused = await service.set_uses(store, customer.id, drive["id"], paused=True)
    assert paused["status"] == "paused"
    with pytest.raises(service.ConnectionRefused):
        await service.set_uses(store, customer.id, drive["id"], cadence_minutes=1)


# --- normalizing records -----------------------------------------------------------

def test_records_from_every_app_normalize_without_a_fixed_schema():
    gmail = catalog.normalize(catalog.get("google-mail"), {
        "id": "m1", "subject": "Invoice 42", "from": "billing@kerning.test", "date": "Fri, 23 Oct 2026 14:05:00 +0000",
        "body": "Please find attached.", "threadId": "t1", "_nango_metadata": {"last_action": "ADDED", "cursor": "c1"}})
    assert gmail["external_id"] == "m1" and gmail["title"] == "Invoice 42" and gmail["author"] == "billing@kerning.test"
    assert gmail["occurred_at"] == datetime(2026, 10, 23, 14, 5, tzinfo=timezone.utc) and gmail["meta"]["threadId"] == "t1"
    slack = catalog.normalize(catalog.get("slack"), {"id": "s1", "text": "ship it", "user": {"real_name": "Theo"},
                                                     "ts": "1761228300.000100", "channel_id": "C123"})
    assert slack["author"] == "Theo" and slack["occurred_at"].year == 2025 and slack["meta"]["channel_id"] == "C123"
    cal = catalog.normalize(catalog.get("google-calendar"), {"id": "e1", "summary": "Halden call",
                                                             "start": {"dateTime": "2026-10-29T15:00:00-07:00"},
                                                             "htmlLink": "https://cal/e1"})
    assert cal["title"] == "Halden call" and cal["url"] == "https://cal/e1" and cal["occurred_at"].hour == 22
    gone = catalog.normalize(catalog.get("google-drive"), {"id": "f1", "_nango_metadata": {"last_action": "DELETED"}})
    assert gone["action"] == "DELETED"


# --- the sync worker -------------------------------------------------------------------

def _records_route(pages):
    """pages: list of record lists served in order regardless of cursor."""
    state = {"i": 0}

    def answer(request, body):
        i = state["i"]
        state["i"] += 1
        recs = pages[i] if i < len(pages) else []
        nxt = "next" if i + 1 < len(pages) else None
        return 200, {"records": recs, "next_cursor": nxt}
    return answer


def _config(integration, syncs):
    return 200, [{"providerConfigKey": integration, "syncs": syncs}]


@pytest.mark.asyncio
async def test_a_sync_watches_everything_but_remembers_only_what_the_rules_allow(store, customer):
    cid = customer.id
    conn = await store.upsert_app_connection(cid, integration="google-mail", nango_connection_id="nc_m")
    await service.set_uses(store, cid, conn["id"], uses={"remember": True}, rules=[
        {"decision": "allow", "field": "subject", "match": "contains", "value": "invoice"}])
    conn = await store.get_app_connection(cid, conn["id"])
    fake = _Nango({
        ("GET", "/scripts/config"): _config("google-mail", [{"name": "emails", "returns": ["GmailEmail"]}]),
        ("GET", "/records"): _records_route([[
            {"id": "m1", "subject": "Invoice 42", "from": "billing@x.test", "body": "Due Nov 1.",
             "_nango_metadata": {"last_action": "ADDED", "cursor": "c1"}},
            {"id": "m2", "subject": "Lunch?", "from": "theo@x.test", "body": "Thursday?",
             "_nango_metadata": {"last_action": "ADDED", "cursor": "c2"}},
        ]]),
    }).install()
    ingested = []

    async def ingest(store_, encoder, vs, fvs, llm, holder, env):
        ingested.append((holder.customer_id, holder.review_mode, env.source_uri, env.payload_bytes.decode()))

    out = await app_sync.sync_connection(store, conn, ingest=ingest)
    assert out == {"seen": 2, "deleted": 0, "remembered": 1, "retired": 0, "failed": 0}
    assert ingested[0][:3] == (cid, "auto", f"nango://google-mail/{conn['id']}/m1") and "Invoice 42" in ingested[0][3]
    items = await store.list_watched_items(cid, connection_id=conn["id"])
    assert {i["external_id"]: bool(i["remembered_document_id"]) for i in items} == {"m1": True, "m2": False}
    assert all(i["expires_at"] for i in items)
    after = await store.get_app_connection(cid, conn["id"])
    assert after["cursors"] == {"GmailEmail": "c2"} and after["sync_due"] is False and after["last_synced_at"]
    rec_call = [c for c in fake.calls if c["path"] == "/records"][0]
    assert rec_call["headers"]["connection-id"] == "nc_m" and rec_call["headers"]["provider-config-key"] == "google-mail"
    assert rec_call["params"]["model"] == "GmailEmail" and "cursor" not in rec_call["params"]


@pytest.mark.asyncio
async def test_a_walled_tenant_still_watches_but_remembers_nothing(store, customer, monkeypatch):
    cid = customer.id
    conn = await store.upsert_app_connection(cid, integration="slack", nango_connection_id="nc_s")
    await service.set_uses(store, cid, conn["id"], uses={"remember": True},
                           rules=[{"decision": "allow", "field": "text", "match": "contains", "value": "ship"}])
    conn = await store.get_app_connection(cid, conn["id"])
    _Nango({("GET", "/scripts/config"): _config("slack", [{"name": "messages", "returns": [{"name": "SlackMessage"}]}]),
            ("GET", "/records"): _records_route([[{"id": "s1", "text": "ship it", "_nango_metadata": {"cursor": "k1"}}]])}).install()

    async def walled(*a, **k):
        raise HTTPException(status_code=402, detail="cap")
    monkeypatch.setattr("crystal_cache.ingress.auth.require_write_capacity", walled)
    calls = []

    async def ingest(*a):
        calls.append(a)
    out = await app_sync.sync_connection(store, conn, ingest=ingest)
    assert out["seen"] == 1 and out["remembered"] == 0 and calls == []


@pytest.mark.asyncio
async def test_a_deleted_record_retires_the_memory_made_from_it(store, customer):
    from crystal_cache.infrastructure.schema import CrystalRow
    cid = customer.id
    conn = await store.upsert_app_connection(cid, integration="notion", nango_connection_id="nc_n")
    uri = app_sync.source_uri("notion", conn["id"], "p1")
    async with store.session() as s:
        s.add(CrystalRow(id="c_page", customer_id=cid, crystal_type="customer:legacy", summary_vector=[], source_uri=uri))
        s.add(CrystalRow(id="c_other", customer_id=cid, crystal_type="customer:legacy", summary_vector=[], source_uri="upload://x"))
    await store.upsert_watched_item(cid, conn["id"], kind="page", external_id="p1", title="Roadmap")
    _Nango({("GET", "/scripts/config"): _config("notion", [{"name": "content-metadata", "returns": ["ContentMetadata"]},
                                                           {"name": "users", "returns": ["User"]}]),
            ("GET", "/records"): _records_route([[{"id": "p1", "_nango_metadata": {"last_action": "DELETED", "cursor": "z"}}]])}).install()
    out = await app_sync.sync_connection(store, conn)
    assert out["deleted"] == 1 and out["retired"] == 1
    assert await store.get_crystal("c_page") is None and await store.get_crystal("c_other") is not None
    assert await store.list_watched_items(cid, connection_id=conn["id"]) == []
    assert len(await store.list_watched_items(cid, connection_id=conn["id"], include_deleted=True)) == 1
    assert await app_sync.models_for("notion") == ["ContentMetadata"]  # the users sync is skipped


@pytest.mark.asyncio
async def test_an_expired_sign_in_parks_the_connection_for_reconnect(store, customer):
    cid = customer.id
    conn = await store.upsert_app_connection(cid, integration="slack", nango_connection_id="nc_x")
    _Nango({("GET", "/scripts/config"): _config("slack", [{"name": "messages", "returns": ["SlackMessage"]}]),
            ("GET", "/records"): (401, {"error": "invalid_credentials"})}).install()
    out = await app_sync.sync_connection(store, conn)
    assert out["error"] == "needs_reauth"
    after = await store.get_app_connection(cid, conn["id"])
    assert after["status"] == "needs_reauth" and "reconnect" in after["last_error"]
    assert await store.list_due_app_connections() == []


@pytest.mark.asyncio
async def test_watched_items_expire_after_seven_days(store, customer):
    cid = customer.id
    conn = await store.upsert_app_connection(cid, integration="slack", nango_connection_id="nc_e")
    await store.upsert_watched_item(cid, conn["id"], kind="message", external_id="old", ttl_days=7)
    now = datetime.now(timezone.utc)
    assert await store.purge_expired_watched_items(now=now + timedelta(days=6)) == 0
    assert await store.purge_expired_watched_items(now=now + timedelta(days=7, minutes=1)) == 1


# --- leaving ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_disconnect_and_purge_revoke_at_nango_and_erase_keeps_the_settings(store, customer):
    cid = customer.id
    fake = _Nango({("DELETE", "/connection/"): (204, b"")}).install()
    a = await store.upsert_app_connection(cid, integration="slack", nango_connection_id="nc_a")
    await store.upsert_app_connection(cid, integration="notion", nango_connection_id="nc_b")
    await store.upsert_watched_item(cid, a["id"], kind="message", external_id="m")
    # "Erase all memories" keeps the connections, drops what they showed.
    await store.erase_tenant_bank(cid)
    assert len(await store.list_app_connections(cid)) == 2
    assert await store.list_watched_items(cid) == []
    assert await service.remove(store, cid, a["id"]) is True
    assert fake.calls[0]["path"] == "/connection/nc_a" and fake.calls[0]["params"] == {"provider_config_key": "slack"}
    assert await service.revoke_all_connections(store, cid) == 1
    assert fake.calls[-1]["path"] == "/connection/nc_b"


# --- Q118 A: only what is configured in Nango is offered -----------------------------------

@pytest.mark.asyncio
async def test_the_catalog_offers_only_apps_configured_in_nango(store, customer):
    from crystal_cache.goals.planner import app_statuses, planner_system
    service.reset_for_tests()
    fake = _Nango({("GET", "/integrations"): (200, {"data": [
        {"unique_key": "slack", "provider": "slack"}, {"unique_key": "github", "provider": "github"},
        {"unique_key": "something-we-do-not-know", "provider": "x"}]})}).install()
    view = await service.catalog_view(store, customer.id)
    assert [a["integration"] for a in view["apps"]] == ["slack", "github"]
    gh = next(a for a in view["apps"] if a["integration"] == "github")
    assert gh["acts"] == [] and gh["group"] == "Engineering" and gh["available"] is True
    # Cached: a second view does not ask Nango again.
    await service.catalog_view(store, customer.id)
    assert len([c for c in fake.calls if c["path"] == "/integrations"]) == 1
    # Not configured: refused before any session is minted.
    with pytest.raises(service.ConnectionRefused) as e:
        await service.start_session(store, customer.id, "notion")
    assert e.value.code == "app_not_configured"
    # An app already connected stays listed (to manage or remove) even if it is no longer configured.
    await store.upsert_app_connection(customer.id, integration="notion", nango_connection_id="nc_n")
    view = await service.catalog_view(store, customer.id)
    notion = next(a for a in view["apps"] if a["integration"] == "notion")
    assert notion["available"] is False and len(notion["connections"]) == 1
    # The planner is offered exactly what can be connected.
    offered = await service.available_apps(store, customer.id)
    statuses = app_statuses(await store.list_app_connections(customer.id), {a.integration for a in offered})
    assert statuses["slack"] == "ready" and statuses["github"] == "ready" and statuses["dropbox"] == "setup"
    system = planner_system(15, statuses)
    assert "- slack: Slack (messages; acts: post_message; ready to connect)" in system
    assert "- dropbox: Dropbox (files; acts: none; not set up on this server yet" in system


@pytest.mark.asyncio
async def test_nango_unreachable_keeps_the_last_known_catalog(store, customer):
    service.reset_for_tests()
    _Nango({("GET", "/integrations"): (200, {"data": [{"unique_key": "dropbox"}]})}).install()
    assert await service.configured_integrations() == {"dropbox"}
    service._configured["at"] = 0.0  # expired
    _Nango({("GET", "/integrations"): (500, {"error": "down"})}).install()
    assert await service.configured_integrations() == {"dropbox"}
