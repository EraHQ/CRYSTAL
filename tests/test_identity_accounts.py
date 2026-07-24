"""Accounts Phase A (2026-07-06) — hosted identity + tenant admin guard.

Covers the ratified design end to end at the policy layer:
  - Firebase-JWT principal resolution (verification seam monkeypatched),
  - admin bootstrap via CC_PLATFORM_ADMIN_EMAILS at first login,
  - D4 presence-as-switch (no CC_FIREBASE_PROJECT_ID => JWTs never valid),
  - the stage-2 tenant guard: own-tenant console routes allowed, foreign
    tenants 404 (never an existence oracle), pinned read-only cognition /
    metacognition views, everything else still platform-admin-only,
  - route-level pinning (list override + detail 404 on foreign env),
  - GET /v1/me for every principal kind.

The CROSS-TENANT TRIPWIRES here are the acceptance gate for Phase A
(B4 posture): the negative cases are asserted as hard as the positives.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import pytest

from crystal_cache.config import Settings
from crystal_cache.ingress import auth as auth_mod


ADMIN_EMAIL = "anthony@erahq.ai"
FAKE_JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1In0.sig"  # shape-valid only


def _use(monkeypatch, **kw) -> None:
    base = dict(environment="development", admin_api_key="",
                api_key_pepper="", firebase_project_id="proj-x",
                platform_admin_emails=ADMIN_EMAIL)
    base.update(kw)
    settings = Settings(**base)
    monkeypatch.setattr(auth_mod, "get_settings", lambda: settings)


def _claims(uid: str, email: str) -> dict:
    return {"sub": uid, "email": email}


def _verify_as(monkeypatch, claims) -> None:
    """Stub the JWT verification seam: any token yields these claims."""
    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt", lambda tok, proj: claims
    )


@pytest.fixture
async def tenants(store):
    """Two tenants (A, B; Key A carried on the created Customer) + an
    owner user each. Uses the house `store` fixture from conftest."""
    a = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="x")
    b = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="x")
    ua = await store.create_user("uid_a", "a@a.test", a.id, "owner")
    ub = await store.create_user("uid_b", "b@b.test", b.id, "owner")
    return {"a": a, "b": b, "key_a": a.api_key, "key_b": b.api_key,
            "ua": ua, "ub": ub}


# --- JWT shape discrimination (D2) -------------------------------------------

@pytest.mark.parametrize("token,expected", [
    (FAKE_JWT, True),
    ("eyJx.y", False),                # 2 segments
    ("cc_sk_abc123", False),          # Key A shape
    ("ck_task_abc", False),           # task key shape
    ("", False),
])
def test_jwt_shape_discrimination(token, expected):
    assert auth_mod._looks_like_firebase_jwt(token) is expected


# --- resolve_firebase_user ----------------------------------------------------

async def test_identity_disabled_means_jwts_never_valid(monkeypatch, store):
    """D4: no CC_FIREBASE_PROJECT_ID => the JWT path simply doesn't exist.
    Self-host carries zero new auth surface."""
    _use(monkeypatch, firebase_project_id="")
    _verify_as(monkeypatch, _claims("uid_a", "a@a.test"))  # even a "valid" one
    assert await auth_mod.resolve_firebase_user(store, FAKE_JWT) is None


async def test_invalid_jwt_resolves_to_none(monkeypatch, store):
    _use(monkeypatch)
    monkeypatch.setattr(auth_mod, "_verify_firebase_jwt",
                        lambda tok, proj: None)
    assert await auth_mod.resolve_firebase_user(store, FAKE_JWT) is None


async def test_known_uid_resolves_to_user(monkeypatch, store, tenants):
    _use(monkeypatch)
    _verify_as(monkeypatch, _claims("uid_a", "a@a.test"))
    user = await auth_mod.resolve_firebase_user(store, FAKE_JWT)
    assert user is not None and user.customer_id == tenants["a"].id
    assert user.role == "owner"


async def test_admin_bootstrap_provisions_platform_admin(monkeypatch, store):
    """First login of an allowlisted email creates the platform_admin
    account (customer_id None) — the ratified bootstrap."""
    _use(monkeypatch)
    _verify_as(monkeypatch, _claims("uid_root", ADMIN_EMAIL))
    user = await auth_mod.resolve_firebase_user(store, FAKE_JWT)
    assert user is not None
    assert user.role == "platform_admin" and user.customer_id is None
    # Idempotent on the second login (row exists, no re-create).
    again = await auth_mod.resolve_firebase_user(store, FAKE_JWT)
    assert again is not None and again.id == user.id


async def test_unknown_nonallowlisted_jwt_is_denied(monkeypatch, store):
    """A valid JWT for an unprovisioned, non-allowlisted account never
    conjures a tenant — signup is a Phase B/C flow, not this resolver."""
    _use(monkeypatch)
    _verify_as(monkeypatch, _claims("uid_stranger", "who@else.test"))
    assert await auth_mod.resolve_firebase_user(store, FAKE_JWT) is None


# --- stage-2 tenant guard: the cross-tenant tripwires -------------------------

async def _guard(store, method, path, bearer):
    return await auth_mod.tenant_admin_error(
        method, path, f"Bearer {bearer}" if bearer else None, store)


async def test_keya_allows_own_tenant_console_route(monkeypatch, store, tenants):
    _use(monkeypatch)
    err, pin = await _guard(
        store, "GET",
        f"/admin/api/customers/{tenants['a'].id}/crystals", tenants["key_a"])
    assert err is None and pin is None


async def test_keya_foreign_tenant_console_route_is_404(
        monkeypatch, store, tenants):
    """TRIPWIRE: tenant A probing tenant B's console gets 404 — the same
    shape as a nonexistent customer (never an existence oracle)."""
    _use(monkeypatch)
    err, pin = await _guard(
        store, "GET",
        f"/admin/api/customers/{tenants['b'].id}/crystals", tenants["key_a"])
    assert err == (404, "Customer not found") and pin is None


async def test_jwt_owner_allows_own_and_404s_foreign(
        monkeypatch, store, tenants):
    _use(monkeypatch)
    _verify_as(monkeypatch, _claims("uid_a", "a@a.test"))
    err, _ = await _guard(
        store, "GET",
        f"/admin/api/customers/{tenants['a'].id}/query_logs", FAKE_JWT)
    assert err is None
    err, _ = await _guard(
        store, "GET",
        f"/admin/api/customers/{tenants['b'].id}/query_logs", FAKE_JWT)
    assert err == (404, "Customer not found")


async def test_tenant_cannot_list_all_customers(monkeypatch, store, tenants):
    """TRIPWIRE: the cross-tenant customer list stays platform-only."""
    _use(monkeypatch)
    err, _ = await _guard(store, "GET", "/admin/api/customers",
                          tenants["key_a"])
    assert err is not None and err[0] == 401


async def test_tenant_readable_cognition_list_is_pinned(
        monkeypatch, store, tenants):
    """Amended D3: tenants may read cognition environments — pinned."""
    _use(monkeypatch)
    err, pin = await _guard(store, "GET", "/admin/api/cognition/environments",
                            tenants["key_a"])
    assert err is None and pin == tenants["a"].id


async def test_tenant_readable_detail_is_pinned(
        monkeypatch, store, tenants):
    _use(monkeypatch)
    err, pin = await _guard(
        store, "GET", "/admin/api/cognition/environments/env_123",
        tenants["key_a"])
    assert err is None and pin == tenants["a"].id

    # C1 (2026-07-08): System Critiques are SUPER-ADMIN only — the
    # substrate endpoints left the tenant allowlist.
    for path in (
        "/admin/api/metacognition/substrate-observations",
        "/admin/api/metacognition/substrate-observations/grouped",
    ):
        err, pin = await _guard(store, "GET", path, tenants["key_a"])
        assert err is not None and err[0] == 401, path


async def test_tenant_readable_is_get_only(monkeypatch, store, tenants):
    """TRIPWIRE: the read allowlist never admits mutations."""
    _use(monkeypatch)
    err, _ = await _guard(store, "POST", "/admin/api/cognition/environments",
                          tenants["key_a"])
    assert err is not None and err[0] == 401


async def test_other_admin_surfaces_stay_platform_only(
        monkeypatch, store, tenants):
    """TRIPWIRE: anything not tenant-pathed or allowlisted denies."""
    _use(monkeypatch)
    for path in ("/admin/api/customers", "/admin/api/metacognition/state"):
        err, _ = await _guard(store, "GET", path, tenants["key_a"])
        assert err is not None and err[0] == 401, path


async def test_platform_admin_user_passes_everything(
        monkeypatch, store, tenants):
    """A platform_admin USER (JWT) is equivalent to the static key."""
    _use(monkeypatch)
    await store.create_user("uid_root", ADMIN_EMAIL, None, "platform_admin")
    _verify_as(monkeypatch, _claims("uid_root", ADMIN_EMAIL))
    for path in ("/admin/api/customers",
                 f"/admin/api/customers/{tenants['b'].id}/crystals",
                 "/admin/api/metacognition/state"):
        err, pin = await _guard(store, "GET", path, FAKE_JWT)
        assert err is None and pin is None, path


async def test_no_bearer_is_401(monkeypatch, store):
    _use(monkeypatch)
    err, _ = await _guard(store, "GET", "/admin/api/customers", None)
    assert err is not None and err[0] == 401


async def test_unknown_key_is_401(monkeypatch, store):
    _use(monkeypatch)
    err, _ = await _guard(store, "GET",
                          "/admin/api/customers/whoever/crystals",
                          "cc_sk_not_a_real_key")
    assert err is not None and err[0] == 401


# --- route-level pinning ------------------------------------------------------

class _PinnedReq:
    """Request stand-in carrying the middleware's stashed pin."""
    class _State:
        pass

    def __init__(self, pin=None):
        self.state = self._State()
        if pin is not None:
            self.state.tenant_pin = pin


async def test_cognition_list_override_ignores_query_param(monkeypatch):
    """TRIPWIRE: a pinned tenant asking for customer_id=B still gets A.
    (S9: the endpoints read cognition_runs via the store now — the
    tripwire's INTENT is unchanged, the seam moved.)"""
    from crystal_cache.cognition import api as cog_api
    seen = {}

    class _FakeStore:
        async def count_open_critiques_by_run(self, run_ids):
            return {}

        async def list_cognition_runs(self, customer_id="", **kw):
            seen["cid"] = customer_id
            return []
    monkeypatch.setattr(cog_api, "get_metadata_store", lambda: _FakeStore())
    await cog_api.list_environments(_PinnedReq(pin="cust_A"),
                                    customer_id="cust_B")
    assert seen["cid"] == "cust_A"


async def test_cognition_detail_foreign_env_is_404(monkeypatch):
    from crystal_cache.cognition import api as cog_api

    class _FakeStore:
        async def count_open_critiques_by_run(self, run_ids):
            return {}

        async def get_cognition_run(self, run_id):
            return {"id": run_id, "customer_id": "cust_B"}
    monkeypatch.setattr(cog_api, "get_metadata_store", lambda: _FakeStore())
    resp = await cog_api.get_environment_detail(
        _PinnedReq(pin="cust_A"), "env_1")
    assert resp.status_code == 404
    # Unpinned (platform admin) sees it fine.
    resp = await cog_api.get_environment_detail(_PinnedReq(), "env_1")
    assert resp.status_code == 200


# --- GET /v1/me ---------------------------------------------------------------

class _MeReq:
    def __init__(self, authorization=None):
        self.headers = ({"authorization": authorization}
                        if authorization else {})


async def test_me_admin_key(monkeypatch, store):
    _use(monkeypatch, admin_api_key="cc_sk_admin_root")
    from crystal_cache.endpoints.me import get_me
    out = await get_me(_MeReq("Bearer cc_sk_admin_root"), store)
    assert out["kind"] == "platform_admin_key"
    assert out["role"] == "platform_admin"


async def test_me_jwt_user(monkeypatch, store, tenants):
    _use(monkeypatch)
    _verify_as(monkeypatch, _claims("uid_a", "a@a.test"))
    from crystal_cache.endpoints.me import get_me
    out = await get_me(_MeReq(f"Bearer {FAKE_JWT}"), store)
    assert out == {"kind": "user", "role": "owner",
                   "customer_id": tenants["a"].id,
                   "user_id": "uid_a", "email": "a@a.test"}


async def test_me_key_a(monkeypatch, store, tenants):
    _use(monkeypatch)
    from crystal_cache.endpoints.me import get_me
    out = await get_me(_MeReq(f"Bearer {tenants['key_a']}"), store)
    assert out["kind"] == "customer_key"
    assert out["customer_id"] == tenants["a"].id and out["role"] == "owner"


async def test_me_unknown_401(monkeypatch, store):
    _use(monkeypatch)
    from crystal_cache.endpoints.me import get_me
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        await get_me(_MeReq("Bearer cc_sk_nope"), store)
    assert e.value.status_code == 401
    with pytest.raises(HTTPException) as e:
        await get_me(_MeReq(None), store)
    assert e.value.status_code == 401


# --- store CRUD ---------------------------------------------------------------

async def test_user_store_roundtrip_and_onboarding(store, tenants):
    u = await store.get_user_by_email("a@a.test")
    assert u is not None and u.id == "uid_a"
    u2 = await store.update_user_onboarding(
        "uid_a", industry="healthcare", building="agents", experience="pro")
    assert (u2.industry, u2.building, u2.experience) == (
        "healthcare", "agents", "pro")
    assert await store.get_user_by_id("uid_missing") is None


# --- tenant-console read sweep (2026-07-07): Cognition/Conflicts/Bank tabs ------

def test_tenant_readable_covers_console_tabs():
    """The live incident: a signed-in tenant 401'd on its own Cognition,
    Conflicts, and Bank-detail reads. These are tenant-readable (GET,
    pinned) now; writes stay platform-only."""
    from crystal_cache.ingress.auth import _tenant_readable

    for p in ("/admin/api/push-queue", "/admin/api/cognition-tasks",
              "/admin/api/knowledge-gaps", "/admin/api/conflicts",
              "/admin/api/backlog", "/admin/api/crystal_types",
              "/admin/api/crystals/crys_abc123"):
        assert _tenant_readable("GET", p), p
        assert not _tenant_readable("POST", p), p  # reads only

    # Writes on those surfaces remain platform-admin-only.
    assert not _tenant_readable("POST", "/admin/api/push-queue/x/approve")
    # Conflict resolve became a tenant write 2026-07-23 (the curation
    # gate incident: valid-token 401s in production) — the negative
    # pin moved to a surface that stays platform-only.
    assert _tenant_readable("POST", "/admin/api/conflicts/x/resolve")


async def test_tenant_gets_pin_for_console_reads(monkeypatch, store):
    """tenant_admin_error allows the new reads WITH a pin (the handler
    override is what stops cross-tenant snooping via ?customer_id=)."""
    from crystal_cache.ingress import auth as auth_mod
    from crystal_cache.ingress.auth import tenant_admin_error

    c = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    err, pin = await tenant_admin_error(
        "GET", "/admin/api/push-queue", f"Bearer {c.api_key}", store)
    assert err is None and pin == c.id

    err, pin = await tenant_admin_error(
        "GET", "/admin/api/crystals/crys_whatever",
        f"Bearer {c.api_key}", store)
    assert err is None and pin == c.id

    # Non-listed admin surface still denied.
    err, pin = await tenant_admin_error(
        "GET", "/admin/api/customers", f"Bearer {c.api_key}", store)
    assert err is not None and err[0] == 401


async def test_crystal_detail_enforces_pin_ownership(store):
    """A pinned tenant opening a FOREIGN crystal gets the identical 404 as
    a nonexistent one — never an existence oracle."""
    from types import SimpleNamespace
    from crystal_cache.endpoints.admin import admin_get_crystal

    a = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    b = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    from crystal_cache.models.crystal import Crystal
    crystal = Crystal(id="crys_owned_by_b", customer_id=b.id,
                      name="Topic X", summary_vector=[0.0] * 8,
                      routing_vector=[0.0] * 8)
    await store.upsert_crystal(crystal)
    crystal_id = crystal.id

    def _req(pin):
        return SimpleNamespace(state=SimpleNamespace(tenant_pin=pin))

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as e:
        await admin_get_crystal(_req(a.id), crystal_id, store)
    assert e.value.status_code == 404

    out = await admin_get_crystal(_req(b.id), crystal_id, store)  # own: OK
    assert out.status_code == 200

    out = await admin_get_crystal(_req(None), crystal_id, store)  # admin: OK
    assert out.status_code == 200


# --- K1 (2026-07-08): console access to /v1 document + subscription routes ----

class _K1Req:
    def __init__(self, authorization=None):
        self.headers = ({"authorization": authorization}
                        if authorization else {})
        class _S: pass
        self.state = _S()


async def test_or_console_without_param_is_key_a_only(store, tenants):
    """No ?customer_id= → require_customer verbatim: Key A resolves, a
    console JWT alone does not (SDK behavior byte-identical)."""
    from crystal_cache.ingress.auth import require_customer_or_console
    from fastapi import HTTPException

    c = await require_customer_or_console(
        _K1Req(authorization=f"Bearer {tenants['key_a']}"), store)
    assert c.id == tenants["a"].id

    try:
        await require_customer_or_console(
            _K1Req(authorization="Bearer not-a-key"), store)
        assert False, "expected 401"
    except HTTPException as e:
        assert e.status_code == 401


async def test_or_console_with_param_enforces_self_or_admin(
        store, tenants, monkeypatch):
    """?customer_id= engages self_or_admin: the customer's own Key A
    passes; a foreign customer's key gets the uniform 404."""
    from crystal_cache.ingress.auth import require_customer_or_console
    from fastapi import HTTPException

    c = await require_customer_or_console(
        _K1Req(authorization=f"Bearer {tenants['key_a']}"), store,
        customer_id=tenants["a"].id)
    assert c.id == tenants["a"].id

    try:
        await require_customer_or_console(
            _K1Req(authorization=f"Bearer {tenants['key_b']}"), store,
            customer_id=tenants["a"].id)
        assert False, "expected 404"
    except HTTPException as e:
        assert e.status_code == 404
