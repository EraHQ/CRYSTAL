"""Accounts Phase C backend enablers (2026-07-06) — signup provisioning,
onboarding capture, the inference_mode toggle, and the tenant spend view.

Signup is the front door of the managed product (ratified plan): a valid
first-time Firebase JWT provisions Customer (inference_mode=managed,
default model, NO Key B) + owner User, returning Key A exactly once.
The negative cases are asserted as hard as the positives.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from crystal_cache.config import Settings
from crystal_cache.endpoints.me import (
    SIGNUP_DEFAULT_MODEL,
    signup,
    update_onboarding,
)
from crystal_cache.endpoints.customers import update_inference_mode
from crystal_cache.endpoints.admin import get_customer_spend
from crystal_cache.ingress import auth as auth_mod


ADMIN_EMAIL = "anthony@erahq.ai"
FAKE_JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1In0.sig"


def _use(monkeypatch, **kw) -> None:
    base = dict(environment="development", admin_api_key="",
                api_key_pepper="", firebase_project_id="proj-x",
                platform_admin_emails=ADMIN_EMAIL)
    base.update(kw)
    settings = Settings(**base)
    monkeypatch.setattr(auth_mod, "get_settings", lambda: settings)
    import crystal_cache.endpoints.me as me_mod
    monkeypatch.setattr(me_mod, "get_settings", lambda: settings)


def _verify_as(monkeypatch, uid, email) -> None:
    import time
    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt",
        # Fresh auth_time by default: most tests model a just-signed-in
        # session; the step-up tripwire overrides this with a stale one.
        lambda tok, proj: {"sub": uid, "email": email,
                           "auth_time": int(time.time())},
    )


class _Req:
    def __init__(self, authorization=None, body=None):
        self.headers = ({"authorization": authorization}
                        if authorization else {})
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


# --- signup --------------------------------------------------------------------

async def test_signup_provisions_managed_tenant_and_owner(monkeypatch, store):
    _use(monkeypatch)
    _verify_as(monkeypatch, "uid_new", "new@user.test")
    out = await signup(_Req(f"Bearer {FAKE_JWT}",
                            body={"industry": "healthcare"}), store)
    assert out["created"] is True and out["role"] == "owner"
    assert out["api_key"], "Key A must be revealed exactly once"

    customer = await store.get_customer_by_id(out["customer_id"])
    assert customer.inference_mode == "managed"
    assert customer.model_routing_config.model_id == SIGNUP_DEFAULT_MODEL
    assert (customer.model_routing_config.api_key_ref or "") == ""

    user = await store.get_user_by_id("uid_new")
    assert user.customer_id == out["customer_id"]
    assert user.industry == "healthcare"


async def test_signup_is_idempotent_and_never_reveals_key_again(
        monkeypatch, store):
    _use(monkeypatch)
    _verify_as(monkeypatch, "uid_new", "new@user.test")
    first = await signup(_Req(f"Bearer {FAKE_JWT}"), store)
    again = await signup(_Req(f"Bearer {FAKE_JWT}"), store)
    assert again["created"] is False
    assert again["customer_id"] == first["customer_id"]
    assert again["api_key"] is None  # TRIPWIRE: one-time reveal only


async def test_signup_admin_email_bootstraps_no_tenant(monkeypatch, store):
    """The platform root never gets a tenant minted by signup."""
    _use(monkeypatch)
    _verify_as(monkeypatch, "uid_root", ADMIN_EMAIL)
    out = await signup(_Req(f"Bearer {FAKE_JWT}"), store)
    assert out["role"] == "platform_admin"
    assert out["customer_id"] is None and out["api_key"] is None


async def test_signup_disabled_without_firebase_config(monkeypatch, store):
    """D4 presence-as-switch: self-host has no signup surface at all."""
    _use(monkeypatch, firebase_project_id="")
    with pytest.raises(HTTPException) as e:
        await signup(_Req(f"Bearer {FAKE_JWT}"), store)
    assert e.value.status_code == 404


async def test_signup_rejects_invalid_jwt(monkeypatch, store):
    _use(monkeypatch)
    monkeypatch.setattr(auth_mod, "_verify_firebase_jwt",
                        lambda tok, proj: None)
    with pytest.raises(HTTPException) as e:
        await signup(_Req(f"Bearer {FAKE_JWT}"), store)
    assert e.value.status_code == 401
    with pytest.raises(HTTPException) as e:
        await signup(_Req(None), store)
    assert e.value.status_code == 401


# --- onboarding ------------------------------------------------------------------

async def test_onboarding_updates_signed_in_user(monkeypatch, store):
    _use(monkeypatch)
    _verify_as(monkeypatch, "uid_new", "new@user.test")
    await signup(_Req(f"Bearer {FAKE_JWT}"), store)
    out = await update_onboarding(
        _Req(f"Bearer {FAKE_JWT}",
             body={"building": "agents", "experience": "pro"}),
        store,
    )
    assert out["building"] == "agents" and out["experience"] == "pro"


async def test_onboarding_rejects_key_principals(monkeypatch, store):
    """Key-based callers have no user row — JWT only."""
    _use(monkeypatch)
    with pytest.raises(HTTPException) as e:
        await update_onboarding(_Req("Bearer cc_sk_whatever",
                                     body={"industry": "x"}), store)
    assert e.value.status_code == 401


# --- inference_mode toggle ---------------------------------------------------------

class _AuthedReq(_Req):
    """Request whose bearer is a customer's own Key A (self-or-admin)."""


async def test_inference_mode_flip_to_managed_and_back(monkeypatch, store):
    _use(monkeypatch)
    c = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    req = _AuthedReq(f"Bearer {c.api_key}",
                     body={"inference_mode": "managed"})
    out = await update_inference_mode(c.id, req, store)
    assert (await store.get_customer_by_id(c.id)).inference_mode == "managed"

    req = _AuthedReq(f"Bearer {c.api_key}", body={"inference_mode": "byok"})
    await update_inference_mode(c.id, req, store)
    assert (await store.get_customer_by_id(c.id)).inference_mode == "byok"


async def test_byok_flip_requires_stored_key_b(monkeypatch, store):
    """TRIPWIRE: no Key B on file → byok flip is a 400, not a broken
    next-call."""
    _use(monkeypatch)
    c = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="")
    await store.set_customer_inference_mode(c.id, "managed")
    req = _AuthedReq(f"Bearer {c.api_key}", body={"inference_mode": "byok"})
    with pytest.raises(HTTPException) as e:
        await update_inference_mode(c.id, req, store)
    assert e.value.status_code == 400


async def test_inference_mode_rejects_garbage_value(monkeypatch, store):
    _use(monkeypatch)
    c = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    req = _AuthedReq(f"Bearer {c.api_key}", body={"inference_mode": "free"})
    with pytest.raises(HTTPException) as e:
        await update_inference_mode(c.id, req, store)
    assert e.value.status_code == 400


async def test_inference_mode_foreign_key_a_is_denied(monkeypatch, store):
    """TRIPWIRE: tenant A cannot flip tenant B's mode (B1 self-or-admin)."""
    _use(monkeypatch)
    a = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    b = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    req = _AuthedReq(f"Bearer {a.api_key}",
                     body={"inference_mode": "managed"})
    with pytest.raises(HTTPException) as e:
        await update_inference_mode(b.id, req, store)
    assert e.value.status_code in (403, 404)


# --- spend view --------------------------------------------------------------------

async def test_spend_view_reports_mtd_against_cap(monkeypatch, store):
    _use(monkeypatch)
    c = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="")
    await store.set_customer_inference_mode(c.id, "managed")
    await store.record_llm_call(
        c.id, model="claude-haiku-4-5", input_tokens=0, output_tokens=0,
        billing="managed", price_table={},
    )
    out = await get_customer_spend(c.id, store)
    assert out["inference_mode"] == "managed"
    assert out["managed_monthly_cap_micro_usd"] > 0
    assert out["managed_month_to_date_micro_usd"] >= 0
    assert "totals" in out


async def test_spend_view_unknown_customer_404(monkeypatch, store):
    _use(monkeypatch)
    with pytest.raises(HTTPException) as e:
        await get_customer_spend("cust_missing", store)
    assert e.value.status_code == 404


# --- JWT principals on the self-or-admin routes (live fix, 2026-07-06) --------
# The pinned console authenticates with the session JWT; these routes must
# accept the customer's OWNER (and platform_admin users) — found live when
# Settings' key-save 404'd for a signed-in tenant.

async def test_self_or_admin_accepts_owner_jwt(monkeypatch, store):
    from crystal_cache.ingress.auth import require_customer_self_or_admin
    _use(monkeypatch)
    _verify_as(monkeypatch, "uid_o", "o@o.test")
    out = await signup(_Req(f"Bearer {FAKE_JWT}"), store)
    got = await require_customer_self_or_admin(
        out["customer_id"], _Req(f"Bearer {FAKE_JWT}"), store)
    assert got.id == out["customer_id"]


async def test_self_or_admin_rejects_foreign_owner_jwt(monkeypatch, store):
    """TRIPWIRE: an owner's JWT is NOT a skeleton key — foreign customer
    ids stay indistinguishable from nonexistent ones."""
    from crystal_cache.ingress.auth import require_customer_self_or_admin
    _use(monkeypatch)
    _verify_as(monkeypatch, "uid_o", "o@o.test")
    await signup(_Req(f"Bearer {FAKE_JWT}"), store)
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    with pytest.raises(HTTPException) as e:
        await require_customer_self_or_admin(
            other.id, _Req(f"Bearer {FAKE_JWT}"), store)
    assert e.value.status_code == 404


async def test_self_or_admin_accepts_platform_admin_jwt(monkeypatch, store):
    from crystal_cache.ingress.auth import require_customer_self_or_admin
    _use(monkeypatch)
    _verify_as(monkeypatch, "uid_root", ADMIN_EMAIL)
    await signup(_Req(f"Bearer {FAKE_JWT}"), store)  # bootstraps admin
    c = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    got = await require_customer_self_or_admin(
        c.id, _Req(f"Bearer {FAKE_JWT}"), store)
    assert got.id == c.id


# --- model updates (hosted parity: tuning at onboarding + Settings) -----------

async def test_signup_honors_model_choice(monkeypatch, store):
    _use(monkeypatch)
    _verify_as(monkeypatch, "uid_m", "m@m.test")
    out = await signup(_Req(f"Bearer {FAKE_JWT}",
                            body={"model": "claude-haiku-4-5"}), store)
    c = await store.get_customer_by_id(out["customer_id"])
    assert c.model_routing_config.model_id == "claude-haiku-4-5"


async def test_signup_unknown_model_falls_back_to_default(monkeypatch, store):
    _use(monkeypatch)
    _verify_as(monkeypatch, "uid_m2", "m2@m.test")
    out = await signup(_Req(f"Bearer {FAKE_JWT}",
                            body={"model": "gpt-999"}), store)
    c = await store.get_customer_by_id(out["customer_id"])
    assert c.model_routing_config.model_id == SIGNUP_DEFAULT_MODEL


async def test_model_patch_managed_restricted_to_allowed_set(
        monkeypatch, store):
    from crystal_cache.endpoints.customers import update_model
    _use(monkeypatch)
    c = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="")
    await store.set_customer_inference_mode(c.id, "managed")

    req = _AuthedReq(f"Bearer {c.api_key}",
                     body={"model_id": "claude-opus-4-8"})
    await update_model(c.id, req, store)
    assert (await store.get_customer_by_id(c.id)) \
        .model_routing_config.model_id == "claude-opus-4-8"

    req = _AuthedReq(f"Bearer {c.api_key}", body={"model_id": "gpt-999"})
    with pytest.raises(HTTPException) as e:
        await update_model(c.id, req, store)
    assert e.value.status_code == 400


async def test_model_patch_byok_is_unrestricted(monkeypatch, store):
    from crystal_cache.endpoints.customers import update_model
    _use(monkeypatch)
    c = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    req = _AuthedReq(f"Bearer {c.api_key}",
                     body={"model_id": "my-finetuned-model-v7"})
    await update_model(c.id, req, store)
    assert (await store.get_customer_by_id(c.id)) \
        .model_routing_config.model_id == "my-finetuned-model-v7"


# --- Key A regeneration (2026-07-07): the lost-key recovery path ---------------

async def test_rotate_kills_old_key_and_mints_working_new_one(
        monkeypatch, store):
    from crystal_cache.endpoints.customers import rotate_api_key
    _use(monkeypatch)
    c = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    old_key = c.api_key

    req = _AuthedReq(f"Bearer {old_key}", body={})
    out = await rotate_api_key(c.id, req, store)
    import json
    payload = json.loads(out.body)
    new_key = payload["api_key"]
    assert new_key and new_key != old_key

    assert await store.get_customer_by_api_key(old_key) is None  # DEAD
    resolved = await store.get_customer_by_api_key(new_key)
    assert resolved is not None and resolved.id == c.id  # ALIVE


async def test_rotate_via_owner_jwt_the_lost_key_case(monkeypatch, store):
    """THE use case: console session, key never copied — the JWT owner
    regenerates from Settings."""
    from crystal_cache.endpoints.customers import rotate_api_key
    _use(monkeypatch)
    _verify_as(monkeypatch, "uid_rot", "rot@t.test")
    signed = await signup(_Req(f"Bearer {FAKE_JWT}"), store)

    req = _Req(f"Bearer {FAKE_JWT}", body={})
    out = await rotate_api_key(signed["customer_id"], req, store)
    import json
    new_key = json.loads(out.body)["api_key"]
    resolved = await store.get_customer_by_api_key(new_key)
    assert resolved.id == signed["customer_id"]


async def test_rotate_foreign_key_denied(monkeypatch, store):
    """TRIPWIRE: tenant A cannot rotate tenant B's key."""
    from crystal_cache.endpoints.customers import rotate_api_key
    _use(monkeypatch)
    a = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    b = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    with pytest.raises(HTTPException) as e:
        await rotate_api_key(b.id, _AuthedReq(f"Bearer {a.api_key}", body={}),
                             store)
    assert e.value.status_code == 404
    # And B's key still works — nothing was rotated.
    assert (await store.get_customer_by_api_key(b.api_key)).id == b.id


async def test_rotate_jwt_requires_fresh_auth_time(monkeypatch, store):
    """STEP-UP TRIPWIRE: an idle session's JWT (stale auth_time) cannot
    rotate — 401 until the user re-verifies."""
    import time
    from crystal_cache.endpoints.customers import rotate_api_key
    _use(monkeypatch)

    stale = {"sub": "uid_stale", "email": "stale@t.test",
             "auth_time": int(time.time()) - 3600}
    monkeypatch.setattr(auth_mod, "_verify_firebase_jwt",
                        lambda tok, proj: dict(stale))
    signed = await signup(_Req(f"Bearer {FAKE_JWT}"), store)

    with pytest.raises(HTTPException) as e:
        await rotate_api_key(signed["customer_id"],
                             _Req(f"Bearer {FAKE_JWT}", body={}), store)
    assert e.value.status_code == 401

    fresh = dict(stale, auth_time=int(time.time()) - 5)
    monkeypatch.setattr(auth_mod, "_verify_firebase_jwt",
                        lambda tok, proj: dict(fresh))
    out = await rotate_api_key(signed["customer_id"],
                               _Req(f"Bearer {FAKE_JWT}", body={}), store)
    import json
    assert json.loads(out.body)["api_key"]


async def test_rotate_key_a_exempt_from_step_up(monkeypatch, store):
    """Key A callers present the credential itself — no auth_time concept;
    rotation proceeds (and kills the presented key)."""
    from crystal_cache.endpoints.customers import rotate_api_key
    _use(monkeypatch)
    c = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    out = await rotate_api_key(c.id, _AuthedReq(f"Bearer {c.api_key}",
                                                body={}), store)
    import json
    assert json.loads(out.body)["api_key"]
