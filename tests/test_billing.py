"""L2-S3 pins (remedy A, 2026-09-08): the money path.

The webhook's signature check is pinned with REAL verification — the
test signs payloads exactly as Stripe does (v1 = HMAC-SHA256(secret,
"{t}.{payload}")) and runs the SDK's construct_event against them, so
the pin exercises the same code path production trusts. Requires the
stripe package (importorskip; if these show as 's', pip install stripe).
"""
import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest
from fastapi import HTTPException

pytest.importorskip("stripe")

from crystal_cache.config import Settings
from crystal_cache.endpoints import billing as billing_mod
from crystal_cache.ingress.auth import trial_expired

SECRET = "whsec_testsecret123"

# RC-13 (2026-10-05): real Stripe events carry a unique id and an
# increasing `created`; checkout sessions carry a payment_status. The
# webhook now dedupes by id, orders by created, and grants only on paid,
# so the fixtures model those three fields.
import itertools

_EVT = itertools.count(1)


def _stamp() -> dict:
    n = next(_EVT)
    return {"id": f"evt_test_{n}", "created": 1_700_000_000 + n}


class _StubRequest:
    def __init__(self, payload: bytes, sig: Optional[str]):
        self._payload = payload
        self.headers = {"stripe-signature": sig} if sig else {}

    async def body(self) -> bytes:
        return self._payload


def _sign(payload: bytes, secret: str = SECRET, t: Optional[int] = None) -> str:
    t = t or int(time.time())
    signed = f"{t}.".encode() + payload
    v1 = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return f"t={t},v1={v1}"


def _event(cid: str) -> bytes:
    return json.dumps({
        **_stamp(),
        "object": "event",
        "api_version": "2024-06-20",
        "type": "checkout.session.completed",
        "data": {"object": {
            "id": "cs_test_1", "object": "checkout.session",
            "client_reference_id": cid,
            "customer": "cus_stripe_abc",
            "payment_status": "paid",
        }},
    }).encode()


@pytest.mark.asyncio
async def test_webhook_rejects_bad_signature(store, customer, monkeypatch):
    monkeypatch.setattr(
        billing_mod, "get_settings",
        lambda: Settings(stripe_webhook_secret=SECRET),
    )
    payload = _event(customer.id)
    with pytest.raises(HTTPException) as e:
        await billing_mod.stripe_webhook(
            _StubRequest(payload, _sign(payload, secret="whsec_WRONG")), store
        )
    assert e.value.status_code == 400
    c = await store.get_customer_by_id(customer.id)
    assert c.subscription_tier != billing_mod.STARTER_TIER  # untouched


@pytest.mark.asyncio
async def test_webhook_good_signature_upgrades_and_clears_trial(
    store, customer, monkeypatch
):
    monkeypatch.setattr(
        billing_mod, "get_settings",
        lambda: Settings(stripe_webhook_secret=SECRET),
    )
    # Put the tenant into an EXPIRED trial first — the webhook is the door out.
    await store.set_customer_subscription(
        customer.id, "trial_29",
        trial_expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    payload = _event(customer.id)
    out = await billing_mod.stripe_webhook(
        _StubRequest(payload, _sign(payload)), store
    )
    assert out == {"received": True}
    c = await store.get_customer_by_id(customer.id)
    assert c.subscription_tier == billing_mod.STARTER_TIER
    assert c.trial_expires_at is None
    assert trial_expired(c) is False  # writes resume
    # L2-S4=B: the portal join persisted from the event.
    assert c.stripe_customer_id == "cus_stripe_abc"


@pytest.mark.asyncio
async def test_portal_409_before_first_payment(store, customer, monkeypatch):
    monkeypatch.setattr(
        billing_mod, "get_settings",
        lambda: Settings(stripe_secret_key="sk_test_x"),
    )
    body = billing_mod.PortalRequest(return_url="https://console.test/billing")
    with pytest.raises(HTTPException) as e:
        await billing_mod.customer_portal(body, (customer, None))
    assert e.value.status_code == 409


@pytest.mark.asyncio
async def test_portal_opens_for_paid_tenant(store, customer, monkeypatch):
    import stripe

    monkeypatch.setattr(
        billing_mod, "get_settings",
        lambda: Settings(stripe_secret_key="sk_test_x"),
    )
    paid = await store.set_customer_subscription(
        customer.id, billing_mod.STARTER_TIER, None,
        stripe_customer_id="cus_stripe_abc",
    )
    captured: dict = {}

    def _fake_portal(**kwargs):
        captured.update(kwargs)

        class _S:
            url = "https://billing.stripe.test/p"

        return _S()

    monkeypatch.setattr(stripe.billing_portal.Session, "create", _fake_portal)
    body = billing_mod.PortalRequest(return_url="https://console.test/billing")
    out = await billing_mod.customer_portal(body, (paid, None))
    assert out["portal_url"].startswith("https://billing.stripe.test")
    assert captured["customer"] == "cus_stripe_abc"
    assert captured["return_url"] == "https://console.test/billing"


@pytest.mark.asyncio
async def test_webhook_404_when_unconfigured(store):
    with pytest.raises(HTTPException) as e:
        await billing_mod.stripe_webhook(_StubRequest(b"{}", None), store)
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_checkout_wires_reference_and_price(customer, monkeypatch):
    import stripe

    monkeypatch.setattr(
        billing_mod, "get_settings",
        lambda: Settings(
            stripe_secret_key="sk_test_x",
            stripe_price_starter="price_starter29",
        ),
    )
    captured: dict = {}

    def _fake_create(**kwargs):
        captured.update(kwargs)

        class _S:
            url = "https://checkout.stripe.test/s"
            id = "cs_test_2"

        return _S()

    monkeypatch.setattr(stripe.checkout.Session, "create", _fake_create)
    body = billing_mod.CheckoutRequest(
        success_url="https://console.test/ok", cancel_url="https://console.test/no"
    )
    out = await billing_mod.create_checkout(body, (customer, None))
    assert out["checkout_url"].startswith("https://checkout.stripe.test")
    assert captured["client_reference_id"] == customer.id
    assert captured["mode"] == "subscription"
    assert captured["line_items"][0]["price"] == "price_starter29"
    # Self-host neutrality pin: with the knob off (default), no MoR
    # parameter reaches Stripe — plain accounts stay plain.
    assert "managed_payments" not in captured


@pytest.mark.asyncio
async def test_checkout_managed_payments_opt_in(customer, monkeypatch):
    import stripe

    monkeypatch.setattr(
        billing_mod, "get_settings",
        lambda: Settings(
            stripe_secret_key="sk_test_x",
            stripe_price_starter="price_starter29",
            stripe_managed_payments=True,
        ),
    )
    captured: dict = {}

    def _fake_create(**kwargs):
        captured.update(kwargs)

        class _S:
            url = "https://checkout.stripe.test/s"
            id = "cs_test_3"

        return _S()

    monkeypatch.setattr(stripe.checkout.Session, "create", _fake_create)
    body = billing_mod.CheckoutRequest(
        success_url="https://console.test/ok", cancel_url="https://console.test/no"
    )
    await billing_mod.create_checkout(body, (customer, None))
    # The MoR contract: explicit opt-in + the pinned API version.
    assert captured["managed_payments"] == {"enabled": True}
    assert stripe.api_version == "2025-03-31.basil"


@pytest.mark.asyncio
async def test_session_principal_resolves_firebase_owner(store, customer, monkeypatch):
    """The 401-on-Upgrade regression (found live 2026-09-22): a signed-in
    console session must resolve to its OWN tenant on the billing
    surfaces. Pinned at the dependency."""
    from crystal_cache.ingress import auth as auth_mod

    await store.create_user("uid_bill_1", "bill1@test.dev", customer.id, "owner")
    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt",
        lambda tok, proj: {"sub": "uid_bill_1", "email": "bill1@test.dev"},
    )
    monkeypatch.setattr(
        auth_mod, "get_settings",
        lambda: Settings(firebase_project_id="test-proj"),
    )

    class _Req:
        headers = {"authorization": "Bearer eyJx.eyJy.sig"}

    resolved, operator = await auth_mod.resolve_principal_or_session(_Req(), store)
    assert resolved.id == customer.id
    assert operator is None


@pytest.mark.asyncio
async def test_session_principal_rejects_unknown_session(store, monkeypatch):
    from fastapi import HTTPException as HTTPExc

    from crystal_cache.ingress import auth as auth_mod

    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt",
        lambda tok, proj: {"sub": "uid_nobody", "email": "nobody@test.dev"},
    )
    monkeypatch.setattr(
        auth_mod, "get_settings",
        lambda: Settings(firebase_project_id="test-proj"),
    )

    class _Req:
        headers = {"authorization": "Bearer eyJx.eyJy.sig"}

    with pytest.raises(HTTPExc) as e:
        await auth_mod.resolve_principal_or_session(_Req(), store)
    assert e.value.status_code == 401


@pytest.mark.asyncio
async def test_stripe_rejection_becomes_clean_502(customer, monkeypatch):
    import stripe

    monkeypatch.setattr(
        billing_mod, "get_settings",
        lambda: Settings(
            stripe_secret_key="sk_test_x",
            stripe_price_starter="price_starter29",
        ),
    )

    def _boom(**kwargs):
        raise stripe.StripeError("the product tax code is missing")

    monkeypatch.setattr(stripe.checkout.Session, "create", _boom)
    body = billing_mod.CheckoutRequest(
        success_url="https://console.test/ok", cancel_url="https://console.test/no"
    )
    with pytest.raises(HTTPException) as e:
        await billing_mod.create_checkout(body, (customer, None))
    assert e.value.status_code == 502
    # Q6=C (2026-09-30): provider text never reaches the user; the fixed
    # support message with a reference id does.
    assert "tax code" not in e.value.detail
    assert "contact support" in e.value.detail
    assert "reference" in e.value.detail


# ---------------------------------------------------------------------------
# Scale-solo + plan lifecycle (ratified 2026-09-30: Scale-solo, Q4=A).
# ---------------------------------------------------------------------------

PRICE_STARTER = "price_starter29"
PRICE_SCALE = "price_scale49"


def _live_settings(**extra) -> Settings:
    base = dict(
        stripe_secret_key="sk_test_x",
        stripe_webhook_secret=SECRET,
        stripe_price_starter=PRICE_STARTER,
        stripe_price_scale=PRICE_SCALE,
    )
    base.update(extra)
    return Settings(**base)


def _capture_checkout(monkeypatch) -> dict:
    import stripe

    captured: dict = {}

    def _fake_create(**kwargs):
        captured.update(kwargs)

        class _S:
            url = "https://checkout.stripe.test/s"
            id = "cs_test_plan"

        return _S()

    monkeypatch.setattr(stripe.checkout.Session, "create", _fake_create)
    return captured


def _checkout_event(cid: str, price_id: Optional[str]) -> bytes:
    obj = {
        "id": "cs_test_meta", "object": "checkout.session",
        "client_reference_id": cid, "customer": "cus_stripe_abc",
    }
    if price_id is not None:
        obj["metadata"] = {"price_id": price_id}
    obj["payment_status"] = "paid"
    return json.dumps({
        **_stamp(), "object": "event",
        "api_version": "2024-06-20",
        "type": "checkout.session.completed",
        "data": {"object": obj},
    }).encode()


def _sub_event(etype: str, status: str, price_id: Optional[str],
               stripe_cid: str = "cus_stripe_abc") -> bytes:
    items = (
        {"object": "list", "data": [{"id": "si_1", "price": {"id": price_id}}]}
        if price_id else {"object": "list", "data": []}
    )
    return json.dumps({
        **_stamp(), "object": "event",
        "api_version": "2024-06-20",
        "type": etype,
        "data": {"object": {
            "id": "sub_test_1", "object": "subscription",
            "customer": stripe_cid, "status": status, "items": items,
        }},
    }).encode()


async def _post(store, payload: bytes) -> dict:
    return await billing_mod.stripe_webhook(
        _StubRequest(payload, _sign(payload)), store
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("plan,price,tier", [
    ("starter", PRICE_STARTER, "starter_29"),
    ("scale", PRICE_SCALE, "scale_49_seat"),
])
async def test_price_tier_mapping_round_trips_both_directions(
    store, customer, monkeypatch, plan, price, tier
):
    """plan -> price at checkout, price -> tier at the webhook, and the
    two agree for every plan: the price checkout charges is exactly the
    price the webhook maps back to that plan's tier."""
    monkeypatch.setattr(billing_mod, "get_settings", lambda: _live_settings())
    captured = _capture_checkout(monkeypatch)
    body = billing_mod.CheckoutRequest(
        success_url="https://console.test/ok",
        cancel_url="https://console.test/no", plan=plan,
    )
    await billing_mod.create_checkout(body, (customer, None))
    assert captured["line_items"][0]["price"] == price
    assert captured["metadata"] == {"price_id": price}
    assert billing_mod.price_tier_map(_live_settings())[price] == tier

    await _post(store, _checkout_event(customer.id, captured["metadata"]["price_id"]))
    c = await store.get_customer_by_id(customer.id)
    assert c.subscription_tier == tier
    assert c.stripe_customer_id == "cus_stripe_abc"


def test_invalid_plan_is_rejected_by_validation():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        billing_mod.CheckoutRequest(
            success_url="https://console.test/ok",
            cancel_url="https://console.test/no", plan="enterprise",
        )


def test_plan_defaults_to_starter_for_older_consoles():
    body = billing_mod.CheckoutRequest(
        success_url="https://console.test/ok", cancel_url="https://console.test/no",
    )
    assert body.plan == "starter"


@pytest.mark.asyncio
async def test_scale_checkout_400_when_scale_price_unset(customer, monkeypatch):
    monkeypatch.setattr(
        billing_mod, "get_settings",
        lambda: _live_settings(stripe_price_scale=""),
    )
    _capture_checkout(monkeypatch)
    body = billing_mod.CheckoutRequest(
        success_url="https://console.test/ok",
        cancel_url="https://console.test/no", plan="scale",
    )
    with pytest.raises(HTTPException) as e:
        await billing_mod.create_checkout(body, (customer, None))
    assert e.value.status_code == 400


@pytest.mark.asyncio
async def test_checkout_409_for_paid_tenant(store, customer, monkeypatch):
    """Q4=A: one subscription per tenant; paid plan changes go through
    the portal, so checkout can never double-bill."""
    monkeypatch.setattr(billing_mod, "get_settings", lambda: _live_settings())
    captured = _capture_checkout(monkeypatch)
    paid = await store.set_customer_subscription(
        customer.id, "starter_29", None, stripe_customer_id="cus_stripe_abc",
    )
    body = billing_mod.CheckoutRequest(
        success_url="https://console.test/ok",
        cancel_url="https://console.test/no", plan="scale",
    )
    with pytest.raises(HTTPException) as e:
        await billing_mod.create_checkout(body, (paid, None))
    assert e.value.status_code == 409
    assert captured == {}  # Stripe was never called


@pytest.mark.asyncio
async def test_webhook_unknown_price_changes_nothing(store, customer, monkeypatch):
    monkeypatch.setattr(billing_mod, "get_settings", lambda: _live_settings())
    await store.set_customer_subscription(customer.id, "free", None)
    out = await _post(store, _checkout_event(customer.id, "price_stranger"))
    assert out == {"received": True}
    c = await store.get_customer_by_id(customer.id)
    assert c.subscription_tier == "free"  # never defaulted to a paid tier


@pytest.mark.asyncio
async def test_subscription_updated_switches_starter_to_scale(
    store, customer, monkeypatch
):
    monkeypatch.setattr(billing_mod, "get_settings", lambda: _live_settings())
    await store.set_customer_subscription(
        customer.id, "starter_29", None, stripe_customer_id="cus_stripe_abc",
    )
    await _post(store, _sub_event(
        "customer.subscription.updated", "active", PRICE_SCALE,
    ))
    c = await store.get_customer_by_id(customer.id)
    assert c.subscription_tier == "scale_49_seat"
    # And back down again through the same mapping.
    await _post(store, _sub_event(
        "customer.subscription.updated", "active", PRICE_STARTER,
    ))
    c = await store.get_customer_by_id(customer.id)
    assert c.subscription_tier == "starter_29"


@pytest.mark.asyncio
async def test_subscription_updated_unknown_price_changes_nothing(
    store, customer, monkeypatch
):
    monkeypatch.setattr(billing_mod, "get_settings", lambda: _live_settings())
    await store.set_customer_subscription(
        customer.id, "starter_29", None, stripe_customer_id="cus_stripe_abc",
    )
    await _post(store, _sub_event(
        "customer.subscription.updated", "active", "price_stranger",
    ))
    c = await store.get_customer_by_id(customer.id)
    assert c.subscription_tier == "starter_29"


@pytest.mark.asyncio
async def test_past_due_keeps_the_paid_tier(store, customer, monkeypatch):
    monkeypatch.setattr(billing_mod, "get_settings", lambda: _live_settings())
    await store.set_customer_subscription(
        customer.id, "scale_49_seat", None, stripe_customer_id="cus_stripe_abc",
    )
    await _post(store, _sub_event(
        "customer.subscription.updated", "past_due", PRICE_SCALE,
    ))
    c = await store.get_customer_by_id(customer.id)
    assert c.subscription_tier == "scale_49_seat"


@pytest.mark.asyncio
@pytest.mark.parametrize("etype,status", [
    ("customer.subscription.deleted", "canceled"),
    ("customer.subscription.updated", "unpaid"),
])
async def test_cancellation_drops_to_free_and_keeps_the_join(
    store, customer, monkeypatch, etype, status
):
    monkeypatch.setattr(billing_mod, "get_settings", lambda: _live_settings())
    await store.set_customer_subscription(
        customer.id, "scale_49_seat", None, stripe_customer_id="cus_stripe_abc",
    )
    await _post(store, _sub_event(etype, status, PRICE_SCALE))
    c = await store.get_customer_by_id(customer.id)
    assert c.subscription_tier == "free"
    assert c.trial_expires_at is None
    # The Stripe join outlives the tier: invoices stay reachable, and
    # a free tenant may check out again.
    assert c.stripe_customer_id == "cus_stripe_abc"


@pytest.mark.asyncio
async def test_lifecycle_event_for_unknown_stripe_customer_is_a_noop(
    store, customer, monkeypatch
):
    monkeypatch.setattr(billing_mod, "get_settings", lambda: _live_settings())
    out = await _post(store, _sub_event(
        "customer.subscription.deleted", "canceled", PRICE_SCALE,
        stripe_cid="cus_nobody",
    ))
    assert out == {"received": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    "https://evil.example/steal",          # wrong host
    "http://console.test/ok",              # plain http off loopback
    "javascript:alert(1)",                 # not a URL we ever send to Stripe
    "https://console.test.evil.example/",  # suffix trick
    "",
])
async def test_checkout_rejects_return_urls_off_the_console_host(
    customer, monkeypatch, bad
):
    """S8: the console supplies Stripe's return URLs; only https on an
    allow-listed host (or http on loopback) passes."""
    monkeypatch.setattr(billing_mod, "get_settings", lambda: _live_settings())
    captured = _capture_checkout(monkeypatch)
    body = billing_mod.CheckoutRequest(
        success_url=bad, cancel_url="https://console.test/no",
    )
    with pytest.raises(HTTPException) as e:
        await billing_mod.create_checkout(body, (customer, None))
    assert e.value.status_code == 400
    assert captured == {}


@pytest.mark.asyncio
async def test_checkout_accepts_loopback_http_for_dev(customer, monkeypatch):
    monkeypatch.setattr(billing_mod, "get_settings", lambda: _live_settings())
    captured = _capture_checkout(monkeypatch)
    body = billing_mod.CheckoutRequest(
        success_url="http://localhost:5173/admin/billing",
        cancel_url="http://127.0.0.1:5173/admin/billing",
    )
    await billing_mod.create_checkout(body, (customer, None))
    assert captured["success_url"] == "http://localhost:5173/admin/billing"


# ---------------------------------------------------------------------------
# 2026-09-30 key-leak incident: secrets never leave in logs or responses,
# and a stray newline in a stored secret never breaks Stripe calls.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stripe_error_detail_never_carries_a_secret():
    leaked = (
        "Network error: InvalidHeader: header value: "
        "'Bearer sk_live_51ABCdef123\\n' whsec_zzz999 rk_test_abc"
    )

    def _boom():
        raise RuntimeError(leaked)

    with pytest.raises(HTTPException) as e:
        await billing_mod._stripe_call(_boom)
    assert e.value.status_code == 502
    for secret in ("sk_live_51ABCdef123", "whsec_zzz999", "rk_test_abc", "Bearer"):
        assert secret not in e.value.detail
    assert "InvalidHeader" not in e.value.detail  # no provider text at all
    assert "contact support" in e.value.detail


def test_redact_scrubs_every_secret_shape_for_the_logs():
    text = "Bearer sk_live_51ABC and whsec_zzz999 and rk_test_abc and pk_live_q"
    out = billing_mod._redact(text)
    for secret in ("sk_live_51ABC", "whsec_zzz999", "rk_test_abc", "pk_live_q"):
        assert secret not in out


@pytest.mark.asyncio
async def test_checkout_strips_whitespace_from_the_secret_key(customer, monkeypatch):
    import stripe

    monkeypatch.setattr(
        billing_mod, "get_settings",
        lambda: _live_settings(stripe_secret_key="sk_test_x\n"),
    )
    seen: dict = {}

    def _fake_create(**kwargs):
        seen["key"] = stripe.api_key

        class _S:
            url = "https://checkout.stripe.test/s"
            id = "cs_test_strip"

        return _S()

    monkeypatch.setattr(stripe.checkout.Session, "create", _fake_create)
    body = billing_mod.CheckoutRequest(
        success_url="https://console.test/ok", cancel_url="https://console.test/no",
    )
    await billing_mod.create_checkout(body, (customer, None))
    assert seen["key"] == "sk_test_x"


@pytest.mark.asyncio
async def test_webhook_verifies_with_a_newline_padded_secret(store, customer, monkeypatch):
    monkeypatch.setattr(
        billing_mod, "get_settings",
        lambda: _live_settings(stripe_webhook_secret=SECRET + "\n"),
    )
    out = await _post(store, _checkout_event(customer.id, PRICE_STARTER))
    assert out == {"received": True}
    c = await store.get_customer_by_id(customer.id)
    assert c.subscription_tier == "starter_29"
