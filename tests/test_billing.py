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
        "id": "evt_test_1",
        "object": "event",
        "api_version": "2024-06-20",
        "type": "checkout.session.completed",
        "data": {"object": {
            "id": "cs_test_1", "object": "checkout.session",
            "client_reference_id": cid,
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
