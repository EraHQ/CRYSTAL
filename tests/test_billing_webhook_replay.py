"""RC-13 (2026-10-05): the Stripe webhook is idempotent, ordered, paid-only,
and a tier change clamps the stored model.

Stripe retries deliveries and does not guarantee order. Before this, a
replayed checkout.session.completed re-granted a tier, a late
subscription.updated could overwrite a newer deleted, an unpaid session
granted, a downgrade left Opus stored on a free tenant, and three model
lists disagreed. Every event here is signed exactly as Stripe signs and
goes through the real route, as in tests/test_billing.py.
"""
import hashlib
import hmac
import json
import time
from typing import Optional

import pytest

pytest.importorskip("stripe")

from crystal_cache.config import Settings
from crystal_cache.control.admission import allowed_models_for
from crystal_cache.endpoints import billing as billing_mod

SECRET = "whsec_testsecret123"
PRICE_STARTER = "price_starter_t"
PRICE_SCALE = "price_scale_t"


class _StubRequest:
    def __init__(self, payload: bytes, sig: Optional[str]):
        self._payload = payload
        self.headers = {"stripe-signature": sig} if sig else {}

    async def body(self) -> bytes:
        return self._payload


def _sign(payload: bytes) -> str:
    t = int(time.time())
    v1 = hmac.new(SECRET.encode(), f"{t}.".encode() + payload, hashlib.sha256).hexdigest()
    return f"t={t},v1={v1}"


def _checkout(event_id, created, cid, price_id, payment_status="paid"):
    return json.dumps({
        "id": event_id, "created": created, "object": "event", "api_version": "2024-06-20",
        "type": "checkout.session.completed",
        "data": {"object": {
            "id": "cs_1", "object": "checkout.session", "client_reference_id": cid,
            "customer": "cus_stripe_rc13", "metadata": {"price_id": price_id},
            "payment_status": payment_status,
        }},
    }).encode()


def _sub(event_id, created, etype, status, price_id):
    items = {"object": "list", "data": [{"id": "si_1", "price": {"id": price_id}}]} if price_id else {"object": "list", "data": []}
    return json.dumps({
        "id": event_id, "created": created, "object": "event", "api_version": "2024-06-20",
        "type": etype,
        "data": {"object": {"id": "sub_1", "object": "subscription", "customer": "cus_stripe_rc13",
                            "status": status, "items": items}},
    }).encode()


async def _post(store, payload):
    return await billing_mod.stripe_webhook(_StubRequest(payload, _sign(payload)), store)


@pytest.fixture
def billing_settings(monkeypatch):
    monkeypatch.setattr(
        billing_mod, "get_settings",
        lambda: Settings(stripe_webhook_secret=SECRET, stripe_price_starter=PRICE_STARTER,
                         stripe_price_scale=PRICE_SCALE),
    )


@pytest.mark.asyncio
async def test_replayed_event_is_a_no_op(store, customer, billing_settings):
    p = _checkout("evt_a", 100, customer.id, PRICE_STARTER)
    assert await _post(store, p) == {"received": True}
    c = await store.get_customer_by_id(customer.id)
    assert c.subscription_tier == "starter_29"
    # Downgrade through a later lifecycle event...
    await _post(store, _sub("evt_b", 200, "customer.subscription.deleted", "canceled", None))
    assert (await store.get_customer_by_id(customer.id)).subscription_tier == "free"
    # ...then Stripe redelivers the old checkout: ignored, by id.
    out = await _post(store, p)
    assert out.get("duplicate") is True
    assert (await store.get_customer_by_id(customer.id)).subscription_tier == "free"


@pytest.mark.asyncio
async def test_out_of_order_lifecycle_does_not_overwrite_newer_state(store, customer, billing_settings):
    await _post(store, _checkout("evt_1", 100, customer.id, PRICE_STARTER))
    # The deleted event (created 300) arrives BEFORE the updated-to-scale
    # event (created 200). The stale updated must not resurrect Scale.
    await _post(store, _sub("evt_3", 300, "customer.subscription.deleted", "canceled", None))
    assert (await store.get_customer_by_id(customer.id)).subscription_tier == "free"
    await _post(store, _sub("evt_2", 200, "customer.subscription.updated", "active", PRICE_SCALE))
    assert (await store.get_customer_by_id(customer.id)).subscription_tier == "free"
    # A genuinely newer update is honoured.
    await _post(store, _sub("evt_4", 400, "customer.subscription.updated", "active", PRICE_SCALE))
    assert (await store.get_customer_by_id(customer.id)).subscription_tier == "scale_49_seat"


@pytest.mark.asyncio
async def test_unpaid_checkout_grants_nothing(store, customer, billing_settings):
    await _post(store, _checkout("evt_u", 100, customer.id, PRICE_STARTER, payment_status="unpaid"))
    assert (await store.get_customer_by_id(customer.id)).subscription_tier in (None, "free")


@pytest.mark.asyncio
async def test_downgrade_clamps_the_stored_model_and_the_next_call_uses_it(store, customer, billing_settings):
    await store.set_customer_inference_mode(customer.id, "managed")
    await _post(store, _checkout("evt_up", 100, customer.id, PRICE_STARTER))
    await store.set_customer_model(customer.id, "claude-opus-4-8")  # allowed on Starter
    assert (await store.get_customer_by_id(customer.id)).model_routing_config.model_id == "claude-opus-4-8"

    await _post(store, _sub("evt_dn", 200, "customer.subscription.deleted", "canceled", None))
    c = await store.get_customer_by_id(customer.id)
    assert c.subscription_tier == "free"
    assert c.model_routing_config.model_id in allowed_models_for(c)
    assert c.model_routing_config.model_id == "claude-sonnet-5"

    # The next model call resolves to the stored (now allowed) model.
    from crystal_cache.control.admission import enforce_managed_model

    enforce_managed_model(c, c.model_routing_config.model_id)  # no raise


def test_one_model_list_everywhere():
    """me.py and customers.py no longer keep their own lists; the console
    reads /v1/me.allowed_models."""
    import inspect

    from crystal_cache.endpoints import customers, me

    for mod in (me, customers):
        assert "MANAGED_ALLOWED_MODELS" not in inspect.getsource(mod), mod.__name__
    assert allowed_models_for("free") == ("claude-haiku-4-5", "claude-sonnet-5")
    assert "claude-opus-4-8" in allowed_models_for("starter_29")
