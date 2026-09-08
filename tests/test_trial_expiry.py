"""L2-S2 pins (Q4=B + wiring=A, 2026-09-07): the trial that expires,
and the invariant that protects everyone who isn't on one.

The single most important pin here is tier-None-never-degrades: every
existing tenant and every self-host deployment has subscription_tier
None, and no expiry logic may ever touch them.
"""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from crystal_cache.ingress.auth import require_active_subscription, trial_expired


class _C:
    def __init__(self, tier=None, exp=None):
        self.subscription_tier = tier
        self.trial_expires_at = exp


def _past():
    return datetime.now(timezone.utc) - timedelta(hours=1)


def _future():
    return datetime.now(timezone.utc) + timedelta(days=6)


def test_tier_none_never_degrades():
    # Existing tenants / self-host / anything unstamped: NEVER expired,
    # even with a (stray) past clock.
    assert trial_expired(_C(None, None)) is False
    assert trial_expired(_C(None, _past())) is False


def test_paid_tier_never_degrades():
    assert trial_expired(_C("starter_29", None)) is False
    assert trial_expired(_C("starter_29", _past())) is False  # stale clock


def test_active_trial_is_not_expired():
    assert trial_expired(_C("trial_29", _future())) is False


def test_expired_trial_is_expired_and_raises_402():
    c = _C("trial_29", _past())
    assert trial_expired(c) is True
    with pytest.raises(HTTPException) as e:
        require_active_subscription(c)
    assert e.value.status_code == 402
    assert "memories are safe" in e.value.detail


def test_naive_datetime_is_treated_as_utc():
    naive_past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=2)
    assert trial_expired(_C("trial_29", naive_past)) is True


@pytest.mark.asyncio
async def test_signup_stamps_the_trial(store, monkeypatch):
    from crystal_cache.config import Settings
    from crystal_cache.endpoints import me as me_mod
    from crystal_cache.ingress import auth as auth_mod

    class _StubRequest:
        def __init__(self, token):
            self.headers = {"authorization": f"Bearer {token}"}

        async def json(self):
            raise ValueError("no body")

    monkeypatch.setattr(
        me_mod, "get_settings",
        lambda: Settings(firebase_project_id="test-proj"),
    )
    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt",
        lambda tok, proj: {"sub": "uid_trial_1", "email": "trial1@test.dev"},
    )
    r = await me_mod.signup(_StubRequest("eyJx.eyJy.sig"), store)
    c = await store.get_customer_by_id(r["customer_id"])
    assert c.subscription_tier == "trial_29"
    assert c.trial_expires_at is not None
    # ~7 days out, and NOT expired. (SQLite may hand back a naive
    # datetime for a tz-aware column — normalize before arithmetic,
    # same rule trial_expired applies.)
    exp = c.trial_expires_at
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    remaining = exp - datetime.now(timezone.utc)
    assert timedelta(days=6) < remaining <= timedelta(days=7)
    assert trial_expired(c) is False


@pytest.mark.asyncio
async def test_set_customer_subscription_clears_the_clock(store, customer):
    # The S3 webhook's upgrade shape: paid tier + None clock.
    await store.set_customer_subscription(
        customer.id, "trial_29", trial_expires_at=_past()
    )
    c = await store.get_customer_by_id(customer.id)
    assert trial_expired(c) is True
    await store.set_customer_subscription(customer.id, "starter_29", None)
    c = await store.get_customer_by_id(customer.id)
    assert c.subscription_tier == "starter_29"
    assert c.trial_expires_at is None
    assert trial_expired(c) is False
