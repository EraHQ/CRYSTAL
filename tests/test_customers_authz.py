"""B1 fix (2026-07-03) — /v1/customers/{id} is self-or-admin, not open.

GET/PATCH /v1/customers/{id} were UNAUTHENTICATED: anyone who knew a
customer_id could read that customer's routing config and overwrite their
upstream key (Key B). The fix requires the caller to be the customer
itself (its Key A) or the platform admin; everyone else gets 404 (the
route is not an existence oracle).

These tests assert the NEGATIVE cases (the security property) as hard as
the positive ones.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from crystal_cache.config import Settings
from crystal_cache.endpoints.customers import get_customer, update_upstream_key
from crystal_cache.ingress import auth as auth_mod


ADMIN_KEY = "cc_sk_admin_test_b1_0000"


class _Req:
    def __init__(self, authorization: str | None = None,
                 body: dict | None = None) -> None:
        self.headers = {"authorization": authorization} if authorization else {}
        self._body = body or {}

    async def json(self) -> dict:
        return self._body


def _use_admin_key(monkeypatch, key: str = ADMIN_KEY) -> None:
    settings = Settings(environment="development", admin_api_key=key,
                        api_key_pepper="")
    monkeypatch.setattr(auth_mod, "get_settings", lambda: settings)


# --- GET: read authorization ------------------------------------------------

async def test_get_customer_self_succeeds(store, customer):
    resp = await get_customer(
        customer.id, _Req(f"Bearer {customer.api_key}"), store,
    )
    assert resp.id == customer.id


async def test_get_customer_unauthenticated_is_404(store, customer):
    """The core regression: no token must NOT read the record."""
    with pytest.raises(HTTPException) as exc:
        await get_customer(customer.id, _Req(None), store)
    assert exc.value.status_code == 404


async def test_get_customer_other_customers_key_is_404(store, customer):
    """A valid Key A for a DIFFERENT customer cannot read this one."""
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="k",
    )
    with pytest.raises(HTTPException) as exc:
        await get_customer(customer.id, _Req(f"Bearer {other.api_key}"), store)
    assert exc.value.status_code == 404


async def test_get_customer_unknown_key_is_404(store, customer):
    with pytest.raises(HTTPException) as exc:
        await get_customer(customer.id, _Req("Bearer cc_sk_bogus"), store)
    assert exc.value.status_code == 404


async def test_get_customer_admin_key_succeeds(monkeypatch, store, customer):
    _use_admin_key(monkeypatch)
    resp = await get_customer(customer.id, _Req(f"Bearer {ADMIN_KEY}"), store)
    assert resp.id == customer.id


# --- PATCH: the sensitive write --------------------------------------------

async def test_patch_upstream_key_unauthenticated_is_404_and_no_write(
    store, customer,
):
    """The finding that mattered most: an anonymous caller must not be able
    to overwrite the upstream key."""
    with pytest.raises(HTTPException) as exc:
        await update_upstream_key(
            customer.id, _Req(None, {"api_key_ref": "attacker_key"}), store,
        )
    assert exc.value.status_code == 404
    # And nothing was written.
    fetched = await store.get_customer_by_id(customer.id)
    assert fetched.model_routing_config.api_key_ref != "attacker_key"


async def test_patch_upstream_key_other_customer_is_404(store, customer):
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="k",
    )
    with pytest.raises(HTTPException) as exc:
        await update_upstream_key(
            customer.id,
            _Req(f"Bearer {other.api_key}", {"api_key_ref": "x"}),
            store,
        )
    assert exc.value.status_code == 404


async def test_patch_upstream_key_self_succeeds(store, customer):
    resp = await update_upstream_key(
        customer.id,
        _Req(f"Bearer {customer.api_key}", {"api_key_ref": "new_key_b"}),
        store,
    )
    import json
    assert json.loads(resp.body)["updated"] is True


async def test_patch_upstream_key_admin_succeeds(monkeypatch, store, customer):
    _use_admin_key(monkeypatch)
    resp = await update_upstream_key(
        customer.id,
        _Req(f"Bearer {ADMIN_KEY}", {"api_key_ref": "admin_set_key"}),
        store,
    )
    import json
    assert json.loads(resp.body)["updated"] is True
