"""RC-06 (2026-10-05): a signup that fails part-way leaves nothing behind.

Signup commits customer, mode, tier, user, seat and onboarding in
separate sessions. A failure after the customer row existed left a
tenant with no user (or a user with no seat), and the next attempt
minted a second tenant beside the orphan. Now any failure after
create_customer compensates by purging that tenant, the request fails
cleanly, and a retry provisions exactly one coherent tenant.

Harness mirrors tests/test_accounts_phase_c.py (settings + JWT seam).
"""
import pytest
from fastapi import HTTPException
from sqlalchemy import func, select

from crystal_cache.config import Settings
from crystal_cache.endpoints.me import signup
from crystal_cache.infrastructure.schema import CustomerRow, OperatorRow, UserRow
from crystal_cache.ingress import auth as auth_mod

FAKE_JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1In0.sig"


def _use(monkeypatch) -> None:
    settings = Settings(environment="development", admin_api_key="", api_key_pepper="",
                        firebase_project_id="proj-x", platform_admin_emails="admin@x.test")
    monkeypatch.setattr(auth_mod, "get_settings", lambda: settings)
    import crystal_cache.endpoints.me as me_mod
    monkeypatch.setattr(me_mod, "get_settings", lambda: settings)


def _verify_as(monkeypatch, uid, email) -> None:
    import time
    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt",
        lambda tok, proj: {"sub": uid, "email": email, "auth_time": int(time.time())},
    )


class _Req:
    def __init__(self, authorization=None, body=None):
        self.headers = {"authorization": authorization} if authorization else {}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


async def _counts(store):
    async with store.session() as s:
        customers = (await s.execute(select(func.count(CustomerRow.id)))).scalar_one()
        users = (await s.execute(select(func.count(UserRow.id)))).scalar_one()
        operators = (await s.execute(select(func.count(OperatorRow.id)))).scalar_one()
    return customers, users, operators


@pytest.mark.asyncio
async def test_failure_after_the_customer_row_leaves_nothing_and_a_retry_succeeds(store, monkeypatch):
    _use(monkeypatch)
    _verify_as(monkeypatch, "uid_rc06", "rc06@user.test")
    before = await _counts(store)

    # Inject the failure AFTER the customer exists: the seat link blows up.
    real_link = store.link_operator_identity

    async def _boom(*a, **k):
        raise RuntimeError("seat link exploded")

    monkeypatch.setattr(store, "link_operator_identity", _boom)
    with pytest.raises(HTTPException) as e:
        await signup(_Req(f"Bearer {FAKE_JWT}"), store)
    assert e.value.status_code == 500
    assert "seat link exploded" not in str(e.value.detail)  # no exception text egress
    # Compensated: no customer, no user, no operator remains from the attempt.
    assert await _counts(store) == before
    assert await store.get_user_by_id("uid_rc06") is None

    # Retry with the failure gone: exactly one coherent tenant.
    monkeypatch.setattr(store, "link_operator_identity", real_link)
    out = await signup(_Req(f"Bearer {FAKE_JWT}"), store)
    assert out["created"] is True
    user = await store.get_user_by_id("uid_rc06")
    assert user is not None and user.customer_id == out["customer_id"]
    c = await store.get_customer_by_id(out["customer_id"])
    assert c is not None and c.inference_mode == "managed" and c.subscription_tier == "free"
    after = await _counts(store)
    assert after == (before[0] + 1, before[1] + 1, before[2] + 1)

    # A second signup for the same uid is the existing-identity path, not a second tenant.
    again = await signup(_Req(f"Bearer {FAKE_JWT}"), store)
    assert again["customer_id"] == out["customer_id"]
    assert await _counts(store) == after
