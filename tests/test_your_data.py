"""Your data (2026-10-03; Q34=A export, Q35=A import-restore, Q36=A erase
and delete, Q37=B seven-day grace with a step-up confirmation).

Real sqlite store, real rows, real routes. The erase is schema-driven, so
the pin seeds rows across tables keyed by customer_id, crystal_id and
the crystal-reference columns (chains, edges), then proves what survives
and what does not, by scope.
"""
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from crystal_cache.infrastructure.schema import (
    CrystalChainRow, CrystalEdgeRow, CrystalRow, FactRow, LlmCallRow,
    OperatorRow, QueryLogRow,
)


async def _seed_bank(store, customer_id, tag):
    async with store.session() as s:
        s.add(CrystalRow(id=f"cr_{tag}_a", customer_id=customer_id, summary_vector=[],
                         summary_text=f"{tag} a"))
        s.add(CrystalRow(id=f"cr_{tag}_b", customer_id=customer_id, summary_vector=[],
                         summary_text=f"{tag} b"))
        await s.flush()
        s.add(FactRow(id=f"f_{tag}_1", crystal_id=f"cr_{tag}_a", claim_text="one"))
        s.add(FactRow(id=f"f_{tag}_2", crystal_id=f"cr_{tag}_b", claim_text="two"))
        s.add(CrystalChainRow(source_crystal_id=f"cr_{tag}_a", target_crystal_id=f"cr_{tag}_b"))
        s.add(CrystalEdgeRow(crystal_a_id=f"cr_{tag}_a", crystal_b_id=f"cr_{tag}_b",
                             edge_type="co_query"))


async def _counts(store, customer_id, tag):
    from sqlalchemy import func, select

    async with store.session() as s:
        crystals = (await s.execute(
            select(func.count(CrystalRow.id)).where(CrystalRow.customer_id == customer_id)
        )).scalar_one()
        facts = (await s.execute(
            select(func.count(FactRow.id)).where(FactRow.crystal_id.like(f"cr_{tag}_%"))
        )).scalar_one()
        chains = (await s.execute(
            select(func.count()).select_from(CrystalChainRow)
            .where(CrystalChainRow.source_crystal_id.like(f"cr_{tag}_%"))
        )).scalar_one()
        edges = (await s.execute(
            select(func.count()).select_from(CrystalEdgeRow)
            .where(CrystalEdgeRow.crystal_a_id.like(f"cr_{tag}_%"))
        )).scalar_one()
        ledger = (await s.execute(
            select(func.count(LlmCallRow.id)).where(LlmCallRow.customer_id == customer_id)
        )).scalar_one()
        operators = (await s.execute(
            select(func.count(OperatorRow.id)).where(OperatorRow.team_id == customer_id)
        )).scalar_one()
    return dict(crystals=crystals, facts=facts, chains=chains, edges=edges,
                ledger=ledger, operators=operators)


@pytest.mark.asyncio
async def test_erase_empties_the_bank_and_keeps_the_account(store, customer):
    other = await store.create_customer(provider="anthropic", model_id="m", api_key_ref="")
    await _seed_bank(store, customer.id, "mine")
    await _seed_bank(store, other.id, "theirs")
    await store.record_llm_call(customer.id, model="claude-haiku-4-5", input_tokens=1,
                                output_tokens=1, billing="managed", price_table={})
    before = await _counts(store, customer.id, "mine")
    assert before["crystals"] == 2 and before["facts"] == 2 and before["chains"] == 1
    assert before["edges"] == 1 and before["ledger"] == 1 and before["operators"] >= 1

    counts = await store.erase_tenant_bank(customer.id)
    assert counts.get("crystals") == 2
    assert counts.get("facts") == 2
    assert counts.get("crystal_chains") == 1
    assert counts.get("crystal_edges") == 1

    after = await _counts(store, customer.id, "mine")
    assert after["crystals"] == 0 and after["facts"] == 0
    assert after["chains"] == 0 and after["edges"] == 0
    # The account survives an erase: ledger, operators (Key A), the row.
    assert after["ledger"] == 1 and after["operators"] >= 1
    assert await store.get_customer_by_id(customer.id) is not None
    # The neighbour is untouched.
    theirs = await _counts(store, other.id, "theirs")
    assert theirs["crystals"] == 2 and theirs["facts"] == 2 and theirs["chains"] == 1


@pytest.mark.asyncio
async def test_export_then_erase_then_import_round_trips(store, customer):
    await _seed_bank(store, customer.id, "rt")
    export = await store.export_bank_topology(customer.id)
    assert len(export["crystals"]) == 2 and len(export["facts"]) == 2
    await store.erase_tenant_bank(customer.id)
    assert (await _counts(store, customer.id, "rt"))["crystals"] == 0
    counts = await store.import_bank_topology(customer.id, export)
    assert counts["crystals"] == 2 and counts["facts"] == 2 and counts["chains"] == 1
    after = await _counts(store, customer.id, "rt")
    assert after["crystals"] == 2 and after["facts"] == 2 and after["chains"] == 1


@pytest.mark.asyncio
async def test_schedule_locks_revokes_and_restore_unlocks(store, customer):
    from crystal_cache.infrastructure.metadata_store_erase_ext import _aware

    purge_after = await store.schedule_account_deletion(customer.id)
    c = await store.get_customer_by_id(customer.id)
    assert c.deletion_scheduled_at is not None
    assert _aware(c.purge_after) == purge_after
    assert purge_after - _aware(c.deletion_scheduled_at) == timedelta(days=7)
    # Idempotent: asking twice keeps the first date.
    assert await store.schedule_account_deletion(customer.id) == purge_after

    old_hash_customer = await store.get_customer_by_api_key(customer.api_key)
    assert old_hash_customer is not None
    revoked = await store.revoke_tenant_access(customer.id)
    assert revoked["api_key_rotated"] == 1
    assert await store.get_customer_by_api_key(customer.api_key) is None  # old key dead

    assert await store.cancel_account_deletion(customer.id) is True
    c = await store.get_customer_by_id(customer.id)
    assert c.deletion_scheduled_at is None and c.purge_after is None
    assert await store.cancel_account_deletion(customer.id) is False


@pytest.mark.asyncio
async def test_lock_refuses_every_route_but_me_restore_and_export(store, customer):
    from crystal_cache.ingress import auth

    await store.schedule_account_deletion(customer.id)
    c = await store.get_customer_by_id(customer.id)
    for path in ("/v1/me", "/v1/me/restore", "/v1/export/topology"):
        auth.refuse_if_deleting(c, path)  # allowed
    for path in ("/v1/billing/checkout", "/v1/store", "/admin/api/crystals", "/v1/me/erase"):
        with pytest.raises(HTTPException) as e:
            auth.refuse_if_deleting(c, path)
        assert e.value.status_code == 403
        assert e.value.headers["X-Account-State"] == "deleting"
    # The admin guard refuses the tenant's console session too.
    err, pin = await auth.tenant_admin_error(
        "GET", "/admin/api/crystals", f"Bearer {customer.api_key}", store,
    )
    assert err is not None and err[0] == 403


@pytest.mark.asyncio
async def test_purge_removes_the_account_and_tombstones_the_ledger(store, customer):
    from crystal_cache.workers.purge import purge_due_accounts

    await _seed_bank(store, customer.id, "pg")
    await store.record_llm_call(customer.id, model="claude-haiku-4-5", input_tokens=1,
                                output_tokens=1, billing="managed", price_table={})
    await store.schedule_account_deletion(customer.id)
    # Not due yet: nothing happens.
    assert await purge_due_accounts(store) == []
    assert await store.get_customer_by_id(customer.id) is not None

    # Force the date into the past and run the pass.
    async with store.session() as s:
        from crystal_cache.infrastructure.schema import CustomerRow

        row = await s.get(CustomerRow, customer.id)
        row.purge_after = datetime.now(timezone.utc) - timedelta(seconds=1)
        await s.commit()
    assert await purge_due_accounts(store) == [customer.id]

    assert await store.get_customer_by_id(customer.id) is None
    after = await _counts(store, customer.id, "pg")
    assert after["crystals"] == 0 and after["facts"] == 0 and after["operators"] == 0
    assert after["ledger"] == 0  # no longer keyed by the tenant...
    from sqlalchemy import func, select

    async with store.session() as s:
        tomb = (await s.execute(
            select(func.count(LlmCallRow.id)).where(LlmCallRow.customer_id.like("deleted:%"))
        )).scalar_one()
    assert tomb == 1  # ...but the accounting record survives, anonymized


# ----------------------------------------------------------------------------
# Routes: owner only, step-up, typed confirmation
# ----------------------------------------------------------------------------

def _session(monkeypatch, store, uid, email, *, age_seconds=5):
    from crystal_cache.config import Settings
    from crystal_cache.ingress import auth as auth_mod

    monkeypatch.setattr(auth_mod, "get_settings", lambda: Settings(firebase_project_id="test-proj"))
    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt",
        lambda tok, proj: {"sub": uid, "email": email, "email_verified": True,
                           "firebase": {"sign_in_provider": "google.com"},
                           "auth_time": int(time.time()) - age_seconds},
    )


class _Req:
    def __init__(self, body=None, path="/v1/me/erase"):
        self.headers = {"authorization": "Bearer eyJx.eyJy.sig"}
        self._body = body or {}
        self.url = SimpleNamespace(path=path)
        self.app = SimpleNamespace(state=SimpleNamespace())

    async def json(self):
        return self._body


@pytest.mark.asyncio
async def test_erase_route_needs_owner_fresh_sign_in_and_the_word(store, customer, monkeypatch):
    from crystal_cache.endpoints import me as me_mod

    await store.create_user("uid_own", "own@test.dev", customer.id, "owner")
    await _seed_bank(store, customer.id, "rt2")

    _session(monkeypatch, store, "uid_own", "own@test.dev", age_seconds=3600)
    with pytest.raises(HTTPException) as e:
        await me_mod.erase_my_memories(_Req({"confirm": "ERASE"}), store)
    assert e.value.status_code == 401 and e.value.headers["X-Step-Up"] == "required"

    _session(monkeypatch, store, "uid_own", "own@test.dev")
    with pytest.raises(HTTPException) as e:
        await me_mod.erase_my_memories(_Req({"confirm": "erase"}), store)
    assert e.value.status_code == 400
    assert (await _counts(store, customer.id, "rt2"))["crystals"] == 2  # nothing happened

    out = await me_mod.erase_my_memories(_Req({"confirm": "ERASE"}), store)
    assert out["erased"] is True and out["rows"]["crystals"] == 2
    assert (await _counts(store, customer.id, "rt2"))["crystals"] == 0


@pytest.mark.asyncio
async def test_delete_route_locks_and_restore_route_unlocks(store, customer, monkeypatch):
    from crystal_cache.endpoints import me as me_mod

    await store.create_user("uid_own2", "own2@test.dev", customer.id, "owner")
    _session(monkeypatch, store, "uid_own2", "own2@test.dev")
    out = await me_mod.delete_my_account(_Req({"confirm": "DELETE"}, path="/v1/me/delete"), store)
    assert out["scheduled"] is True and out["api_key_rotated"] == 1
    c = await store.get_customer_by_id(customer.id)
    assert c.deletion_scheduled_at is not None
    # /v1/me still reports, with the dates.
    me = await me_mod.get_me(_Req(path="/v1/me"), store)
    assert me["deletion_scheduled_at"] and me["purge_after"]
    out = await me_mod.restore_my_account(_Req(path="/v1/me/restore"), store)
    assert out["restored"] is True
    assert (await store.get_customer_by_id(customer.id)).deletion_scheduled_at is None


@pytest.mark.asyncio
async def test_non_owner_cannot_erase_or_delete(store, customer, monkeypatch):
    """User.role is owner or platform_admin. A platform admin attached to
    a tenant is NOT its owner, and must not be able to erase or delete it
    through the owner routes."""
    from crystal_cache.endpoints import me as me_mod

    await store.create_user("uid_padmin", "padmin@test.dev", customer.id, "platform_admin")
    _session(monkeypatch, store, "uid_padmin", "padmin@test.dev")
    with pytest.raises(HTTPException) as e:
        await me_mod.erase_my_memories(_Req({"confirm": "ERASE"}), store)
    assert e.value.status_code == 403
    with pytest.raises(HTTPException) as e:
        await me_mod.delete_my_account(_Req({"confirm": "DELETE"}, path="/v1/me/delete"), store)
    assert e.value.status_code == 403
