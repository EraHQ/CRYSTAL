"""Growth G2 — control-plane command channel (ControlExtensionsMixin).

The outbound-poll command state machine: an operator writes a command
(approve/deny / terminate); the agent claims it first-wins (pending→consumed);
staleness voids pending commands. Direct against the in-memory store fixture;
asyncio_mode=auto. (Signature verification is tested separately in
test_control_signing.py — the agent verifies the signed blob this channel
carries.)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from crystal_cache.infrastructure.schema import ControlCommandRow


async def _age_command(store, command_id, seconds):
    """Push a command's created_at into the past (test-only) so claim
    ordering (oldest-first) is deterministic regardless of insert timing."""
    async with store.session() as session:
        row = await session.get(ControlCommandRow, command_id)
        row.created_at = datetime.now(timezone.utc) - timedelta(seconds=seconds)
        await session.commit()


async def test_create_and_claim_first_wins(store, customer):
    sid = "sess_ctl_1"
    cmd = await store.create_control_command(
        sid, customer.id, "req_1", "approval_decision",
        decision="approve", signature="sig-blob", nonce="n1",
    )
    assert cmd["status"] == "pending"
    assert cmd["decision"] == "approve"
    assert cmd["command_type"] == "approval_decision"

    claimed = await store.claim_next_command_for_session(sid)
    assert claimed is not None
    assert claimed["id"] == cmd["id"]
    assert claimed["status"] == "consumed"
    assert claimed["consumed_at"] is not None
    # Carries the signed-authorization envelope the agent verifies.
    assert claimed["signature"] == "sig-blob"
    assert claimed["nonce"] == "n1"

    # First-wins: once consumed there is nothing left to claim.
    assert await store.claim_next_command_for_session(sid) is None


async def test_claim_is_oldest_first(store, customer):
    sid = "sess_ctl_2"
    c1 = await store.create_control_command(
        sid, customer.id, "r1", "approval_decision", decision="deny",
    )
    c2 = await store.create_control_command(
        sid, customer.id, "r2", "terminate",
    )
    # Make c1 unambiguously older.
    await _age_command(store, c1["id"], 5)

    first = await store.claim_next_command_for_session(sid)
    assert first["id"] == c1["id"]
    second = await store.claim_next_command_for_session(sid)
    assert second["id"] == c2["id"]
    assert await store.claim_next_command_for_session(sid) is None


async def test_void_pending_commands(store, customer):
    sid = "sess_ctl_3"
    await store.create_control_command(
        sid, customer.id, "r1", "approval_decision", decision="approve",
    )
    await store.create_control_command(
        sid, customer.id, "r2", "approval_decision", decision="deny",
    )
    voided = await store.void_pending_commands_for_session(sid)
    assert voided == 2

    # Nothing claimable after a void (crash-while-awaiting reclamation).
    assert await store.claim_next_command_for_session(sid) is None
    assert await store.list_commands_for_session(sid, status="pending") == []
    assert len(await store.list_commands_for_session(sid, status="voided")) == 2


async def test_terminate_dependency_command(store, customer):
    sid = "sess_ctl_4"
    cmd = await store.create_control_command(
        sid, customer.id, "term_1", "terminate_dependency",
        dependency_id="sdep_abc",
    )
    assert cmd["command_type"] == "terminate_dependency"
    assert cmd["dependency_id"] == "sdep_abc"
    claimed = await store.claim_next_command_for_session(sid)
    assert claimed["dependency_id"] == "sdep_abc"


async def test_list_commands_newest_first(store, customer):
    sid = "sess_ctl_5"
    older = await store.create_control_command(
        sid, customer.id, "r1", "terminate",
    )
    await _age_command(store, older["id"], 5)
    newer = await store.create_control_command(
        sid, customer.id, "r2", "terminate",
    )
    rows = await store.list_commands_for_session(sid)
    assert [r["id"] for r in rows] == [newer["id"], older["id"]]
