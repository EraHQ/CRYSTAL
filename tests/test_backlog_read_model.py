"""Never-Idle Convergence — unified backlog read-model (list_backlog, D6).

Aggregates waiting work across six queues into one ranked, normalized view.
Covers: empty bank → empty; all six sources represented; terminal/in-progress
states excluded; ranking (priority desc, then oldest-first within a priority);
tenant scoping; and the limit cap.

Rows are inserted directly with controlled status + created_at so ranking and
status filtering are deterministic (conftest in-memory store; asyncio_mode=auto).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from crystal_cache.infrastructure.schema import (
    AgentTaskRow,
    CognitionTaskRow,
    KnowledgeConflictRow,
    KnowledgeGapRow,
    PushReviewQueueRow,
    VerificationTaskRow,
)

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _at(minutes: int) -> datetime:
    return _T0 + timedelta(minutes=minutes)


# ---------------------------------------------------------------------------
# Seed helpers — one row per source, status + created_at controllable
# ---------------------------------------------------------------------------

async def _gap(store, cid, *, gid, status="open", priority="medium", subject="a gap", at=0):
    async with store.session() as s:
        s.add(KnowledgeGapRow(
            id=gid, customer_id=cid, domain="d", subject=subject, missing="missing X",
            priority=priority, status=status, source="llm_observation", created_at=_at(at),
        ))


async def _conflict(store, cid, *, kid, status="open", subject="a subject", at=0):
    async with store.session() as s:
        s.add(KnowledgeConflictRow(
            id=kid, customer_id=cid, fact_a_id="fa", fact_b_id="fb",
            claim_a="A", claim_b="B", subject=subject, pair_key=f"pk_{kid}",
            status=status, created_at=_at(at),
        ))


async def _cog(store, cid, *, tid, status="pending", priority="background", topic="research T", at=0):
    async with store.session() as s:
        s.add(CognitionTaskRow(
            id=tid, customer_id=cid, task_type="research", payload={"topic": topic},
            priority=priority, status=status, created_at=_at(at),
        ))


async def _agent(store, cid, *, aid, status="queued", task="build the thing", at=0):
    async with store.session() as s:
        s.add(AgentTaskRow(
            id=aid, customer_id=cid, project_dir="/p", task=task,
            status=status, source="cli", created_at=_at(at),
        ))


async def _push(store, cid, *, pid, status="pending", key="some key", at=0):
    async with store.session() as s:
        s.add(PushReviewQueueRow(
            id=pid, customer_id=cid, key=key, value="v", confidence=0.7,
            source="llm_observation", status=status, created_at=_at(at),
        ))


async def _verif(store, cid, *, vid, status="pending", priority=0.5, claim="verify this", at=0):
    async with store.session() as s:
        s.add(VerificationTaskRow(
            id=vid, customer_id=cid, candidate_claim=claim, candidate_vector=[],
            priority=priority, status=status, created_at=_at(at),
        ))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_empty_backlog_is_empty(store, customer):
    assert await store.list_backlog(customer.id) == []


async def test_aggregates_all_six_sources(store, customer):
    await _gap(store, customer.id, gid="g1", at=1)
    await _conflict(store, customer.id, kid="c1", at=2)
    await _cog(store, customer.id, tid="t1", at=3)
    await _agent(store, customer.id, aid="a1", at=4)
    await _push(store, customer.id, pid="p1", at=5)
    await _verif(store, customer.id, vid="v1", at=6)

    backlog = await store.list_backlog(customer.id)
    kinds = {it["kind"] for it in backlog}
    assert kinds == {
        "gap", "conflict", "cognition_task", "agent_task",
        "push_review", "verification",
    }
    assert len(backlog) == 6
    # Common shape present on every item.
    for it in backlog:
        assert set(it) == {"kind", "id", "subject", "status", "priority_score", "created_at"}


async def test_excludes_terminal_and_in_progress(store, customer):
    # Terminal / in-progress rows across every source — none should surface.
    await _gap(store, customer.id, gid="g_filled", status="filled", at=1)
    await _conflict(store, customer.id, kid="c_resolved", status="resolved", at=2)
    await _cog(store, customer.id, tid="t_running", status="running", at=3)
    await _cog(store, customer.id, tid="t_done", status="complete", at=4)
    await _agent(store, customer.id, aid="a_done", status="done", at=5)
    await _push(store, customer.id, pid="p_approved", status="approved", at=6)
    await _verif(store, customer.id, vid="v_resolved", status="resolved", at=7)

    assert await store.list_backlog(customer.id) == []


async def test_ranking_priority_then_age(store, customer):
    # Scores: gap_high=3, conflict=2 (@1), verification(0.5)=2 (@3), gap_low=1.
    await _gap(store, customer.id, gid="g_low", priority="low", at=0)        # score 1
    await _conflict(store, customer.id, kid="c_mid", at=1)                   # score 2 @1
    await _verif(store, customer.id, vid="v_mid", priority=0.5, at=3)        # score 2 @3
    await _gap(store, customer.id, gid="g_high", priority="high", at=2)      # score 3

    backlog = await store.list_backlog(customer.id)
    order = [it["id"] for it in backlog]
    # priority desc; within the two score-2 items, oldest (@1) before (@3).
    assert order == ["g_high", "c_mid", "v_mid", "g_low"]


async def test_tenant_scoped(store, customer):
    other = await store.create_customer(
        provider="anthropic", model_id="claude-sonnet-4-5-20250929",
        api_key_ref="sk-test-other",
    )
    await _gap(store, other.id, gid="g_other", at=1)
    await _gap(store, customer.id, gid="g_mine", at=1)

    backlog = await store.list_backlog(customer.id)
    assert [it["id"] for it in backlog] == ["g_mine"]


async def test_limit_caps_after_ranking(store, customer):
    await _gap(store, customer.id, gid="g_low", priority="low", at=0)
    await _gap(store, customer.id, gid="g_high", priority="high", at=1)
    await _conflict(store, customer.id, kid="c_mid", at=2)

    top2 = await store.list_backlog(customer.id, limit=2)
    assert len(top2) == 2
    # Highest-priority two survive the cap.
    assert [it["id"] for it in top2] == ["g_high", "c_mid"]
