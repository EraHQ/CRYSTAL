"""Memory-blend Increment 3 — session consumption.

Covers:
  - `retrieval/session_dispatch.session_subject_from_last_log` (pure)
  - `MetadataStore.get_last_query_log_for_sequence` (async, DB-backed)

D-MB3 / D-MB4, see docs/MEMORY_BLEND_PLAN.md. The DB-backed carry-forward
replaces v1's module-global session dict.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from crystal_cache.models import QueryLog
from crystal_cache.retrieval.session_dispatch import (
    is_followup_no_retrieval_needed,
    session_subject_from_last_log,
)


# ---------------------------------------------------------------------------
# session_subject_from_last_log — pure
# ---------------------------------------------------------------------------

def test_subject_none_when_no_prior_log():
    assert session_subject_from_last_log(None) is None


def test_subject_from_matched_facts():
    log = SimpleNamespace(
        query_text="how does crystallization work",
        matched_facts=["fact_1"],
        routed_crystal_id=None,
    )
    assert session_subject_from_last_log(log) == "how does crystallization work"


def test_subject_from_routed_crystal_only():
    log = SimpleNamespace(
        query_text="the cognition loop",
        matched_facts=[],
        routed_crystal_id="crys_42",
    )
    assert session_subject_from_last_log(log) == "the cognition loop"


def test_no_subject_when_prior_matched_nothing():
    log = SimpleNamespace(
        query_text="random unmatched question",
        matched_facts=[],
        routed_crystal_id=None,
    )
    assert session_subject_from_last_log(log) is None


def test_subject_carries_forward_into_followup_detection():
    """A longer, intent-free query that would otherwise retrieve becomes a
    follow-up once a subject is carried forward from the prior matched turn."""
    convo = [
        {"role": "user", "content": "how does crystallization work"},
        {"role": "assistant", "content": "It clusters related facts."},
        {"role": "user", "content": (
            "and then walk me through the part after that in a lot more "
            "detail than you just did please"
        )},
    ]
    # Without a subject: long + no intent → retrieve.
    assert is_followup_no_retrieval_needed(convo) is False
    # With a carried-forward subject → follow-up.
    assert is_followup_no_retrieval_needed(
        convo, session_subject="how does crystallization work"
    ) is True


# ---------------------------------------------------------------------------
# get_last_query_log_for_sequence — async, DB-backed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_last_query_log_returns_latest_turn(customer, store):
    """Two logged turns in one sequence → the method returns the later one."""
    seq = "seq_inc3_latest"

    await store.write_query_log(QueryLog(
        id="qlog_inc3_t0",
        customer_id=customer.id,
        query_text="first turn",
        match_type="none",
        sequence_id=seq,
        turn_index=0,
    ))
    await store.write_query_log(QueryLog(
        id="qlog_inc3_t1",
        customer_id=customer.id,
        query_text="second turn",
        match_type="high",
        matched_facts=["fact_x"],
        routed_crystal_id="crys_x",
        sequence_id=seq,
        turn_index=1,
    ))

    last = await store.get_last_query_log_for_sequence(
        customer_id=customer.id,
        sequence_id=seq,
    )
    assert last is not None
    assert last.id == "qlog_inc3_t1"
    assert last.query_text == "second turn"
    # And it feeds a usable subject signal.
    assert session_subject_from_last_log(last) == "second turn"


@pytest.mark.asyncio
async def test_get_last_query_log_none_for_unknown_sequence(customer, store):
    """No logs for the sequence → None."""
    last = await store.get_last_query_log_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_inc3_does_not_exist",
    )
    assert last is None
