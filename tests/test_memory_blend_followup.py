"""Memory-blend Increment 2 — follow-up retrieval-skip gate.

Covers `retrieval/session_dispatch.is_followup_no_retrieval_needed`
(D-MB2, see docs/MEMORY_BLEND_PLAN.md). Pure unit tests, no network.
"""
from __future__ import annotations

from crystal_cache.retrieval.session_dispatch import (
    FOLLOWUP_MAX_WORDS,
    is_followup_no_retrieval_needed,
)


def _turn(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def _conversation(last_user: str) -> list[dict]:
    """A live conversation (one prior round) ending in `last_user`."""
    return [
        _turn("user", "where is generate_sparse_key defined"),
        _turn("assistant", "It lives in the sparse_keys module."),
        _turn("user", last_user),
    ]


def test_short_followup_skips_retrieval():
    """A short, intent-free follow-up is detected → skip retrieval."""
    assert is_followup_no_retrieval_needed(
        _conversation("yes, give me the exact one")
    ) is True


def test_retrieval_intent_always_retrieves():
    """An explicit lookup mid-conversation still retrieves."""
    assert is_followup_no_retrieval_needed(
        _conversation("what is the crystal bank")
    ) is False


def test_where_is_identity_query_retrieves():
    """Identity/location queries ('where is X') must retrieve even when short."""
    assert is_followup_no_retrieval_needed(
        _conversation("where is the reader module")
    ) is False


def test_first_turn_never_skips():
    """No assistant turn yet → must retrieve (not a follow-up)."""
    assert is_followup_no_retrieval_needed(
        [_turn("user", "thanks, got it")]
    ) is False


def test_long_query_without_intent_retrieves():
    """A long query (over the word cap) without intent is not treated as a
    short follow-up → retrieve."""
    long_q = " ".join(["word"] * (FOLLOWUP_MAX_WORDS + 5))
    assert is_followup_no_retrieval_needed(_conversation(long_q)) is False


def test_session_subject_makes_it_a_followup():
    """A carried-forward subject (Inc 3) turns even a longer intent-free query
    into a follow-up."""
    long_q = " ".join(["word"] * (FOLLOWUP_MAX_WORDS + 5))
    assert is_followup_no_retrieval_needed(
        _conversation(long_q), session_subject="the reader module"
    ) is True
