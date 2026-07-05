"""Slice 3: LearningService._call_combined_bf runs through the seam.

The codebase's one formerly-async LLM call site now wraps the sync seam in
asyncio.to_thread (tier small) with json_schema structured output. The model
knob (DEFAULT_MODEL) and the lazy AsyncAnthropic client are gone; tests
inject a seam-shaped fake via set_llm_client.

R14 note: these assertions are verified by `pytest`; they describe expected
behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

import json

from crystal_cache.learning.learning_service import (
    COMBINED_SCHEMA,
    LearningService,
)
from crystal_cache.llm import reset_llm_client, set_llm_client


class _RecordingSeam:
    """Seam-shaped fake: records the call kwargs, returns a canned JSON doc."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.last_kwargs = None

    def is_ready(self) -> bool:
        return True

    def complete(self, **kwargs):
        self.last_kwargs = kwargs
        return json.dumps(self._payload)


class _RaisingSeam:
    """Seam-shaped fake whose complete always raises (fail-safe path)."""

    def is_ready(self) -> bool:
        return True

    def complete(self, **kwargs):
        raise RuntimeError("simulated upstream failure")


def _service() -> LearningService:
    # _call_combined_bf touches only the LLM client, so the store/encoder/
    # vector-store collaborators are irrelevant here.
    return LearningService(store=None, encoder=None, vector_store=None)


async def test_call_combined_bf_runs_through_seam():
    payload = {
        "reflection": "Always check the return type before indexing.",
        "category": "API_BEHAVIOR",
        "knowledge": "The call returns a tuple, not a list.",
        "inference": "NO_INFERENCE",
    }
    fake = _RecordingSeam(payload)
    set_llm_client(fake)
    try:
        out = await _service()._call_combined_bf(
            prompt="the task",
            response="def task_func(): ...",
            failure_signal="AssertionError: expected tuple",
            prior_rules=["an earlier rule that failed"],
        )
    finally:
        reset_llm_client()

    assert out == payload
    assert fake.last_kwargs["tier"] == "small"
    assert fake.last_kwargs["max_tokens"] == 768
    assert fake.last_kwargs["json_schema"] is COMBINED_SCHEMA
    # The prompt sections all made it into the single user message.
    user = fake.last_kwargs["messages"][0]["content"]
    assert "the task" in user
    assert "an earlier rule that failed" in user


async def test_call_combined_bf_fail_safe_returns_none():
    set_llm_client(_RaisingSeam())
    try:
        out = await _service()._call_combined_bf(
            prompt="t", response="x", failure_signal="boom", prior_rules=[],
        )
    finally:
        reset_llm_client()
    assert out is None


async def test_call_combined_bf_bad_json_returns_none():
    """A non-JSON completion trips the same fail-safe as an API error."""

    class _ProseSeam:
        def is_ready(self) -> bool:
            return True

        def complete(self, **kwargs):
            return "sorry, here is some prose instead of JSON"

    set_llm_client(_ProseSeam())
    try:
        out = await _service()._call_combined_bf(
            prompt="t", response="x", failure_signal="boom", prior_rules=[],
        )
    finally:
        reset_llm_client()
    assert out is None
