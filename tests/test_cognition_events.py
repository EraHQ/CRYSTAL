"""Lifecycle events for the Inspector (ratified 2026-07-14, Q1C).

The machinery narrates itself: continuation, empty-retry, agentic
sessions, envelope digests, contract amendments, research degrades —
previously log-only — now land on env.events (capped, serialized in
to_dict, flowing to the detail endpoint via the snapshot untouched).

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

from crystal_cache.cognition.models import CognitionEnvironment


def test_record_event_shape_cap_and_serialization():
    env = CognitionEnvironment(customer_id="c")
    env.record_event("composition_continued", step_id=3,
                     continuations=2, chars=41000)
    e = env.events[0]
    assert e["kind"] == "composition_continued"
    assert e["step_id"] == 3
    assert e["continuations"] == 2
    assert "ts" in e
    # step_id omitted when None.
    env.record_event("validator_envelopes", parts=2, chars=250000)
    assert "step_id" not in env.events[1]
    # Cap: a pathological loop can't bloat the snapshot.
    for i in range(300):
        env.record_event("x", n=i)
    assert len(env.events) == 200
    # to_dict serializes the tail only.
    d = env.to_dict()
    assert len(d["events"]) == 100
    assert d["events"][-1]["n"] == 299


async def test_events_emitted_through_the_seats(monkeypatch):
    """One integration pass: continuation + empty-retry events emit
    from the composition path; the engine's amendment stanza and the
    envelope digester carry record_event calls (source anchors)."""
    import inspect
    from types import SimpleNamespace

    import crystal_cache.config as config_mod
    from crystal_cache.cognition.models import (
        Plan, PlanStep, StepAction, StepOutput, StepStatus,
    )
    from crystal_cache.cognition.roles import _worker_llm_step
    from crystal_cache.llm import reset_llm_client, set_llm_client
    from crystal_cache.llm.client import LLMResult

    # Self-contained fakes (2026-07-15 fix: this file previously
    # imported them from tests.test_cognition_agentic, which resolves
    # only when the repo root happens to be on sys.path — it failed on
    # Windows. Test modules never import test modules.)
    def _flag(mp, on: bool):
        mp.setattr(config_mod, "get_settings",
                   lambda: SimpleNamespace(cognition_agentic_workers=on))

    def _analyze_env():
        e = CognitionEnvironment(customer_id="c")
        e.plan = Plan(steps=[
            PlanStep(id=1, action=StepAction.ANALYZE, description="a")])
        return e

    class _EmptyThenGood:
        def __init__(self):
            self.calls = 0
            self._script = [("", "end_turn"), ("ok", "end_turn")]

        def complete_detailed(self, **kw):
            self.calls += 1
            text, stop = self._script.pop(0)
            return LLMResult(text=text, model="fake", input_tokens=1,
                             output_tokens=1, stop_reason=stop)

        def is_ready(self):
            return True

    _flag(monkeypatch, False)
    env = _analyze_env()
    set_llm_client(_EmptyThenGood())
    try:
        await _worker_llm_step(
            env, env.plan.steps[0],
            StepOutput(step_id=1, action="analyze",
                       status=StepStatus.RUNNING),
        )
    finally:
        reset_llm_client()
    kinds = [e["kind"] for e in env.events]
    assert "composition_empty_retry" in kinds

    # Source anchors for the seats not exercised here.
    import crystal_cache.cognition.engine as engine_mod
    import crystal_cache.cognition.roles as roles_mod
    import crystal_cache.cognition.agentic as agentic_mod
    assert 'record_event("contract_amended"' in inspect.getsource(engine_mod)
    assert 'record_event("validator_envelopes"' in inspect.getsource(roles_mod)
    assert 'record_event("agentic_step"' in inspect.getsource(roles_mod)
    assert 'record_event("agentic_fallback"' in inspect.getsource(roles_mod)
    assert 'record_event("research_degraded"' in inspect.getsource(agentic_mod)


def test_salvaged_flag_flows_from_render_to_finding(monkeypatch):
    from crystal_cache.search import fetch as fetch_mod
    from crystal_cache.search import render as render_mod

    def fake_render(url, *, timeout_seconds=20.0, resolver=None):
        return {"url": url, "title": "t",
                "content": "x" * 900, "salvaged": True}

    monkeypatch.setattr(render_mod, "render_and_extract", fake_render)
    payload = {"results": [
        {"title": "", "url": "https://spa.example.com/page",
         "snippet": "", "content": None},
    ]}

    def fake_fetch(url, **kw):
        # Static extract thin -> render fallback fires.
        return {"url": url, "title": "", "content": "thin"}

    monkeypatch.setattr(fetch_mod, "fetch_and_extract", fake_fetch,
                        raising=False)
    out = fetch_mod.fill_missing_content(
        payload, max_pages=1, content_cap=30000,
        render_enabled=True, render_timeout_seconds=5.0,
    )
    r = out["results"][0]
    assert r.get("rendered") is True
    assert r.get("salvaged") is True
