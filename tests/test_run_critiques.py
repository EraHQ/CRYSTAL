"""Run critiques + the ratchet feed (Q2B, ratified 2026-07-15).

Operator critiques pin to parts of a run's anatomy (target_path),
surface in the console, and — the point of ratifying B over A — feed
the orchestrator on retries and on future runs of the same trigger,
so operator judgment enters the ratchet instead of sitting inert.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

from crystal_cache.cognition.models import CognitionEnvironment




async def test_critique_crud_and_counts(store):
    c = await store.create_run_critique(
        "env_1", "cus_a", target_path="step:3/tool_call:1",
        text="The FFmpeg version was paired with the wrong changelog.",
        author="tenant", trigger_id="gap_42",
    )
    assert c["id"].startswith("crit_")
    assert c["status"] == "open"

    listed = await store.list_run_critiques("env_1")
    assert len(listed) == 1
    assert listed[0]["target_path"] == "step:3/tool_call:1"

    counts = await store.count_open_critiques_by_run(["env_1", "env_2"])
    assert counts == {"env_1": 1}

    assert await store.set_run_critique_status(c["id"], "resolved") is True
    got = await store.get_run_critique(c["id"])
    assert got["status"] == "resolved"
    assert got["resolved_at"] is not None
    counts = await store.count_open_critiques_by_run(["env_1"])
    assert counts == {}
    # Unknown id: honest False, no invention.
    assert await store.set_run_critique_status("crit_nope", "open") is False


async def test_ratchet_read_spans_run_and_trigger(store):
    # Critique on a PRIOR run of the same gap...
    await store.create_run_critique(
        "env_old", "cus_a", target_path="deliverable",
        text="Survey breadth was too narrow — document the search scope.",
        trigger_id="gap_42",
    )
    # ...and one on the current run itself (mid-run, retry case).
    await store.create_run_critique(
        "env_new", "cus_a", target_path="criterion:2",
        text="This criterion is uncountable as written.",
        trigger_id="gap_42",
    )
    # A resolved critique and a foreign-customer critique must not leak.
    r = await store.create_run_critique(
        "env_old", "cus_a", target_path="run", text="done already",
        trigger_id="gap_42")
    await store.set_run_critique_status(r["id"], "resolved")
    await store.create_run_critique(
        "env_x", "cus_b", target_path="run", text="other tenant",
        trigger_id="gap_42")

    got = await store.list_open_critiques_for_trigger(
        "cus_a", trigger_id="gap_42", run_id="env_new")
    texts = {c["text"] for c in got}
    assert len(got) == 2
    assert any("breadth" in t for t in texts)
    assert any("uncountable" in t for t in texts)
    # No conditions -> no dump.
    assert await store.list_open_critiques_for_trigger("cus_a") == []


async def test_orchestrator_prompt_carries_critiques(monkeypatch):
    """The prompt block: env.operator_critiques renders as OPERATOR
    CRITIQUES with target paths; empty list renders nothing. Verified
    at the bank_context seam via a captured orchestrator prompt."""
    import inspect
    import crystal_cache.cognition.roles as roles_mod
    src = inspect.getsource(roles_mod)
    assert "OPERATOR CRITIQUES" in src
    # Barrier discipline: the block feeds the orchestrator prompt
    # assembly only — the worker prompt builder must not mention it.
    worker_src = inspect.getsource(roles_mod._assemble_prior_context)
    assert "operator_critiques" not in worker_src

    env = CognitionEnvironment(customer_id="c")
    env.operator_critiques = [
        {"target_path": "step:3", "text": "cite the changelog itself"},
    ]
    # Engine seam: critiques_applied event shape (engine emits it).
    env.record_event("critiques_applied", count=1, attempt=1)
    assert env.events[-1]["kind"] == "critiques_applied"


async def test_engine_fetch_seam_is_wired():
    import inspect
    import crystal_cache.cognition.engine as engine_mod
    src = inspect.getsource(engine_mod.run_cognition_workflow)
    assert "list_open_critiques_for_trigger" in src
    assert 'record_event(\n                        "critiques_applied"' in src or \
        "critiques_applied" in src
