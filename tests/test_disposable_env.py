"""Disposable execution environment — Phase 1 (2026-07-03, step 5 part 2).

The lifecycle contract + local workspace: seed a copy of the working dir,
run an autonomous worker in the COPY (never the original), enforce time and
cost limits, and tear the workspace down UNCONDITIONALLY — even on crash.

Imports the CRYS module directly (CRYS CRYS has no pytest
suite of its own; its security-critical pure logic is tested from here).

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_CA = Path(__file__).resolve().parents[1] / "CRYS"
if str(_CA) not in sys.path:
    sys.path.insert(0, str(_CA))

from crystal_code.disposable import (  # noqa: E402
    DisposableLimits,
    LocalWorkspaceEnv,
    SeedMode,
    TaskOutcome,
    run_disposable_task,
)


# --- seeding ---------------------------------------------------------------

async def test_copy_seed_gives_worker_the_working_dir(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "app.py").write_text("print(1)\n")
    (proj / "data").mkdir()
    (proj / "data" / "x.txt").write_text("hello")

    seen = {}

    async def worker(handle, stop):
        seen["app"] = (handle.root / "app.py").read_text()
        seen["nested"] = (handle.root / "data" / "x.txt").read_text()
        seen["root"] = handle.root
        return {"ok": True}

    env = LocalWorkspaceEnv(proj, seed_mode=SeedMode.COPY)
    result = await run_disposable_task(env, worker)

    assert result.outcome == TaskOutcome.COMPLETED
    assert seen["app"] == "print(1)\n"
    assert seen["nested"] == "hello"


async def test_worker_edits_do_not_touch_the_original(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "app.py").write_text("original\n")

    async def worker(handle, stop):
        # Wreck the copy.
        (handle.root / "app.py").write_text("WRECKED\n")
        (handle.root / "new_junk.txt").write_text("junk")
        return None

    env = LocalWorkspaceEnv(proj, seed_mode=SeedMode.COPY)
    await run_disposable_task(env, worker)

    # The original is untouched; the junk never appeared there.
    assert (proj / "app.py").read_text() == "original\n"
    assert not (proj / "new_junk.txt").exists()


async def test_empty_seed_is_empty(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "app.py").write_text("x")

    async def worker(handle, stop):
        return {"contents": [p.name for p in handle.root.iterdir()]}

    env = LocalWorkspaceEnv(proj, seed_mode=SeedMode.EMPTY)
    result = await run_disposable_task(env, worker)
    assert result.harvest["contents"] == []


async def test_path_seed_copies_named_path(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "irrelevant.txt").write_text("no")
    other = tmp_path / "other"
    other.mkdir()
    (other / "wanted.txt").write_text("yes")

    async def worker(handle, stop):
        return {"has_wanted": (handle.root / "wanted.txt").exists(),
                "has_irrelevant": (handle.root / "irrelevant.txt").exists()}

    env = LocalWorkspaceEnv(proj, seed_mode=SeedMode.PATH, seed_path=other)
    result = await run_disposable_task(env, worker)
    assert result.harvest["has_wanted"] is True
    assert result.harvest["has_irrelevant"] is False


# --- teardown is unconditional ---------------------------------------------

async def test_teardown_destroys_workspace_on_success(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "f.txt").write_text("x")
    captured = {}

    async def worker(handle, stop):
        captured["root"] = handle.root
        assert handle.root.exists()
        return None

    env = LocalWorkspaceEnv(proj)
    await run_disposable_task(env, worker)
    # The workspace is gone.
    assert not captured["root"].exists()


async def test_teardown_runs_even_when_worker_crashes(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    captured = {}

    async def worker(handle, stop):
        captured["root"] = handle.root
        raise RuntimeError("worker blew up")

    env = LocalWorkspaceEnv(proj)
    result = await run_disposable_task(env, worker)

    assert result.outcome == TaskOutcome.FAILED
    assert "blew up" in result.error
    assert not captured["root"].exists()  # torn down despite the crash


# --- limits ----------------------------------------------------------------

async def test_deadline_stops_the_task(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()

    async def slow_worker(handle, stop):
        # A cooperative worker polls the stop-check between steps.
        for _ in range(100):
            if stop() is not None:
                return None
            await asyncio.sleep(0.02)
        return {"finished": True}

    env = LocalWorkspaceEnv(proj)
    result = await run_disposable_task(
        env, slow_worker,
        limits=DisposableLimits(deadline_seconds=0.05),
    )
    assert result.outcome == TaskOutcome.DEADLINE_EXCEEDED
    assert result.harvest is None  # nothing harvested from a killed task


async def test_budget_stops_the_task(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()

    # Cost climbs each time it's read.
    calls = {"n": 0}

    async def cost_reader():
        calls["n"] += 1
        return calls["n"] * 1000  # micro-USD

    async def worker(handle, stop):
        for _ in range(100):
            if stop() is not None:
                return None
            await asyncio.sleep(0.001)
        return {"finished": True}

    env = LocalWorkspaceEnv(proj)
    result = await run_disposable_task(
        env, worker,
        limits=DisposableLimits(budget_micro_usd=2500),
        cost_reader=cost_reader,
    )
    # The live monitor catches the climbing cost during the run.
    assert result.outcome == TaskOutcome.BUDGET_EXCEEDED


async def test_already_over_budget_stops_before_work(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    ran = {"v": False}

    async def cost_reader():
        return 10_000  # already way over

    async def worker(handle, stop):
        ran["v"] = True
        return {"finished": True}

    env = LocalWorkspaceEnv(proj)
    result = await run_disposable_task(
        env, worker,
        limits=DisposableLimits(budget_micro_usd=5000),
        cost_reader=cost_reader,
    )
    assert result.outcome == TaskOutcome.BUDGET_EXCEEDED
    assert ran["v"] is False  # worker never ran


async def test_no_limits_runs_to_completion(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()

    async def worker(handle, stop):
        return {"done": True}

    env = LocalWorkspaceEnv(proj)
    result = await run_disposable_task(env, worker)
    assert result.outcome == TaskOutcome.COMPLETED
    assert result.harvest == {"done": True}
    assert result.duration_seconds >= 0
