"""Phase 3 slice 2: TaskSpec + run_remote_task vs fakes (2026-07-03).

Proves the plane-side remote shape before any GCP exists: seed packing
reuses the SHARED seed_workspace (semantics identical to local boxes),
harvest unpacking is traversal-safe, and the runner owns the limits —
cancel on trip, revoke + prefix-delete on EVERY exit path.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import asyncio
import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "CRYS"))

from crystal_code.disposable import DisposableLimits, SeedMode, TaskOutcome
from crystal_code.remote_env import (
    JobState,
    TaskSpec,
    pack_seed,
    run_remote_task,
    unpack_harvest,
)


# --- fakes --------------------------------------------------------------------

class InMemoryTaskStore:
    def __init__(self, harvest: bytes | None = None):
        self.seeds: dict[str, bytes] = {}
        self._harvest = harvest
        self.deleted: list[str] = []

    async def put_seed(self, task_id: str, data: bytes) -> str:
        self.seeds[task_id] = data
        return f"mem://seeds/{task_id}"

    async def harvest_ref(self, task_id: str) -> str:
        return f"mem://harvest/{task_id}"

    async def get_harvest(self, task_id: str):
        return self._harvest

    async def delete_task_prefix(self, task_id: str) -> None:
        self.deleted.append(task_id)


class FakeJobBackend:
    """Scripted job lifecycle: runs for `polls_until_done` polls then ends
    in `final` state. `never_finish` runs until cancelled."""

    def __init__(self, *, polls_until_done: int = 2,
                 final: JobState = JobState.SUCCEEDED,
                 never_finish: bool = False,
                 launch_error: bool = False):
        self.polls = 0
        self.cancelled: list[str] = []
        self.launched: list[TaskSpec] = []
        self._until = polls_until_done
        self._final = final
        self._never = never_finish
        self._launch_error = launch_error

    async def launch(self, spec: TaskSpec) -> str:
        if self._launch_error:
            raise RuntimeError("no capacity")
        self.launched.append(spec)
        return f"job-{spec.task_id}"

    async def poll(self, job_id: str) -> JobState:
        self.polls += 1
        if self._never or self.polls < self._until:
            return JobState.RUNNING
        return self._final

    async def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)


def _spec(task_id="t1", **kw) -> TaskSpec:
    return TaskSpec(
        task_id=task_id, image="crys-runner:test", command=["run"],
        seed_ref="mem://seeds/t1", harvest_ref="mem://harvest/t1",
        task_key="ck_task_test", **kw,
    )


def _tar_gz_of(name: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


# --- seed packing: shared semantics with local boxes ---------------------------

def test_pack_seed_copy_roundtrips(tmp_path):
    src = tmp_path / "proj"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("alpha")
    (src / "sub" / "b.txt").write_text("beta")

    data = pack_seed(SeedMode.COPY, working_dir=src)
    out = tmp_path / "out"
    unpack_harvest(data, out)   # same tar format both directions
    assert (out / "a.txt").read_text() == "alpha"
    assert (out / "sub" / "b.txt").read_text() == "beta"


def test_pack_seed_empty_and_missing_source_fail_safe(tmp_path):
    for data in (
        pack_seed(SeedMode.EMPTY),
        pack_seed(SeedMode.COPY, working_dir=tmp_path / "does-not-exist"),
    ):
        out = tmp_path / f"o{len(data)}"
        unpack_harvest(data, out)
        assert list(out.iterdir()) == []   # empty box, no raise


def test_unpack_harvest_refuses_traversal(tmp_path):
    """A hostile harvest (path traversal) is REFUSED LOUDLY — the data
    filter raises rather than extracting outside dest. The harvest crossed
    the trust boundary; an explicit failure beats a silent drop."""
    evil = _tar_gz_of("../escape.txt", b"nope")
    dest = tmp_path / "dest"
    with pytest.raises(tarfile.OutsideDestinationError):
        unpack_harvest(evil, dest)
    assert not (tmp_path / "escape.txt").exists()


# --- the runner: limits, harvest, unconditional teardown -----------------------

async def _revoker(log):
    async def _r():
        log.append("revoked")
    return _r


async def test_success_path_harvests_revokes_and_cleans(tmp_path):
    harvest = _tar_gz_of("report.md", b"# done")
    store = InMemoryTaskStore(harvest=harvest)
    backend = FakeJobBackend(polls_until_done=3)
    log: list[str] = []

    result = await run_remote_task(
        _spec(), backend, store,
        revoke_key=await _revoker(log), poll_seconds=0.001,
    )
    assert result.outcome == TaskOutcome.COMPLETED
    assert result.harvest is not None
    out = tmp_path / "h"
    unpack_harvest(result.harvest["tar_gz"], out)
    assert (out / "report.md").read_text() == "# done"
    assert log == ["revoked"]
    assert store.deleted == ["t1"]
    assert backend.cancelled == []   # clean finish needs no cancel


async def test_deadline_trip_cancels_revokes_cleans():
    store = InMemoryTaskStore()
    backend = FakeJobBackend(never_finish=True)
    log: list[str] = []

    result = await run_remote_task(
        _spec(limits=DisposableLimits(deadline_seconds=0.05)),
        backend, store, revoke_key=await _revoker(log), poll_seconds=0.001,
    )
    assert result.outcome == TaskOutcome.DEADLINE_EXCEEDED
    assert backend.cancelled          # the box was stopped
    assert log == ["revoked"]
    assert store.deleted == ["t1"]


async def test_budget_trip_reads_the_cost_reader():
    store = InMemoryTaskStore()
    backend = FakeJobBackend(never_finish=True)
    spend = {"v": 0}

    async def reader() -> int:
        spend["v"] += 400   # grows every poll
        return spend["v"]

    result = await run_remote_task(
        _spec(limits=DisposableLimits(budget_micro_usd=1000)),
        backend, store, cost_reader=reader, poll_seconds=0.001,
    )
    assert result.outcome == TaskOutcome.BUDGET_EXCEEDED
    assert result.cost_micro_usd >= 1000
    assert backend.cancelled


async def test_job_failure_is_failed_but_still_torn_down():
    store = InMemoryTaskStore()
    backend = FakeJobBackend(polls_until_done=2, final=JobState.FAILED)
    log: list[str] = []

    result = await run_remote_task(
        _spec(), backend, store,
        revoke_key=await _revoker(log), poll_seconds=0.001,
    )
    assert result.outcome == TaskOutcome.FAILED
    assert result.harvest is None
    assert log == ["revoked"]
    assert store.deleted == ["t1"]


async def test_launch_crash_still_revokes_and_cleans():
    store = InMemoryTaskStore()
    backend = FakeJobBackend(launch_error=True)
    log: list[str] = []

    result = await run_remote_task(
        _spec(), backend, store,
        revoke_key=await _revoker(log), poll_seconds=0.001,
    )
    assert result.outcome == TaskOutcome.FAILED
    assert "no capacity" in (result.error or "")
    assert log == ["revoked"]
    assert store.deleted == ["t1"]
