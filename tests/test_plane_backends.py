"""Phase 3 §9 item 5: the live-GCP backend shapes vs fake SDK clients
(2026-07-05).

No network, no GCP: fake storage/run clients prove the layout, signing
fallback, override contents, state mapping, and idempotent teardown —
the live proof against the real project is the operator's runbook step.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from crystal_cache.plane import (
    CloudRunJobsBackend,
    DisposableLimits,
    GcsTaskStore,
    JobState,
    TaskSpec,
)


# --- GCS store vs a fake storage client -----------------------------------------

class FakeBlob:
    def __init__(self, store, path):
        self.store, self.path = store, path
        self.signed = []

    def upload_from_string(self, data, content_type=None):
        self.store.data[self.path] = bytes(data, "utf8") if isinstance(data, str) else data

    def exists(self):
        return self.path in self.store.data

    def download_as_bytes(self):
        return self.store.data[self.path]

    def delete(self):
        del self.store.data[self.path]

    def generate_signed_url(self, **kw):
        self.signed.append(kw)
        self.store.signed.append((self.path, kw))
        return f"https://signed.example/{self.path}?method={kw.get('method')}"


class FakeBucket:
    def __init__(self, store):
        self.store = store

    def blob(self, path):
        return FakeBlob(self.store, path)


class FakeStorageClient:
    def __init__(self, *, keyless=True):
        self.data: dict[str, bytes] = {}
        self.signed: list = []
        # keyless creds (no sign_bytes) exercise the IAM signBlob kwargs path
        if keyless:
            self._credentials = SimpleNamespace(
                sign_bytes=None, valid=True, token="tok-123",
                service_account_email="plane@crystal.iam.gserviceaccount.com",
            )
        else:
            self._credentials = SimpleNamespace(sign_bytes=lambda b: b"sig")

    def bucket(self, name):
        return FakeBucket(self)

    def list_blobs(self, bucket, prefix=""):
        return [FakeBlob(self, p) for p in sorted(self.data) if p.startswith(prefix)]


def _store(client) -> GcsTaskStore:
    return GcsTaskStore("crys-tasks", tenant_id="cus_abc", client=client)


async def test_seed_layout_and_readonly_signed_get():
    client = FakeStorageClient()
    url = await _store(client).put_seed("t1", b"tarball-bytes")
    assert "tasks/cus_abc/t1/seed.tar.gz" in client.data
    path, kw = client.signed[-1]
    assert kw["method"] == "GET" and kw["version"] == "v4"
    assert url.startswith("https://signed.example/tasks/cus_abc/t1/seed.tar.gz")


async def test_keyless_signing_adds_iam_kwargs():
    client = FakeStorageClient(keyless=True)
    await _store(client).put_seed("t1", b"x")
    _, kw = client.signed[-1]
    assert kw["service_account_email"].endswith("gserviceaccount.com")
    assert kw["access_token"] == "tok-123"


async def test_keyed_signing_omits_iam_kwargs():
    client = FakeStorageClient(keyless=False)
    await _store(client).put_seed("t1", b"x")
    _, kw = client.signed[-1]
    assert "service_account_email" not in kw


async def test_harvest_ref_is_writeonly_put_with_pinned_content_type():
    client = FakeStorageClient()
    await _store(client).harvest_ref("t1")
    path, kw = client.signed[-1]
    assert path.endswith("t1/harvest.tar.gz")
    assert kw["method"] == "PUT" and kw["content_type"] == "application/gzip"


async def test_harvest_roundtrip_and_absent_is_none():
    client = FakeStorageClient()
    store = _store(client)
    assert await store.get_harvest("t1") is None
    client.data["tasks/cus_abc/t1/harvest.tar.gz"] = b"harvest-bytes"
    assert await store.get_harvest("t1") == b"harvest-bytes"


async def test_delete_task_prefix_is_scoped_and_idempotent():
    client = FakeStorageClient()
    client.data["tasks/cus_abc/t1/seed.tar.gz"] = b"a"
    client.data["tasks/cus_abc/t1/harvest.tar.gz"] = b"b"
    client.data["tasks/cus_abc/t2/seed.tar.gz"] = b"KEEP"
    store = _store(client)
    await store.delete_task_prefix("t1")
    assert list(client.data) == ["tasks/cus_abc/t2/seed.tar.gz"]
    await store.delete_task_prefix("t1")   # second call: nothing, no raise


# --- Cloud Run Jobs backend vs fake run clients ----------------------------------

class FakeOperation:
    def __init__(self, exec_name):
        self.metadata = SimpleNamespace(name=exec_name)


class FakeJobsClient:
    def __init__(self):
        self.requests = []

    def run_job(self, request):
        self.requests.append(request)
        return FakeOperation("projects/p/locations/r/jobs/j/executions/e-1")


class FakeExecsClient:
    def __init__(self):
        self.state = SimpleNamespace(succeeded_count=0, failed_count=0, cancelled_count=0)
        self.cancelled = []

    def get_execution(self, name):
        return self.state

    def cancel_execution(self, name):
        self.cancelled.append(name)


def _spec(**kw) -> TaskSpec:
    base = dict(
        task_id="t1", image="ignored", command=["python", "run.py"],
        seed_ref="https://signed/seed", harvest_ref="https://signed/harvest",
        task_key="ck_task_x",
    )
    base.update(kw)
    return TaskSpec(**base)


def _backend(jobs, execs) -> CloudRunJobsBackend:
    return CloudRunJobsBackend(
        "crystal-501323", "us-east5", jobs_client=jobs, executions_client=execs,
    )


async def test_launch_overrides_carry_the_ratified_contract():
    jobs, execs = FakeJobsClient(), FakeExecsClient()
    exec_id = await _backend(jobs, execs).launch(
        _spec(limits=DisposableLimits(deadline_seconds=120)),
    )
    assert exec_id.endswith("executions/e-1")
    req = jobs.requests[0]
    assert req.name.endswith("/jobs/crys-task-runner")
    ov = req.overrides.container_overrides[0]
    env = {e.name: e.value for e in ov.env}
    assert env == {
        "CRYS_SEED_REF": "https://signed/seed",
        "CRYS_HARVEST_REF": "https://signed/harvest",
        "CRYS_TASK_KEY": "ck_task_x",
        "CRYS_TASK_ID": "t1",
    }
    assert list(ov.args) == ["python", "run.py"]
    assert req.overrides.timeout.seconds == 180   # deadline + 60s harvest slack


async def test_gpu_spec_targets_the_gpu_job_definition():
    jobs, execs = FakeJobsClient(), FakeExecsClient()
    await _backend(jobs, execs).launch(_spec(gpu=True))
    assert jobs.requests[0].name.endswith("/jobs/crys-task-runner-gpu")


async def test_poll_maps_execution_counters_to_states():
    jobs, execs = FakeJobsClient(), FakeExecsClient()
    backend = _backend(jobs, execs)
    assert await backend.poll("e") == JobState.RUNNING
    execs.state.succeeded_count = 1
    assert await backend.poll("e") == JobState.SUCCEEDED
    execs.state = SimpleNamespace(succeeded_count=0, failed_count=1, cancelled_count=0)
    assert await backend.poll("e") == JobState.FAILED
    execs.state = SimpleNamespace(succeeded_count=0, failed_count=0, cancelled_count=1)
    assert await backend.poll("e") == JobState.FAILED


async def test_cancel_is_idempotent_when_already_terminal():
    jobs, execs = FakeJobsClient(), FakeExecsClient()

    def boom(name):
        raise RuntimeError("execution already completed")

    execs.cancel_execution = boom
    await _backend(jobs, execs).cancel("e-1")   # swallowed, no raise
