"""Cloud Run Jobs JobBackend (Phase 3 §9 item 5, ratified G1/G3/G5).

One execution per task against PRE-CREATED job definitions (the infra
runbook creates them once):

    crys-task-runner       the standard runner image
    crys-task-runner-gpu   the GPU variant (tier-gated at admission)

Two definitions because per-execution overrides can change env, args,
timeout, and task count — but not attach accelerators; GPU is part of
the job's template. Everything task-specific rides the per-execution
override (ratified G2/G3):

    env   CRYS_SEED_REF / CRYS_HARVEST_REF (per-task signed URLs),
          CRYS_TASK_KEY (the box's ONLY credential), CRYS_TASK_ID
    args  the task command, handed to the runner entrypoint

launch() starts the execution WITHOUT waiting (run_remote_task owns the
monitor loop); the execution resource name — available immediately on
the operation's metadata — is the job id for poll/cancel. State mapping
reads the execution's terminal counters: any succeeded task = SUCCEEDED,
any failed/cancelled = FAILED, otherwise RUNNING (task_count is 1, so
the counters are effectively booleans).

Blocking SDK calls are wrapped in asyncio.to_thread, same as the store.
"""
from __future__ import annotations

import asyncio

import structlog

from .remote import JobState, TaskSpec

logger = structlog.get_logger()


class CloudRunJobsBackend:
    """JobBackend protocol implementation over Cloud Run Jobs."""

    def __init__(
        self,
        project: str,
        region: str,
        *,
        job_name: str = "crys-task-runner",
        gpu_job_name: str = "crys-task-runner-gpu",
        jobs_client=None,
        executions_client=None,
    ) -> None:
        self._project = project
        self._region = region
        self._job_name = job_name
        self._gpu_job_name = gpu_job_name
        self._jobs = jobs_client
        self._execs = executions_client

    # -- plumbing ---------------------------------------------------------

    def _jobs_client(self):
        if self._jobs is None:
            from google.cloud import run_v2

            self._jobs = run_v2.JobsClient()
        return self._jobs

    def _execs_client(self):
        if self._execs is None:
            from google.cloud import run_v2

            self._execs = run_v2.ExecutionsClient()
        return self._execs

    def _job_path(self, gpu: bool) -> str:
        name = self._gpu_job_name if gpu else self._job_name
        return f"projects/{self._project}/locations/{self._region}/jobs/{name}"

    # -- JobBackend protocol -------------------------------------------------

    async def launch(self, spec: TaskSpec) -> str:
        def _launch() -> str:
            from google.cloud import run_v2
            from google.protobuf import duration_pb2

            env = [
                run_v2.EnvVar(name="CRYS_SEED_REF", value=spec.seed_ref),
                run_v2.EnvVar(name="CRYS_HARVEST_REF", value=spec.harvest_ref),
                run_v2.EnvVar(name="CRYS_TASK_KEY", value=spec.task_key),
                run_v2.EnvVar(name="CRYS_TASK_ID", value=spec.task_id),
            ]
            override = run_v2.RunJobRequest.Overrides.ContainerOverride(
                env=env, args=list(spec.command), clear_args=not spec.command,
            )
            overrides = run_v2.RunJobRequest.Overrides(
                container_overrides=[override], task_count=1,
            )
            # The job-level timeout backstops the plane's own deadline
            # enforcement (the monitor cancels first in the normal case;
            # this catches a dead plane). +60s slack so the runner can
            # upload a harvest after a task-level trip.
            if spec.limits and spec.limits.deadline_seconds:
                overrides.timeout = duration_pb2.Duration(
                    seconds=int(spec.limits.deadline_seconds) + 60,
                )
            request = run_v2.RunJobRequest(
                name=self._job_path(spec.gpu), overrides=overrides,
            )
            operation = self._jobs_client().run_job(request=request)
            # metadata IS the Execution resource, named immediately —
            # no waiting on the LRO (the monitor loop polls instead).
            return operation.metadata.name

        exec_name = await asyncio.to_thread(_launch)
        logger.info(
            "cloud_run_jobs.launched",
            task_id=spec.task_id, execution=exec_name, gpu=spec.gpu,
        )
        return exec_name

    async def poll(self, job_id: str) -> JobState:
        def _poll() -> JobState:
            e = self._execs_client().get_execution(name=job_id)
            if getattr(e, "succeeded_count", 0) >= 1:
                return JobState.SUCCEEDED
            if getattr(e, "failed_count", 0) >= 1 or getattr(e, "cancelled_count", 0) >= 1:
                return JobState.FAILED
            return JobState.RUNNING

        return await asyncio.to_thread(_poll)

    async def cancel(self, job_id: str) -> None:
        """Idempotent: cancelling a finished or already-cancelled execution
        raises server-side; teardown treats that as done."""
        def _cancel() -> None:
            try:
                self._execs_client().cancel_execution(name=job_id)
            except Exception as e:  # noqa: BLE001 — already terminal is fine
                logger.info("cloud_run_jobs.cancel_noop", execution=job_id, note=str(e)[:120])

        await asyncio.to_thread(_cancel)
