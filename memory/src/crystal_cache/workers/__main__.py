"""Standalone worker process: ``python -m crystal_cache.workers``.

Runs the background workers (crystallization, drive sync, cognition, and the
gated metacognition worker) WITHOUT the HTTP API — the "worker" container in
the docker-compose split, where the API container runs with
``CC_RUN_WORKERS=false`` and serves requests only.

The dependencies are built by ``crystal_cache.runtime.build_core_runtime``,
the same bundle the API lifespan uses, so the two processes can't drift
(A.5 / WS E, decision D3). Schema is owned by migrations (``alembic upgrade
head``) run before this starts, exactly as for the API process — this
entrypoint does not create tables.

Stops cleanly on SIGTERM / SIGINT (``docker stop``) by setting the shared
shutdown event the workers poll, then draining the tasks.
"""
from __future__ import annotations

import asyncio
import signal

import structlog

from ..config import settings
from ..llm import get_llm_client
from ..runtime import build_core_runtime
from . import (
    run_cognition_worker,
    run_crystallization_worker,
    run_drive_sync_worker,
    run_metacognition_worker,
)

logger = structlog.get_logger(__name__)


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop, shutdown_event: asyncio.Event
) -> None:
    """Wire SIGTERM/SIGINT to the shutdown event for a clean container stop.

    ``loop.add_signal_handler`` is the preferred path (POSIX / the Linux
    container). It raises NotImplementedError on Windows' Proactor loop, so
    fall back to ``signal.signal`` there — enough for a local
    ``python -m crystal_cache.workers`` smoke run on Windows.
    """
    def _request_shutdown() -> None:
        logger.info("worker_process.signal_received")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except NotImplementedError:
            signal.signal(sig, lambda *_: shutdown_event.set())


async def _run() -> None:
    logger.info("worker_process.startup", environment=settings.environment)

    core = await build_core_runtime()
    shutdown_event = asyncio.Event()
    _install_signal_handlers(asyncio.get_running_loop(), shutdown_event)

    worker_tasks: list[tuple[asyncio.Task, str]] = [
        (
            asyncio.create_task(run_crystallization_worker(
                store=core.store,
                encoder=core.encoder,
                vector_store=core.vector_store,
                shutdown_event=shutdown_event,
            )),
            "crystallization",
        ),
        (
            asyncio.create_task(run_drive_sync_worker(
                store=core.store,
                shutdown_event=shutdown_event,
            )),
            "drive_sync",
        ),
        (
            asyncio.create_task(run_cognition_worker(
                store=core.store,
                fact_vector_store=core.fact_vector_store,
                encoder=core.encoder,
                shutdown_event=shutdown_event,
            )),
            "cognition",
        ),
    ]

    # Metacognition worker — gated identically to the API lifespan.
    if settings.enable_metacognition_worker:
        worker_tasks.append((
            asyncio.create_task(run_metacognition_worker(
                store=core.store,
                shutdown_event=shutdown_event,
            )),
            "metacognition",
        ))
        logger.info(
            "worker_process.metacog_wired",
            provider_ready=get_llm_client().is_ready(),
        )

    logger.info("worker_process.running", workers=len(worker_tasks))

    try:
        # Block until a signal sets the event; the workers run in the
        # background polling the same event.
        await shutdown_event.wait()
    finally:
        logger.info("worker_process.shutting_down")
        shutdown_event.set()
        for task, name in worker_tasks:
            try:
                await asyncio.wait_for(task, timeout=10)
            except asyncio.TimeoutError:
                logger.warning("worker.shutdown_timeout", worker=name)
                task.cancel()
        await core.store.dispose()
        logger.info("worker_process.stopped")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
