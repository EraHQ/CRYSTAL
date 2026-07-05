"""The CRYS daemon — a persistent worker that drains the agent_tasks queue.

`python -m crystal_code --daemon` starts a long-running process that
polls the `agent_tasks` table (the cognition-worker pattern: database
tables ARE the message queues), claims the oldest queued task, executes
the proven F8 composed background run, and writes the report back to
the row. Producers are the CLI (`--queue --task "..."`) and the agent
itself (the guarded `queue_task` tool) — which is the point: work the
agent queues gets completed without user intervention.

Safety posture is inherited wholesale from background.py, which is
already fully non-interactive: auto-approved on a fresh git branch
ONLY, shell and browser denied, block_paths enforced, ground-truth
verification by the CLI, and the user's branch restored no matter what.
The daemon adds queue hygiene on top: any task still marked 'running'
at startup belonged to a daemon that died mid-run — it is marked
FAILED with a note, never silently re-executed against the partial
git state the dead run may have left.

Per-task output goes to ~/.crystal-code/tasks/<id>.log (the full
transcript) and the tail is stored on the row as the report.

v1 scope, stated: one task at a time (the encoder and the LLM budget
are shared resources — sequential is the honest default); a foreground
process (detach it with your platform's tools: `start` on Windows,
`nohup`/`&` on unix); one daemon per database (the claim is
select-then-update without row locking — the seam is documented in
metadata_store_agent_ext.py). The agent is rebuilt per task (~20-30s
encoder load) — correctness over warm-start cleverness in v1.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import time
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone

from . import config_store
from . import schedule, style

POLL_SECONDS = 5

# Daemon liveness signal (June 2026, CRYS live test). A queued task on
# a daemon-less machine sat silently forever — the "start a daemon"
# hint was one dim line. The daemon ticks this file from an independent
# asyncio task (so it stays fresh even while a long run executes), and
# both producers check it AT QUEUE TIME to warn loudly. File, not DB
# row: the queue is machine-local by design (project_dir is a local
# path), so a machine-local signal is honest — and a crashed daemon
# stops looking alive within HEARTBEAT_STALE_SECONDS.
HEARTBEAT_FILE = config_store.CONFIG_DIR / "daemon.heartbeat"
HEARTBEAT_STALE_SECONDS = 30


def _write_heartbeat(db: Optional[str]) -> None:
    try:
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT_FILE.write_text(
            f"{db or 'local (default store in the launch folder)'}\n",
            encoding="utf-8",
        )
    except OSError:
        pass  # a heartbeat hiccup must never touch the daemon's work


def daemon_running() -> bool:
    """Best-effort: a daemon on THIS machine ticked within the stale
    window. Used for warnings only — never to gate queueing."""
    try:
        return (time.time() - HEARTBEAT_FILE.stat().st_mtime) < HEARTBEAT_STALE_SECONDS
    except OSError:
        return False


async def _heartbeat_loop(db: Optional[str]) -> None:
    while True:
        _write_heartbeat(db)
        await asyncio.sleep(POLL_SECONDS)

# Recurring series park after this many CONSECUTIVE failures. The
# failure mode this prevents: a broken recurring task silently burning
# LLM budget on schedule, forever. Three is enough to rule out flake;
# the park is loud (a 'series parked' line + note on the row).
SERIES_FAILURE_CAP = 3


async def _recur_if_scheduled(store, task: dict, status: str) -> None:
    """After a recurring occurrence finishes, enqueue its successor.

    Fixed-rate wall-clock recurrence (see crystal_code/schedule.py):
    the next row's run_at lands on the anchor grid, skipping anything
    missed. Failures increment series_failures (carried on the child
    row, reset on success); at SERIES_FAILURE_CAP the series parks
    instead of recurring — the last row's report says so.
    """
    if not task.get("recur_seconds"):
        return
    # A --cancel between claim and finish nulls recur_seconds to stop the
    # series; re-read so cancelling the RUNNING occurrence takes effect (the
    # claim-time dict still carries the pre-cancel value).
    current = await store.get_agent_task(task["id"])
    if current is not None and not current.get("recur_seconds"):
        print(f"{time.strftime('%H:%M:%S')}  "
              + style.dim(f"series cancelled — not rescheduling {task['id']}"))
        return
    failures = 0 if status == "done" else int(task.get("series_failures") or 0) + 1
    if failures >= SERIES_FAILURE_CAP:
        print(
            f"{time.strftime('%H:%M:%S')}  "
            + style.yellow(
                f"series parked  {task['id']}  — {failures} consecutive "
                "failures; not rescheduling. Fix the task and re-queue it."
            )
        )
        return
    anchor = task.get("run_at") or task.get("created_at")
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=dt_timezone.utc)
    next_run = schedule.next_occurrence(
        anchor, int(task["recur_seconds"]), datetime.now(dt_timezone.utc)
    )
    child = await store.create_agent_task(
        task["customer_id"],
        project_dir=task["project_dir"],
        task=task["task"],
        branch=None,  # each occurrence gets its own default branch name
        source=task["source"],
        run_at=next_run,
        recur_seconds=int(task["recur_seconds"]),
        parent_task_id=task["id"],
        series_failures=failures,
    )
    when = schedule.describe_schedule(next_run, None)
    line = f"recurs → {child['id']}  {when}"
    print(f"{time.strftime('%H:%M:%S')}  {style.dim(line)}")


REPORT_TAIL_CHARS = 4000

TASKS_LOG_DIR = config_store.CONFIG_DIR / "tasks"


def _tail(path: Path, chars: int = REPORT_TAIL_CHARS) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(log unreadable)"
    text = text.strip()
    if len(text) > chars:
        return f"... (truncated to the last {chars} chars)\n" + text[-chars:]
    return text or "(no output)"


async def _execute_task(task: dict, db: Optional[str]) -> tuple[str, str, str]:
    """Run one claimed task through the F8 background runner.

    Returns (status, report, log_path). All output the run prints —
    plan pass, execute pass, verification, report — is captured to the
    per-task log file; the tail becomes the stored report.
    """
    from .background import run_background_task

    TASKS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = TASKS_LOG_DIR / f"{task['id']}.log"

    project_dir = Path(task["project_dir"])
    status = "failed"
    with open(log_path, "w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            try:
                if not project_dir.is_dir():
                    print(f"refused: project folder does not exist: {project_dir}")
                    rc = 2
                else:
                    rc = await run_background_task(
                        project_dir,
                        task["task"],
                        task["branch"] or None,
                        db,
                        task["customer_id"],
                        # A failed gap retry must not mint a SECOND gap —
                        # that's the loop the one-retry cap exists to prevent.
                        write_gap_on_failure=(task.get("source") != "gap_retry"),
                        source_task_id=task["id"],
                    )
                status = "done" if rc == 0 else "failed"
            except Exception as e:  # noqa: BLE001 — one bad task must not kill the daemon
                print(f"\ndaemon: task crashed — {type(e).__name__}: {e}")
                status = "failed"
    return status, _tail(log_path), str(log_path)


async def run_daemon(db: Optional[str]) -> int:
    """The poll loop. Ctrl+C to stop (finishes nothing mid-flight —
    the current task's run_background_task already finalizes/restores
    on KeyboardInterrupt via its own try/finally)."""
    # Late imports: the daemon shares the CLI's store plumbing.
    from crystal_cache.infrastructure.metadata_store import set_metadata_store

    from .runtime import _make_store, _resolve_db_url

    store = _make_store(_resolve_db_url(db))
    await store.init()  # creates agent_tasks (and any missing tables) on local stores
    if await _refuse_if_schema_stale(store, "daemon refused"):
        await store.dispose()
        return 2
    set_metadata_store(store)
    stale = await store.fail_stale_running_tasks(
        "daemon restarted while this task was running — review the task's "
        "branch manually before re-queueing"
    )
    reopened_gaps = await store.reopen_stale_retrying_gaps()
    gap_prefix = store.gap_retry_branch_prefix()

    print()
    print(style.rule())
    print(style.bold("CRYS daemon — background task worker"))
    print(f"  queue   : {db or 'local (default store in the launch folder)'}")
    print(f"  logs    : {TASKS_LOG_DIR}")
    if stale:
        print(style.yellow(f"  cleanup : {stale} task(s) were stuck 'running' from a dead daemon — marked failed"))
    if reopened_gaps:
        print(style.yellow(f"  cleanup : {reopened_gaps} gap(s) were stuck 'retrying' from a dead daemon — reopened"))
    print(style.dim("  polling every 5s — queue work with --queue or the agent's queue_task tool. Ctrl+C to stop."))
    print(style.rule())
    print()

    hb_task = asyncio.create_task(_heartbeat_loop(db))
    done = failed = 0
    try:
        while True:
            task = await store.claim_next_agent_task()
            if task is None:
                # Idle pass (Phase C): no due work → give one open gap its
                # single retry. Enqueued as a normal task (source
                # 'gap_retry', fresh branch named after the gap, parent
                # lineage to the failed run) so the ordinary claim/
                # execute/finish machinery handles it — the queue never
                # waits on gap work, because gaps are only touched when
                # the queue is empty.
                gap = await store.claim_next_open_agent_gap()
                if gap is not None:
                    parsed = store.parse_agent_gap_missing(gap["missing"])
                    if not parsed["project"] or not parsed["task"]:
                        await store.resolve_agent_gap(gap["id"], status="needs_operator")
                        print(style.yellow(
                            f"{time.strftime('%H:%M:%S')}  ⚠ gap {gap['id']} is missing its "
                            "PROJECT/TASK sections — marked needs_operator"
                        ))
                        continue
                    retry_prompt = (
                        f"{parsed['task']}\n\n"
                        "NOTE: a previous attempt at this task failed verification. "
                        f"The failed work is preserved on branch '{parsed['branch']}' for reference. "
                        "Its failing verify output was:\n\n"
                        f"{parsed['failure'][:1500]}\n\n"
                        "Investigate the cause of that failure FIRST, then attempt the task."
                    )
                    parent = parsed["task_id"] if parsed["task_id"].startswith("atask_") else None
                    retry = await store.create_agent_task(
                        gap["customer_id"],
                        project_dir=parsed["project"],
                        task=retry_prompt,
                        branch=f"{gap_prefix}{gap['id']}",
                        source="gap_retry",
                        parent_task_id=parent,
                    )
                    print(f"{time.strftime('%H:%M:%S')}  {style.bold('gap retry')} queued {retry['id']} "
                          f"{style.dim('for ' + gap['id'])}")
                    continue
                await asyncio.sleep(POLL_SECONDS)
                continue
            label = task["task"] if len(task["task"]) <= 70 else task["task"][:67] + "..."
            print(f"{time.strftime('%H:%M:%S')}  {style.bold('claimed')} {task['id']}  {style.dim(label)}")
            status, report, log_path = await _execute_task(task, db)
            await store.finish_agent_task(
                task["id"],
                status=status,
                report=report,
                error=None if status == "done" else "see report/log",
                log_path=log_path,
            )
            if status == "done":
                done += 1
                print(f"{time.strftime('%H:%M:%S')}  {style.green('done')}    {task['id']}  {style.dim(log_path)}")
            else:
                failed += 1
                print(f"{time.strftime('%H:%M:%S')}  {style.red('failed')}  {task['id']}  {style.dim(log_path)}")
            await _recur_if_scheduled(store, task, status)
            # Gap retry resolution: the branch name carries the gap id.
            if task.get("source") == "gap_retry" and (task["branch"] or "").startswith(gap_prefix):
                gap_id = task["branch"][len(gap_prefix):]
                if status == "done":
                    await store.resolve_agent_gap(gap_id, status="filled")
                    print(f"{time.strftime('%H:%M:%S')}  {style.green('gap filled')} {gap_id}")
                else:
                    await store.resolve_agent_gap(gap_id, status="needs_operator")
                    print(style.yellow(style.rule()))
                    print(style.yellow(f"⚠ NEEDS OPERATOR — gap {gap_id}"))
                    print(style.yellow(
                        "  The retry also failed verification. The agent is missing something "
                        "a third identical run won't find."
                    ))
                    print(style.yellow(f"  Both attempts are preserved on their branches; see --tasks for the gap list."))
                    print(style.yellow(style.rule()))
    except KeyboardInterrupt:
        print(f"\ndaemon stopped. completed {done}, failed {failed} this session.")
        return 0
    finally:
        hb_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await hb_task
        # Remove the heartbeat on the way out so producers see the truth
        # immediately instead of waiting out the stale window.
        with contextlib.suppress(OSError):
            HEARTBEAT_FILE.unlink()
        await store.dispose()


def format_task_line(t: dict) -> str:
    """One queue row for /tasks and --tasks listings."""
    status = t["status"]
    colored = {
        "queued": style.dim(status),
        "running": style.yellow(status),
        "done": style.green(status),
        "failed": style.red(status),
        "cancelled": style.dim(status),
    }.get(status, status)
    label = t["task"] if len(t["task"]) <= 60 else t["task"][:57] + "..."
    src = style.dim(f"[{t['source']}]")
    when = schedule.describe_schedule(t.get("run_at"), t.get("recur_seconds"))
    sched = style.dim(f" ↻ {when}") if t.get("recur_seconds") else (
        style.dim(f" ({when})") if when and status == "queued" else ""
    )
    return f"  {colored:<18} {t['id']}  {src} {label}{sched}"


def _jsonable_task(t: dict) -> dict:
    """Tool outputs must be JSON-serializable — datetimes to ISO strings."""
    out = dict(t)
    for k in ("created_at", "started_at", "finished_at", "run_at"):
        if out.get(k) is not None:
            out[k] = out[k].isoformat()
    return out


async def _refuse_if_schema_stale(store, doing: str) -> bool:
    """True (and prints the fix) when the store predates the ORM schema.

    Every entry point that opens a store runs this. The rule earned its
    generality the hard way: v1 of the detection covered only the REPL
    and the daemon, calling the enqueue path 'overkill' — and the first
    live `--queue --every` against a stale launch-folder store hit a
    raw OperationalError instead of the message we'd already built.
    """
    check = await store.check_schema_compatibility()
    if not check["mismatches"]:
        return False
    from .runtime import schema_mismatch_message
    print(style.yellow(f"{doing}: " + schema_mismatch_message(check)))
    return True


async def enqueue_cli(
    db: Optional[str],
    project_dir: Path,
    task: str,
    branch: Optional[str],
    customer_id: Optional[str],
    at: Optional[str] = None,
    every: Optional[str] = None,
) -> int:
    """`--queue --task "..."` (+ optional --at/--every): enqueue and exit.

    Schedule parsing happens HERE, at the boundary, so a bad schedule
    string is a clean refusal before any row exists. Local → UTC
    conversion per the schedule.py boundary rule. `--every` without
    `--at` anchors the series at now + interval (the first run is one
    interval from now — 'every 4h' starting immediately would surprise).
    """
    from .runtime import LOCAL_CUSTOMER_ID, _make_store, _resolve_db_url

    try:
        run_at_utc, recur_seconds = _parse_schedule_args(at, every)
    except ValueError as e:
        print(style.yellow(f"not queued: {e}"))
        return 2

    store = _make_store(_resolve_db_url(db))
    try:
        await store.init()
        if await _refuse_if_schema_stale(store, "not queued"):
            return 2
        row = await store.create_agent_task(
            customer_id or LOCAL_CUSTOMER_ID,
            project_dir=str(project_dir),
            task=task,
            branch=branch,
            source="cli",
            run_at=run_at_utc,
            recur_seconds=recur_seconds,
        )
    finally:
        await store.dispose()
    when = schedule.describe_schedule(row.get("run_at"), row.get("recur_seconds"))
    print(f"queued {row['id']}" + (f"  ({when})" if when else ""))
    if daemon_running():
        print(style.dim("  a running daemon will pick it up."))
    else:
        print(style.yellow(
            "  ⚠ no daemon is running — this task will NOT execute until "
            "you start one: python -m crystal_code --daemon"
        ))
    return 0


def _parse_schedule_args(
    at: Optional[str], every: Optional[str]
) -> tuple[Optional[datetime], Optional[int]]:
    """(--at, --every) → (run_at UTC | None, recur_seconds | None).

    Shared by the CLI flags and the queue_task tool so both surfaces
    accept and refuse exactly the same strings.
    """
    recur_seconds = schedule.parse_every(every) if every else None
    if at:
        run_at_utc = schedule.local_to_utc(schedule.parse_at(at))
    elif recur_seconds:
        run_at_utc = datetime.now(dt_timezone.utc) + timedelta(seconds=recur_seconds)
    else:
        run_at_utc = None
    return run_at_utc, recur_seconds


async def list_tasks_cli(db: Optional[str]) -> int:
    """`--tasks`: print the queue, newest first, and exit."""
    from .runtime import _make_store, _resolve_db_url

    store = _make_store(_resolve_db_url(db))
    try:
        await store.init()
        if await _refuse_if_schema_stale(store, "can't read the queue"):
            return 2
        rows = await store.list_agent_tasks(limit=30)
        gaps = await store.list_agent_gaps(statuses=["open", "retrying", "needs_operator"], limit=20)
    finally:
        await store.dispose()
    if not rows and not gaps:
        print("no background tasks yet — queue one with --queue --task \"...\" or the agent's queue_task tool.")
        return 0
    print()
    for t in rows:
        print(format_task_line(t))
    if gaps:
        print()
        print(style.bold("  knowledge gaps") + style.dim("  (failed runs awaiting their one retry, or your attention)"))
        for g in gaps:
            mark = {"open": "○", "retrying": "↻", "needs_operator": style.yellow("⚠")}.get(g["status"], "?")
            line = f"  {mark} {g['id']}  [{g['status']}]  {(g['subject'] or '')[:70]}"
            print(style.yellow(line) if g["status"] == "needs_operator" else line)
        if any(g["status"] == "needs_operator" for g in gaps):
            print(style.yellow("  ⚠ needs_operator gaps have exhausted their retry — both branches are preserved for review."))
    print()
    print(style.dim(f"  full logs: {TASKS_LOG_DIR}"))
    return 0


async def cancel_cli(db: Optional[str], task_id: str) -> int:
    """`--cancel <task_id>`: cancel a queued task or stop a recurring series.

    A running occurrence can't be stopped mid-flight (the daemon owns the
    process), but its series is stopped so it won't recur. Find the id in
    `--tasks`.
    """
    from .runtime import _make_store, _resolve_db_url

    store = _make_store(_resolve_db_url(db))
    try:
        await store.init()
        if await _refuse_if_schema_stale(store, "can't cancel"):
            return 2
        res = await store.cancel_agent_task(task_id)
    finally:
        await store.dispose()

    outcome = res["outcome"]
    if outcome == "not_found":
        print(style.yellow(f"no task {task_id!r} — check --tasks for the id."))
        return 2
    if outcome == "cancelled":
        tail = " Recurring series stopped." if res["was_recurring"] else ""
        print(f"{style.green('cancelled')} {task_id}.{tail}")
        return 0
    if outcome == "recurrence_stopped":
        print(style.yellow(
            f"task {task_id} is running — the daemon will finish this occurrence, "
            "but the recurring series is stopped (no future occurrences)."
        ))
        return 0
    if outcome == "running_uncancelable":
        print(style.yellow(
            f"task {task_id} is already running and can't be cancelled mid-run; "
            "the daemon will finish it. (Not a recurring task — nothing to stop.)"
        ))
        return 0
    # already_terminal
    print(f"task {task_id} is already {res['status']} — nothing to cancel.")
    return 0


def register_queue_tools(project_dir: Path, store_ref: dict) -> None:
    """Register `queue_task` and `get_task_status` for the REPL agent.

    store_ref is the mutable-reference pattern (like the subagent's
    parent_ref): the tools are registered BEFORE build_agent runs, and
    the CLI fills store_ref["store"] right after — the impls read it at
    call time.

    Guard interaction, by design rather than accident: `queue_task`
    classifies as 'unknown' (fail-closed) so the guard PROMPTS —
    correct, because queueing IS approving a future auto-approved
    headless run on this project. `get_task_status` starts with `get_`
    so it classifies as a read and flows freely.

    Schema staleness: the REPL's store passed build_agent's check
    before it ever reaches store_ref, so the tools don't re-check.
    """
    from crystal_cache.agent import Tool, get_registry

    registry = get_registry()
    if "queue_task" in registry:
        return

    async def _queue_impl(
        customer_id: str, task: str = "", branch: str = "",
        at: str = "", every: str = "", **kwargs: Any,
    ) -> dict:
        store = store_ref.get("store")
        if store is None:
            return {"error": "the task queue is not available yet", "is_error": True}
        if not task.strip():
            return {"error": "queue_task requires a non-empty 'task'", "is_error": True}
        try:
            run_at_utc, recur_seconds = _parse_schedule_args(
                at.strip() or None, every.strip() or None
            )
        except ValueError as e:
            return {"error": str(e), "is_error": True}
        row = await store.create_agent_task(
            customer_id,
            project_dir=str(project_dir),
            task=task.strip(),
            branch=branch.strip() or None,
            source="agent",
            run_at=run_at_utc,
            recur_seconds=recur_seconds,
        )
        when = schedule.describe_schedule(row.get("run_at"), row.get("recur_seconds"))
        alive = daemon_running()
        if not alive:
            # Print directly too — the user must see this even if the
            # model fails to relay the note (found live 2026-06-12: an
            # 8 AM recurring task was queued with no daemon anywhere,
            # and the only warning was buried mid-paragraph).
            print(style.yellow(
                "    ⚠ no daemon is running — queued work will NOT execute "
                "until one starts: python -m crystal_code --daemon"
            ))
        if alive:
            note = (
                "queued. A running daemon will execute it on a fresh git "
                "branch and store a report; check with get_task_status "
                "or /tasks."
            )
        else:
            note = (
                "queued — but NO DAEMON IS RUNNING on this machine, so "
                "this task will NOT execute until the user starts one "
                "with: python -m crystal_code --daemon. Tell the user "
                "this plainly and prominently — do not bury it."
            )
        return {
            "task_id": row["id"],
            "status": row["status"],
            "schedule": when or "asap",
            "daemon_running": alive,
            "note": note,
        }

    async def _status_impl(customer_id: str, task_id: str = "", **kwargs: Any) -> dict:
        store = store_ref.get("store")
        if store is None:
            return {"error": "the task queue is not available yet", "is_error": True}
        if task_id.strip():
            row = await store.get_agent_task(task_id.strip())
            if row is None:
                return {"error": f"no task {task_id!r}", "is_error": True}
            return {"task": _jsonable_task(row)}
        rows = await store.list_agent_tasks(limit=10)
        return {"tasks": [_jsonable_task(r) for r in rows], "count": len(rows)}

    registry.register(Tool(
        name="queue_task",
        description=(
            "Queue a coding task for the background daemon to execute later "
            "without supervision (on a fresh git branch, with the project's "
            "verify command as ground truth). Use when the user wants work "
            "done in the background, later, at a specific time, or on a "
            "repeating schedule — the user approves each queued task. "
            "Provide a complete, self-contained task description; the "
            "daemon run starts with no conversation context. Scheduling: "
            "'at' for a start time, 'every' for recurrence (each occurrence "
            "is a separate full run — when proposing a recurring task, tell "
            "the user their one approval covers ALL future occurrences)."
        ),
        contexts=frozenset({"agent"}),
        parameters_schema={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Complete, self-contained description of the work."},
                "branch": {"type": "string", "description": "Optional git branch name (default: agent/task-<timestamp>)."},
                "at": {"type": "string", "description": "Optional start time, local: 'HH:MM' (next occurrence of that time) or 'YYYY-MM-DD HH:MM'. Omit for ASAP."},
                "every": {"type": "string", "description": "Optional recurrence interval: '30m', '4h', '1d', '2w', or 'hourly'/'daily'/'weekly' (min 60s). With 'at', the series anchors there; without, first run is one interval from now."},
            },
            "required": ["task"],
        },
        impl=_queue_impl,
    ))
    registry.register(Tool(
        name="get_task_status",
        description=(
            "Check the background task queue: pass task_id for one task's "
            "status and report, or omit it to list the 10 most recent "
            "tasks (queued/running/done/failed)."
        ),
        contexts=frozenset({"agent"}),
        parameters_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Optional task id from queue_task."},
            },
        },
        impl=_status_impl,
    ))
