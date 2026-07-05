"""F8 — Background agents: hand off a whole task, review a branch.

    python -m crystal_code <project> --task "fix the flaky retry test" \
        --branch agent/fix-retry

Headless, no REPL. The run composes everything F1–F7 shipped:

  1. PREFLIGHT — must be a git repo with a CLEAN worktree (uncommitted
     work would otherwise ride along into the agent's branch and get
     committed under its name — refusing is the only honest answer).
     A fresh branch is created; your original branch is restored at the
     end no matter what.
  2. PLAN PASS — the agent investigates in plan mode (read-only,
     subagents available) and produces a numbered plan.
  3. EXECUTE PASS — auto-approved writes (the branch is the safety
     boundary; block_paths still hold), verify-loop enforcement
     included.
  4. FINALIZE — the CLI runs the verify command ITSELF for a
     ground-truth PASS/FAIL (the agent's claim is not the evidence),
     commits the changes to the agent branch, switches you back, and
     prints the report: plan, diffstat, verification verdict.

v1 is a foreground process — park it in another terminal. A detached
daemon is a later iteration.
"""
from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path
from typing import Any, Optional


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=120
    )


# ---------------------------------------------------------------------------
# Git mechanics (pure; validated standalone)
# ---------------------------------------------------------------------------

def preflight(project_dir: Path, branch: str) -> dict:
    """Checks + branch creation. Returns {ok, error, original_branch}."""
    if _git(["rev-parse", "--git-dir"], project_dir).returncode != 0:
        return {"ok": False, "error": "not a git repository — background agents are branch-scoped by design", "original_branch": None}
    dirty = _git(["status", "--porcelain"], project_dir).stdout.strip()
    if dirty:
        n = len(dirty.splitlines())
        return {"ok": False, "error": f"the worktree has {n} uncommitted change(s) — commit or stash first, so your work can't end up in the agent's commit", "original_branch": None}
    head = _git(["rev-parse", "--abbrev-ref", "HEAD"], project_dir)
    original = head.stdout.strip() if head.returncode == 0 else None
    if not original:
        return {"ok": False, "error": "could not determine the current branch (empty repository?)", "original_branch": None}
    if _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], project_dir).returncode == 0:
        return {"ok": False, "error": f"branch '{branch}' already exists — pick another name", "original_branch": original}
    if _git(["checkout", "-b", branch], project_dir).returncode != 0:
        return {"ok": False, "error": f"could not create branch '{branch}'", "original_branch": original}
    return {"ok": True, "error": None, "original_branch": original}


def finalize(project_dir: Path, branch: str, original_branch: str, task: str) -> dict:
    """Commit the agent's changes, diffstat, and restore the user's branch.

    Returns {committed, diffstat, restore_ok}.
    """
    out: dict = {"committed": False, "diffstat": "", "restore_ok": False}
    if _git(["status", "--porcelain"], project_dir).stdout.strip():
        _git(["add", "-A"], project_dir)
        msg = f"crystal-code background agent: {' '.join(task.split())[:120]}"
        commit = _git(
            ["-c", "user.name=crystal-code",
             "-c", "user.email=agent@crystal-code.local",
             "commit", "-m", msg],
            project_dir,
        )
        out["committed"] = commit.returncode == 0
    diff = _git(["diff", "--stat", f"{original_branch}..{branch}"], project_dir)
    out["diffstat"] = diff.stdout.strip()
    out["restore_ok"] = _git(["checkout", original_branch], project_dir).returncode == 0
    return out


async def run_verify_for_report(project_dir: Path, verify_command: str) -> dict:
    """Ground-truth verification for the final report — the CLI runs the
    command itself rather than trusting the agent's claim (R14).

    E1c (2026-07-03): routed through the shared sandbox chokepoint so the
    headless background agent's ground-truth verify is contained like every
    other model-influenced execution. The verify command is
    operator-configured (standing consent) and may use shell features, so
    allow_shell_features=True.
    """
    from .sandbox import SandboxProfile, sandbox_run_async

    result = await sandbox_run_async(
        verify_command, project_dir, 600,
        allow_shell_features=True,
        profile=SandboxProfile.CPU,
    )
    if result.timed_out:
        return {"passed": False, "tail": "(verify command timed out after 600s)"}
    return {"passed": not result.is_error, "tail": result.output[-1200:]}


# ---------------------------------------------------------------------------
# The headless run
# ---------------------------------------------------------------------------

async def run_background_task(
    project_dir: Path,
    task: str,
    branch: Optional[str],
    db: Optional[str],
    customer_id: Optional[str],
    *,
    write_gap_on_failure: bool = True,
    source_task_id: Optional[str] = None,
) -> int:
    # Imports here keep `preflight`/`finalize` importable without the
    # full library (and standalone-testable).
    from . import config_store, style
    from .checkpoints import CheckpointManager
    from .cli import _RECALL_ENABLED, _system_for, _turn_heartbeat, _stop_heartbeat
    from .guard import Guard, WRITE_TOOLS
    from .instructions import ProjectInstructions
    from .mcp_hands import MCPHands, MCPHandsError
    from .project_config import load_project_config
    from .recall import maybe_recall
    from .runtime import LOCAL_CUSTOMER_ID, build_agent, resolve_models
    from .session_registry import SessionHandle
    from .subagent import register_subagent_tool
    from .verify import register_verify_tool, set_verify_result_sink

    from crystal_cache.agent.turn_finalize import finalize_agent_turn

    creds = config_store.resolve_credentials()
    if creds is None:
        print("No credentials found. Run interactively once (python -m crystal_code) to set up.")
        return 2

    branch = branch or f"agent/task-{int(time.time())}"
    print()
    print(style.rule())
    print(style.bold("crystal-code — background agent"))
    print(f"  task   : {task}")
    print(f"  branch : {branch}")
    print(style.rule())
    print()

    pf = preflight(project_dir, branch)
    if not pf["ok"]:
        print(f"refused: {pf['error']}")
        return 2
    original_branch = pf["original_branch"]
    print(f"  (branch created; your branch '{original_branch}' will be restored at the end)\n")

    # From here the user is ON the agent branch — every exit path below,
    # including aborts and crashes, must flow through finalize() so their
    # branch is restored. (A refused run once left the user stranded on
    # an empty agent branch; never again.)
    aborted: Optional[str] = None
    project_cfg = None
    agent = None
    models = None
    session = None  # Unify-Agents (P1d): registered once the agent's store exists
    # Phase C: every failing run_verify output lands here via the sink;
    # fail→pass is the reflection trigger, the first failure is the lesson.
    verify_failures: list[str] = []
    set_verify_result_sink(
        lambda rc, out: verify_failures.append(out) if rc != 0 else None
    )
    hands = MCPHands(project_dir, skip=("browser",), headless=True)  # no unwatched browsing; untrusted MCP refused
    try:
        try:
            await hands.open()
        except MCPHandsError as e:
            # Partial failure is survivable: open() registers each
            # server's tools as it goes, so an optional server failing
            # (the shell server, commonly) leaves the file tools live.
            # Abort only when NOTHING connected — a background agent
            # without file hands can't do its job.
            if not hands.registered_tools:
                aborted = f"file tools failed to connect — {e}"
            else:
                print(style.yellow(f"  (warning: {e} — continuing with {len(hands.registered_tools)} tools from {', '.join(hands.connected_servers)})"))

        if aborted is None:
            project_cfg = load_project_config(project_dir)
            # Project instructions ride in the headless prompts too — a
            # background task must obey the same standing rules as a
            # live session.
            instructions = ProjectInstructions(project_dir)
            if instructions.startup_notice:
                print(f"  ({instructions.startup_notice})")
            guard = Guard(project_dir, hooks=project_cfg.hooks, shell_config=project_cfg.shell)
            guard.auto = True  # the branch is the safety boundary; block_paths still hold
            # Shell and browser are denied outright in headless runs —
            # the branch protects the repo, not the machine, and nobody
            # is watching. Their tools aren't registered here either
            # (shell: not registered; browser: server skipped above);
            # the guard denies are the backstop.
            guard.shell_mode = "deny"
            guard.browser_mode = "deny"
            if project_cfg.verify_command:
                register_verify_tool(project_dir, project_cfg.verify_command)
            models = resolve_models(creds, project_cfg.models)
            checkpoints = CheckpointManager(project_dir)
            guard.notify_write = checkpoints.on_write
            parent_ref: dict = {"agent": None}
            register_subagent_tool(parent_ref, lambda: models["fast"], guard)

            # Unify-Agents (P1d): a background run is just as visible as the
            # REPL — register a session and emit the same tool + turn events,
            # so daemon work shows up in the Agents view with its turn-by-turn
            # activity, cost, and tools (decision 6: everything visible, incl.
            # the daemon). Wrap intercept once (it rides the build below, the
            # same chokepoint that captures reads/writes/shell/subagent/
            # crystal writes); fail-safe via record_event.
            session = SessionHandle(project_dir=str(project_dir), model=models["main"])
            _raw_intercept = guard.intercept

            async def _intercept_traced(tool_name: str, tool_input: dict) -> dict:
                res = await _raw_intercept(tool_name, tool_input)
                ok = res.get("action") == "allow"
                await session.record_event(
                    event_type="tool_called" if ok else "tool_denied",
                    phase="subagent" if tool_name == "subagent" else "tool",
                    label=style.humanize_call(tool_name, tool_input, project_dir),
                    status="ok" if ok else "denied",
                    payload={"tool": tool_name, **({} if ok else {"reason": res.get("reason")})},
                )
                return res

            guard.intercept = _intercept_traced  # type: ignore[method-assign]

            agent = await build_agent(
                creds, db=db, customer_id=customer_id,
                on_status=lambda s: print(f"  [{s}]"),
                intercept=guard.intercept, after_tool=guard.after_tool,
                model=models["main"],
            )
            parent_ref["agent"] = agent
            await session.bind(agent.tool_state["store"], agent.customer.id)
            for _srv in hands.connected_servers:
                await session.register_dependency(kind="mcp_server", descriptor=_srv)

            # One run-pass = one turn in the session timeline. Heartbeats
            # 'running' DURING the (possibly multi-minute) headless pass — the
            # event loop is free while agent.run awaits, so a long build isn't
            # false-marked crashed — and records turn_started/turn_completed
            # with tokens, cost, tools, and files written.
            async def _pass(messages: list, *, action: str, recall_block: Optional[str] = None) -> dict:
                turn_no = session.begin_turn()
                await session.beat(status="running", current_action=action)
                await session.record_event(
                    event_type="turn_started", phase="turn", label=action,
                    payload={"phase": action}, turn_index=turn_no,
                )
                hb = asyncio.create_task(_turn_heartbeat(session))
                t0 = asyncio.get_running_loop().time()
                try:
                    res = await agent.run(
                        messages=messages,
                        system=_system_for(
                            agent, project_cfg, guard, instructions,
                            recall_block=recall_block,
                        ),
                    )
                finally:
                    await _stop_heartbeat(hb)
                ti = res.get("prompt_tokens")
                to = res.get("completion_tokens")
                tcr = res.get("cache_creation_tokens") or 0
                trd = res.get("cache_read_tokens") or 0

                # Post-turn universal signal set (Bucket 1): the SAME shared
                # layer the agent endpoint calls — a background pass now emits a
                # cost-ledger row, grounded citations + marketplace credit + the
                # uncited-answer coverage gap, and an MCR trace + self-critique,
                # the half the coding surfaces used to lack. origin="coding-bg";
                # sequence_id is the session id (the per-agent cost rollups'
                # unit). Fail-safe inside finalize.
                finalized = await finalize_agent_turn(
                    store=agent.tool_state["store"],
                    encoder=agent.tool_state["encoder"],
                    customer=agent.customer,
                    anthropic_client=agent.llm,
                    result=res,
                    user_query=task,
                    sequence_id=session.session_id if session is not None else None,
                    origin="coding-bg",
                    turn_index=turn_no,
                )
                # Reuse the cost the ledger just computed for the timeline event
                # below (one compute, two consumers); fall back to the
                # standalone arithmetic when cost accounting is off. Cache-aware
                # now that C1 surfaces the cache_* fields — cache reads bill ~0.1x.
                cost = finalized.get("cost_micro_usd")
                if cost is None:
                    try:
                        from crystal_cache.cost.pricing import compute_cost_micro_usd
                        cost = compute_cost_micro_usd(
                            getattr(agent, "model", "") or "",
                            input_tokens=ti or 0,
                            output_tokens=to or 0,
                            cache_creation_tokens=tcr,
                            cache_read_tokens=trd,
                        )
                    except Exception:  # noqa: BLE001
                        cost = None
                # Per-pass token line — makes C1 caching visible: cache_read
                # climbs as the trajectory is re-sent across passes/iterations
                # (input drops to the delta; the rest is a cheap cache read).
                print(style.dim(
                    f"  · tokens: in={ti} out={to} "
                    f"cache_read={trd} cache_write={tcr}"
                ))
                final = (res.get("final_text") or "").strip()
                await session.record_event(
                    event_type="turn_completed", phase="turn", label=action,
                    payload={
                        "summary": final.splitlines()[0][:200] if final else "",
                        "tools_called": list(guard.turn_calls),
                        "files_written": [str(p) for p in guard.turn_written_paths],
                        "iterations": res.get("iterations"),
                        "cache_read_tokens": trd,
                        "cache_creation_tokens": tcr,
                    },
                    status="ok", turn_index=turn_no,
                    duration_ms=int((asyncio.get_running_loop().time() - t0) * 1000),
                    tokens_input=ti, tokens_output=to, cost_micro_usd=cost,
                )
                return res

            # Pre-run self-recall (ONCE per task): consult the agent's own
            # bank for patterns relevant to the task and carry them into the
            # PLAN pass only — the plan threads recalled knowledge into
            # execution via result["messages"], so once is enough (the three
            # passes don't each re-recall). Same gate + fail-safe as the REPL
            # (recall.py); skipped when CC_AGENT_RECALL is off.
            recall_block = None
            if _RECALL_ENABLED:
                recall_block = await maybe_recall(
                    agent=agent, user_input=task, fast_model=models["fast"],
                )
                if recall_block:
                    print(style.dim("  (recalled relevant knowledge from the bank)"))
                    await session.record_event(
                        event_type="knowledge_recalled", phase="knowledge",
                        label="recalled from own bank",
                    )

            # Phase A — plan (read-only investigation).
            print("\n" + style.rule())
            print(style.bold("PLAN PASS") + style.dim("  (read-only investigation)"))
            print(style.rule())
            guard.plan_mode = True
            checkpoints.begin_turn(task)
            guard.begin_turn()
            result = await _pass(
                [{"role": "user", "content": task}],
                action="plan pass", recall_block=recall_block,
            )
            plan_text = result["final_text"]
            print(plan_text)

            # Phase B — execute (auto-approved, verify loop enforced).
            print("\n" + style.rule())
            print(style.bold("EXECUTE PASS") + style.dim("  (auto-approved on the agent branch)"))
            print(style.rule())
            guard.plan_mode = False
            guard.begin_turn()
            instructions.refresh()  # pick up anything persisted during planning
            result = await _pass(
                result["messages"] + [{
                    "role": "user",
                    "content": (
                        "The plan is approved. Execute it now, step by step, "
                        "following the verification contract."
                    ),
                }],
                action="execute pass",
            )
            if (
                project_cfg.verify_command
                and any(t in WRITE_TOOLS for t in guard.turn_calls)
                and "run_verify" not in guard.turn_calls
            ):
                print("  (files changed but not verified — one enforcement pass)")
                result = await _pass(
                    result["messages"] + [{
                        "role": "user",
                        "content": (
                            "You modified files but did not call run_verify. Run it "
                            "now; if it fails, fix the failures and run it again "
                            "before summarizing."
                        ),
                    }],
                    action="verify enforcement",
                )
            print(result["final_text"])
    except Exception as e:  # noqa: BLE001 — a crash must still restore the branch
        aborted = f"{type(e).__name__}: {e}"
    finally:
        set_verify_result_sink(None)
        await hands.close()

    if aborted:
        print("\n" + style.yellow(f"aborted: {aborted}"))

    # Ground-truth verification BEFORE switching branches (pointless on
    # an aborted run — nothing to vouch for).
    verdict = None
    if not aborted and project_cfg is not None and project_cfg.verify_command:
        print("\n" + style.rule())
        print(style.bold("VERIFICATION") + style.dim("  (ground truth, run by the CLI — not the agent's claim)"))
        print(style.rule())
        if session is not None:
            await session.beat(status="running", current_action="verifying")
        verdict = await run_verify_for_report(project_dir, project_cfg.verify_command)
        print("  " + (style.green("PASSED") if verdict["passed"] else style.red("FAILED")))
        if not verdict["passed"] and verdict["tail"]:
            print("  " + verdict["tail"].replace("\n", "\n  "))

    # ----- Phase C: learn from the run (never fails the run) -----------
    # Fail→pass → reflect: the agent hit a wall AND found the way past —
    # the only validated-lesson shape (BCB Level B). Terminal verdict
    # failure → record a knowledge gap for the daemon's idle pass.
    # Aborted/crashed runs write neither: no verified evidence either way.
    cid = customer_id or LOCAL_CUSTOMER_ID
    if agent is not None and not aborted and verdict is not None:
        if session is not None:
            await session.beat(status="running", current_action="reflecting")
        if verdict["passed"] and verify_failures:
            from .reflection import run_reflection
            stored = await run_reflection(
                store=agent.tool_state["store"],
                encoder=agent.tool_state["encoder"],
                client=agent.llm,
                model=(models or {}).get("fast") or creds.model,
                customer_id=cid,
                task=task,
                failing_tail=verify_failures[0],
                diffstat=_git(["diff", "--stat", f"{original_branch}..HEAD"], project_dir).stdout.strip(),
            )
            if stored:
                print(style.dim(f"  (reflection stored: {stored['key']} — \"{stored['claim']}\")"))
                if session is not None:
                    await session.record_event(
                        event_type="crystal_written", phase="knowledge",
                        label=stored.get("key") or "reflection",
                        payload={"key": stored.get("key"), "claim": stored.get("claim")},
                    )
        elif not verdict["passed"] and write_gap_on_failure:
            try:
                gap = await agent.tool_state["store"].create_agent_gap(
                    cid,
                    task=task,
                    task_id=source_task_id or "(foreground)",
                    branch=branch,
                    failing_tail=(verify_failures[0] if verify_failures else verdict["tail"]),
                    project_dir=str(project_dir),
                )
                print(style.dim(f"  (recorded as knowledge gap {gap['id']} — an idle daemon will retry it once)"))
                if session is not None:
                    await session.record_event(
                        event_type="gap_recorded", phase="gap", status="error",
                        label=task[:120], payload={"gap_id": gap["id"]},
                    )
            except Exception as e:  # noqa: BLE001 — bookkeeping must not mask the run's outcome
                print(style.yellow(f"  (could not record knowledge gap: {e})"))

    # Close the session before the git dance — the agent's work is done; the
    # verdict + events carry the outcome, the session just goes 'exited'.
    if session is not None:
        await session.close(status="exited")

    # finalize ALWAYS runs once preflight created the branch: it commits
    # whatever work exists (quarantined on the agent branch — partial
    # work from a crash included, on purpose) and restores the user's
    # branch.
    fin = finalize(project_dir, branch, original_branch, task)

    # An abort that produced no work should leave no residue — drop the
    # empty agent branch so a retry can reuse the name.
    branch_removed = False
    if aborted and not fin["committed"] and not fin["diffstat"] and fin["restore_ok"]:
        branch_removed = _git(["branch", "-D", branch], project_dir).returncode == 0

    print("\n" + style.rule())
    print(style.bold("REPORT"))
    print(style.rule())
    if branch_removed:
        print(f"  branch    : '{branch}' removed (run aborted with no changes)")
    else:
        print(f"  branch    : {branch}" + ("" if fin["committed"] else "  (no changes were committed)"))
    if fin["diffstat"]:
        print("  changes   :")
        print("    " + fin["diffstat"].replace("\n", "\n    "))
    if verdict is not None:
        print("  verified  : " + (style.green("PASS") if verdict["passed"] else style.red("FAIL — review before merging")))
    elif aborted:
        print("  verified  : skipped (run aborted)")
    else:
        print("  verified  : no verify_command configured — review manually")
    print(f"  restored  : you are back on '{original_branch}'"
          if fin["restore_ok"] else f"  WARNING: could not switch back to '{original_branch}'")
    if fin["diffstat"] and not branch_removed:
        print(f"\n  review with: git diff {original_branch}..{branch}")
    if aborted:
        return 2
    # R14: when a verify command exists, the CLI's ground-truth verdict IS
    # the run's outcome — a committed-but-failing run returns 1 (the daemon
    # marks it failed, recurrence counts the failure, a gap retry resolves
    # needs_operator). Committed work stays quarantined on the branch
    # either way. (Found live: a FAIL-verdict run was reported 'done'.)
    if verdict is not None and not verdict["passed"]:
        return 1
    return 0 if (fin["committed"] or not fin["diffstat"]) else 1
