"""The terminal loop for the coding doorway.

Startup: resolve credentials (config_store, with first-time setup if
needed), connect the file "hands" (the filesystem MCP server, scoped to
the project folder), build the agent, then run a read/answer loop. The
agent is stateless — it takes the full message history each turn — so the
running conversation is held here in `messages`.

Step 2b: the agent now has FILE hands (read/write/edit/list/move),
limited to one project folder passed via --project. Code search and shell
are Step 2c.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys
from pathlib import Path
from typing import Any, Optional

from . import config_store
from . import style
from .checkpoints import CheckpointManager
from .daemon import format_task_line, register_queue_tools
from .guard import Guard, WRITE_TOOLS
from .ingest import run_ingest_headless, run_ingest_wizard, resync_written_files, _first_line
from .instructions import ProjectInstructions
from .mcp_hands import MCPHands, MCPHandsError
from .onboarding import run_setup
from .project_config import load_project_config
from .recall import maybe_recall
from .runtime import build_agent, diagnose_store, resolve_customer_by_key, resolve_models, run_audit
from .session_registry import SessionHandle
from .shell import register_shell_tool
from .subagent import register_subagent_tool
from .verify import register_verify_tool
from .transcript import cap_transcript, format_ago

from crystal_cache.agent.system_prompt import build_system_prompt
from crystal_cache.agent.turn_finalize import finalize_agent_turn


_COMMANDS_LINE = (
    "/info  /login  /logout  /ingest  /auto  /plan  /go  /model  "
    "/checkpoints  /rewind [n]  /tasks  /setup  /resume  /reset  /exit"
)


def _print_banner() -> None:
    print()
    print(style.rule())
    print(style.bold("CRYS") + style.dim(" — the coding agent with a real memory (Crystal Cache)"))
    print(style.dim(_COMMANDS_LINE))
    print(style.rule())

# A short note about where the key came from, shown once at startup.
_SOURCE_NOTE = {
    "env": "using the API key from your environment",
    "config": "using your saved setup",
    "dotenv": "using the key from the project .env",
    "setup": "setup complete",
}


def _status(msg: str) -> None:
    print(style.dim(f"  … {msg}"))


def _resolve_or_setup() -> config_store.Credentials:
    """Resolve credentials; run first-time setup if none are found."""
    creds = config_store.resolve_credentials()
    if creds is None:
        creds = run_setup()
    return creds


# F3: appended to the system prompt when the project has a verify command.
_VERIFY_ADDENDUM = (
    "\n\nVERIFICATION CONTRACT: this project's verify command is `{cmd}`, "
    "exposed to you as the run_verify tool. After you create or modify ANY "
    "file you MUST call run_verify, read any failures, fix them, and run it "
    "again. Never declare the work finished while verification fails."
)


# F4: appended instead while plan mode is on — propose, don't execute.
_PLAN_ADDENDUM = (
    "\n\nPLAN MODE is active: investigate using your read-only tools (read, "
    "list, search, knowledge), but do NOT modify files or run commands — "
    "those tools will refuse while plan mode is on. End your reply with a "
    "concise NUMBERED PLAN of the changes you propose, step by step. The "
    "user will approve it with /go or discard it."
)


_BANK_ADDENDUM = (
    "\n\nKNOWLEDGE BANK: you have persistent knowledge tools backed by a bank "
    "that can hold ingested codebases and documents, INCLUDING files that "
    "are not in the current project folder. Ingested code is keyed as "
    "Code|<file path>|<symbol>. Pick the tool by question shape: for "
    "identity/enumeration questions ('what does <file> define', 'list every "
    "X') use key_scan with subject_contains=<file path or name> — it "
    "returns ALL matching facts, not just the best one. For 'find the "
    "passage about X' use content_search; for entity/Q&A lookups use "
    "knowledge_search. When asked about code or documents you cannot find "
    "in the project folder, search the bank before concluding the "
    "information is unavailable. And before researching from scratch or "
    "building something you may have tackled before, do a quick "
    "knowledge_search or key_scan of your OWN saved patterns and "
    "Reflections first — reuse a hard-won lesson instead of relearning it."
)


# Pre-turn self-recall (recall.py): default ON; CC_AGENT_RECALL=0/off
# disables it (e.g. to isolate behavior or skip the fast-model call).
_RECALL_ENABLED = os.environ.get("CC_AGENT_RECALL", "1").strip().lower() not in {
    "0", "false", "no", "off",
}

# Project conversation memory (P5): persist the transcript per project so a
# relaunch can resume context (the "which file did I last work on?" gap).
# Default ON; CC_PROJECT_MEMORY=0/off disables persist + recap + /resume.
_PROJECT_MEMORY_ENABLED = os.environ.get("CC_PROJECT_MEMORY", "1").strip().lower() not in {
    "0", "false", "no", "off",
}
# Cap the persisted transcript to the last N messages. Messages include
# tool-use/result blocks, so this is a coarse size bound, not a turn count.
_TRANSCRIPT_MAX_MESSAGES = 120


def _print_conversation_recap(convo: dict) -> None:
    """One-line launch recap of the prior session in this project."""
    turns = convo.get("turn_count") or 0
    ago = format_ago(convo.get("updated_at"))
    meta = convo.get("meta") or {}
    files = meta.get("last_files") if isinstance(meta, dict) else None
    bits = [f"{turns} turn{'s' if turns != 1 else ''}"]
    if isinstance(files, list) and files:
        bits.append("last touched " + ", ".join(str(f) for f in files[:3]))
    print(style.dim(
        f"  · last session here ({ago}): {'; '.join(bits)} — /resume to reload it"
    ))


async def _persist_conversation(
    store: Any,
    *,
    customer_id: str,
    conversation_key: str,
    messages: list,
    turn_count: int,
    last_summary: Optional[str],
    last_files: list,
) -> None:
    """Write the rolling conversation for this project so the next launch can
    resume. Fail-safe — persistence must never break a turn (the session-
    registry posture). The caller caps; we also tail-cap defensively."""
    try:
        await store.upsert_conversation(
            customer_id,
            conversation_key=conversation_key,
            transcript=cap_transcript(messages, _TRANSCRIPT_MAX_MESSAGES),
            turn_count=turn_count,
            last_summary=((last_summary or "").strip()[:240] or None),
            mode="coding",
            meta={"last_files": last_files[-8:]} if last_files else None,
        )
    except Exception as e:  # noqa: BLE001 — never break a turn over memory
        print(style.dim(f"    · (project memory not saved: {type(e).__name__})"))


def _system_for(
    agent: Any,
    project_cfg: Any,
    guard: Guard,
    instructions: ProjectInstructions,
    recall_block: Optional[str] = None,
) -> str:
    """Default system prompt + the knowledge-bank affordance + the project
    instructions section (always present — full rules with a file, the
    create-CRYSTAL.md policy without one) + optional recalled knowledge for
    THIS turn + the active contract (plan mode wins; otherwise verify)."""
    base = build_system_prompt(agent.customer, agent.tools) + _BANK_ADDENDUM + instructions.addendum()
    if recall_block:
        base += recall_block
    if guard.plan_mode:
        return base + _PLAN_ADDENDUM
    if project_cfg.verify_command:
        return base + _VERIFY_ADDENDUM.format(cmd=project_cfg.verify_command)
    return base


def _print_info(
    project_dir: Path,
    creds: config_store.Credentials,
    hands: MCPHands,
    agent: Any,
    db: Optional[str],
    login_source: str,
    guard: Guard,
    models: dict,
    instructions: ProjectInstructions,
) -> None:
    print("\nWhere things stand right now:")
    print(f"  running from    : {os.getcwd()}")
    print(f"  project folder  : {project_dir}")
    if hands.registered_tools:
        print("  file access     : ON, limited to the project folder above")
        print(f"  servers connected: {', '.join(hands.connected_servers)}")
        shown = ', '.join(sorted(hands.registered_tools)[:6])
        more = len(hands.registered_tools) - 6
        suffix = f", +{more} more" if more > 0 else ""
        print(f"  file tools      : {len(hands.registered_tools)} ({shown}{suffix})")
    else:
        print("  file access     : OFF this session (no file tools connected)")
    print(f"  knowledge store : {db or 'local (default, in the launch folder)'}")
    print(f"  customer (memory): {agent.customer.id}")
    print(f"  login           : {login_source}")
    print(f"  approvals       : {'AUTO (no prompts this session)' if guard.auto else 'ask before writes and shell commands'}")
    browser_state = {None: 'available (asks y/n at first use)', True: 'allowed this session', False: 'declined this session'}[guard.browser_consent]
    print(f"  shell           : ON — every command individually approved (/auto never applies)")
    print(f"  browser         : {browser_state}")
    print(f"  instructions    : {instructions.path.name + ' (in every prompt; the agent records new standing rules there)' if instructions.path else 'none (the agent creates CRYSTAL.md when you set a standing rule)'}")
    print(f"  mode            : {'PLAN (read-only — /go to execute, /plan off to discard)' if guard.plan_mode else 'normal'}")
    print(f"  answering model : {models['main']} (via {creds.provider}, from {models['main_source']})")
    print(f"  fast model      : {models['fast']} (for delegated work, from {models['fast_source']})")
    print(f"  total tools     : {len(agent.tools)}")
    print()


# In-turn liveness. The REPL beats at turn boundaries — 'idle' before each
# input(), 'running' before each agent.run — but a single long turn (a big
# build, or a truncate-and-retry) can outrun the server's 90s stale window
# with no beat in between, so the Inspector would false-mark a working agent
# 'crashed' and orphan its deps. This timer beats 'running' DURING the turn:
# the event loop is free while agent.run awaits the LLM/tools, so the task is
# scheduled normally; input() only blocks the loop BETWEEN turns, when this
# task isn't running. Cancelled the moment the turn ends.
_TURN_HEARTBEAT_SECONDS = 30


async def _turn_heartbeat(session: SessionHandle) -> None:
    while True:
        await asyncio.sleep(_TURN_HEARTBEAT_SECONDS)
        await session.beat(status="running")


async def _stop_heartbeat(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _amain(
    project_dir: Path, flag_db: Optional[str], flag_customer: Optional[str]
) -> int:
    _print_banner()

    creds = _resolve_or_setup()
    greeting = f"Hi {creds.name}! " if creds.name else ""
    print(f"{greeting}({_SOURCE_NOTE.get(creds.source, 'ready')})\n")

    # Knowledge login precedence: explicit flags > saved /login > local
    # default. Flags are a per-session override and never touch the
    # saved login.
    saved_db, saved_customer = config_store.load_login()
    if flag_db or flag_customer:
        db, customer_id = flag_db, flag_customer
        login_source = "command-line flags (this session only)"
    elif saved_db or saved_customer:
        db, customer_id = saved_db, saved_customer
        login_source = "saved login (/logout to stop using it)"
        print(f"  Using your saved Crystal Cache login: customer {customer_id}\n")
    else:
        db, customer_id = None, None
        login_source = "none (local default; /login to connect your knowledge)"

    # Connect the file hands FIRST, so the agent picks them up when built.
    # If the bridge can't start, we keep going without file tools rather
    # than failing the whole agent — and say so plainly.
    def _approve_untrusted_mcp(name: str, command: str) -> bool:
        # F2 (2026-07-03): a non-built-in MCP server wants to spawn a local
        # command. Show the human the exact command and require approval —
        # a repo-level or tampered config can't silently start a process.
        print("")
        print(style.yellow(f"  [approval needed] MCP server {name!r} wants to start"))
        print("  " + style.dim("|") + " " + style.bold(command))
        print(style.dim("  (this runs a local process from your MCP config; approve only if you trust it)"))
        try:
            answer = input("  start it? y = yes / n = no: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")

    hands = MCPHands(project_dir, approve_untrusted=_approve_untrusted_mcp)
    try:
        _status(f"Connecting file tools, limited to {project_dir}")
        await hands.open()
        _status(f"File tools ready ({len(hands.registered_tools)})")
    except MCPHandsError as e:
        # open() registers tools server-by-server, so a late failure
        # (commonly the shell server) can leave earlier servers live.
        # Say what actually happened rather than declaring tools OFF.
        if hands.registered_tools:
            print(style.yellow(
                f"  {len(hands.registered_tools)} file tools ready from "
                f"{', '.join(hands.connected_servers)}; one server failed — {e}"
            ))
        else:
            print(style.yellow(f"  File tools OFF - {e}"))

    # F3/F5: per-project config first — it feeds both the guard (hooks)
    # and the verify tool. Registered BEFORE the agent is built so the
    # tool lands in its toolset.
    project_cfg = load_project_config(project_dir)
    if project_cfg.error:
        print(f"  (warning: {project_cfg.error} — continuing without it)\n")
    if project_cfg.verify_command:
        register_verify_tool(project_dir, project_cfg.verify_command)
        print(f"  (verify loop ON — the agent runs `{project_cfg.verify_command}` after edits)\n")

    # Shell v1: the CRYS-native run_command tool. Always registered in
    # the REPL — the guard makes every single command an explicit y/n,
    # immune to /auto and session approvals (see shell.py).
    register_shell_tool(project_dir, project_cfg.shell)
    print(style.dim("  (shell ON — every command is shown to you and individually approved)") + "\n")

    # Project instructions (CRYSTAL.md > AGENTS.md > CLAUDE.md): standing
    # rules injected into every turn's system prompt, re-read on change
    # (per-turn mtime check), and persisted by the agent through its
    # normal guarded file tools when the user sets a new standing rule.
    instructions = ProjectInstructions(project_dir)
    if instructions.startup_notice:
        print(f"  ({instructions.startup_notice})\n")

    # F1: the approval guard — every tool call the agent makes passes
    # through guard.intercept (the F0 seam). Writes show a diff and ask;
    # /auto on lifts the prompts for the session. F5 hooks ride along:
    # block_paths deny in every mode; on_file_edited runs after writes.
    guard = Guard(project_dir, hooks=project_cfg.hooks, shell_config=project_cfg.shell)
    if guard.block_paths:
        print(f"  (blocked paths: {', '.join(guard.block_paths)})\n")
    if guard.on_file_edited:
        print(f"  (post-edit hooks: {', '.join(guard.on_file_edited)})\n")

    # F6: model routing — main from project config or credentials; fast
    # for delegated work (F7 subagents). /model shows it.
    models = resolve_models(creds, project_cfg.models)
    if models["main_source"] == "project config":
        print(f"  (model override from project config: {models['main']})\n")

    # F7: the subagent tool — registered before the agent builds so it
    # lands in the toolset. parent_ref stays pointed at the LIVE agent
    # (reassigned after every rebuild) so workers inherit the current
    # customer; the fast model comes from F6 routing.
    parent_ref: dict = {"agent": None}
    register_subagent_tool(parent_ref, lambda: models["fast"], guard)

    # Daemon queue tools — queue_task prompts via the guard (queueing IS
    # approving a future unattended run); get_task_status reads freely.
    # store_ref is filled right after build_agent (same late-binding
    # pattern as parent_ref).
    store_ref: dict = {"store": None}
    register_queue_tools(project_dir, store_ref)

    # F2: checkpoints — a snapshot lands before the first approved write
    # of each turn, via the guard's notify_write seam. /rewind undoes.
    checkpoints = CheckpointManager(project_dir)
    guard.notify_write = checkpoints.on_write
    if not checkpoints.available:
        print("  (note: not a git repository — /rewind is unavailable here)\n")

    # Foundation F4: register this session so it surfaces in the Inspector.
    # Fail-safe throughout (observability never breaks the agent); bound
    # once the agent's store exists. D4: writes land in whatever store CRYS
    # uses — the local default when offline, the team DB when logged in — so
    # offline sessions stay local automatically.
    session = SessionHandle(project_dir=str(project_dir), model=models["main"])

    # Unify-Agents (P1c): persist every tool call as an event so the Agents
    # timeline shows the agent's fine-grained activity, not just turn
    # boundaries. The guard is the chokepoint every action funnels through —
    # reads, writes, shell, browser, subagent delegations, and knowledge/
    # crystal writes are all tool calls through intercept — so wrapping it
    # once captures the whole trace with the same human label the terminal
    # shows. Wrap the bound method here (the wrapper rides all agent
    # (re)builds below — /login and /logout rebuild but reuse this guard);
    # fail-safe via session.record_event, which no-ops until the session is
    # bound and never raises into a turn.
    _raw_intercept = guard.intercept

    async def _intercept_traced(tool_name: str, tool_input: dict[str, Any]) -> dict:
        result = await _raw_intercept(tool_name, tool_input)
        allowed = result.get("action") == "allow"
        await session.record_event(
            event_type="tool_called" if allowed else "tool_denied",
            phase="subagent" if tool_name == "subagent" else "tool",
            label=style.humanize_call(tool_name, tool_input, project_dir),
            status="ok" if allowed else "denied",
            payload={
                "tool": tool_name,
                **({} if allowed else {"reason": result.get("reason")}),
            },
        )
        return result

    guard.intercept = _intercept_traced  # type: ignore[method-assign]

    try:
        agent = await build_agent(
            creds, db=db, customer_id=customer_id, on_status=_status,
            intercept=guard.intercept, after_tool=guard.after_tool,
            model=models["main"],
        )
        parent_ref["agent"] = agent
        store_ref["store"] = agent.tool_state["store"]
        await session.bind(agent.tool_state["store"], agent.customer.id)
        for _srv in hands.connected_servers:
            await session.register_dependency(kind="mcp_server", descriptor=_srv)
    except Exception as e:  # noqa: BLE001 — surface any startup failure plainly
        print(f"\nCould not start: {type(e).__name__}: {e}")
        await hands.close()
        return 1

    # P5: project conversation memory. Load this project's saved rolling
    # conversation (recap now; /resume reloads it). Keyed by the resolved
    # project_dir; the store is mode-agnostic, coding mode here. Works offline
    # too (the local default store is itself a db). Fail-safe.
    conversation_key = str(project_dir)
    saved_conversation: Optional[dict] = None
    conversation_turns = 0
    last_files: list[str] = []
    if _PROJECT_MEMORY_ENABLED:
        try:
            saved_conversation = await agent.tool_state["store"].get_conversation(
                agent.customer.id, conversation_key=conversation_key,
            )
        except Exception:  # noqa: BLE001 — memory load must never block startup
            saved_conversation = None
        if saved_conversation and (
            saved_conversation.get("turn_count")
            or saved_conversation.get("transcript")
        ):
            _print_conversation_recap(saved_conversation)

    print("\nReady. Talk to the agent below.\n")

    messages: list[dict[str, Any]] = []
    try:
        while True:
            await session.beat(status="idle")
            try:
                user_input = input(style.bold(style.cyan("you > "))).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye.")
                return 0

            if not user_input:
                continue
            if user_input in ("/exit", "/quit"):
                print("bye.")
                return 0
            if user_input == "/reset":
                messages = []
                conversation_turns = 0
                last_files = []
                if _PROJECT_MEMORY_ENABLED:
                    try:
                        await agent.tool_state["store"].delete_conversation(
                            agent.customer.id, conversation_key=conversation_key,
                        )
                    except Exception:  # noqa: BLE001 — best-effort clear
                        pass
                    saved_conversation = None
                    print("  (conversation cleared — including this project's saved memory)\n")
                else:
                    print("  (conversation cleared)\n")
                continue
            if user_input == "/resume":
                if not _PROJECT_MEMORY_ENABLED:
                    print("  (project memory is off — set CC_PROJECT_MEMORY=1 to use /resume)\n")
                    continue
                if not saved_conversation or not saved_conversation.get("transcript"):
                    print("  (nothing to resume for this project yet)\n")
                    continue
                messages = list(saved_conversation["transcript"])
                conversation_turns = saved_conversation.get("turn_count") or 0
                _meta = saved_conversation.get("meta") or {}
                last_files = (
                    list(_meta["last_files"])
                    if isinstance(_meta.get("last_files"), list) else []
                )
                print(style.dim(
                    f"  (resumed {conversation_turns} turn(s) from your last session here — "
                    "the agent now has that context)\n"
                ))
                continue
            if user_input in ("/info", "/where"):
                _print_info(project_dir, creds, hands, agent, db, login_source, guard, models, instructions)
                continue
            if user_input == "/tasks":
                rows = await agent.tool_state["store"].list_agent_tasks(limit=15)
                if not rows:
                    print("\n  no background tasks yet — ask the agent to queue one, or use --queue.\n")
                else:
                    print()
                    for t in rows:
                        print(format_task_line(t))
                    print(style.dim("\n  daemon: python -m crystal_code --daemon   logs: ~/.crystal-code/tasks/\n"))
                continue
            if user_input == "/model":
                print(f"\n  main (this agent)    : {models['main']}  [{models['main_source']}]")
                print(f"  fast (delegated work): {models['fast']}  [{models['fast_source']}]")
                print('  Change these in .crystal-code.json under "models" and restart.\n')
                continue
            if user_input.startswith("/auto"):
                arg = user_input.removeprefix("/auto").strip().lower()
                if arg == "on":
                    guard.auto = True
                    print("  (auto-approve ON — the agent edits and runs without asking this session)\n")
                elif arg == "off":
                    guard.auto = False
                    print("  (auto-approve OFF — back to asking before writes and shell commands)\n")
                else:
                    state = "ON" if guard.auto else "OFF"
                    print(f"  auto-approve is {state}. Use /auto on or /auto off.\n")
                continue
            if user_input.startswith("/plan"):
                arg = user_input.removeprefix("/plan").strip().lower()
                if arg == "off":
                    guard.plan_mode = False
                    print("  (plan mode OFF — discarded without executing)\n")
                else:
                    guard.plan_mode = True
                    print("  (plan mode ON — the agent investigates and proposes a numbered plan; /go to execute, /plan off to discard)\n")
                continue
            if user_input == "/go":
                if not guard.plan_mode:
                    print("  (nothing to execute — /plan first, ask for what you want, then /go)\n")
                    continue
                guard.plan_mode = False
                print("  (plan approved — executing)\n")
                # Fall through to the normal turn path with a synthetic
                # prompt: the agent executes its own proposed plan, with
                # gates, checkpoints, and the verify loop all active again.
                user_input = (
                    "The plan you proposed is approved. Execute it now, step "
                    "by step, following the verification contract."
                )
            if user_input.startswith("/checkpoints"):
                items = checkpoints.list()
                if not checkpoints.available:
                    print("  (not a git repository — checkpoints are off here)\n")
                elif not items:
                    print("  no checkpoints yet — one is taken before the agent's first edit each turn.\n")
                else:
                    print("")
                    for i, item in enumerate(items):
                        print(f"  [{i}] {item['when']:<18} {item['label']}")
                    print("  /rewind [n] restores files to one of these (0 = most recent).\n")
                continue
            if user_input.startswith("/rewind"):
                arg = user_input.removeprefix("/rewind").strip()
                idx = int(arg) if arg.isdigit() else 0
                items = checkpoints.list()
                if items and 0 <= idx < len(items):
                    sure = input(
                        f"  restore files to checkpoint [{idx}] "
                        f"({items[idx]['label']})? y/n: "
                    ).strip().lower()
                    if sure not in ("y", "yes"):
                        print("  (rewind cancelled)\n")
                        continue
                print("  " + checkpoints.rewind(idx).replace("\n", "\n  ") + "\n")
                continue
            if user_input == "/login":
                default_db = db or saved_db or ""
                hint = f" [{default_db}]" if default_db else ""
                db_in = input(f"  Database path{hint}: ").strip() or default_db
                if not db_in:
                    print("  (a database path is required — it's where your crystals live)\n")
                    continue
                try:
                    key_in = getpass.getpass("  Customer API key (Key A, hidden): ").strip()
                except Exception:
                    key_in = input("  Customer API key (Key A): ").strip()
                if not key_in:
                    print("  (no key entered)\n")
                    continue
                _status("Looking up that customer")
                res = await resolve_customer_by_key(db_in, key_in)
                if not res["customer_id"]:
                    print(f"  Login failed: {res['error']}\n")
                    continue
                # Persist db + RESOLVED id only — Key A is never written.
                config_store.save_login(db_in, res["customer_id"])
                saved_db, saved_customer = db_in, res["customer_id"]
                db, customer_id = db_in, res["customer_id"]
                login_source = "saved login (/logout to stop using it)"
                try:
                    agent = await build_agent(
                        creds, db=db, customer_id=customer_id, on_status=_status,
                        intercept=guard.intercept, after_tool=guard.after_tool,
                        model=models["main"],
                    )
                    parent_ref["agent"] = agent
                except Exception as e:  # noqa: BLE001
                    print(f"  [logged in, but could not restart the agent: {e}]\n")
                    continue
                messages = []
                n = res["crystals"]
                extra = f" ({n} crystals)" if isinstance(n, int) and n >= 0 else ""
                print(
                    f"  Logged in as {customer_id}{extra}. Saved — future "
                    "launches use this automatically.\n"
                )
                continue
            if user_input == "/logout":
                config_store.clear_login()
                saved_db, saved_customer = None, None
                if flag_db or flag_customer:
                    db, customer_id = flag_db, flag_customer
                    login_source = "command-line flags (this session only)"
                else:
                    db, customer_id = None, None
                    login_source = "none (local default; /login to connect your knowledge)"
                try:
                    agent = await build_agent(
                        creds, db=db, customer_id=customer_id, on_status=_status,
                        intercept=guard.intercept, after_tool=guard.after_tool,
                        model=models["main"],
                    )
                    parent_ref["agent"] = agent
                except Exception as e:  # noqa: BLE001
                    print(f"  [could not restart after logout: {e}]\n")
                    return 1
                messages = []
                print("  (logged out — saved login cleared, conversation cleared)\n")
                continue
            if user_input == "/ingest":
                # Whole-folder ingestion into the live agent's customer,
                # via the server pipeline (see ingest.py). Synchronous —
                # the CLI has no background worker, and doesn't need one.
                try:
                    await run_ingest_wizard(
                        project_dir, agent, guard,
                        db_label=db or "the local default store",
                    )
                except (EOFError, KeyboardInterrupt):
                    print("\n  (ingest cancelled)\n")
                continue
            if user_input == "/setup":
                creds = run_setup()
                models = resolve_models(creds, project_cfg.models)
                try:
                    agent = await build_agent(
                        creds, db=db, customer_id=customer_id, on_status=_status,
                        intercept=guard.intercept, after_tool=guard.after_tool,
                        model=models["main"],
                    )
                    parent_ref["agent"] = agent
                except Exception as e:  # noqa: BLE001
                    print(f"  [could not restart with new setup: {e}]\n")
                    return 1
                messages = []
                print("  (setup updated, conversation cleared)\n")
                continue

            # A real prompt for the agent — label the turn so the
            # checkpoint (taken at its first write, if any) is findable.
            # Instructions are re-checked first so an edit to the file
            # (by you, or by the agent last turn) lands in THIS prompt.
            note = instructions.refresh()
            if note:
                print(f"  ({note})")
            checkpoints.begin_turn(user_input)
            guard.begin_turn()
            turn_no = session.begin_turn()
            turn_t0 = asyncio.get_running_loop().time()
            await session.beat(status="running", current_action=user_input)
            await session.record_event(
                event_type="turn_started", phase="turn", turn_index=turn_no,
                label=user_input[:120], payload={"prompt": user_input},
            )
            messages.append({"role": "user", "content": user_input})

            # Pre-turn self-recall: CRYS decides whether consulting its own
            # bank helps here and, if so, what to search — the hits ride in
            # THIS turn's system prompt (recall.py). Agent-controlled (it can
            # decline) and fail-safe (None on any trouble); not appended to
            # `messages`, so it never accretes across turns.
            recall_block = None
            if _RECALL_ENABLED:
                recall_block = await maybe_recall(
                    agent=agent, user_input=user_input, fast_model=models["fast"],
                )
                if recall_block:
                    print(style.dim("    · recalled relevant knowledge from your bank"))
                    await session.record_event(
                        event_type="knowledge_recalled", phase="knowledge",
                        turn_index=turn_no, label="recalled from own bank",
                    )

            # Keep the session live for the whole turn (see _turn_heartbeat).
            hb = asyncio.create_task(_turn_heartbeat(session))
            try:
                result = await agent.run(
                    messages=messages,
                    system=_system_for(
                        agent, project_cfg, guard, instructions,
                        recall_block=recall_block,
                    ),
                )
            except Exception as e:  # noqa: BLE001 — keep the REPL alive on errors
                print(f"  [error talking to the agent: {type(e).__name__}: {e}]\n")
                messages.pop()  # drop the just-added turn so history stays clean
                await session.record_event(
                    event_type="turn_failed", phase="turn", turn_index=turn_no,
                    status="error", label=f"{type(e).__name__}: {e}",
                    payload={"error": f"{type(e).__name__}: {e}"},
                )
                await _stop_heartbeat(hb)
                continue

            # F3 enforcement: the agent edited files but never verified —
            # one automatic follow-up turn before the answer is shown.
            if (
                project_cfg.verify_command
                and any(t in WRITE_TOOLS for t in guard.turn_calls)
                and "run_verify" not in guard.turn_calls
            ):
                print("  (files changed but not verified — asking the agent to run the verify command)")
                nudge = result["messages"] + [{
                    "role": "user",
                    "content": (
                        "You modified files but did not call run_verify. Run "
                        "it now; if it fails, fix the failures and run it "
                        "again before summarizing your work."
                    ),
                }]
                try:
                    result = await agent.run(
                        messages=nudge, system=_system_for(agent, project_cfg, guard, instructions)
                    )
                except Exception as e:  # noqa: BLE001 — the first answer stands
                    print(f"  [verify nudge failed: {type(e).__name__}: {e}]")

            # Turn work is done — stop the in-turn heartbeat. The bank sync
            # below is brief, and the top-of-loop 'idle' beat refreshes the
            # row right after.
            await _stop_heartbeat(hb)

            # Post-turn universal signal set (Bucket 1): the SAME shared layer
            # the agent endpoint calls (crystal_cache.agent.turn_finalize), so
            # a coding turn now emits what it used to lack — a cost-ledger row,
            # grounded citations + marketplace credit + the uncited-answer
            # coverage gap, and an MCR reasoning trace + self-critique. Runs on
            # the FINAL result (after any verify-enforcement pass) so the
            # signals describe the answer actually shown. origin="coding";
            # sequence_id is the session id (the coding agent's unit is the
            # session, the way the per-agent cost rollups group). Fail-safe.
            finalized = await finalize_agent_turn(
                store=agent.tool_state["store"],
                encoder=agent.tool_state["encoder"],
                customer=agent.customer,
                anthropic_client=agent.llm,
                result=result,
                user_query=user_input,
                sequence_id=session.session_id,
                origin="coding",
                turn_index=turn_no,
            )

            # Record the turn's trajectory + cost into the event stream (the
            # Agents timeline + unified log read it). Fail-safe inside the
            # handle. Reuse the cost the ledger just computed (one compute, two
            # consumers); fall back to the standalone arithmetic when cost
            # accounting is off, so the timeline figure is never lost.
            _summary = (result.get("final_text") or "").strip()
            _in_tok = result.get("prompt_tokens")
            _out_tok = result.get("completion_tokens")
            _cost = finalized.get("cost_micro_usd")
            if _cost is None:
                from crystal_cache.cost.pricing import compute_cost_micro_usd
                _cost = compute_cost_micro_usd(
                    getattr(agent, "model", "") or "",
                    input_tokens=_in_tok or 0,
                    output_tokens=_out_tok or 0,
                )
            await session.record_event(
                event_type="turn_completed", phase="turn", turn_index=turn_no,
                status="ok", label=_summary[:120],
                payload={
                    "summary": _summary[:2000],
                    "tools_called": list(guard.turn_calls),
                    "files_written": [str(p) for p in guard.turn_written_paths],
                    "iterations": result.get("iterations"),
                },
                tokens_input=_in_tok,
                tokens_output=_out_tok,
                cost_micro_usd=_cost,
                duration_ms=int(
                    (asyncio.get_running_loop().time() - turn_t0) * 1000
                ),
            )

            # Bank freshness: re-sync tracked files the agent changed
            # this turn, so the bank never serves pre-edit knowledge.
            # Tracked-only + hash-skip (see resync_written_files); a
            # sync failure is reported but never breaks the turn.
            if guard.turn_written_paths:
                try:
                    sync_lines = await resync_written_files(
                        project_dir, list(guard.turn_written_paths),
                        store=agent.tool_state["store"],
                        encoder=agent.tool_state["encoder"],
                        vector_store=agent.tool_state["vector_store"],
                        fact_vector_store=agent.tool_state["fact_vector_store"],
                        customer_id=agent.customer.id,
                        client=agent.llm,
                    )
                    for ln in sync_lines:
                        print(style.dim(f"    · {ln}"))
                except Exception as e:  # noqa: BLE001 — freshness is best-effort
                    print(style.dim(f"    · bank sync skipped: {type(e).__name__}: {_first_line(e)}"))

            print()
            print(f"{style.bold(style.green('agent >'))} {result['final_text']}")
            print(style.rule())
            print()
            # Carry the full trajectory (including tool turns) forward so
            # the agent has complete context on the next turn.
            messages = result["messages"]

            # P5: update last-touched files (only on turns that wrote any, so
            # the recap reflects the most recent edits) and persist the rolling
            # conversation for this project so the next launch can resume.
            if guard.turn_written_paths:
                last_files = [str(p) for p in guard.turn_written_paths]
            conversation_turns += 1
            if _PROJECT_MEMORY_ENABLED:
                await _persist_conversation(
                    agent.tool_state["store"],
                    customer_id=agent.customer.id,
                    conversation_key=conversation_key,
                    messages=messages,
                    turn_count=conversation_turns,
                    last_summary=result.get("final_text"),
                    last_files=last_files,
                )
    finally:
        await session.close()
        await hands.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="crys",
        description="CRYS — the Crystal Cache coding agent (terminal). Also runnable as `python -m crystal_code`.",
    )
    # The project folder can be given either as a bare path or with
    # --project; both work, and a missing value defaults to the current
    # directory. File access is limited to whichever folder is chosen.
    parser.add_argument(
        "project_path",
        nargs="?",
        default=None,
        help=(
            "The folder the agent may work in (default: current directory). "
            "File access is limited to this folder."
        ),
    )
    parser.add_argument(
        "--project",
        dest="project_flag",
        default=None,
        help="Same as the positional folder argument (either form works).",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "Database for knowledge/memory: a file path (local SQLite) or "
            "a full URL (e.g. Postgres). Default: a local store in the "
            "launch folder."
        ),
    )
    parser.add_argument(
        "--customer",
        dest="customer_id",
        default=None,
        help=(
            "Existing customer id whose full memory the agent should use. "
            "Default: a local in-memory customer with no prior knowledge."
        ),
    )
    parser.add_argument(
        "--list-customers",
        dest="list_customers",
        action="store_true",
        help=(
            "List the customers (and crystal counts) in the chosen --db, "
            "then exit. Testing helper for picking a --customer."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help=(
            "Show the library's full structured logs (tool registrations, "
            "router internals, per-iteration telemetry). Default is "
            "warnings only, with the CLI's own activity trace instead."
        ),
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help=(
            "Ingest the project folder into the knowledge bank and exit "
            "(no REPL). Honors .gitignore; combine with --exclude / "
            "--code-only / --db / --customer. Re-running is an idempotent "
            "sync: unchanged files are skipped, changed files replace "
            "their prior crystals."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Glob or folder to exclude from --ingest (repeatable), "
            "e.g. --exclude 'tests/**' --exclude docs/legacy"
        ),
    )
    parser.add_argument(
        "--code-only",
        dest="code_only",
        action="store_true",
        help="With --ingest: skip prose docs (md/txt/rst), ingest code only.",
    )
    parser.add_argument(
        "--task",
        default="",
        help=(
            "F8: run headless on this task instead of the REPL — plan pass, "
            "auto-approved execution on a fresh git branch, verify, commit, "
            "report. Requires a clean git worktree."
        ),
    )
    parser.add_argument(
        "--branch",
        default="",
        help="Branch name for --task (default: agent/task-<timestamp>).",
    )
    parser.add_argument(
        "--at",
        default="",
        help=(
            "With --queue: start time, local — 'HH:MM' (next occurrence "
            "of that time) or 'YYYY-MM-DD HH:MM'."
        ),
    )
    parser.add_argument(
        "--every",
        default="",
        help=(
            "With --queue: recurrence interval — '30m', '4h', '1d', '2w', "
            "or 'hourly'/'daily'/'weekly'. Fixed-rate wall-clock schedule; "
            "with --at, the series anchors there."
        ),
    )
    parser.add_argument(
        "--seed-general",
        default="",
        metavar="FILE",
        help=(
            "Import a pattern seed file (JSONL of {key, claim}) into the "
            "GENERAL knowledge bank and subscribe the active customer, "
            "then exit. Re-running replaces the bank for that type. "
            "Try: --seed-general seeds/general_swe_patterns.jsonl"
        ),
    )
    parser.add_argument(
        "--general-type",
        default="",
        help="Crystal type for --seed-general (default: general:swe_patterns).",
    )
    parser.add_argument(
        "--queue",
        action="store_true",
        help=(
            "With --task: enqueue the task for the background daemon "
            "instead of running it now, and exit immediately."
        ),
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help=(
            "Run the CRYS daemon: a persistent worker that executes queued "
            "tasks (from --queue or the agent's queue_task tool) one at a "
            "time on fresh git branches, unattended. Ctrl+C to stop."
        ),
    )
    parser.add_argument(
        "--tasks",
        action="store_true",
        help="List the background task queue (newest first) and exit.",
    )
    parser.add_argument(
        "--cancel",
        default="",
        metavar="TASK_ID",
        help=(
            "Cancel a queued background task by id (from --tasks), or stop a "
            "recurring series. A running occurrence finishes but won't recur. "
            "Then exit."
        ),
    )
    parser.add_argument(
        "--showcase",
        action="store_true",
        help=(
            "Run the CRYS capability showcase: one command spins up a fresh "
            "in-process CRYS over a demo fixture and walks the whole surface "
            "(ingest, retrieval + citations, build, learn/reuse, delegate, "
            "self-heal, share, govern), printing panels and a saved report."
        ),
    )
    parser.add_argument(
        "--showcase-acts",
        dest="showcase_acts",
        default="",
        help="With --showcase: run a subset, e.g. --showcase-acts 0,1,5 (default: all).",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help=(
            "Scan the knowledge bank for contradictions and print the "
            "conflicts it surfaces, then exit. Scans the --customer bank "
            "(or the local agent's own). On-demand convergence; "
            "surfacing-only — nothing is deleted or overwritten."
        ),
    )
    parser.add_argument(
        "--audit-max-calls",
        dest="audit_max_calls",
        type=int,
        default=50,
        help="With --audit: max discriminator calls (budget). Default 50.",
    )
    parser.add_argument(
        "--audit-max-pairs",
        dest="audit_max_pairs",
        type=int,
        default=200,
        help="With --audit: max candidate pairs considered. Default 200.",
    )
    ns = parser.parse_args()

    # Idempotent re-apply for direct cli.main() invocation — the module
    # entry (__main__.py) already configured this before imports, which
    # is what silences import-time registration logs.
    style.quiet_library_logs(ns.verbose)

    # Testing helper: list customers in the store, then exit. Doesn't need
    # a project folder, the encoder, or the agent — just the store.
    if ns.list_customers:
        diag = asyncio.run(diagnose_store(ns.db))
        where = ns.db or "the default local store"
        if diag["customers"]:
            print(f"\nCustomers in {where}:")
            for cid, count in diag["customers"]:
                print(f"  {cid}  ({count} crystals)")
            print()
        else:
            print(f"\nNo customers readable in {where} via the v2 schema.")
            if diag["v2_error"]:
                print(f"  reason: {diag['v2_error']}")
            tables = diag["tables"]
            if tables:
                print("  Tables actually in that database (raw, read-only):")
                for t, n in tables:
                    print(f"    {t}: {n} rows")
            elif tables is None:
                print("  (could not open it as a local SQLite file to inspect)")
            print()
        sys.exit(0)

    # Daemon + queue listing — neither needs a project folder (tasks
    # carry their own project_dir on the row).
    if ns.daemon:
        from .daemon import run_daemon
        sys.exit(asyncio.run(run_daemon(ns.db)))
    if ns.tasks:
        from .daemon import list_tasks_cli
        sys.exit(asyncio.run(list_tasks_cli(ns.db)))
    if ns.cancel.strip():
        from .daemon import cancel_cli
        sys.exit(asyncio.run(cancel_cli(ns.db, ns.cancel.strip())))
    if ns.showcase or ns.showcase_acts.strip():
        from .showcase import run_showcase
        acts = None
        if ns.showcase_acts.strip():
            acts = [int(x) for x in ns.showcase_acts.split(",") if x.strip().isdigit()]
        sys.exit(run_showcase(acts=acts))
    if ns.audit:
        sys.exit(asyncio.run(run_audit(
            ns.db, ns.customer_id,
            max_pairs=ns.audit_max_pairs, max_calls=ns.audit_max_calls,
        )))

    chosen = ns.project_flag or ns.project_path
    project_dir = Path(chosen).resolve() if chosen else Path.cwd()
    if not project_dir.is_dir():
        # A non-existent folder otherwise surfaces later as a cryptic
        # "Connection closed" from the file server. Catch it here with
        # the most common cause: Git Bash on Windows strips backslashes
        # from an unquoted path before Python ever sees it.
        print(
            f"That project folder doesn't exist:\n  {project_dir}\n\n"
            "If you're in Git Bash on Windows, backslashes in a path get\n"
            "eaten by the shell. Use forward slashes, quote it, or use a\n"
            "relative path:\n"
            "  python -m crystal_code C:/Users/you/project\n"
            "  python -m crystal_code ../docs\n"
        )
        sys.exit(2)
    # Headless ingestion — no REPL, no agent build.
    if ns.ingest:
        sys.exit(asyncio.run(run_ingest_headless(
            project_dir,
            excludes=ns.exclude,
            include_docs=not ns.code_only,
            db=ns.db,
            customer_id=ns.customer_id,
        )))

    # F8: headless background-agent mode — no REPL.
    if ns.task.strip() and not ns.queue:
        from .background import run_background_task
        sys.exit(asyncio.run(run_background_task(
            project_dir, ns.task.strip(), ns.branch.strip() or None,
            ns.db, ns.customer_id,
        )))

    # General-bank seeding: import-and-exit (no project needed).
    if ns.seed_general.strip():
        from .general_seed import DEFAULT_GENERAL_TYPE, run_seed_import
        sys.exit(asyncio.run(run_seed_import(
            ns.db, Path(ns.seed_general.strip()).expanduser(),
            crystal_type=ns.general_type.strip() or DEFAULT_GENERAL_TYPE,
            customer_id=ns.customer_id,
        )))

    # Daemon queue: enqueue-and-exit.
    if ns.queue:
        if not ns.task.strip():
            print("--queue requires --task \"...\" (the work to enqueue).")
            sys.exit(2)
        from .daemon import enqueue_cli
        sys.exit(asyncio.run(enqueue_cli(
            ns.db, project_dir, ns.task.strip(),
            ns.branch.strip() or None, ns.customer_id,
            at=ns.at.strip() or None, every=ns.every.strip() or None,
        )))

    sys.exit(asyncio.run(_amain(project_dir, ns.db, ns.customer_id)))


if __name__ == "__main__":
    main()
