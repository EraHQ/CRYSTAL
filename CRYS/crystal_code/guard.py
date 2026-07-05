"""F1 — Approval gates + diff preview (the trust layer).

The guard is the CLI's tool interceptor (the F0 seam on Agent): every
tool call the agent makes passes through `Guard.intercept` before it
executes. Read-only tools flow freely. Before a file WRITE you see a
unified diff of exactly what would change and approve it; before a
SHELL command you see the command. A denial isn't an exception — the
reason goes back to the model as an error tool_result, so the agent
adapts ("the user declined this edit") instead of crashing.

Classification is fail-closed: a tool the guard doesn't recognize as
read-only is treated like a write and prompted. Approving with 'a'
remembers that tool name for the session; `/auto on` approves
everything for the session.

The prompt uses blocking input() inside the agent's async turn. That
is deliberate and safe here: this is a single-user local REPL — while
the agent waits for your y/n there is nothing else the event loop
should be doing.

Seam for F2 (checkpoints): `notify_write` is called with the file path
right after a write is approved, BEFORE it executes — the checkpoint
manager snapshots there.
"""
from __future__ import annotations

import asyncio
import difflib
import fnmatch
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from . import style


def drain_pending_stdin() -> bool:
    """Discard input already buffered on stdin. Returns True if anything
    was discarded.

    Why: approval prompts can appear mid-turn, while the user's multi-
    line PASTE is still sitting in the stdin buffer — every leftover
    line then gets consumed as a y/n answer and rejected, producing a
    wall of "(please answer y or n)" (found live 2026-06-12, ~35 lines
    of it). Draining immediately before the prompt means the question
    is answered by what the user types AFTER seeing it.

    Tradeoff accepted: deliberate type-ahead answers are discarded too.
    Given the failure mode, that's the right side to err on — and the
    caller prints a notice whenever something was dropped.

    Covers the three terminal realities on our platforms:
      - Windows console (cmd / Windows Terminal): msvcrt.kbhit loop
      - Git Bash / MINGW64 (mintty): stdin is a PIPE — kbhit doesn't
        see it; PeekNamedPipe reports pending bytes to read off
      - POSIX: zero-timeout select
    Every branch fails silent — a drain hiccup must never break an
    approval prompt.
    """
    drained = False
    try:
        fd = sys.stdin.fileno()
    except (ValueError, OSError, AttributeError):
        return False  # stdin closed/replaced (tests, redirection)
    if os.name == "nt":
        try:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getwch()
                drained = True
        except Exception:  # noqa: BLE001
            pass
        try:
            import ctypes
            import ctypes.wintypes as wt
            import msvcrt
            handle = msvcrt.get_osfhandle(fd)
            avail = wt.DWORD(0)
            while (
                ctypes.windll.kernel32.PeekNamedPipe(
                    handle, None, 0, None, ctypes.byref(avail), None
                )
                and avail.value > 0
            ):
                if not os.read(fd, avail.value):
                    break
                drained = True
                avail = wt.DWORD(0)
        except Exception:  # noqa: BLE001
            pass
    else:
        try:
            import select
            while select.select([sys.stdin], [], [], 0)[0]:
                if not os.read(fd, 4096):
                    break
                drained = True
        except Exception:  # noqa: BLE001
            pass
    return drained


# Mutating filesystem-server tools (names register verbatim from
# @modelcontextprotocol/server-filesystem — see mcp_servers.json).
WRITE_TOOLS = frozenset({
    "write_file",
    "edit_file",
    "create_directory",
    "move_file",
})

# The shell hand: the CRYS-native run_command tool (shell.py) plus the
# legacy third-party server's tool name, kept so it stays gated if the
# server is ever re-enabled.
SHELL_TOOLS = frozenset({"run_command", "shell_execute"})

# Tools that execute but carry standing user consent: run_verify runs
# ONLY the command the user wrote into .crystal-code.json themselves.
ALWAYS_ALLOWED = frozenset({"run_verify"})

# Pure-research delegation (F7). Classified as read so the parent never
# prompts for it. The subagent's own interceptor enforces its read-only
# containment.
RESEARCH_TOOLS = frozenset({"subagent"})

# The browser hand (@playwright/mcp) — every tool it registers starts
# with this prefix. Consent model: ONE y/n per session before any
# browser tool runs (Chromium opens on first use); after consent,
# tools in _BROWSER_READ flow freely while everything else (click,
# type, fill, evaluate, upload, dialogs) prompts like a write.
BROWSER_PREFIX = "browser_"
_BROWSER_READ = frozenset({
    "browser_navigate",
    "browser_navigate_back",
    "browser_snapshot",
    "browser_take_screenshot",
    "browser_console_messages",
    "browser_network_requests",
    "browser_wait_for",
    "browser_tabs",
    "browser_resize",
    "browser_close",
})

# Read-only name shapes: the filesystem/ripgrep read tools plus the
# library's retrieval tools all match one of these.
_READ_PREFIXES = (
    "read_", "list_", "list-", "get_", "search", "advanced-search",
    "count", "directory_", "tree",
)
_READ_SUFFIXES = ("_search", "_scan", "_lookup", "_invoke")


def classify(tool_name: str) -> str:
    """'write' | 'shell' | 'read' | 'unknown' — unknown prompts (fail closed)."""
    if tool_name in RESEARCH_TOOLS:
        return "read"
    if tool_name in ALWAYS_ALLOWED:
        return "read"
    if tool_name in WRITE_TOOLS:
        return "write"
    if tool_name in SHELL_TOOLS:
        return "shell"
    low = tool_name.lower()
    if low.startswith(_READ_PREFIXES) or low.endswith(_READ_SUFFIXES):
        return "read"
    return "unknown"


def _truncation_guard(tool_name: str, tool_input: dict[str, Any]) -> Optional[str]:
    """Deny-reason if a write looks truncated/empty, else None.

    Belt to the agent loop's truncation check (H2): a write_file whose
    `content` arrived empty/missing — or an edit_file with no usable
    edits — is the signature of a tool call cut off at the output token
    limit (the 2026-06-13 MMORPG write_file failure). Refusing it here,
    before it reaches the third-party filesystem server, stops an empty
    file from being written and hands the model an actionable reason
    instead of a silent no-op. Deliberate empty-file creation is the
    rare loser in this tradeoff; the expensive failure is the common one.
    """
    if tool_name == "write_file":
        content = tool_input.get("content")
        if not isinstance(content, str) or content == "":
            return (
                "write_file arrived with empty content, which usually means "
                "the call was cut off at the output token limit. The file "
                "was NOT written. If it's large, write it in sections: "
                "create it with the first portion, then append the rest "
                "with smaller edit_file calls."
            )
    if tool_name == "edit_file":
        edits = tool_input.get("edits")
        usable = [
            e for e in (edits or [])
            if isinstance(e, dict)
            and (str(e.get("oldText", "")) or str(e.get("newText", "")))
        ]
        if not usable:
            return (
                "edit_file arrived with no usable edits, which usually means "
                "the call was cut off at the output token limit. Nothing was "
                "changed. Re-issue the edit in smaller pieces."
            )
    return None


def _read_current(path_str: str) -> Optional[str]:
    try:
        p = Path(path_str)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return None


def _unified(old: str, new: str, path: str) -> str:
    lines = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"{path} (current)",
        tofile=f"{path} (proposed)",
    )
    return "".join(lines)


def _clip(text: str, max_lines: int = 60) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    hidden = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n  ... ({hidden} more lines)"


def describe(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Human-readable preview of what the call would do."""
    if tool_name == "write_file":
        path = str(tool_input.get("path", "?"))
        new = str(tool_input.get("content", ""))
        old = _read_current(path)
        if old is None:
            n = len(new.splitlines())
            return f"NEW FILE {path} ({n} lines):\n{_clip(new, 30)}"
        diff = _unified(old, new, path)
        return _clip(diff) if diff.strip() else f"{path}: no content change"

    if tool_name == "edit_file":
        path = str(tool_input.get("path", "?"))
        edits = tool_input.get("edits") or []
        old = _read_current(path)
        if old is not None:
            # Apply the edits to a copy to render a true unified diff.
            new, applied = old, True
            for e in edits:
                o = str(e.get("oldText", ""))
                if o and o in new:
                    new = new.replace(o, str(e.get("newText", "")), 1)
                else:
                    applied = False
                    break
            if applied:
                diff = _unified(old, new, path)
                return _clip(diff) if diff.strip() else f"{path}: no content change"
        # Fall back to showing the raw edits.
        parts = [f"EDITS to {path}:"]
        for e in edits:
            parts.append("--- remove:\n" + _clip(str(e.get("oldText", "")), 12))
            parts.append("+++ insert:\n" + _clip(str(e.get("newText", "")), 12))
        return "\n".join(parts)

    if tool_name == "move_file":
        return f"MOVE {tool_input.get('source', '?')} -> {tool_input.get('destination', '?')}"
    if tool_name == "create_directory":
        return f"CREATE DIRECTORY {tool_input.get('path', '?')}"
    if tool_name in SHELL_TOOLS:
        cmd = tool_input.get("command")
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        return f"SHELL: {cmd}"
    # Unknown tool — show name + input so the user can judge.
    return f"{tool_name} with input: {tool_input}"


def _paths_from_input(tool_input: dict[str, Any]) -> list[str]:
    """Every path-shaped value a tool call carries (read or write)."""
    out: list[str] = []
    for key in ("path", "source", "destination"):
        v = tool_input.get(key)
        if isinstance(v, str) and v:
            out.append(v)
    v = tool_input.get("paths")
    if isinstance(v, list):
        out.extend(p for p in v if isinstance(p, str) and p)
    return out


class Guard:
    """Session approval policy, used as the Agent's interceptor."""

    def __init__(
        self,
        project_dir: Path,
        *,
        hooks: Optional[dict] = None,
        shell_config: Optional[dict] = None,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[[str], None] = print,
    ) -> None:
        self.project_dir = project_dir
        self.auto = False  # /auto on -> approve everything this session
        self._session_approved: set[str] = set()  # 'a' answers, per tool name
        self._input = input_fn
        self._print = print_fn
        # F2 seam: called with the file path of an approved mutating call,
        # before it executes. The checkpoint manager hangs here.
        self.notify_write: Optional[Callable[[str], None]] = None
        # F3: every tool the agent actually used this turn (approved
        # calls only — a denied write changed nothing). The CLI reads
        # this to enforce the verify loop. Reset via begin_turn().
        self.turn_calls: list[str] = []
        # Bank freshness: file paths of approved writes this turn — the
        # CLI re-syncs the tracked ones into the knowledge bank at turn
        # end (see ingest.resync_written_files). Reset via begin_turn().
        self.turn_written_paths: list[str] = []
        # F4: plan mode — investigate and propose, execute nothing.
        # Denies writes, shell, and browser interactions. run_verify IS
        # allowed (reversed 2026-06-12, phase C live test): observing the
        # failing test output is investigation — the agent kept reaching
        # for verify-first during planning and was held, leaving plans
        # built on inferred failures and starving the reflection loop of
        # its fail half. The command is user-authored (standing consent);
        # the plan-mode promise is "the agent's hands change nothing",
        # and verify commands are expected to be side-effect-safe.
        self.plan_mode = False
        # F5: project hooks from .crystal-code.json. block_paths are
        # fnmatch patterns (matched against the project-relative path
        # AND the bare filename; '*' crosses directories) that the agent
        # may never touch — reads included, because ".env" in a block
        # list means "don't read my secrets", not just "don't edit
        # them". on_file_edited commands run after each approved write
        # ({file} substituted); their output is fed back to the model.
        hooks = hooks or {}
        bp = hooks.get("block_paths")
        self.block_paths: list[str] = [p for p in bp if isinstance(p, str)] if isinstance(bp, list) else []
        fe = hooks.get("on_file_edited")
        self.on_file_edited: list[str] = [c for c in fe if isinstance(c, str)] if isinstance(fe, list) else []
        # Shell policy (see shell.py's policy amendment). "prompt": every
        # shell command is individually approved — /auto and session
        # approvals NEVER apply to shell. "deny": shell is unavailable
        # outright (headless background runs, where nobody is watching).
        self.shell_mode: str = "prompt"
        shell_cfg = shell_config or {}
        sa = shell_cfg.get("allow")
        self.shell_allow: Optional[list[str]] = [str(a) for a in sa] if isinstance(sa, list) and sa else None
        sd = shell_cfg.get("deny")
        self.shell_deny: list[str] = [str(d) for d in sd] if isinstance(sd, list) else []
        # Browser policy. "prompt": one consent question per session,
        # asked the FIRST time the agent reaches for a browser tool —
        # the user's tweak: no manual config edit, no reprompting the
        # task. "deny": browser unavailable (headless runs, which also
        # skip starting the server). None = not asked yet.
        self.browser_mode: str = "prompt"
        self.browser_consent: Optional[bool] = None

    def begin_turn(self) -> None:
        self.turn_calls.clear()
        self.turn_written_paths.clear()

    def _fresh_prompt(self) -> None:
        """Drop any stdin the user buffered BEFORE this prompt existed
        (mid-turn pastes), so the next read is a real answer. Only
        touches the process's actual stdin when the guard is wired to
        the builtin input — injected input_fn doubles (tests) skip."""
        if self._input is not input:
            return
        if drain_pending_stdin():
            self._print(style.dim(
                "  (pending pasted input discarded — it can't answer an "
                "approval prompt; paste it as a message after answering)"
            ))

    async def intercept(self, tool_name: str, tool_input: dict[str, Any]) -> dict:
        # F5 block_paths — first check, every mode: a blocked path is
        # blocked in plan mode, in /auto, everywhere.
        for p in _paths_from_input(tool_input):
            rule = self._blocked_by(p)
            if rule:
                self._trace(tool_name, tool_input, note="blocked path")
                return {
                    "action": "deny",
                    "reason": (
                        f"project hook block_paths: '{p}' matches the rule "
                        f"'{rule}' in .crystal-code.json. This path is "
                        "off-limits to the agent — do not read, modify, or "
                        "work around it; tell the user if you believe you "
                        "need it."
                    ),
                }
        kind = classify(tool_name)
        # Browser tools carry their own consent + plan logic — dispatch
        # BEFORE the generic plan-mode check, which would otherwise deny
        # browser READS during planning (they classify as 'unknown').
        if tool_name.startswith(BROWSER_PREFIX):
            return await self._intercept_browser(tool_name, tool_input)
        if self.plan_mode and kind != "read":
            self._trace(tool_name, tool_input, note="held: plan mode")
            return {
                "action": "deny",
                "reason": (
                    "plan mode is active: investigate with read-only tools "
                    "and propose a numbered plan — do not execute changes "
                    "or commands. The user will approve with /go."
                ),
            }
        # H3 (2026-06-13): refuse a truncated/empty write before it
        # reaches the filesystem server, with an actionable reason. Belt
        # to the agent loop's max_tokens check (some truncations leave a
        # parseable-but-empty content field that slips past it).
        if kind == "write":
            truncated = _truncation_guard(tool_name, tool_input)
            if truncated:
                self._trace(tool_name, tool_input, note="empty write refused")
                return {"action": "deny", "reason": truncated}
        if kind == "read":
            self._trace(tool_name, tool_input)
            self.turn_calls.append(tool_name)
            return {"action": "allow"}

        if kind == "shell":
            # Shell never rides /auto or session approvals — every
            # command is seen and answered individually, or (headless)
            # denied outright. See shell.py's policy amendment.
            return await self._intercept_shell(tool_name, tool_input)

        if self.auto or tool_name in self._session_approved:
            self._trace(tool_name, tool_input)
            self.turn_calls.append(tool_name)
            self._note_write(tool_name, tool_input)
            return {"action": "allow"}

        # Show what would happen, then ask.
        self._print("")
        self._print(style.yellow(f"  [approval needed] {tool_name}"))
        for line in describe(tool_name, tool_input).splitlines():
            self._print("  " + style.dim("|") + " " + style.color_diff_line(line))
        self._fresh_prompt()
        while True:
            answer = self._input(
                "  approve? y = yes / n = no / a = yes for this tool all session: "
            ).strip().lower()
            if answer in ("y", "yes"):
                self.turn_calls.append(tool_name)
                self._note_write(tool_name, tool_input)
                return {"action": "allow"}
            if answer in ("a", "all"):
                self._session_approved.add(tool_name)
                self.turn_calls.append(tool_name)
                self._note_write(tool_name, tool_input)
                return {"action": "allow"}
            if answer in ("n", "no"):
                return {
                    "action": "deny",
                    "reason": (
                        "the user reviewed this action and declined it. Do "
                        "not retry the same change; ask the user how they "
                        "want to proceed, or take a different approach."
                    ),
                }
            self._print("  (please answer y, n, or a)")

    async def _intercept_shell(self, tool_name: str, tool_input: dict[str, Any]) -> dict:
        from .shell import screen_command

        command = str(tool_input.get("command", ""))
        if self.shell_mode == "deny":
            self._trace(tool_name, tool_input, note="denied: headless")
            return {
                "action": "deny",
                "reason": (
                    "shell commands are not available in headless background "
                    "runs — there is no user present to review them. Use the "
                    "file tools and run_verify, or note the command in your "
                    "summary for the user to run themselves."
                ),
            }
        reason = screen_command(command, self.shell_allow, self.shell_deny)
        if reason:
            self._trace(tool_name, tool_input, note="screened out")
            return {
                "action": "deny",
                "reason": (
                    f"this command was refused before review: {reason}. "
                    "Propose a different command, or tell the user what you "
                    "were trying to do."
                ),
            }
        self._print("")
        self._print(style.yellow(f"  [approval needed] shell command"))
        self._print("  " + style.dim("|") + " " + style.bold(command))
        self._fresh_prompt()
        while True:
            answer = self._input(
                "  run it? y = yes / n = no  (shell is always asked — /auto doesn't apply): "
            ).strip().lower()
            if answer in ("y", "yes"):
                self.turn_calls.append(tool_name)
                return {"action": "allow"}
            if answer in ("n", "no"):
                return {
                    "action": "deny",
                    "reason": (
                        "the user reviewed this command and declined it. Do "
                        "not retry the same command; ask how they want to "
                        "proceed, or take a different approach."
                    ),
                }
            self._print("  (please answer y or n — shell has no approve-all)")

    async def _intercept_browser(self, tool_name: str, tool_input: dict[str, Any]) -> dict:
        if self.browser_mode == "deny":
            self._trace(tool_name, tool_input, note="denied: headless")
            return {
                "action": "deny",
                "reason": (
                    "the browser is not available in headless background "
                    "runs — web content is untrusted input and nobody is "
                    "watching. Note what you wanted to look up in your "
                    "summary for the user."
                ),
            }
        if self.browser_consent is None:
            # The one-time session consent — asked at the moment of
            # first need, so the user's task continues without a
            # config edit or a reprompt.
            self._print("")
            self._print(style.yellow("  [browser] the agent wants to use the web browser"))
            self._print(style.dim("  (opens a local Chromium window; web pages are untrusted input — treat with care)"))
            self._fresh_prompt()
            while True:
                answer = self._input("  allow the browser for this session? y = yes / n = no: ").strip().lower()
                if answer in ("y", "yes"):
                    self.browser_consent = True
                    break
                if answer in ("n", "no"):
                    self.browser_consent = False
                    break
                self._print("  (please answer y or n)")
        if not self.browser_consent:
            self._trace(tool_name, tool_input, note="browser declined")
            return {
                "action": "deny",
                "reason": (
                    "the user declined browser use for this session. Work "
                    "without it, or tell the user what you would have "
                    "looked up."
                ),
            }
        if self.plan_mode and tool_name not in _BROWSER_READ:
            self._trace(tool_name, tool_input, note="held: plan mode")
            return {
                "action": "deny",
                "reason": (
                    "plan mode is active: browse and read freely, but page "
                    "interactions wait for the approved plan."
                ),
            }
        if tool_name in _BROWSER_READ:
            self._trace(tool_name, tool_input)
            self.turn_calls.append(tool_name)
            return {"action": "allow"}
        # Page interaction — prompted like a write ('a' and /auto apply:
        # the session consent already happened, and per-click prompts on
        # an approved flow would be unbearable without an opt-out).
        if self.auto or tool_name in self._session_approved:
            self._trace(tool_name, tool_input)
            self.turn_calls.append(tool_name)
            return {"action": "allow"}
        self._print("")
        self._print(style.yellow(f"  [approval needed] browser action"))
        self._print("  " + style.dim("|") + " " + style.humanize_call(tool_name, tool_input, self.project_dir))
        self._fresh_prompt()
        while True:
            answer = self._input(
                "  approve? y = yes / n = no / a = yes for this action all session: "
            ).strip().lower()
            if answer in ("y", "yes"):
                self.turn_calls.append(tool_name)
                return {"action": "allow"}
            if answer in ("a", "all"):
                self._session_approved.add(tool_name)
                self.turn_calls.append(tool_name)
                return {"action": "allow"}
            if answer in ("n", "no"):
                return {
                    "action": "deny",
                    "reason": (
                        "the user reviewed this browser action and declined "
                        "it. Do not retry it; ask how they want to proceed."
                    ),
                }
            self._print("  (please answer y, n, or a)")

    def _trace(self, tool_name: str, tool_input: dict[str, Any], note: str = "") -> None:
        """One dim plain-language line per tool call — the CLI's activity
        feed. With library logs quieted by default, this is how the user
        watches the agent work, in words rather than function calls."""
        doing = style.humanize_call(tool_name, tool_input, self.project_dir)
        suffix = f"  ({note})" if note else ""
        self._print(style.dim(f"    · {doing}{suffix}"))

    def _note_write(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        if tool_name not in WRITE_TOOLS:
            return
        path = tool_input.get("path") or tool_input.get("destination")
        if not path:
            return
        self.turn_written_paths.append(str(path))
        if self.notify_write is not None:
            try:
                self.notify_write(str(path))
            except Exception:  # noqa: BLE001 — a checkpoint hiccup must
                pass  # never block an approved write.

    # -- F5: block matching + post-edit hooks ---------------------------

    def _blocked_by(self, path_str: str) -> Optional[str]:
        """The block_paths rule this path matches, or None.

        A relative path is treated as project-relative directly — the
        filesystem server resolves relative paths against its working
        directory (the project folder), so routing them through
        os.path.relpath (which resolves against the CLI's CWD) would
        let `secrets/key.txt` slip past a `secrets/*` rule that
        /abs/project/secrets/key.txt correctly trips."""
        if not self.block_paths:
            return None
        if os.path.isabs(path_str):
            try:
                rel = os.path.relpath(path_str, self.project_dir)
            except ValueError:  # e.g. different drive on Windows
                rel = path_str
        else:
            rel = path_str
        rel = rel.replace("\\", "/")
        base = os.path.basename(rel)
        for pat in self.block_paths:
            if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(base, pat):
                return pat
        return None

    async def after_tool(self, tool_name: str, tool_input: dict[str, Any]) -> Optional[str]:
        """Agent's post-call observer (the F5 seam): run on_file_edited
        commands after a successful write; the returned note is appended
        to the tool output so the model sees what the hooks changed."""
        if tool_name not in WRITE_TOOLS or not self.on_file_edited:
            return None
        path = tool_input.get("path") or tool_input.get("destination")
        if not path:
            return None
        notes: list[str] = []
        for template in self.on_file_edited:
            cmd = template.replace("{file}", str(path))
            # E1c Phase 0 (2026-07-03): route hook execution through the
            # shared sandbox chokepoint. Hooks are operator-configured in
            # .crystal-code.json and may use shell features.
            from .sandbox import SandboxProfile, sandbox_run_async

            result = await sandbox_run_async(
                cmd, self.project_dir, 120,
                allow_shell_features=True,
                profile=SandboxProfile.CPU,
            )
            if result.timed_out:
                notes.append(f"`{cmd}` timed out after 120s")
                continue
            out = result.output.strip()
            if len(out) > 1500:
                out = out[:1500] + " ..."
            status = "ok" if result.exit_code == 0 else f"exit {result.exit_code}"
            notes.append(f"`{cmd}` ({status})" + (f": {out}" if out else ""))
        if not notes:
            return None
        return (
            "project on_file_edited hooks ran — if a hook reformatted the "
            "file, re-read it before further edits. " + " | ".join(notes)
        )
