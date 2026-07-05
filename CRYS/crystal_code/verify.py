"""F3 — the `run_verify` tool: the agent checks its own work.

The shell hand is deliberately allowlisted to read-only commands, so
the agent cannot run tests through it — and widening that allowlist
was ruled out without an OS sandbox (see mcp_servers.json). This tool
is the honest alternative: it runs EXACTLY the verify command the user
wrote into `.crystal-code.json` (e.g. "pytest -q"), with no parameters
the model can vary. The user authoring that command is standing
consent, which is why the guard auto-allows this one tool.

Output is tail-truncated: test runners print the failures at the end,
and token economics say don't ship ten thousand lines of dots.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

from crystal_cache.agent import Tool, get_registry

VERIFY_TOOL_NAME = "run_verify"
_TIMEOUT_SECONDS = 600
_MAX_OUTPUT_CHARS = 4000

# Per-run result sink (Phase C, reflection loop). The tool registers
# ONCE per process but the daemon runs many tasks per process, so the
# observer can't be baked into the closure at registration time — it's
# swapped per run. background.py sets it at run start and clears it in
# its finally; the REPL never sets it, so interactive sessions are
# untouched. Single sink (not a list) because runs are sequential
# within a process by design.
_RESULT_SINK: dict = {"cb": None}


def set_verify_result_sink(cb) -> None:
    """Install (or clear, with None) the per-run observer for verify
    results. The callback receives (exit_code: int, output: str)."""
    _RESULT_SINK["cb"] = cb


def register_verify_tool(project_dir: Path, verify_command: str) -> bool:
    """Register `run_verify` into the shared registry (once per process)."""
    registry = get_registry()
    if VERIFY_TOOL_NAME in registry:
        return False

    async def _impl(customer_id: str, **kwargs: Any) -> dict[str, Any]:
        # customer_id is required by the registry contract; verification
        # is project-scoped, not customer-scoped.
        #
        # E1c Phase 0 (2026-07-03): route through the shared sandbox
        # chokepoint like the shell tool, so the verify path can't be used
        # to smuggle unsandboxed execution once the backend lands. The
        # verify command is operator-configured and may use shell features
        # (pipes, &&), so allow_shell_features=True.
        from .sandbox import SandboxProfile, sandbox_run_async

        result = await sandbox_run_async(
            verify_command, project_dir, _TIMEOUT_SECONDS,
            allow_shell_features=True,
            profile=SandboxProfile.CPU,
        )
        combined = result.output
        rc = -1 if result.timed_out else result.exit_code
        if _RESULT_SINK["cb"] is not None:
            try:
                _RESULT_SINK["cb"](rc, combined)
            except Exception:  # noqa: BLE001 — observation must never break verification
                pass
        return {
            "exit_code": rc,
            "output": combined or "(no output)",
            "is_error": result.is_error,
        }

    registry.register(Tool(
        name=VERIFY_TOOL_NAME,
        description=(
            f"Run the project's verification command (`{verify_command}`) "
            "and return its output and exit code. Takes no parameters. "
            "ALWAYS run this after creating or modifying any file, read "
            "any failures, fix them, and run it again — only conclude "
            "your work when it passes."
        ),
        contexts=frozenset({"agent"}),
        parameters_schema={"type": "object", "properties": {}},
        impl=_impl,
    ))
    return True
