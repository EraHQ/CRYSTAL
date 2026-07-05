"""Connect to the standard MCP servers and register their tools (Step 2b).

This is the "hands" bridge. It reads mcp_servers.json, connects to each
ENABLED MCP server over stdio (via the official `mcp` Python SDK), asks
each server what tools it has, and registers those tools into the agent's
tool registry — so the agent can call them exactly like its built-in
tools.

Step 2b connects the FILESYSTEM server only (read/write/edit/list/move),
scoped to one project folder. Code search and shell (Step 2c) stay
disabled in the config until the sandbox is in place.

Segmentation: this lives in the CRYS agent segment, imports the
library's PUBLIC tool registry, and never touches src/.

Lifecycle: each MCP server is a child process that must stay alive for
the whole session. `MCPHands.open()` starts the enabled servers and
registers their tools; `MCPHands.close()` shuts them down. The CLI holds
one MCPHands for the session and closes it on exit.

Graceful degradation: if a server can't start (missing `mcp` SDK,
missing Node/npx, a bad path), `open()` raises MCPHandsError. The CLI
catches it, tells the user plainly, and runs the agent WITHOUT file
tools rather than failing entirely.
"""
from __future__ import annotations

import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Callable, Optional

import structlog

from crystal_cache.agent import Tool, get_registry

logger = structlog.get_logger(__name__)

# mcp_hands.py -> crystal_code -> CRYS / mcp_servers.json
CONFIG_PATH = Path(__file__).resolve().parents[1] / "mcp_servers.json"

# F2 hardening (2026-07-03): the first-party servers that ship WITH the
# agent in the package-level config. These are trusted by provenance (we
# author them) and spawn without an approval prompt. ANY server whose name
# is not in this set — e.g. one introduced by a repo-level or tampered
# config — is treated as untrusted: it requires explicit human approval
# before its command is spawned (interactive), and is refused outright in
# headless runs where no human is present to approve it.
_TRUSTED_BUILTIN_SERVERS = frozenset({
    "filesystem", "ripgrep", "browser", "shell",
})


class MCPHandsError(RuntimeError):
    """Raised when a server can't be started or the config can't be read."""


class MCPHands:
    """Connects enabled MCP servers and registers their tools."""

    def __init__(
        self,
        project_dir: Path,
        skip: tuple[str, ...] = (),
        *,
        headless: bool = False,
        approve_untrusted: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        self.project_dir = project_dir
        # Server names to leave unstarted even when enabled in config —
        # headless background runs skip the browser this way.
        self._skip = set(skip)
        # F2: headless refuses untrusted (non-built-in) servers outright —
        # no human is present to approve a spawn. Interactive runs call
        # approve_untrusted(name, command) and spawn only on a True return.
        self._headless = headless
        self._approve_untrusted = approve_untrusted
        self._stack = AsyncExitStack()
        self._sessions: list[Any] = []  # keep live references for the session
        self.connected_servers: list[str] = []
        self.registered_tools: list[str] = []
        self.skipped_untrusted: list[str] = []

    async def open(self) -> None:
        """Connect to every enabled server and register its tools.

        Must be called BEFORE the Agent is constructed, so the agent
        picks up these tools when it reads the registry.
        """
        # Import the SDK lazily so a missing dependency is a clear,
        # contained message rather than a crash at program import.
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as e:
            raise MCPHandsError(
                "the MCP client SDK isn't installed (run: pip install mcp)"
            ) from e

        config = _load_config()
        servers = config.get("mcpServers", {})
        for name, spec in servers.items():
            if not spec.get("_enabled", False) or name in self._skip:
                continue
            command = spec.get("command")
            args = [self._substitute(a) for a in spec.get("args", [])]

            # F2 trust gate (2026-07-03): a server not among the first-party
            # built-ins is untrusted (it could come from a repo-level or
            # tampered config). Headless refuses it — nobody can approve a
            # spawn. Interactive asks the human, showing the exact command,
            # and spawns only on approval. Built-ins spawn without a prompt.
            if name not in _TRUSTED_BUILTIN_SERVERS:
                cmd_display = " ".join(
                    [str(command), *[str(a) for a in args]]
                ).strip()
                if self._headless:
                    self.skipped_untrusted.append(name)
                    logger.warning(
                        "mcp_hands.untrusted_skipped_headless",
                        server=name, command=cmd_display,
                    )
                    continue
                approved = (
                    self._approve_untrusted is not None
                    and self._approve_untrusted(name, cmd_display)
                )
                if not approved:
                    self.skipped_untrusted.append(name)
                    logger.warning(
                        "mcp_hands.untrusted_declined",
                        server=name, command=cmd_display,
                    )
                    continue

            # Merge any server-specific env (e.g. the shell allowlist) ON TOP
            # of the real environment, so the child still inherits PATH and
            # can find its executable and the commands it runs. Passing only
            # the spec's env would wipe PATH.
            spec_env = spec.get("env")
            env = {**os.environ, **spec_env} if spec_env else None
            # Run each server with its working directory set to the project
            # folder. The filesystem server is already hard-scoped by its
            # path arg; this matters for the ripgrep search server, whose
            # search defaults to the working directory when no path is given.
            params = StdioServerParameters(
                command=command,
                args=args,
                env=env,
                cwd=str(self.project_dir),
            )
            try:
                read, write = await self._stack.enter_async_context(
                    stdio_client(params)
                )
                session = await self._stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()
                listed = await session.list_tools()
            except Exception as e:  # noqa: BLE001 — report which server failed
                raise MCPHandsError(
                    f"couldn't start server {name!r} "
                    f"({command} {' '.join(args)}): {type(e).__name__}: {e}"
                ) from e

            self._sessions.append(session)
            self.connected_servers.append(name)
            self._register_tools(name, session, listed.tools)
            logger.info(
                "mcp_hands.connected",
                server=name,
                tools=len(listed.tools),
            )

    def _substitute(self, arg: str) -> str:
        """Fill placeholders in a server's args (currently just PROJECT_DIR)."""
        return arg.replace("${PROJECT_DIR}", str(self.project_dir))

    def _register_tools(self, server_name: str, session: Any, tools: Any) -> None:
        registry = get_registry()
        for t in tools:
            if t.name in registry:
                # Don't clobber an existing tool of the same name.
                logger.warning(
                    "mcp_hands.tool_name_collision",
                    tool=t.name,
                    server=server_name,
                )
                continue
            registry.register(Tool(
                name=t.name,
                description=t.description or f"{server_name} tool",
                contexts=frozenset({"agent"}),
                parameters_schema=(
                    t.inputSchema or {"type": "object", "properties": {}}
                ),
                impl=self._make_impl(session, t.name),
            ))
            self.registered_tools.append(t.name)

    @staticmethod
    def _make_impl(session: Any, tool_name: str) -> Any:
        """Build the proxy that forwards a tool call to the MCP server."""
        async def _impl(customer_id: str, **kwargs: Any) -> dict[str, Any]:
            # customer_id is required by the registry contract, but file
            # operations aren't customer-scoped, so it's ignored here.
            result = await session.call_tool(tool_name, kwargs)
            return {
                "result": _result_text(result),
                "is_error": bool(getattr(result, "isError", False)),
            }
        return _impl

    async def close(self) -> None:
        """Shut down every connected server. Safe to call even if open() failed."""
        await self._stack.aclose()


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise MCPHandsError(f"config not found at {CONFIG_PATH}")
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise MCPHandsError(f"config at {CONFIG_PATH} is not valid JSON: {e}") from e


def _result_text(result: Any) -> str:
    """Flatten an MCP tool result's content blocks into plain text."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)
