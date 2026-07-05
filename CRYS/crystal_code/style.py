"""Terminal styling + library log quieting for the coding agent CLI.

Dependency-free ANSI. Color is enabled only on a real terminal and
respects NO_COLOR; everything degrades to plain text in pipes, tests,
and captured output. On Windows consoles, `os.system("")` switches on
VT processing (the documented no-op trick) — Git Bash/mintty handles
ANSI natively anyway.

quiet_library_logs() exists because structlog runs UNCONFIGURED in the
library (PrintLogger, no level filter), which is why the CLI used to
open with 35 tool_registry.registered lines and interleave router
internals into conversations. Default is warnings+ only; --verbose
restores the full firehose for debugging. It must run BEFORE the
library modules are imported (registration logs fire at import time),
which is why __main__.py calls it from a raw sys.argv scan before
importing cli.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from typing import Any

_enabled: bool | None = None


def enabled() -> bool:
    global _enabled
    if _enabled is None:
        if os.name == "nt":
            os.system("")  # enable VT processing on Windows consoles
        _enabled = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    return _enabled


def _wrap(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled() else text


def dim(text: str) -> str:
    return _wrap("2", text)


def bold(text: str) -> str:
    return _wrap("1", text)


def green(text: str) -> str:
    return _wrap("32", text)


def red(text: str) -> str:
    return _wrap("31", text)


def yellow(text: str) -> str:
    return _wrap("33", text)


def cyan(text: str) -> str:
    return _wrap("36", text)


def rule(char: str = "─") -> str:
    width = min(shutil.get_terminal_size((80, 20)).columns, 100)
    return dim(char * width)


def color_diff_line(line: str) -> str:
    """Unified-diff line coloring for the approval preview."""
    if line.startswith(("+++", "---")):
        return bold(line)
    if line.startswith("+"):
        return green(line)
    if line.startswith("-"):
        return red(line)
    if line.startswith("@@"):
        return cyan(line)
    return line


def trace_args(tool_input: dict[str, Any], limit: int = 90) -> str:
    """One compact line of a tool call's arguments for the activity
    trace — every value flattened and clipped, the whole thing capped."""
    parts: list[str] = []
    used = 0
    for key, value in tool_input.items():
        s = str(value).replace("\n", " ")
        if len(s) > 40:
            s = s[:37] + "..."
        piece = f"{key}={s!r}" if isinstance(value, str) else f"{key}={s}"
        if used + len(piece) > limit and parts:
            parts.append("…")
            break
        parts.append(piece)
        used += len(piece)
    return " ".join(parts)


def quiet_library_logs(verbose: bool) -> None:
    """Warnings+ only from the library by default; --verbose restores
    everything. Also silences the model-weights progress bars when
    quiet — the CLI's own status line covers that wait."""
    import structlog

    level = logging.DEBUG if verbose else logging.WARNING
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )
    if not verbose:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


# ---------------------------------------------------------------------------
# Plain-language activity narration
# ---------------------------------------------------------------------------

def short_path(value: str, project_dir: Any = None) -> str:
    """A path the way a person would say it: project-relative when it's
    inside the project, just the tail of it otherwise."""
    s = str(value).replace("\\", "/")
    if project_dir is not None:
        root = str(project_dir).replace("\\", "/").rstrip("/")
        if s.lower().startswith(root.lower() + "/"):
            return s[len(root) + 1:]
        if s.lower() == root.lower():
            return "."
    parts = s.rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) > 2 else s


def _first(tool_input: dict[str, Any], *keys: str) -> str:
    for k in keys:
        v = tool_input.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _clip_text(s: str, n: int = 60) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def humanize_call(tool_name: str, tool_input: dict[str, Any], project_dir: Any = None) -> str:
    """One plain-language line for a tool call — what the agent is doing,
    in words a person would use. Unknown tools fall back to the raw
    name + compact args so future MCP tools still surface."""
    def p(*keys: str) -> str:
        return short_path(_first(tool_input, *keys) or "?", project_dir)

    q = _clip_text(_first(tool_input, "query", "query_text", "pattern", "q", "key", "task", "prompt", "command", "text"))

    # Files and folders
    if tool_name in ("read_file", "read_text_file", "read_media_file"):
        return f"reading {p('path')}"
    if tool_name == "read_multiple_files":
        paths = tool_input.get("paths") or []
        return f"reading {len(paths)} files" if paths else "reading files"
    if tool_name == "write_file":
        return f"writing {p('path')}"
    if tool_name == "edit_file":
        return f"editing {p('path')}"
    if tool_name == "create_directory":
        return f"creating folder {p('path')}"
    if tool_name == "move_file":
        return f"moving {p('source')} → {p('destination')}"
    if tool_name in ("list_directory", "list_directory_with_sizes"):
        return f"listing {p('path')}"
    if tool_name == "directory_tree":
        return f"mapping the folder tree of {p('path')}"
    if tool_name == "get_file_info":
        return f"checking {p('path')}"
    if tool_name == "search_files":
        return f"searching file names for {q!r}" if q else "searching file names"
    if tool_name == "list_allowed_directories":
        return "checking which folders are accessible"

    # Ripgrep (code search)
    if tool_name in ("search", "advanced-search"):
        return f"searching file contents for {q!r}" if q else "searching file contents"
    if tool_name == "count-matches":
        return f"counting matches for {q!r}"
    if tool_name == "list-files":
        return "listing files"
    if tool_name == "list-file-types":
        return "listing file types"

    # Knowledge bank
    if tool_name == "knowledge_search":
        return f"searching the knowledge bank for {q!r}"
    if tool_name == "content_search":
        return f"looking for passages about {q!r}"
    if tool_name == "navigation_search":
        return "surveying what the knowledge bank covers"
    if tool_name == "key_scan":
        what = _first(tool_input, "subject_contains", "key_prefix")
        return f"listing everything filed under {_clip_text(what)!r}" if what else "listing the bank's index"
    if tool_name == "depth_search":
        return f"analyzing connections around {q!r}"
    if tool_name == "crystal_recall":
        return f"recalling stored knowledge about {q!r}" if q else "recalling stored knowledge"
    if tool_name == "crystal_write":
        return f"saving to the knowledge bank: {q!r}" if q else "saving to the knowledge bank"
    if tool_name == "mem0_recall":
        return "recalling session memory"
    if tool_name == "mem0_write":
        return "noting this in session memory"
    if tool_name == "document_upload":
        return "uploading a document to the knowledge bank"

    # Work delegation + verification
    if tool_name == "run_verify":
        return "running the project's verify command"
    if tool_name == "subagent":
        return f"delegating research: {q!r}" if q else "delegating research to a helper"
    if tool_name == "cognition_run":
        return "starting a deeper research task"
    if tool_name == "llm_invoke":
        return "consulting a model"
    if tool_name == "decompose":
        return "breaking the question into parts"
    if tool_name == "web_search":
        return f"searching the web for {q!r}"
    if tool_name == "queue_task":
        return f"queueing a background task: {q!r}" if q else "queueing a background task"
    if tool_name == "get_task_status":
        return "checking the background task queue"
    if tool_name in ("shell_execute", "run_command"):
        return f"running: {q}" if q else "running a shell command"

    # Browser (@playwright/mcp)
    if tool_name.startswith("browser_"):
        url = _clip_text(_first(tool_input, "url"))
        element = _clip_text(_first(tool_input, "element", "selector", "ref"))
        if tool_name == "browser_navigate":
            return f"opening {url}" if url else "opening a page"
        if tool_name == "browser_navigate_back":
            return "going back a page"
        if tool_name == "browser_snapshot":
            return "reading the page"
        if tool_name == "browser_take_screenshot":
            return "taking a screenshot"
        if tool_name == "browser_click":
            return f"clicking {element!r}" if element else "clicking on the page"
        if tool_name == "browser_type":
            return f"typing into {element!r}" if element else "typing on the page"
        if tool_name == "browser_fill_form":
            return "filling out a form"
        if tool_name == "browser_select_option":
            return f"selecting an option in {element!r}" if element else "selecting an option"
        if tool_name == "browser_press_key":
            key = _first(tool_input, "key")
            return f"pressing {key!r}" if key else "pressing a key"
        if tool_name == "browser_hover":
            return f"hovering over {element!r}" if element else "hovering on the page"
        if tool_name == "browser_evaluate":
            return "running JavaScript on the page"
        if tool_name == "browser_file_upload":
            return "uploading a file to the page"
        if tool_name == "browser_console_messages":
            return "reading the browser console"
        if tool_name == "browser_network_requests":
            return "reading the page's network activity"
        if tool_name == "browser_wait_for":
            return "waiting for the page"
        if tool_name == "browser_tabs":
            return "managing browser tabs"
        if tool_name == "browser_handle_dialog":
            return "answering a browser dialog"
        if tool_name == "browser_resize":
            return "resizing the browser window"
        if tool_name == "browser_close":
            return "closing the browser"
        if tool_name == "browser_install":
            return "installing the browser (one-time download)"
        return f"using the browser: {tool_name.removeprefix('browser_').replace('_', ' ')}"

    # Unknown (future MCP tools) — raw but compact.
    return f"{tool_name} {trace_args(tool_input)}".rstrip()
