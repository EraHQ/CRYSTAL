"""Read-only source connector — grounded access to actual source code.

Gives cognition a way to READ real source (this repo, or a GitHub repo)
instead of reconstructing file paths and code from memory. This is the
fix behind the C2/C3 groundedness work: a path/code question can now be
answered against the real thing, and the answerability gate (C2) +
groundedness gate (C3) can treat a source read as grounding evidence.

Two backends behind one interface, selected by CC_SOURCE_BACKEND:

  * LocalFsSourceConnector — reads files under a configured root on the
    local filesystem (the deployment's own repo). No network, no auth.
  * GitHubSourceConnector — reads files from a GitHub repo via the REST
    contents + code-search APIs. Optional token for private repos and
    higher rate limits.

Both expose the same three READ-ONLY operations, each returning a
JSON-serializable dict:

  read(path)    -> {op, backend, path, content, truncated, size} | {error}
  list(path)    -> {op, backend, path, entries:[{name,type,size}]}  | {error}
  search(query) -> {op, backend, query, matches:[{path,line,text}], truncated} | {error}

`build_source_connector(settings)` returns the configured connector, or
None when nothing is configured. When None, the source_lookup tool
reports unavailable rather than fabricating an answer.

SECURITY (local_fs):
  - The root must be explicitly configured (never defaults to / or cwd).
  - Every path is resolved with realpath and confined to the root
    (traversal guard) — `../` cannot escape.
  - Reads are size-capped; binary and oversized files are skipped.
  - The connector is READ-ONLY: no write/delete/move operations exist.
"""
from __future__ import annotations

import asyncio
import base64
import os
from typing import Any, Optional, Protocol, runtime_checkable

import httpx
import structlog

logger = structlog.get_logger(__name__)


# Caps shared by both backends.
MAX_READ_BYTES = 256_000          # ~256 KB per file read
SEARCH_MAX_MATCHES = 80           # stop collecting after this many hits
SEARCH_MAX_FILES_SCANNED = 5_000  # bound the local walk
SNIPPET_MAX_CHARS = 200           # per matched line

# Directories never worth scanning for source grounding.
IGNORE_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".next", ".turbo", "target", ".idea", ".vscode",
})


class SourceAccessError(Exception):
    """Raised when a requested path escapes the allowed root or is denied."""


@runtime_checkable
class SourceConnector(Protocol):
    """Read-only source access. Implementations must be safe to call
    concurrently and must never write to the source."""

    backend: str

    async def read(self, path: str) -> dict[str, Any]: ...
    async def list(self, path: str = "") -> dict[str, Any]: ...
    async def search(self, query: str, *, path_prefix: str = "") -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _looks_binary(chunk: bytes) -> bool:
    """A NUL byte in the first chunk is a reliable binary signal."""
    return b"\x00" in chunk


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------

class LocalFsSourceConnector:
    """Reads files confined to a single configured root directory."""

    backend = "local_fs"

    def __init__(self, root: str) -> None:
        if not root:
            raise ValueError("LocalFsSourceConnector requires a non-empty root")
        # realpath up front so the traversal guard compares canonical paths.
        self._root = os.path.realpath(root)

    # -- path safety --------------------------------------------------------

    def _resolve(self, rel_path: str) -> str:
        """Resolve a caller-supplied path against the root and confirm it
        stays inside the root. Raises SourceAccessError on escape."""
        rel = (rel_path or "").lstrip("/\\")
        target = os.path.realpath(os.path.join(self._root, rel))
        if target != self._root and not target.startswith(self._root + os.sep):
            raise SourceAccessError(
                f"path {rel_path!r} resolves outside the configured root"
            )
        return target

    # -- sync bodies (run off the event loop via asyncio.to_thread) ---------

    def _read_sync(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if not os.path.isfile(target):
            return {"op": "read", "backend": self.backend, "path": path,
                    "error": "not a file or does not exist"}
        size = os.path.getsize(target)
        with open(target, "rb") as fh:
            raw = fh.read(MAX_READ_BYTES + 1)
        if _looks_binary(raw[:4096]):
            return {"op": "read", "backend": self.backend, "path": path,
                    "size": size, "error": "binary file (not readable as text)"}
        truncated = len(raw) > MAX_READ_BYTES or size > MAX_READ_BYTES
        return {
            "op": "read", "backend": self.backend, "path": path,
            "size": size, "truncated": truncated,
            "content": _decode(raw[:MAX_READ_BYTES]),
        }

    def _list_sync(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if not os.path.isdir(target):
            return {"op": "list", "backend": self.backend, "path": path,
                    "error": "not a directory or does not exist"}
        entries = []
        for name in sorted(os.listdir(target)):
            if name in IGNORE_DIRS:
                continue
            full = os.path.join(target, name)
            is_dir = os.path.isdir(full)
            entries.append({
                "name": name,
                "type": "dir" if is_dir else "file",
                "size": 0 if is_dir else os.path.getsize(full),
            })
        return {"op": "list", "backend": self.backend, "path": path,
                "entries": entries}

    def _search_sync(self, query: str, path_prefix: str) -> dict[str, Any]:
        if not query:
            return {"op": "search", "backend": self.backend, "query": query,
                    "matches": [], "truncated": False,
                    "error": "search needs a non-empty query"}
        start = self._resolve(path_prefix)
        needle = query.lower()
        matches: list[dict[str, Any]] = []
        files_scanned = 0
        truncated = False

        for dirpath, dirnames, filenames in os.walk(start):
            # Prune ignored dirs in place so os.walk skips them.
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            for fname in sorted(filenames):
                if files_scanned >= SEARCH_MAX_FILES_SCANNED:
                    truncated = True
                    break
                files_scanned += 1
                full = os.path.join(dirpath, fname)
                try:
                    with open(full, "rb") as fh:
                        raw = fh.read(MAX_READ_BYTES)
                except OSError:
                    continue
                if _looks_binary(raw[:4096]):
                    continue
                rel = os.path.relpath(full, self._root)
                for lineno, line in enumerate(_decode(raw).splitlines(), start=1):
                    if needle in line.lower():
                        matches.append({
                            "path": rel.replace(os.sep, "/"),
                            "line": lineno,
                            "text": line.strip()[:SNIPPET_MAX_CHARS],
                        })
                        if len(matches) >= SEARCH_MAX_MATCHES:
                            truncated = True
                            break
                if len(matches) >= SEARCH_MAX_MATCHES:
                    break
            if truncated or len(matches) >= SEARCH_MAX_MATCHES:
                break

        return {"op": "search", "backend": self.backend, "query": query,
                "matches": matches, "truncated": truncated,
                "files_scanned": files_scanned}

    # -- async interface ----------------------------------------------------

    async def read(self, path: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(self._read_sync, path)
        except SourceAccessError as e:
            return {"op": "read", "backend": self.backend, "path": path,
                    "error": str(e)}

    async def list(self, path: str = "") -> dict[str, Any]:
        try:
            return await asyncio.to_thread(self._list_sync, path)
        except SourceAccessError as e:
            return {"op": "list", "backend": self.backend, "path": path,
                    "error": str(e)}

    async def search(self, query: str, *, path_prefix: str = "") -> dict[str, Any]:
        try:
            return await asyncio.to_thread(self._search_sync, query, path_prefix)
        except SourceAccessError as e:
            return {"op": "search", "backend": self.backend, "query": query,
                    "matches": [], "truncated": False, "error": str(e)}


# ---------------------------------------------------------------------------
# GitHub backend
# ---------------------------------------------------------------------------

_GITHUB_API = "https://api.github.com"


class GitHubSourceConnector:
    """Reads a GitHub repo via the REST contents + code-search APIs.

    Token is optional (public repos work unauthenticated, subject to
    tighter rate limits). `ref` may be a branch, tag, or commit SHA;
    when empty the repo's default branch is used. Note: GitHub code
    search only indexes the default branch, so `search` ignores `ref`.
    """

    backend = "github"

    def __init__(
        self, owner: str, repo: str, ref: str = "", token: Optional[str] = None,
        *, timeout: float = 15.0,
    ) -> None:
        if not owner or not repo:
            raise ValueError("GitHubSourceConnector requires owner and repo")
        self._owner = owner
        self._repo = repo
        self._ref = ref or ""
        self._token = token or ""
        self._timeout = timeout

    def _headers(self, *, text_match: bool = False) -> dict[str, str]:
        accept = (
            "application/vnd.github.text-match+json"
            if text_match else "application/vnd.github+json"
        )
        headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _contents(self, path: str) -> Any:
        url = f"{_GITHUB_API}/repos/{self._owner}/{self._repo}/contents/{path.lstrip('/')}"
        params = {"ref": self._ref} if self._ref else None
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, headers=self._headers(), params=params)
        resp.raise_for_status()
        return resp.json()

    async def read(self, path: str) -> dict[str, Any]:
        try:
            data = await self._contents(path)
        except httpx.HTTPStatusError as e:
            return {"op": "read", "backend": self.backend, "path": path,
                    "error": f"github contents request failed: {e.response.status_code}"}
        except Exception as e:
            return {"op": "read", "backend": self.backend, "path": path,
                    "error": f"github contents request failed: {e}"}

        if isinstance(data, list):
            return {"op": "read", "backend": self.backend, "path": path,
                    "error": "path is a directory; use op=list"}
        if data.get("type") != "file":
            return {"op": "read", "backend": self.backend, "path": path,
                    "error": f"not a file (type={data.get('type')!r})"}

        size = int(data.get("size", 0))
        encoded = data.get("content", "") or ""
        try:
            raw = base64.b64decode(encoded)
        except Exception:
            raw = b""
        if _looks_binary(raw[:4096]):
            return {"op": "read", "backend": self.backend, "path": path,
                    "size": size, "error": "binary file (not readable as text)"}
        truncated = size > MAX_READ_BYTES
        return {
            "op": "read", "backend": self.backend, "path": path,
            "size": size, "truncated": truncated,
            "content": _decode(raw[:MAX_READ_BYTES]),
        }

    async def list(self, path: str = "") -> dict[str, Any]:
        try:
            data = await self._contents(path)
        except httpx.HTTPStatusError as e:
            return {"op": "list", "backend": self.backend, "path": path,
                    "error": f"github contents request failed: {e.response.status_code}"}
        except Exception as e:
            return {"op": "list", "backend": self.backend, "path": path,
                    "error": f"github contents request failed: {e}"}

        if not isinstance(data, list):
            return {"op": "list", "backend": self.backend, "path": path,
                    "error": "path is a file; use op=read"}
        entries = [
            {
                "name": item.get("name", ""),
                "type": "dir" if item.get("type") == "dir" else "file",
                "size": int(item.get("size", 0)),
            }
            for item in data
            if item.get("name") not in IGNORE_DIRS
        ]
        return {"op": "list", "backend": self.backend, "path": path,
                "entries": entries}

    async def search(self, query: str, *, path_prefix: str = "") -> dict[str, Any]:
        if not query:
            return {"op": "search", "backend": self.backend, "query": query,
                    "matches": [], "truncated": False,
                    "error": "search needs a non-empty query"}
        # GitHub code search: q = <query> repo:owner/repo [path:prefix].
        q = f"{query} repo:{self._owner}/{self._repo}"
        if path_prefix:
            q += f" path:{path_prefix.lstrip('/')}"
        url = f"{_GITHUB_API}/search/code"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    url, headers=self._headers(text_match=True),
                    params={"q": q, "per_page": SEARCH_MAX_MATCHES},
                )
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPStatusError as e:
            # 403/422 are common (auth required, rate limited, or query
            # rejected) — surface a clear note rather than raising.
            return {"op": "search", "backend": self.backend, "query": query,
                    "matches": [], "truncated": False,
                    "error": (
                        f"github code search failed: {e.response.status_code} "
                        f"(code search needs auth and only indexes the default branch)"
                    )}
        except Exception as e:
            return {"op": "search", "backend": self.backend, "query": query,
                    "matches": [], "truncated": False,
                    "error": f"github code search failed: {e}"}

        matches: list[dict[str, Any]] = []
        for item in payload.get("items", []):
            path = item.get("path", "")
            fragments = item.get("text_matches", []) or []
            if fragments:
                for frag in fragments:
                    snippet = (frag.get("fragment", "") or "").strip()
                    matches.append({
                        "path": path, "line": None,
                        "text": snippet[:SNIPPET_MAX_CHARS],
                    })
            else:
                matches.append({"path": path, "line": None, "text": ""})
            if len(matches) >= SEARCH_MAX_MATCHES:
                break
        return {"op": "search", "backend": self.backend, "query": query,
                "matches": matches,
                "truncated": payload.get("total_count", 0) > len(matches)}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_source_connector(settings: Any) -> Optional[SourceConnector]:
    """Return the configured source connector, or None when source access
    is not configured. Never raises on misconfiguration — logs and returns
    None so the source_lookup tool reports 'unavailable' cleanly."""
    backend = (getattr(settings, "source_backend", "") or "").strip().lower()

    if backend == "local_fs":
        root = getattr(settings, "source_fs_root", None)
        if not root:
            logger.warning(
                "source_connector.local_fs_unconfigured",
                note="CC_SOURCE_BACKEND=local_fs but CC_SOURCE_FS_ROOT is unset",
            )
            return None
        try:
            return LocalFsSourceConnector(root)
        except Exception as e:
            logger.warning("source_connector.local_fs_init_failed", error=str(e))
            return None

    if backend == "github":
        owner = getattr(settings, "source_github_owner", None)
        repo = getattr(settings, "source_github_repo", None)
        if not owner or not repo:
            logger.warning(
                "source_connector.github_unconfigured",
                note="CC_SOURCE_BACKEND=github but owner/repo are unset",
            )
            return None
        try:
            return GitHubSourceConnector(
                owner=owner, repo=repo,
                ref=getattr(settings, "source_github_ref", "") or "",
                token=getattr(settings, "source_github_token", None),
            )
        except Exception as e:
            logger.warning("source_connector.github_init_failed", error=str(e))
            return None

    return None
