"""Git source handler — Gate M slice 3, the registry's first tenant.

Implements the two-method contract for GitHub-hosted repos:

  check:  one branches API call comparing the head SHA against
          last_state["head"]. Unchanged head = one request, None.
          First sync (no state, M-Q4) = the full recursive tree,
          scope-filtered. Moved head = the compare API's exact
          added/modified/removed/renamed file list.
  fetch:  one contents API call -> SourceEnvelope with the D6
          identity: repo://<source_name>/<path> — the watch's source
          name IS the authority, paths are ground truth from the
          repo root. The whole point of M: no pick-depth, no prompt,
          no drift.

Scope (M-Q4=C): include/exclude fnmatch globs from watch config,
defaulting to the supported-extension set + the D5 junk filter.
Credentials (M-Q5=C): resolved per-watch token, else CC_GITHUB_TOKEN
from the environment, else unauthenticated (public repos).

The HTTP seam (`self._get`) is injectable — tests fake it; only the
live worker talks to api.github.com.
"""
from __future__ import annotations

import base64
import fnmatch
import os
from typing import Any, Optional

from .source_handlers import ChangeSet, SourceEnvelope

_API = "https://api.github.com"

# Mirrors the Inspector upload accept list — the lanes ingestion
# actually has today. Widens as gates E-H land their formats.
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md",
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs",
    ".java", ".rb", ".c", ".h", ".cpp", ".cs", ".php", ".swift",
    ".kt", ".sh",
}

# The D5 junk filter, server-side.
_JUNK_SEGMENTS = {"__pycache__", "node_modules", "dist", "build"}


def _is_junk(path: str) -> bool:
    return any(
        seg.startswith(".") or seg in _JUNK_SEGMENTS
        or seg.endswith(".egg-info")
        for seg in path.split("/")
    )


def _within_scope(path: str, config: dict) -> bool:
    if _is_junk(path):
        return False
    include = config.get("include") or []
    exclude = config.get("exclude") or []
    if any(fnmatch.fnmatch(path, g) for g in exclude):
        return False
    if include:
        return any(fnmatch.fnmatch(path, g) for g in include)
    ext = "." + path.rsplit(".", 1)[-1] if "." in path.rsplit("/", 1)[-1] else ""
    return ext.lower() in SUPPORTED_EXTENSIONS


def _repo_slug(config: dict) -> str:
    """'owner/name' from either the bare slug or a github.com URL."""
    repo = (config.get("repo") or "").strip().rstrip("/")
    if repo.endswith(".git"):
        repo = repo[:-4]
    if "github.com" in repo:
        repo = repo.split("github.com", 1)[1].lstrip("/:")
    parts = [p for p in repo.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"git watch config needs owner/name, got {repo!r}")
    return "/".join(parts[-2:])


class GitSourceHandler:
    scheme = "git"

    def __init__(self, http_get=None):
        # Injectable seam: async (url, token) -> parsed JSON dict.
        self._get = http_get or self._default_get

    @staticmethod
    async def _default_get(url: str, token: Optional[str]) -> Any:
        import httpx
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _token(token: Optional[str]) -> Optional[str]:
        # M-Q5 layering: per-watch token wins; env is the self-host
        # fallback; None is fine for public repos.
        return token or os.environ.get("CC_GITHUB_TOKEN") or None

    async def check(self, watch, token: Optional[str]) -> Optional[ChangeSet]:
        tok = self._token(token)
        slug = _repo_slug(watch.config or {})
        branch = (watch.config or {}).get("branch") or "master"

        head_info = await self._get(
            f"{_API}/repos/{slug}/branches/{branch}", tok,
        )
        head = head_info["commit"]["sha"]
        last = (watch.last_state or {}).get("head")
        if last == head:
            return None

        cfg = watch.config or {}
        if last is None:
            # First sync (M-Q4): the full tree, bounded by scope.
            tree = await self._get(
                f"{_API}/repos/{slug}/git/trees/{head}?recursive=1", tok,
            )
            changed = [
                item["path"]
                for item in tree.get("tree", [])
                if item.get("type") == "blob"
                and _within_scope(item["path"], cfg)
            ]
            return ChangeSet(new_state={"head": head}, changed=changed)

        # Moved head: the compare API names exactly what moved.
        cmp = await self._get(
            f"{_API}/repos/{slug}/compare/{last}...{head}", tok,
        )
        changed: list[str] = []
        removed: list[str] = []
        for f in cmp.get("files", []):
            status = f.get("status")
            path = f.get("filename") or ""
            if status in ("added", "modified", "changed"):
                if _within_scope(path, cfg):
                    changed.append(path)
            elif status == "removed":
                if _within_scope(path, cfg):
                    removed.append(path)
            elif status == "renamed":
                prev = f.get("previous_filename") or ""
                if prev and _within_scope(prev, cfg):
                    removed.append(prev)
                if _within_scope(path, cfg):
                    changed.append(path)
        return ChangeSet(
            new_state={"head": head}, changed=changed, removed=removed,
        )

    async def fetch(
        self, watch, path: str, token: Optional[str],
    ) -> SourceEnvelope:
        tok = self._token(token)
        slug = _repo_slug(watch.config or {})
        head = (watch.last_state or {}).get("head")
        ref = f"?ref={head}" if head else ""
        data = await self._get(
            f"{_API}/repos/{slug}/contents/{path}{ref}", tok,
        )
        payload = base64.b64decode(data.get("content") or "")
        import mimetypes
        mime = mimetypes.guess_type(path)[0] or "text/plain"
        return SourceEnvelope(
            payload_bytes=payload,
            mime_type=mime,
            # D6 grammar, ground truth: authority = the watch's source
            # name, path measured from the repo root. Always.
            source_uri=f"repo://{watch.source_name}/{path}",
            label=f"{watch.source_name}/{path}",
            extra={"scheme": "git", "repo": slug, "path": path},
        )
