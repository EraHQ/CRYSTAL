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
Credentials (M-Q5=C, narrowed by Q53=A on 2026-10-08): the per-watch
token only. There is no environment fallback and no unauthenticated
mode: a watch without a token is reported, not synced.

The HTTP seam (`self._get`) is injectable — tests fake it; only the
live worker talks to api.github.com.
"""
from __future__ import annotations

import base64
import fnmatch
import re
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


# Lockdown PR-2 (Q53=A, AUDIT_LAUNCH_VERIFY B1-4): repo and branch are
# interpolated into GitHub URLs. Each segment is a closed character set
# with no `.`/`..` segment, so nothing in a watch config can leave
# /repos/{owner}/{name}/. The character sets are URL-safe, so validation
# is the encoding guarantee for these two; file paths (which come from
# GitHub's own tree listing and may carry spaces or `#`) are quoted.
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]{1,255}$")


def _repo_slug(config: dict) -> str:
    """'owner/name' from either the bare slug or a github.com URL,
    validated segment by segment."""
    repo = (config.get("repo") or "").strip().rstrip("/")
    if repo.endswith(".git"):
        repo = repo[:-4]
    if "github.com" in repo:
        repo = repo.split("github.com", 1)[1].lstrip("/:")
    parts = [p for p in repo.split("/") if p]
    if len(parts) < 2:
        raise ValueError("git watch config needs owner/name")
    owner, name = parts[-2:]
    for seg in (owner, name):
        if not _SEGMENT_RE.match(seg) or seg in (".", ".."):
            raise ValueError("git watch repo must be owner/name using letters, digits, '.', '_' and '-'")
    return f"{owner}/{name}"


def _branch_name(config: dict) -> str:
    branch = (config.get("branch") or "master").strip()
    if (not _BRANCH_RE.match(branch) or branch.startswith("/")
            or any(seg in ("", ".", "..") for seg in branch.split("/"))):
        raise ValueError("git watch branch must use letters, digits, '.', '_', '-' and '/' with no empty or '..' segment")
    return branch


def validate_git_config(config: dict) -> dict:
    """The watch-create route calls this before storing a git config:
    repo and branch as above, include/exclude at most 50 globs of at
    most 256 characters each. Returns the normalized config."""
    slug = _repo_slug(config)
    branch = _branch_name(config)
    out = {"repo": slug, "branch": branch}
    for key in ("include", "exclude"):
        globs = config.get(key) or []
        if not isinstance(globs, list) or len(globs) > 50:
            raise ValueError(f"git watch {key} must be a list of at most 50 patterns")
        cleaned = []
        for g in globs:
            if not isinstance(g, str) or not g.strip() or len(g) > 256:
                raise ValueError(f"git watch {key} patterns must be non-empty strings of at most 256 characters")
            cleaned.append(g.strip())
        if cleaned:
            out[key] = cleaned
    return out


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
    def _token(token: Optional[str]) -> str:
        # Lockdown PR-2 (Q53=A): the per-watch token is the ONLY token.
        # The CC_GITHUB_TOKEN fallback is gone: it was set on the
        # production worker, so any tenant could register a watch on a
        # repository only the platform's token could read and have it
        # ingested into their own bank.
        if not token:
            raise ValueError(
                "This watch has no access token. Open the watch and add a "
                "GitHub token for the repository."
            )
        return token

    async def check(self, watch, token: Optional[str]) -> Optional[ChangeSet]:
        tok = self._token(token)
        slug = _repo_slug(watch.config or {})
        branch = _branch_name(watch.config or {})

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
        from urllib.parse import quote

        tok = self._token(token)
        slug = _repo_slug(watch.config or {})
        head = (watch.last_state or {}).get("head")
        if head is not None and not re.fullmatch(r"[0-9a-fA-F]{1,64}", str(head)):
            raise ValueError("git watch state carries an invalid head")
        ref = f"?ref={head}" if head else ""
        if any(seg in ("", ".", "..") for seg in path.split("/")):
            raise ValueError("git file path must not contain empty, '.' or '..' segments")
        data = await self._get(
            f"{_API}/repos/{slug}/contents/{quote(path, safe='/')}{ref}", tok,
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
