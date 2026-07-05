"""Tests for the read-only source connector (VS-D5).

local_fs is exercised for real against a tmp_path tree (the security
properties — traversal confinement, ignore-dir pruning, binary skip —
matter most here). The factory is checked for off-by-default and
backend selection. The GitHub backend's response parsing is covered
with a mocked client so no network call is made; its live behavior is
exercised the first time CC_SOURCE_BACKEND=github points at a repo.
"""
from __future__ import annotations

import base64
from typing import Any

import pytest

from crystal_cache.infrastructure import source_connector as sc
from crystal_cache.infrastructure.source_connector import (
    GitHubSourceConnector,
    LocalFsSourceConnector,
    build_source_connector,
)


# ---------------------------------------------------------------------------
# local_fs — real filesystem against tmp_path
# ---------------------------------------------------------------------------

def _make_tree(root) -> None:
    (root / "a.py").write_text(
        "import os\n\ndef generate_sparse_key(text):\n    return text.upper()\n"
    )
    (root / "sub").mkdir()
    (root / "sub" / "b.txt").write_text("hello\ngenerate_sparse_key appears here too\n")
    (root / ".git").mkdir()
    (root / ".git" / "secret").write_text("generate_sparse_key must NOT be found here\n")
    (root / "bin.dat").write_bytes(b"\x00\x01generate_sparse_key\x00")


@pytest.mark.asyncio
async def test_local_fs_read_returns_content(tmp_path):
    _make_tree(tmp_path)
    c = LocalFsSourceConnector(str(tmp_path))
    r = await c.read("a.py")
    assert "def generate_sparse_key" in r["content"]
    assert r["size"] > 0
    assert r["truncated"] is False


@pytest.mark.asyncio
async def test_local_fs_read_missing_and_binary(tmp_path):
    _make_tree(tmp_path)
    c = LocalFsSourceConnector(str(tmp_path))
    assert "error" in await c.read("nope.py")
    assert (await c.read("bin.dat"))["error"].startswith("binary")


@pytest.mark.asyncio
async def test_local_fs_traversal_is_blocked(tmp_path):
    _make_tree(tmp_path)
    # write a secret OUTSIDE the root
    (tmp_path.parent / "outside.txt").write_text("TOPSECRET\n")
    c = LocalFsSourceConnector(str(tmp_path))
    for escape in ("../outside.txt", "../../etc/passwd", "/etc/passwd"):
        r = await c.read(escape)
        assert "error" in r, f"{escape} should be denied, got {r}"


@pytest.mark.asyncio
async def test_local_fs_list_excludes_ignore_dirs(tmp_path):
    _make_tree(tmp_path)
    c = LocalFsSourceConnector(str(tmp_path))
    r = await c.list("")
    by_name = {e["name"]: e["type"] for e in r["entries"]}
    assert by_name.get("a.py") == "file"
    assert by_name.get("sub") == "dir"
    assert ".git" not in by_name


@pytest.mark.asyncio
async def test_local_fs_search_finds_symbol_skips_ignored_and_binary(tmp_path):
    _make_tree(tmp_path)
    c = LocalFsSourceConnector(str(tmp_path))
    r = await c.search("generate_sparse_key")
    paths = {m["path"] for m in r["matches"]}
    assert "a.py" in paths
    assert "sub/b.txt" in paths
    assert not any(".git" in p for p in paths)   # ignore dir pruned
    assert "bin.dat" not in paths                 # binary skipped
    assert all(isinstance(m["line"], int) for m in r["matches"])


@pytest.mark.asyncio
async def test_local_fs_empty_search_errors(tmp_path):
    c = LocalFsSourceConnector(str(tmp_path))
    assert "error" in await c.search("")


def test_local_fs_requires_root():
    with pytest.raises(ValueError):
        LocalFsSourceConnector("")


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

class _Cfg:
    """Minimal settings stand-in (factory uses getattr with defaults)."""
    source_backend = ""
    source_fs_root = None
    source_github_owner = None
    source_github_repo = None
    source_github_ref = ""
    source_github_token = None


def test_factory_off_by_default():
    assert build_source_connector(_Cfg()) is None


def test_factory_builds_local_fs(tmp_path):
    cfg = _Cfg()
    cfg.source_backend = "local_fs"
    cfg.source_fs_root = str(tmp_path)
    conn = build_source_connector(cfg)
    assert conn is not None and conn.backend == "local_fs"


def test_factory_local_fs_without_root_is_none():
    cfg = _Cfg()
    cfg.source_backend = "local_fs"
    cfg.source_fs_root = None
    assert build_source_connector(cfg) is None


def test_factory_builds_github():
    cfg = _Cfg()
    cfg.source_backend = "github"
    cfg.source_github_owner = "EraHQ"
    cfg.source_github_repo = "crystal-cache"
    conn = build_source_connector(cfg)
    assert conn is not None and conn.backend == "github"


def test_factory_github_without_repo_is_none():
    cfg = _Cfg()
    cfg.source_backend = "github"
    cfg.source_github_owner = "EraHQ"
    cfg.source_github_repo = None
    assert build_source_connector(cfg) is None


# ---------------------------------------------------------------------------
# github — response parsing (mocked, no network)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_github_read_decodes_base64():
    conn = GitHubSourceConnector("o", "r")

    async def fake_contents(path: str) -> Any:
        return {
            "type": "file", "size": 11,
            "content": base64.b64encode(b"hello world").decode(),
        }

    conn._contents = fake_contents  # type: ignore[assignment]
    r = await conn.read("README.md")
    assert r["content"] == "hello world"
    assert r["backend"] == "github"


@pytest.mark.asyncio
async def test_github_read_on_directory_errors():
    conn = GitHubSourceConnector("o", "r")

    async def fake_contents(path: str) -> Any:
        return [{"name": "a.py", "type": "file", "size": 1}]

    conn._contents = fake_contents  # type: ignore[assignment]
    r = await conn.read("src")
    assert "error" in r and "directory" in r["error"]


@pytest.mark.asyncio
async def test_github_list_maps_entries():
    conn = GitHubSourceConnector("o", "r")

    async def fake_contents(path: str) -> Any:
        return [
            {"name": "a.py", "type": "file", "size": 10},
            {"name": "sub", "type": "dir", "size": 0},
            {"name": ".git", "type": "dir", "size": 0},
        ]

    conn._contents = fake_contents  # type: ignore[assignment]
    r = await conn.list("")
    by_name = {e["name"]: e["type"] for e in r["entries"]}
    assert by_name == {"a.py": "file", "sub": "dir"}  # .git pruned


@pytest.mark.asyncio
async def test_github_search_parses_matches(monkeypatch):
    payload = {
        "total_count": 1,
        "items": [
            {"path": "src/x.py", "text_matches": [{"fragment": "def generate_sparse_key():"}]},
        ],
    }

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return payload

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k): return _Resp()

    monkeypatch.setattr(sc.httpx, "AsyncClient", _Client)
    conn = GitHubSourceConnector("o", "r")
    r = await conn.search("generate_sparse_key")
    assert r["matches"][0]["path"] == "src/x.py"
    assert "generate_sparse_key" in r["matches"][0]["text"]


@pytest.mark.asyncio
async def test_github_search_empty_query_errors():
    conn = GitHubSourceConnector("o", "r")
    assert "error" in await conn.search("")
