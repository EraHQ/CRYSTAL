"""Enrichment deadline + render fallback (2026-07-11, ratified Q2A/Q3A).

Rematch #5 forensics: enrichment had no total deadline — 15s per
redirect hop, sequential pages, failures not consuming max_pages slots —
stacking to 200-250s per search. And static fetch reads the
server-rendered shell, so GitHub-class JS pages arrived as "Uh oh!
There was an error while loading" (verbatim in rematch #4 outputs).
Pins: the wall-clock deadline stops the pass; thin/SPA-marker static
extractions trigger the headless render fallback; render failure or
unavailability keeps the static result (enrichment never gets worse).

R14: verified by pytest. Real chromium rendering is smoke-tested on
deploy (the build container cannot download browser binaries).
"""
from __future__ import annotations

import pytest

from crystal_cache.search.fetch import (
    _looks_unrendered,
    fill_missing_content,
)
from crystal_cache.search.render import _is_blocked_host


# ---------------------------------------------------------------------------
# Unrendered detection
# ---------------------------------------------------------------------------

def test_spa_markers_and_thin_content_look_unrendered():
    assert _looks_unrendered("")
    assert _looks_unrendered("short shell")
    assert _looks_unrendered(
        "Uh oh! There was an error while loading. Please reload this page. "
        * 20
    )
    assert not _looks_unrendered("real extracted article content " * 40)


# ---------------------------------------------------------------------------
# Deadline
# ---------------------------------------------------------------------------

def _payload(n: int) -> dict:
    return {"results": [
        {"title": f"t{i}", "url": f"https://example.test/{i}",
         "snippet": "s", "content": None}
        for i in range(n)
    ]}


def test_deadline_stops_the_enrichment_pass(monkeypatch):
    from crystal_cache.search import fetch as fetch_mod

    clock = {"now": 0.0}

    def _slow_fetch(clk, url):
        clk["now"] += 30.0  # each page "costs" 30s
        return {"url": url, "title": "t",
                "content": "real extracted article content " * 40}

    monkeypatch.setattr(fetch_mod, "fetch_and_extract",
                        lambda url, **kw: _slow_fetch(clock, url))
    import time
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    payload = fill_missing_content(
        _payload(5), max_pages=5, content_cap=10_000,
        http_client=object(), deadline_seconds=45.0,
    )
    enriched = [r for r in payload["results"] if r["content"]]
    # 30s + 30s = 60s > 45s deadline before page 3 — exactly 2 enriched.
    assert len(enriched) == 2


def test_deadline_zero_means_unbounded(monkeypatch):
    from crystal_cache.search import fetch as fetch_mod
    monkeypatch.setattr(
        fetch_mod, "fetch_and_extract",
        lambda url, **kw: {"url": url, "title": "t",
                           "content": "real extracted article content " * 40})
    payload = fill_missing_content(
        _payload(3), max_pages=3, content_cap=10_000,
        http_client=object(), deadline_seconds=0.0,
    )
    assert all(r["content"] for r in payload["results"])


# ---------------------------------------------------------------------------
# Render fallback
# ---------------------------------------------------------------------------

def test_spa_shell_triggers_render_and_uses_rendered_content(monkeypatch):
    from crystal_cache.search import fetch as fetch_mod
    from crystal_cache.search import render as render_mod

    monkeypatch.setattr(
        fetch_mod, "fetch_and_extract",
        lambda url, **kw: {"url": url, "title": "t",
                           "content": "error while loading " * 30})
    rendered_calls = []

    def fake_render(url, *, timeout_seconds, resolver=None):
        rendered_calls.append(url)
        return {"url": url, "title": "t",
                "content": "FULL RENDERED RELEASE DATA " * 60}

    monkeypatch.setattr(render_mod, "render_and_extract", fake_render)
    payload = fill_missing_content(
        _payload(1), max_pages=1, content_cap=100_000,
        http_client=object(), render_enabled=True,
    )
    assert rendered_calls == ["https://example.test/0"]
    r = payload["results"][0]
    assert "FULL RENDERED RELEASE DATA" in r["content"]
    assert r["rendered"] is True


def test_render_failure_keeps_static_result(monkeypatch):
    from crystal_cache.search import fetch as fetch_mod
    from crystal_cache.search import render as render_mod

    monkeypatch.setattr(
        fetch_mod, "fetch_and_extract",
        lambda url, **kw: {"url": url, "title": "t",
                           "content": "error while loading " * 30})
    monkeypatch.setattr(render_mod, "render_and_extract",
                        lambda url, **kw: None)
    payload = fill_missing_content(
        _payload(1), max_pages=1, content_cap=100_000,
        http_client=object(), render_enabled=True,
    )
    assert "error while loading" in payload["results"][0]["content"]


def test_good_static_content_never_renders(monkeypatch):
    from crystal_cache.search import fetch as fetch_mod
    from crystal_cache.search import render as render_mod

    monkeypatch.setattr(
        fetch_mod, "fetch_and_extract",
        lambda url, **kw: {"url": url, "title": "t",
                           "content": "real extracted article content " * 40})

    def _boom(url, **kw):
        raise AssertionError("render must not be called for good static")

    monkeypatch.setattr(render_mod, "render_and_extract", _boom)
    fill_missing_content(
        _payload(1), max_pages=1, content_cap=100_000,
        http_client=object(), render_enabled=True,
    )


def test_render_disabled_never_renders(monkeypatch):
    from crystal_cache.search import fetch as fetch_mod
    from crystal_cache.search import render as render_mod

    monkeypatch.setattr(
        fetch_mod, "fetch_and_extract",
        lambda url, **kw: {"url": url, "title": "t", "content": "thin"})

    def _boom(url, **kw):
        raise AssertionError("render disabled")

    monkeypatch.setattr(render_mod, "render_and_extract", _boom)
    fill_missing_content(
        _payload(1), max_pages=1, content_cap=100_000,
        http_client=object(), render_enabled=False,
    )


# ---------------------------------------------------------------------------
# Render SSRF trims
# ---------------------------------------------------------------------------

def test_blocked_hosts_for_browser_subrequests():
    assert _is_blocked_host("metadata.google.internal")
    assert _is_blocked_host("169.254.169.254")
    assert _is_blocked_host("127.0.0.1")
    assert _is_blocked_host("10.1.2.3")
    assert _is_blocked_host("localhost")
    assert _is_blocked_host("")
    assert not _is_blocked_host("github.com")
    assert not _is_blocked_host("93.184.216.34")  # public literal
