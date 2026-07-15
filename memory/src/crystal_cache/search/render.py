"""Headless render fallback for page enrichment (2026-07-11, ratified Q2A).

Static fetch + extraction (search/fetch.py) reads the server-rendered
shell; GitHub-class pages assemble the payload — releases, contributor
panes, activity — with JavaScript, so the shell arrives as "Uh oh!
There was an error while loading. Please reload this page" (verbatim
from rematch #4's step outputs). This module re-fetches such pages
through headless Chromium and extracts the RENDERED DOM: "seeing the
page like a person browsing", without interaction. Interactive browsing
(click, paginate, decide) needs an agent driving it and rides the
workers-as-CRYS slice.

Availability is a capability, not a requirement: playwright + chromium
ship in the container image; environments without them (dev boxes, the
build container) report render_available() == False and the enrichment
pipeline silently keeps its static results. CC_WEB_RENDER_ENABLED=false
is the operator kill switch.

SSRF posture (documented honestly):
  * The TARGET URL is pre-validated with the same assert_public_url
    guard as static fetch — private/link-local/metadata targets never
    reach the browser.
  * Subresource + navigation requests are intercepted and aborted when
    the hostname is a private/loopback/link-local IP literal or a known
    metadata hostname. Chromium performs its own DNS, so a hostname
    that RESOLVES private (DNS rebinding) is a residual risk the
    interceptor cannot fully close; the blast radius is limited to
    triggering internal GETs — response bodies stay inside the browser
    (no CORS) and rendered extraction only returns the page's own DOM.
"""
from __future__ import annotations

import ipaddress
from typing import Optional
from urllib.parse import urlparse

import structlog

logger = structlog.get_logger(__name__)

_BLOCKED_HOSTNAMES = frozenset({
    "metadata.google.internal",
    "metadata",
    "localhost",
})

_render_unavailable_logged = False


def render_available() -> bool:
    """True when playwright is importable (chromium presence is checked
    at launch; a missing browser degrades to unavailable at call time)."""
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _is_blocked_host(hostname: str) -> bool:
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return True
    if host in _BLOCKED_HOSTNAMES:
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False  # non-literal hostname: allowed (residual rebinding risk documented)
    return (
        addr.is_private or addr.is_loopback or addr.is_link_local
        or addr.is_reserved or addr.is_multicast or addr.is_unspecified
    )


def render_and_extract(
    url: str,
    *,
    timeout_seconds: float = 20.0,
    resolver=None,
) -> Optional[dict]:
    """Render one guarded URL in headless Chromium and extract its main
    text from the rendered DOM.

    Returns {"url", "title", "content"} like fetch_and_extract, or None
    when rendering is unavailable or fails — callers keep their static
    result; enrichment never gets WORSE because a render failed.
    """
    global _render_unavailable_logged

    from .fetch import assert_public_url, extract_main_text

    # Same front gate as static fetch: the target itself must be public.
    assert_public_url(url, resolver=resolver)

    try:
        from playwright.sync_api import sync_playwright
    except Exception:  # noqa: BLE001
        if not _render_unavailable_logged:
            logger.info("web_render.unavailable",
                        note="playwright not installed; render fallback off")
            _render_unavailable_logged = True
        return None

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    java_script_enabled=True,
                    bypass_csp=False,
                )

                def _route(route):
                    host = urlparse(route.request.url).hostname or ""
                    if _is_blocked_host(host):
                        route.abort()
                    else:
                        route.continue_()

                context.route("**/*", _route)
                page = context.new_page()
                page.set_default_timeout(timeout_seconds * 1000)
                # 2026-07-13 (rematch #8 forensics): "networkidle" never
                # fires on GitHub-class pages (persistent connections),
                # so goto raised Timeout and we DISCARDED a fully
                # rendered DOM — seven for seven in the logs. Wait for
                # "load" + a short JS settle instead, and SALVAGE the
                # DOM on timeout: whatever rendered by the deadline is
                # the result, not an error.
                salvaged = False
                try:
                    page.goto(url, wait_until="load",
                              timeout=timeout_seconds * 1000)
                except Exception as nav_err:  # noqa: BLE001
                    salvaged = True
                    logger.info("web_render.nav_timeout_salvaging",
                                url=url, error=str(nav_err)[:200])
                settle_ms = min(2500, int(timeout_seconds * 1000 / 4))
                try:
                    page.wait_for_timeout(settle_ms)
                except Exception:  # noqa: BLE001
                    pass
                html = page.content()
                final_url = page.url
            finally:
                browser.close()
    except Exception as e:  # noqa: BLE001 — a failed render never kills a search
        logger.warning("web_render.failed", url=url, error=str(e))
        return None

    title, text = extract_main_text(html)
    if not (text or "").strip():
        return None
    # "salvaged" (2026-07-14, Q1C): the DOM survived a navigation
    # timeout — content is real but the page never finished loading;
    # downstream stamps findings so the Inspector can badge them.
    return {"url": final_url, "title": title, "content": text,
            "salvaged": salvaged}
