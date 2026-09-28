"""Drive watcher knob pin (Q1=B, 2026-09-28).

The REAL off switch behind the launch UI-hide: with
CC_DRIVE_WATCHER_ENABLED unset (default false) the acquisition routes
refuse with a clear 403 and the sync worker leaves scheme=gdrive
watches dormant; flipping the knob resumes both without touching the
recorded watches. Connection list + disconnect deliberately bypass the
gate so existing grants can always be removed.

R14 note: verified by pytest; describes expected behavior.
"""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from crystal_cache import config as config_mod
from crystal_cache.config import Settings
from crystal_cache.endpoints import drive as drive_mod
from crystal_cache.workers import source_sync


def _settings(monkeypatch, enabled: bool) -> None:
    s = Settings(drive_watcher_enabled=enabled)
    monkeypatch.setattr(config_mod, "get_settings", lambda: s)


def test_gate_refuses_by_default_and_opens_on_the_knob(monkeypatch):
    _settings(monkeypatch, enabled=False)
    with pytest.raises(HTTPException) as exc:
        drive_mod._require_drive_enabled()
    assert exc.value.status_code == 403
    assert "disabled" in str(exc.value.detail)

    _settings(monkeypatch, enabled=True)
    drive_mod._require_drive_enabled()  # no raise


class _StubStore:
    """list_source_watches_due is real; anything else unexpected the
    loop touches becomes an async no-op, so the pin doesn't crumble on
    incidental bookkeeping calls."""

    def __init__(self, watches):
        self._watches = watches

    async def list_source_watches_due(self, now):
        return self._watches

    def __getattr__(self, name):
        async def _noop(*a, **k):
            return None
        return _noop


@pytest.mark.asyncio
async def test_worker_leaves_gdrive_dormant_until_the_knob_flips(monkeypatch):
    from crystal_cache.workers import budget as budget_mod

    async def _not_exhausted(*a, **k):
        return False

    monkeypatch.setattr(budget_mod, "llm_budget_exhausted", _not_exhausted)
    monkeypatch.setattr(
        budget_mod, "customer_llm_budget_exhausted", _not_exhausted
    )

    synced = []

    async def _record(*a, **kwargs):
        for x in list(a) + list(kwargs.values()):
            s = getattr(x, "scheme", None)
            if isinstance(s, str):
                synced.append(s)
                return
        synced.append("<no-watch-arg>")

    monkeypatch.setattr(source_sync, "sync_one_watch", _record)
    store = _StubStore([SimpleNamespace(scheme="gdrive", customer_id="c1")])

    _settings(monkeypatch, enabled=False)
    await source_sync._sync_due_watches(store, None, None, None, None)
    assert synced == []  # dormant

    _settings(monkeypatch, enabled=True)
    await source_sync._sync_due_watches(store, None, None, None, None)
    assert synced == ["gdrive"]  # resumed, watch untouched in between
