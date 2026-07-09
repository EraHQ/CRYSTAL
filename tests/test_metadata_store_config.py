"""Resilience pair (2026-07-09): engine pool config by backend."""
import os

import pytest

from crystal_cache.infrastructure.metadata_store import MetadataStore
from crystal_cache.config import Settings


def _settings(url: str) -> Settings:
    return Settings(database_url=url)


def test_postgres_engine_gets_pre_ping_and_recycle():
    store = MetadataStore(
        settings_override=_settings(
            "postgresql+asyncpg://u:p@localhost:1/db"))
    assert store.engine.pool._pre_ping is True
    assert store.engine.pool._recycle == 1800


def test_sqlite_engine_unchanged():
    store = MetadataStore(
        settings_override=_settings("sqlite+aiosqlite:///:memory:"))
    # SQLite keeps SQLAlchemy defaults — no pre-ping, no recycle.
    assert getattr(store.engine.pool, "_pre_ping", False) is False
    assert store.engine.pool._recycle == -1
