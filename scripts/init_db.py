"""Initialize the dev database schema (create all tables).

v2 builds its schema via MetadataStore.init() (SQLAlchemy create_all).
The FastAPI app does NOT auto-create tables on startup — production uses
Alembic migrations, so the app assumes the schema already exists. For
local dev against a fresh SQLite DB, run this once to create every table.

Importing crystal_cache.app first forces every ORM model to register on
Base.metadata (through the router/endpoint imports), so create_all below
sees the full table set — documents, drive watches, cognition tasks,
reasoning traces, critiques, and the core tables alike.

Usage (from the v2 root, with the venv active):
    python scripts/init_db.py

Re-running is safe: create_all only creates tables that don't already
exist. It does NOT seed the crystal_type registry rows — the app's
startup hook seeds those (customer:legacy, general:legacy) on the next
boot once the tables exist.
"""
from __future__ import annotations

import asyncio

# Imported for its registration side effects: pulling in the app wires up
# every endpoint/router, which in turn imports every ORM model so they all
# register on Base.metadata before create_all runs.
import crystal_cache.app  # noqa: F401
from crystal_cache.infrastructure import MetadataStore


async def main() -> None:
    store = MetadataStore()
    await store.init()
    await store.dispose()
    print("Database initialized: all tables created (or already present).")


if __name__ == "__main__":
    asyncio.run(main())
