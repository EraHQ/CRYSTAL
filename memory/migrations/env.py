"""Alembic environment.

Wires Alembic to our SQLAlchemy Base metadata and our app settings.

Alembic runs offline (generates SQL) or online (applies to a live DB).
We support both. The database URL comes from crystal_cache.config, not
from alembic.ini, so it respects the CC_DATABASE_URL env var.

Async-safe: we use an async engine and run migrations inside an async
context since the rest of the app is async.
"""
from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import our metadata — Alembic uses this for autogenerate.
from crystal_cache.infrastructure.schema import Base

try:
    # Module-level settings instance (documented public API).
    from crystal_cache.config import settings
except ImportError:  # pragma: no cover - fall back to the factory
    from crystal_cache.config import get_settings

    settings = get_settings()

# Alembic Config object, provides access to values within alembic.ini.
config = context.config

# Override sqlalchemy.url from app settings (respects CC_DATABASE_URL).
config.set_main_option("sqlalchemy.url", settings.database_url)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL, no live connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine, open a connection, run migrations."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (apply to the live DB)."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
