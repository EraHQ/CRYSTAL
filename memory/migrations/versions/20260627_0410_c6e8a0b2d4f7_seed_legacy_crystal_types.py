"""seed legacy crystal_types (customer:legacy, general:legacy)

Revision ID: c6e8a0b2d4f7
Revises: b5d7e9c1f3a6
Create Date: 2026-06-27 04:10:00.000000

The two legacy catch-all crystal_type rows (`customer:legacy`,
`general:legacy`) are foundational registry data — every legacy crystal
points at one of them. They belong in a one-time migration, NOT a per-process
startup hook: MetadataStore.init()'s own docstring says seeding is
"migration-equivalent" work that must not run in every process. The v2
baseline created the crystal_types TABLE but never seeded these rows, so the
runtime bootstrap was quietly inserting them on boot — and under the
docker-compose split (API + worker against one Postgres) both processes raced
the insert, the loser tripping the unique PK with a red first-boot ERROR.
Seeding here, once, in the single migrate step removes the race at its source.

Idempotent via INSERT ... ON CONFLICT (id) DO NOTHING, so it's a safe no-op on
databases the old bootstrap already seeded (existing dev / deployed DBs). ON
CONFLICT is supported by both backends CRYS targets (Postgres, and SQLite
>= 3.24, which ships in the stdlib sqlite3 on the supported Pythons).
Downgrade removes the two rows.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c6e8a0b2d4f7'
down_revision: Union[str, None] = 'b5d7e9c1f3a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# (id, display_name, scope) — capacity_default/autosplit_policy/pair_schema_dsl
# are set to their model defaults explicitly so a seeded row is identical to
# one the application would create.
_LEGACY_TYPES = (
    ("customer:legacy", "Customer (legacy catch-all)", "customer"),
    ("general:legacy", "General (legacy catch-all)", "general"),
)


def upgrade() -> None:
    now = datetime.now(timezone.utc)
    stmt = sa.text(
        """
        INSERT INTO crystal_types
            (id, display_name, scope, capacity_default, autosplit_policy,
             routing_threshold, cleanup_threshold, pair_schema_dsl, created_at)
        VALUES
            (:id, :display_name, :scope, 50, 'split', NULL, NULL, '', :created_at)
        ON CONFLICT (id) DO NOTHING
        """
    )
    bind = op.get_bind()
    for type_id, display_name, scope in _LEGACY_TYPES:
        bind.execute(
            stmt,
            {
                "id": type_id,
                "display_name": display_name,
                "scope": scope,
                "created_at": now,
            },
        )


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text("DELETE FROM crystal_types WHERE id IN (:a, :b)"),
        {"a": "customer:legacy", "b": "general:legacy"},
    )
