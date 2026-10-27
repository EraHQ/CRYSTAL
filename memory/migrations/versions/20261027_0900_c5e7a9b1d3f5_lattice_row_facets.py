"""Lattice rows answer from indexes (2026-10-26; Q133 A).

A lattice now holds up to 250,000 rows (was 50,000). Measured on Postgres
16 at 250,000 rows: an entity filter read every row's JSON (288 ms local,
3 ms with this), a search scanned every text column (3.7 s). Two derived
columns: facets (data with text lowercased; JSONB with a GIN
jsonb_path_ops index for containment) and search_text (every text value,
lowercased; GIN full-text index). Backfilled here; written by
write_lattice_rows from now on. Plus an index on (lattice_id, created_at,
row_key): the default browse order and the CSV export's keyset walk.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "c5e7a9b1d3f5"
down_revision = "e9b1d3f5a7c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    pg = bind.dialect.name == "postgresql"
    op.add_column("lattice_rows", sa.Column("facets", postgresql.JSONB() if pg else sa.JSON(), nullable=True))
    op.add_column("lattice_rows", sa.Column("search_text", sa.Text(), nullable=True))
    # The browse order (created_at, row_key) and the export's keyset walk.
    op.create_index("ix_lattice_rows_lattice_order", "lattice_rows", ["lattice_id", "created_at", "row_key"])
    if pg:
        op.execute("""
            UPDATE lattice_rows SET
              facets = (SELECT coalesce(jsonb_object_agg(key, CASE WHEN jsonb_typeof(value) = 'string'
                                                                  THEN to_jsonb(lower(value #>> '{}')) ELSE value END),
                                        '{}'::jsonb)
                        FROM jsonb_each(data::jsonb)),
              search_text = (SELECT lower(string_agg(value #>> '{}', ' '))
                             FROM jsonb_each(data::jsonb) WHERE jsonb_typeof(value) = 'string')
        """)
        op.execute("CREATE INDEX ix_lattice_rows_facets ON lattice_rows USING gin (facets jsonb_path_ops)")
        op.execute("CREATE INDEX ix_lattice_rows_search ON lattice_rows USING gin (to_tsvector('simple', coalesce(search_text, '')))")
    else:
        import json
        rows = bind.execute(sa.text("SELECT id, data FROM lattice_rows")).fetchall()
        for rid, data in rows:
            d = json.loads(data) if isinstance(data, str) else (data or {})
            facets = {k: (v.lower() if isinstance(v, str) else v) for k, v in d.items()}
            text = " ".join(v for v in d.values() if isinstance(v, str)).lower() or None
            bind.execute(sa.text("UPDATE lattice_rows SET facets = :f, search_text = :t WHERE id = :i"),
                         {"f": json.dumps(facets), "t": text, "i": rid})


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_lattice_rows_search")
        op.execute("DROP INDEX IF EXISTS ix_lattice_rows_facets")
    op.drop_index("ix_lattice_rows_lattice_order", table_name="lattice_rows")
    op.drop_column("lattice_rows", "search_text")
    op.drop_column("lattice_rows", "facets")
