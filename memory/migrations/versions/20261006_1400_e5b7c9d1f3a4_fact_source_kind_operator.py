"""facts.source_kind 'operator_response' -> 'operator_stated' (2026-10-06).

v110's respond feature wrote the operator's response as a fact with
source_kind='operator_response', a value outside the Fact model's closed
set. Loading those rows raised a validation error, which broke the
assumptions worker's cycle and the idle-phase gap discovery for every
bank holding one. The write now uses 'operator_stated'; this corrects
the rows already written. Data only; no schema change.
"""
from alembic import op

revision = "e5b7c9d1f3a4"
down_revision = "d4a6b8c0e2f3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "UPDATE facts SET source_kind = 'operator_stated' "
        "WHERE source_kind = 'operator_response'"
    )


def downgrade() -> None:
    # The old value was never valid; nothing to restore.
    pass
