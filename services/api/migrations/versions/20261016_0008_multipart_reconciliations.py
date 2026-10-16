"""multipart reconciliations: aborted multipart uploads the worker keeps checking

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "multipart_reconciliations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("object_key", sa.Text(), nullable=False),
        sa.Column("upload_id", sa.String(length=1024), nullable=False),
        sa.Column("last_aborted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("aborts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("clean_checks", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "next_check_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_multipart_reconciliations")),
        sa.UniqueConstraint("object_key", "upload_id", name="uq_multipart_reconciliations_upload"),
    )
    op.create_index(
        "ix_multipart_reconciliations_due", "multipart_reconciliations", ["next_check_at"]
    )


def downgrade() -> None:
    # Once cleanup has cleared an upload's multipart ID or scrubbed a deleted project, an
    # entry here is the only record of it, and late parts it still gets would stay stored
    # for good: refuse rather than lose them.
    rows = (
        op.get_bind()
        .execute(sa.text("SELECT count(*) FROM multipart_reconciliations"))
        .scalar_one()
    )
    if rows:
        raise RuntimeError(
            f"cannot downgrade: {rows} multipart upload(s) are still reconciled from "
            "multipart_reconciliations, the only record of their upload IDs; abort each "
            "one, confirm ListParts finds nothing, and delete its row first"
        )
    op.drop_index("ix_multipart_reconciliations_due", table_name="multipart_reconciliations")
    op.drop_table("multipart_reconciliations")
