"""upload_cleanup maintenance stage

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-14
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD = (
    "'validating', 'preparing', 'camera_reconstruction', 'training', 'normalizing', "
    "'geometry_proposal', 'packaging', 'scene_validation', 'upload_inspection', 'project_cleanup'"
)
_NEW = _OLD + ", 'upload_cleanup'"


def upgrade() -> None:
    op.drop_constraint(op.f("ck_jobs_stage"), "jobs", type_="check")
    op.create_check_constraint(op.f("ck_jobs_stage"), "jobs", f"stage IN ({_NEW})")
    op.create_check_constraint(
        op.f("ck_jobs_upload_cleanup_has_upload"),
        "jobs",
        "stage <> 'upload_cleanup' OR upload_id IS NOT NULL",
    )


def downgrade() -> None:
    op.execute("DELETE FROM jobs WHERE stage = 'upload_cleanup'")
    op.drop_constraint(op.f("ck_jobs_upload_cleanup_has_upload"), "jobs", type_="check")
    op.drop_constraint(op.f("ck_jobs_stage"), "jobs", type_="check")
    op.create_check_constraint(op.f("ck_jobs_stage"), "jobs", f"stage IN ({_OLD})")
