"""artifact_cleanup maintenance stage

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-15
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD = (
    "'validating', 'preparing', 'camera_reconstruction', 'training', 'normalizing', "
    "'geometry_proposal', 'packaging', 'scene_validation', 'upload_inspection', 'project_cleanup', "
    "'upload_cleanup'"
)
_NEW = _OLD + ", 'artifact_cleanup'"


def upgrade() -> None:
    op.drop_constraint(op.f("ck_jobs_stage"), "jobs", type_="check")
    op.create_check_constraint(op.f("ck_jobs_stage"), "jobs", f"stage IN ({_NEW})")


def downgrade() -> None:
    # job_events cascade with their jobs.
    op.execute("DELETE FROM jobs WHERE stage = 'artifact_cleanup'")
    op.drop_constraint(op.f("ck_jobs_stage"), "jobs", type_="check")
    op.create_check_constraint(op.f("ck_jobs_stage"), "jobs", f"stage IN ({_OLD})")
