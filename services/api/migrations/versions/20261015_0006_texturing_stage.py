"""texturing run stage; training is retired but its historical rows stay valid

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-15
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Run stages before and after the change. ``training`` is kept in the new list only so
# that rows written by the splat pipeline still satisfy both constraints.
_OLD_RUN = (
    "'validating', 'preparing', 'camera_reconstruction', 'training', 'normalizing', "
    "'geometry_proposal', 'packaging', 'scene_validation'"
)
_NEW_RUN = (
    "'validating', 'preparing', 'camera_reconstruction', 'normalizing', "
    "'geometry_proposal', 'texturing', 'packaging', 'scene_validation', 'training'"
)
_MAINTENANCE = "'upload_inspection', 'project_cleanup', 'upload_cleanup', 'artifact_cleanup'"


def upgrade() -> None:
    op.drop_constraint(op.f("ck_jobs_stage"), "jobs", type_="check")
    op.create_check_constraint(
        op.f("ck_jobs_stage"), "jobs", f"stage IN ({_NEW_RUN}, {_MAINTENANCE})"
    )
    op.drop_constraint(op.f("ck_jobs_run_stage_has_run"), "jobs", type_="check")
    op.create_check_constraint(
        op.f("ck_jobs_run_stage_has_run"),
        "jobs",
        f"NOT (stage IN ({_NEW_RUN})) OR run_id IS NOT NULL",
    )


def downgrade() -> None:
    # The 0005 lists have no texturing; its jobs cannot be kept. job_events cascade.
    op.execute("DELETE FROM jobs WHERE stage = 'texturing'")
    op.drop_constraint(op.f("ck_jobs_run_stage_has_run"), "jobs", type_="check")
    op.create_check_constraint(
        op.f("ck_jobs_run_stage_has_run"),
        "jobs",
        f"NOT (stage IN ({_OLD_RUN})) OR run_id IS NOT NULL",
    )
    op.drop_constraint(op.f("ck_jobs_stage"), "jobs", type_="check")
    op.create_check_constraint(
        op.f("ck_jobs_stage"), "jobs", f"stage IN ({_OLD_RUN}, {_MAINTENANCE})"
    )
