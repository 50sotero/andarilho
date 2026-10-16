"""texturing run stage; training is retired but its historical rows stay valid

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
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


def _refuse_active(stage: str) -> None:
    """Stop the downgrade while any run, or any texturing job, can still run.

    A run is pinned to the recipe it was submitted with, and the recipes published since 0006 reach
    texturing even when their texturing job is not queued yet. After the downgrade a
    current worker could not insert that job, and a rolled-back worker cannot run the
    pinned recipe, so either way the run fails. The database does not record which
    stages a recipe has, so every queued or running run blocks the downgrade. Layout
    saves queue texturing on finished runs, so active jobs of the stage are counted
    too: deleting one would strand its revision. Let them finish, or cancel them, first.
    """
    bind = op.get_bind()
    runs = bind.execute(
        sa.text("SELECT count(*) FROM reconstruction_runs WHERE status IN ('queued', 'running')")
    ).scalar_one()
    jobs = bind.execute(
        sa.text(
            "SELECT count(*) FROM jobs "
            "WHERE stage = :stage AND state IN ('queued', 'running', 'retry_wait')"
        ),
        {"stage": stage},
    ).scalar_one()
    if runs or jobs:
        raise RuntimeError(
            f"cannot downgrade: {runs} run(s) and {jobs} texturing job(s) are still active; "
            "let them finish or cancel them first"
        )


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
    _refuse_active("texturing")
    # The 0005 lists have no texturing; its finished jobs cannot be kept. job_events cascade.
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
