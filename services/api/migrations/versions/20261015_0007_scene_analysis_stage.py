"""scene_analysis run stage: rooms, doors and furniture from an AI model

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Run stages before and after the change; ``training`` stays valid for historical rows.
_OLD_RUN = (
    "'validating', 'preparing', 'camera_reconstruction', 'normalizing', "
    "'geometry_proposal', 'texturing', 'packaging', 'scene_validation', 'training'"
)
_NEW_RUN = (
    "'validating', 'preparing', 'camera_reconstruction', 'normalizing', 'scene_analysis', "
    "'geometry_proposal', 'texturing', 'packaging', 'scene_validation', 'training'"
)
_MAINTENANCE = "'upload_inspection', 'project_cleanup', 'upload_cleanup', 'artifact_cleanup'"


def _refuse_active(stage: str) -> None:
    """Stop the downgrade while any run, or any scene_analysis job, can still run.

    A run is pinned to the recipe it was submitted with, and the recipes published since 0007 reach
    scene_analysis even when their scene_analysis job is not queued yet. After the downgrade a
    current worker could not insert that job, and a rolled-back worker cannot run the
    pinned recipe, so either way the run fails. The database does not record which
    stages a recipe has, so every queued or running run blocks the downgrade. Layout
    saves queue scene_analysis on finished runs, so active jobs of the stage are counted
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
            f"cannot downgrade: {runs} run(s) and {jobs} scene_analysis job(s) are still active; "
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
    _refuse_active("scene_analysis")
    # The 0006 lists have no scene_analysis; its finished jobs cannot be kept. job_events cascade.
    op.execute("DELETE FROM jobs WHERE stage = 'scene_analysis'")
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
