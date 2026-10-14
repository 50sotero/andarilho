"""Run submission and cancellation (README section 6).

Submission inserts the run, its first job and its usage reservation in one
transaction. Cancellation sets the run-level marker, cancels queued/retry_wait
jobs atomically and marks running jobs with ``cancel_requested_at``; workers
acknowledge by moving their job to ``cancelled`` with their current fencing
token. When no running job remains, the run becomes ``cancelled`` and its
reservation is released exactly once (``state = 'reserved'`` guard).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from andarilho_api.domain.recipes import get_recipe
from andarilho_api.errors import ApiError
from andarilho_api.models import (
    ACTIVE_RUN_STATUSES,
    CaptureSet,
    Job,
    Project,
    ReconstructionRun,
    UsageLedger,
)
from andarilho_api.settings import Settings

ACTIVE_RUN_INDEX = "uq_reconstruction_runs_one_active"


def active_run_exists() -> ApiError:
    return ApiError(
        409,
        "ACTIVE_RUN_EXISTS",
        "This project already has a reconstruction in progress.",
        suggested_action="Wait for the current run to finish or cancel it.",
    )


def constraint_name(exc: IntegrityError) -> str | None:
    diag = getattr(exc.orig, "diag", None)
    return getattr(diag, "constraint_name", None)


def submit_run(
    db: Session,
    settings: Settings,
    *,
    project: Project,
    subject: str,
    capture_set: CaptureSet,
    quality: str,
    include_supplementary_photos: bool,
    retry_of_run_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """Insert run + first job + ledger reservation. Caller commits and handles
    ``IntegrityError`` on the one-active-run index (races)."""
    if quality == "synthetic" and not settings.synthetic_recipe_enabled:
        raise ApiError(422, "VALIDATION_FAILED", "This quality tier is not available.")
    active = db.execute(
        select(ReconstructionRun.id).where(
            ReconstructionRun.project_id == project.id,
            ReconstructionRun.status.in_(ACTIVE_RUN_STATUSES),
        )
    ).first()
    if active is not None:
        raise active_run_exists()

    recipe = get_recipe(quality)
    digest = recipe.digest
    budget = recipe.budget(settings.run_max_seconds)
    run = ReconstructionRun(
        id=uuid.uuid4(),
        project_id=project.id,
        capture_set_id=capture_set.id,
        retry_of_run_id=retry_of_run_id,
        created_by=subject,
        recipe_name=recipe.name,
        recipe_digest=digest,
        quality=quality,
        include_supplementary_photos=include_supplementary_photos,
        resource_budget=budget,
        status="queued",
    )
    db.add(run)
    db.flush()
    first_stage = recipe.stages[0]
    job = Job(
        id=uuid.uuid4(),
        project_id=project.id,
        run_id=run.id,
        capability="cpu",
        stage=first_stage,
        state="queued",
        input_manifest={
            "runId": str(run.id),
            "projectId": str(project.id),
            "captureSetId": str(capture_set.id),
            "quality": quality,
            "includeSupplementaryPhotos": include_supplementary_photos,
            "recipe": {
                "name": recipe.name,
                "digest": digest,
                "stages": list(recipe.stage_list),
                "budget": budget,
                "params": recipe.params,
            },
        },
    )
    db.add(job)
    db.add(
        UsageLedger(
            project_id=project.id,
            run_id=run.id,
            reserved_gpu_seconds=budget["reservedGpuSeconds"],
            reserved_cpu_seconds=budget["reservedCpuSeconds"],
            state="reserved",
        )
    )
    db.flush()
    return {"runId": str(run.id), "jobId": str(job.id), "state": "queued", "stage": first_stage}


def _finalize_cancelled(db: Session, run: ReconstructionRun, now: datetime) -> None:
    run.status = "cancelled"
    run.finished_at = now
    db.execute(
        update(UsageLedger)
        .where(UsageLedger.run_id == run.id, UsageLedger.state == "reserved")
        .values(state="released", settled_at=now)
    )


def cancel_jobs(
    db: Session, *, now: datetime, fence_running: bool, run_id=None, project_id=None
) -> None:
    """Cancel queued/retry_wait jobs and mark running ones (optionally bumping fences)."""
    scope = Job.run_id == run_id if run_id is not None else Job.project_id == project_id
    db.execute(
        update(Job)
        .where(scope, Job.state.in_(("queued", "retry_wait")))
        .values(
            state="cancelled",
            finished_at=now,
            cancel_requested_at=func.coalesce(Job.cancel_requested_at, now),
            updated_at=now,
        )
        .execution_options(synchronize_session=False)
    )
    values: dict[str, Any] = {
        "cancel_requested_at": func.coalesce(Job.cancel_requested_at, now),
        "updated_at": now,
    }
    if fence_running:
        values["fencing_token"] = Job.fencing_token + 1
    db.execute(
        update(Job)
        .where(scope, Job.state == "running")
        .values(**values)
        .execution_options(synchronize_session=False)
    )


def request_run_cancellation(
    db: Session, run: ReconstructionRun, *, now: datetime, fence_running: bool = False
) -> None:
    """Apply README section 6 cancellation to one active run (caller holds locks)."""
    if run.status not in ACTIVE_RUN_STATUSES:
        return
    if run.cancel_requested_at is None:
        run.cancel_requested_at = now
    cancel_jobs(db, now=now, fence_running=fence_running, run_id=run.id)
    finalize_if_idle(db, run, now=now)


def finalize_if_idle(db: Session, run: ReconstructionRun, *, now: datetime) -> None:
    running = db.execute(
        select(func.count()).where(Job.run_id == run.id, Job.state == "running")
    ).scalar_one()
    if running == 0:
        _finalize_cancelled(db, run, now)
