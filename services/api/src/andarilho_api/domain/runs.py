"""Run submission and cancellation (README section 6).

Submission inserts the run, its first job and its usage reservation in one
transaction. Cancellation sets the run-level marker, cancels queued/retry_wait
jobs atomically and marks running jobs with ``cancel_requested_at``; workers
acknowledge by moving their job to ``cancelled`` with their current fencing
token. When no running job remains, the run becomes ``cancelled`` and its
reservation is released exactly once (``state = 'reserved'`` guard). Storage that
the cancelled work may have written is queued for deletion as ``artifact_cleanup``
jobs, as the worker queues them.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
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
    SceneRevision,
    Upload,
    UsageLedger,
)
from andarilho_api.settings import Settings
from andarilho_api.storage import attempt_prefix, scene_prefix, tenant_prefix

ACTIVE_RUN_INDEX = "uq_reconstruction_runs_one_active"
# Shown on a run's first revision when the run is cancelled before it is ready
# (the worker writes the same finding; see queue.RUN_CANCELLED_FINDING).
RUN_CANCELLED_FINDING = {
    "code": "RUN_CANCELLED",
    "message": "Processing was cancelled before this scene was ready.",
}
# The worker's delay before ``artifact_cleanup`` runs (queue.ARTIFACT_CLEANUP_DELAY_SECONDS),
# so that an attempt which was still writing has stopped by then.
ARTIFACT_CLEANUP_DELAY = timedelta(seconds=300)


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


def run_inputs(
    inputs: list[dict[str, Any]], include_supplementary_photos: bool
) -> list[dict[str, Any]]:
    """The capture-set inputs a run reads.

    A video run without supplementary photos never reads its photos: the worker's
    validation drops them, and the upload removal rule lets them be removed.
    """
    if include_supplementary_photos or not any(i.get("mediaKind") == "video" for i in inputs):
        return list(inputs)
    return [i for i in inputs if i.get("mediaKind") != "photo"]


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

    upload_ids = {
        uuid.UUID(str(item["uploadId"]))
        for item in run_inputs(capture_set.inputs, include_supplementary_photos)
    }
    if capture_set.floor_plan_upload_id is not None:
        upload_ids.add(capture_set.floor_plan_upload_id)
    if upload_ids:
        usable = db.execute(
            select(func.count()).where(
                Upload.project_id == project.id,
                Upload.id.in_(upload_ids),
                Upload.status == "verified",
            )
        ).scalar_one()
        if usable != len(upload_ids):
            raise ApiError(
                409,
                "INVALID_STATE",
                "This capture set includes a file that was removed.",
                suggested_action="Create a new capture set from the current files.",
            )

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


def _queue_artifact_cleanup(
    db: Session,
    *,
    project_id: uuid.UUID,
    prefixes: Sequence[str],
    now: datetime,
    reason: str,
    scene_revision_id: uuid.UUID | None = None,
) -> None:
    """Queue the worker's ``artifact_cleanup`` job for storage a cancelled run left behind.

    Same contract as the worker's ``queue.queue_artifact_cleanup``: no run, revision or
    upload, claimable after the delay, and the manifest the worker validates. A request
    identical to one still queued or retrying is not queued again. Prefixes outside the
    project's tenant directory are a programming error.
    """
    if not prefixes:
        return
    tenant = tenant_prefix(project_id)
    for prefix in prefixes:
        if not (prefix.startswith(tenant) and prefix.endswith("/")):
            raise ValueError(f"artifact prefix {prefix!r} is not a directory of the project")
    unique = sorted(set(prefixes))
    pending = db.execute(
        select(Job.id).where(
            Job.project_id == project_id,
            Job.stage == "artifact_cleanup",
            Job.state.in_(("queued", "retry_wait")),
            Job.input_manifest["prefixes"] == unique,
        )
    ).first()
    if pending is not None:
        return
    manifest: dict[str, Any] = {"projectId": str(project_id), "prefixes": unique, "reason": reason}
    if scene_revision_id is not None:
        manifest["sceneRevisionId"] = str(scene_revision_id)
    db.add(
        Job(
            id=uuid.uuid4(),
            project_id=project_id,
            capability="cpu",
            stage="artifact_cleanup",
            state="queued",
            priority=5,
            attempts=0,
            max_attempts=3,
            fencing_token=0,
            available_at=now + ARTIFACT_CLEANUP_DELAY,
            input_manifest=manifest,
        )
    )


def _finalize_cancelled(db: Session, run: ReconstructionRun, now: datetime) -> None:
    run.status = "cancelled"
    run.finished_at = now
    db.execute(
        update(UsageLedger)
        .where(UsageLedger.run_id == run.id, UsageLedger.state == "reserved")
        .values(state="released", settled_at=now)
    )
    # The revision this run's pipeline created (if it got that far) will never be
    # validated: stop it pending, as the worker does for runs that end unfinished. Its
    # packaged files are unreferenced from now on, so they are queued for deletion.
    stopped = db.execute(
        update(SceneRevision)
        .where(
            SceneRevision.run_id == run.id,
            SceneRevision.revision == 1,
            SceneRevision.validation_state == "pending",
        )
        .values(validation_state="invalid", validation_findings=[RUN_CANCELLED_FINDING])
        .returning(SceneRevision.id, SceneRevision.project_id)
        .execution_options(synchronize_session=False)
    ).all()
    for revision_id, project_id in stopped:
        _queue_artifact_cleanup(
            db,
            project_id=project_id,
            prefixes=[scene_prefix(project_id, revision_id)],
            now=now,
            reason="revision_invalid",
            scene_revision_id=revision_id,
        )


def cancel_jobs(
    db: Session, *, now: datetime, fence_running: bool, run_id=None, project_id=None
) -> None:
    """Cancel queued/retry_wait jobs and mark running ones (optionally bumping fences).

    A run's queued job that was claimed before (fence > 0) may have written outputs
    under an earlier fence, and no worker will claim it again, so those attempts'
    prefixes are queued for deletion. A project's jobs need no such cleanup: deleting
    the project removes its whole tenant prefix.
    """
    scope = Job.run_id == run_id if run_id is not None else Job.project_id == project_id
    unclaimed = []
    if run_id is not None:
        # Locked, so that no worker claims one of these jobs between this read and the update.
        unclaimed = db.execute(
            select(Job.id, Job.project_id, Job.run_id, Job.stage, Job.fencing_token)
            .where(scope, Job.state.in_(("queued", "retry_wait")), Job.fencing_token > 0)
            .with_for_update()
        ).all()
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
    by_project: dict[uuid.UUID, list[str]] = {}
    for job in unclaimed:
        by_project.setdefault(job.project_id, []).extend(
            attempt_prefix(job.project_id, job.run_id, job.stage, job.id, fence)
            for fence in range(1, job.fencing_token + 1)
        )
    for owner, prefixes in by_project.items():
        _queue_artifact_cleanup(
            db,
            project_id=owner,
            prefixes=prefixes,
            now=now,
            reason="cancelled_before_claim",
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
