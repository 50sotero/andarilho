"""Reconstruction runs (submit, cancel, retry) and job progress."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from andarilho_api import serializers as ser
from andarilho_api.auth import current_subject
from andarilho_api.db import get_db
from andarilho_api.deps import get_settings_dep, utcnow
from andarilho_api.domain import audit, idempotency
from andarilho_api.domain.membership import WRITERS, load_project, load_scoped, lock_row
from andarilho_api.domain.runs import (
    ACTIVE_RUN_INDEX,
    active_run_exists,
    constraint_name,
    request_run_cancellation,
    submit_run,
)
from andarilho_api.errors import invalid_state, not_found
from andarilho_api.models import CaptureSet, Job, JobEvent, ReconstructionRun
from andarilho_api.schemas.api import EmptyBody, RunCreate
from andarilho_api.settings import Settings

router = APIRouter(tags=["runs"])

Subject = Annotated[str, Depends(current_subject)]
Db = Annotated[Session, Depends(get_db)]
Config = Annotated[Settings, Depends(get_settings_dep)]
IdemKey = Annotated[str | None, Header(alias="Idempotency-Key")]

IDEMPOTENCY_INDEX = "uq_idempotency_scope"


def _idempotent_submit(
    db: Session,
    settings: Settings,
    *,
    project_id: uuid.UUID,
    subject: str,
    operation: str,
    key: str,
    req_hash: str,
    action: Callable[[], dict[str, Any]],
) -> JSONResponse:
    """Replay, reject reuse, or run ``action`` and store its response atomically."""
    record = idempotency.find(
        db, project_id=project_id, operation=operation, key=key, subject=subject, req_hash=req_hash
    )
    if record is not None:
        return JSONResponse(status_code=record.response_status, content=record.response_body)
    try:
        body = action()
        idempotency.store(
            db,
            project_id=project_id,
            operation=operation,
            key=key,
            subject=subject,
            req_hash=req_hash,
            status=202,
            body=body,
            ttl_seconds=settings.idempotency_ttl_seconds,
        )
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        name = constraint_name(exc)
        if name == IDEMPOTENCY_INDEX:
            # A concurrent request with the same key won the race.
            record = idempotency.find(
                db,
                project_id=project_id,
                operation=operation,
                key=key,
                subject=subject,
                req_hash=req_hash,
            )
            if record is not None:
                return JSONResponse(
                    status_code=record.response_status, content=record.response_body
                )
        if name == ACTIVE_RUN_INDEX:
            raise active_run_exists() from None
        raise
    return JSONResponse(status_code=202, content=body)


@router.post("/projects/{project_id}/runs", status_code=202)
def create_run(
    project_id: uuid.UUID,
    body: RunCreate,
    subject: Subject,
    db: Db,
    settings: Config,
    idempotency_key: IdemKey = None,
):
    key = idempotency.require_key(idempotency_key)
    project, _ = load_project(
        db, project_id, subject, roles=WRITERS, lock=True, require_active=True
    )
    operation = "runs.create"
    req_hash = idempotency.request_hash(operation, str(project.id), body.model_dump(mode="json"))

    def action() -> dict[str, Any]:
        capture_set = db.execute(
            select(CaptureSet).where(
                CaptureSet.project_id == project.id, CaptureSet.id == body.capture_set_id
            )
        ).scalar_one_or_none()
        if capture_set is None:
            raise not_found("capture set")
        result = submit_run(
            db,
            settings,
            project=project,
            subject=subject,
            capture_set=capture_set,
            quality=body.quality,
            include_supplementary_photos=body.include_supplementary_photos,
        )
        audit.record(
            db,
            project_id=project.id,
            actor=subject,
            action="run.submitted",
            resource_type="run",
            resource_id=result["runId"],
            data={"quality": body.quality},
        )
        return result

    return _idempotent_submit(
        db,
        settings,
        project_id=project.id,
        subject=subject,
        operation=operation,
        key=key,
        req_hash=req_hash,
        action=action,
    )


@router.post("/runs/{run_id}/cancel", status_code=202)
def cancel_run(run_id: uuid.UUID, subject: Subject, db: Db):
    scoped = load_scoped(db, ReconstructionRun, run_id, subject, "run")
    load_project(db, scoped.project_id, subject, roles=WRITERS, lock=True, require_active=True)
    run = lock_row(db, ReconstructionRun, scoped.id)
    if run.status in ("succeeded", "failed"):
        raise invalid_state("This run has already finished.")
    if run.status != "cancelled":
        request_run_cancellation(db, run, now=utcnow())
        audit.record(
            db,
            project_id=run.project_id,
            actor=subject,
            action="run.cancel_requested",
            resource_type="run",
            resource_id=run.id,
        )
        db.commit()
    return {"runId": str(run.id), "status": run.status, "cancelRequested": True}


@router.post("/runs/{run_id}/retry", status_code=202)
def retry_run(
    run_id: uuid.UUID,
    subject: Subject,
    db: Db,
    settings: Config,
    body: EmptyBody | None = None,
    idempotency_key: IdemKey = None,
):
    key = idempotency.require_key(idempotency_key)
    scoped = load_scoped(db, ReconstructionRun, run_id, subject, "run")
    project, _ = load_project(
        db, scoped.project_id, subject, roles=WRITERS, lock=True, require_active=True
    )
    operation = "runs.retry"
    req_hash = idempotency.request_hash(operation, str(scoped.id), {})

    def action() -> dict[str, Any]:
        previous = lock_row(db, ReconstructionRun, scoped.id)
        if previous.status not in ("failed", "cancelled"):
            raise invalid_state("Only failed or cancelled runs can be retried.")
        capture_set = db.execute(
            select(CaptureSet).where(
                CaptureSet.project_id == project.id, CaptureSet.id == previous.capture_set_id
            )
        ).scalar_one()
        result = submit_run(
            db,
            settings,
            project=project,
            subject=subject,
            capture_set=capture_set,
            quality=previous.quality,
            include_supplementary_photos=previous.include_supplementary_photos,
            retry_of_run_id=previous.id,
        )
        audit.record(
            db,
            project_id=project.id,
            actor=subject,
            action="run.retried",
            resource_type="run",
            resource_id=result["runId"],
            data={"retryOf": str(previous.id)},
        )
        return result

    return _idempotent_submit(
        db,
        settings,
        project_id=project.id,
        subject=subject,
        operation=operation,
        key=key,
        req_hash=req_hash,
        action=action,
    )


@router.get("/jobs/{job_id}")
def get_job(
    job_id: uuid.UUID,
    subject: Subject,
    db: Db,
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
):
    job = load_scoped(db, Job, job_id, subject, "job")
    events = (
        db.execute(
            select(JobEvent)
            .where(JobEvent.job_id == job.id, JobEvent.id > after)
            .order_by(JobEvent.id)
            .limit(limit + 1)
        )
        .scalars()
        .all()
    )
    page = events[:limit]
    next_cursor = page[-1].id if len(events) > limit else None
    return {
        "job": ser.job(job),
        "events": [ser.job_event(e) for e in page],
        "nextCursor": next_cursor,
    }
