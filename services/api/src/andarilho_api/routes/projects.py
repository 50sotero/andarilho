"""Projects: create, list, read, update, delete."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy import and_, select, tuple_
from sqlalchemy.orm import Session

from andarilho_api import serializers as ser
from andarilho_api.auth import current_subject
from andarilho_api.db import get_db
from andarilho_api.deps import check_if_match, get_storage, utcnow
from andarilho_api.domain import audit
from andarilho_api.domain.deletion import request_deletion
from andarilho_api.domain.membership import OWNER, WRITERS, load_project
from andarilho_api.errors import validation_failed
from andarilho_api.models import (
    CaptureSet,
    Job,
    Project,
    ProjectMember,
    ReconstructionRun,
    Scene,
    SceneRevision,
    ShareLink,
    Upload,
)
from andarilho_api.schemas.api import ProjectCreate, ProjectUpdate
from andarilho_api.storage import Storage, StorageError

logger = logging.getLogger("andarilho_api.projects")
router = APIRouter(tags=["projects"])

Subject = Annotated[str, Depends(current_subject)]
Db = Annotated[Session, Depends(get_db)]


def _encode_cursor(created_at: datetime, project_id: uuid.UUID) -> str:
    raw = json.dumps([created_at.isoformat(), str(project_id)]).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        created, pid = json.loads(base64.urlsafe_b64decode(padded.encode()))
        return datetime.fromisoformat(created), uuid.UUID(pid)
    except (ValueError, TypeError, binascii.Error, json.JSONDecodeError):
        raise validation_failed("The cursor is invalid.") from None


@router.post("/projects", status_code=201)
def create_project(body: ProjectCreate, response: Response, subject: Subject, db: Db):
    project = Project(id=uuid.uuid4(), title=body.title, owner_subject=subject, lifecycle="active")
    db.add(project)
    db.flush()
    db.add(ProjectMember(project_id=project.id, subject=subject, role="owner"))
    audit.record(
        db,
        project_id=project.id,
        actor=subject,
        action="project.created",
        resource_type="project",
        resource_id=project.id,
    )
    db.commit()
    db.refresh(project)
    response.headers["ETag"] = ser.etag(project.version)
    return ser.project(project, "owner")


@router.get("/projects")
def list_projects(
    subject: Subject,
    db: Db,
    cursor: Annotated[str | None, Query(max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    stmt = (
        select(Project, ProjectMember.role)
        .join(
            ProjectMember,
            and_(ProjectMember.project_id == Project.id, ProjectMember.subject == subject),
        )
        .where(Project.lifecycle != "deleted")
        .order_by(Project.created_at.desc(), Project.id.desc())
        .limit(limit + 1)
    )
    if cursor:
        created_at, pid = _decode_cursor(cursor)
        stmt = stmt.where(tuple_(Project.created_at, Project.id) < tuple_(created_at, pid))
    rows = db.execute(stmt).all()
    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1][0]
        next_cursor = _encode_cursor(last.created_at, last.id)
    return {"items": [ser.project(p, role) for p, role in page], "nextCursor": next_cursor}


def _project_detail(db: Session, project: Project, role: str) -> dict[str, Any]:
    pid = project.id
    uploads = db.execute(
        select(Upload).where(Upload.project_id == pid).order_by(Upload.created_at, Upload.id)
    ).scalars()
    capture_sets = db.execute(
        select(CaptureSet)
        .where(CaptureSet.project_id == pid)
        .order_by(CaptureSet.created_at, CaptureSet.id)
    ).scalars()
    runs = (
        db.execute(
            select(ReconstructionRun)
            .where(ReconstructionRun.project_id == pid)
            .order_by(ReconstructionRun.created_at.desc(), ReconstructionRun.id)
        )
        .scalars()
        .all()
    )
    jobs_by_run: dict[uuid.UUID, list[Job]] = {}
    if runs:
        for job in db.execute(
            select(Job).where(
                Job.project_id == pid,
                Job.run_id.in_([r.id for r in runs]),
                Job.scene_revision_id.is_(None),
            )
        ).scalars():
            jobs_by_run.setdefault(job.run_id, []).append(job)  # type: ignore[arg-type]

    def scene_of(revision_id: uuid.UUID | None) -> Scene | None:
        if revision_id is None:
            return None
        return db.execute(
            select(Scene)
            .join(SceneRevision, SceneRevision.scene_id == Scene.id)
            .where(SceneRevision.id == revision_id, Scene.project_id == pid)
        ).scalar_one_or_none()

    # Viewers only walk the published revision, so their scene is the one it belongs to,
    # even when a newer draft scene exists.
    viewer = role == "viewer"
    scene = scene_of(project.published_revision_id if viewer else project.active_draft_revision_id)
    if scene is None and not viewer:
        scene = db.execute(
            select(Scene)
            .where(Scene.project_id == pid)
            .order_by(Scene.created_at.desc(), Scene.id)
            .limit(1)
        ).scalar_one_or_none()
    shares = db.execute(
        select(ShareLink)
        .where(ShareLink.project_id == pid)
        .order_by(ShareLink.created_at.desc(), ShareLink.id)
    ).scalars()
    return {
        "project": ser.project(project, role),
        "uploads": [ser.upload(u) for u in uploads],
        "captureSets": [ser.capture_set(c) for c in capture_sets],
        "runs": [ser.run(r, jobs_by_run.get(r.id, [])) for r in runs],
        "scene": ser.scene_ref(scene) if scene else None,
        "shares": [ser.share(s) for s in shares],
    }


@router.get("/projects/{project_id}")
def get_project(project_id: uuid.UUID, response: Response, subject: Subject, db: Db):
    project, role = load_project(db, project_id, subject)
    response.headers["ETag"] = ser.etag(project.version)
    return _project_detail(db, project, role)


@router.patch("/projects/{project_id}")
def update_project(
    project_id: uuid.UUID,
    body: ProjectUpdate,
    request: Request,
    response: Response,
    subject: Subject,
    db: Db,
):
    project, role = load_project(
        db, project_id, subject, roles=WRITERS, lock=True, require_active=True
    )
    check_if_match(request, project.version, "project")
    project.title = body.title
    project.version += 1
    project.updated_at = utcnow()
    db.commit()
    db.refresh(project)
    response.headers["ETag"] = ser.etag(project.version)
    return ser.project(project, role)


@router.delete("/projects/{project_id}", status_code=202)
def delete_project(
    project_id: uuid.UUID,
    subject: Subject,
    db: Db,
    storage: Annotated[Storage, Depends(get_storage)],
):
    project, _ = load_project(db, project_id, subject, roles=OWNER, lock=True)
    if project.lifecycle == "active":
        to_abort = request_deletion(db, project, subject=subject, now=utcnow())
        db.commit()
        for key, multipart_id in to_abort:
            try:
                storage.abort_multipart_upload(key, multipart_id)
            except StorageError:
                # The project_cleanup job and bucket lifecycle rules reconcile leftovers.
                logger.warning("multipart abort failed during project deletion")
    return {"projectId": str(project.id), "lifecycle": "deleting"}
