"""Capture sets: immutable snapshots of verified inputs."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from andarilho_api import serializers as ser
from andarilho_api.auth import current_subject
from andarilho_api.db import get_db
from andarilho_api.domain.canonical import sha256_hex
from andarilho_api.domain.membership import WRITERS, load_project
from andarilho_api.errors import validation_failed
from andarilho_api.models import CaptureSet, Upload
from andarilho_api.schemas.api import CaptureSetCreate

router = APIRouter(tags=["capture-sets"])

Subject = Annotated[str, Depends(current_subject)]
Db = Annotated[Session, Depends(get_db)]


def _snapshot(upload: Upload) -> dict[str, Any]:
    return {
        "uploadId": str(upload.id),
        "mediaKind": upload.media_kind,
        "key": upload.immutable_key,
        "storageVersion": upload.storage_version,
        "sha256": upload.sha256,
        "bytes": upload.actual_bytes or upload.expected_bytes,
        "contentType": upload.declared_content_type,
        "metadata": upload.media_metadata or {},
    }


@router.post("/projects/{project_id}/capture-sets", status_code=201)
def create_capture_set(project_id: uuid.UUID, body: CaptureSetCreate, subject: Subject, db: Db):
    project, _ = load_project(
        db, project_id, subject, roles=WRITERS, lock=True, require_active=True
    )
    wanted = list(body.upload_ids)
    if body.floor_plan_upload_id is not None:
        if body.floor_plan_upload_id in wanted:
            raise validation_failed("The floor plan must not also be listed in uploadIds.")
        wanted.append(body.floor_plan_upload_id)
    uploads = {
        u.id: u
        for u in db.execute(
            select(Upload)
            .where(Upload.project_id == project.id, Upload.id.in_(wanted))
            .with_for_update()
        ).scalars()
    }
    if len(uploads) != len(wanted):
        raise validation_failed("Every upload must belong to this project.")
    if any(u.status != "verified" for u in uploads.values()):
        raise validation_failed("Every upload must be verified before it can be used.")
    media = [uploads[i] for i in body.upload_ids]
    if any(u.media_kind not in ("video", "photo") for u in media):
        raise validation_failed("uploadIds may only contain videos and photos.")
    inputs = [_snapshot(u) for u in media]
    if body.floor_plan_upload_id is not None:
        plan = uploads[body.floor_plan_upload_id]
        if plan.media_kind != "floor_plan":
            raise validation_failed("floorPlanUploadId must reference a floor plan upload.")
        inputs.append(_snapshot(plan))
    inputs.sort(key=lambda i: i["uploadId"])

    capture_set = CaptureSet(
        id=uuid.uuid4(),
        project_id=project.id,
        created_by=subject,
        inputs=inputs,
        floor_plan_upload_id=body.floor_plan_upload_id,
        capture_metadata=body.metadata,
        content_sha256=sha256_hex(inputs),
    )
    db.add(capture_set)
    db.commit()
    db.refresh(capture_set)
    return ser.capture_set(capture_set)
