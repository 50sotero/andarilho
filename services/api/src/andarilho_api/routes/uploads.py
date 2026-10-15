"""Resumable multipart uploads (README section 7, "Multipart upload protocol")."""

from __future__ import annotations

import logging
import uuid
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Response
from sqlalchemy import exists, or_, select
from sqlalchemy.orm import Session

from andarilho_api import serializers as ser
from andarilho_api.auth import current_subject
from andarilho_api.db import get_db
from andarilho_api.deps import get_settings_dep, get_storage, utcnow
from andarilho_api.domain.membership import WRITERS, load_project, load_scoped, lock_row
from andarilho_api.domain.quotas import check_upload, intake_policy
from andarilho_api.errors import ApiError, invalid_state, validation_failed
from andarilho_api.models import (
    ACTIVE_RUN_STATUSES,
    CaptureSet,
    Job,
    ReconstructionRun,
    Upload,
)
from andarilho_api.schemas.api import EmptyBody, PartsRequest, UploadCreate
from andarilho_api.settings import Settings
from andarilho_api.storage import (
    ObjectInfo,
    PreconditionFailed,
    Storage,
    StorageError,
    immutable_key,
    staging_key,
)

logger = logging.getLogger("andarilho_api.uploads")
router = APIRouter(tags=["uploads"])

Subject = Annotated[str, Depends(current_subject)]
Db = Annotated[Session, Depends(get_db)]
Store = Annotated[Storage, Depends(get_storage)]
Config = Annotated[Settings, Depends(get_settings_dep)]


def _storage_unavailable() -> ApiError:
    return ApiError(
        503,
        "STORAGE_UNAVAILABLE",
        "Storage is temporarily unavailable.",
        retryable=True,
        suggested_action="Try again in a moment.",
    )


def _expired() -> ApiError:
    return ApiError(
        409,
        "UPLOAD_EXPIRED",
        "This upload has expired.",
        suggested_action="Start a new upload for this file.",
    )


@router.get("/intake-policy")
def get_intake_policy(subject: Subject, settings: Config) -> dict[str, int]:
    """The deployment's capture limits, so the browser checks a selection before uploading."""
    return intake_policy(settings)


@router.post("/projects/{project_id}/uploads", status_code=201)
def create_upload(
    project_id: uuid.UUID,
    body: UploadCreate,
    subject: Subject,
    db: Db,
    storage: Store,
    settings: Config,
):
    project, _ = load_project(
        db, project_id, subject, roles=WRITERS, lock=True, require_active=True
    )
    part_size, part_count = check_upload(
        db,
        settings,
        project_id=project.id,
        media_kind=body.media_kind,
        content_type=body.content_type,
        size=body.bytes,
    )
    upload_id = uuid.uuid4()
    key = staging_key(project.id, upload_id)
    try:
        multipart_id = storage.create_multipart_upload(key, body.content_type)
    except Exception:
        logger.exception("create_multipart_upload failed")
        raise _storage_unavailable() from None
    upload = Upload(
        id=upload_id,
        project_id=project.id,
        created_by=subject,
        media_kind=body.media_kind,
        display_filename=body.filename,
        declared_content_type=body.content_type,
        expected_bytes=body.bytes,
        staging_key=key,
        multipart_upload_id=multipart_id,
        part_size=part_size,
        part_count=part_count,
        status="initiated",
        expires_at=utcnow() + timedelta(seconds=settings.upload_ttl_seconds),
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return {"upload": ser.upload(upload), "partSize": part_size, "partCount": part_count}


@router.get("/uploads/{upload_id}")
def get_upload(upload_id: uuid.UUID, subject: Subject, db: Db, storage: Store):
    upload = load_scoped(db, Upload, upload_id, subject, "upload")
    completed: list[dict] = []
    if upload.status == "initiated" and upload.multipart_upload_id:
        try:
            parts = storage.list_parts(upload.staging_key, upload.multipart_upload_id)
        except Exception:
            logger.exception("list_parts failed")
            raise _storage_unavailable() from None
        completed = [{"partNumber": p.part_number, "etag": p.etag, "size": p.size} for p in parts]
    return {"upload": ser.upload(upload), "completedParts": completed}


def _require_open(upload: Upload) -> None:
    if upload.status != "initiated" or not upload.multipart_upload_id:
        raise invalid_state("This upload is no longer accepting parts.")
    if upload.expires_at <= utcnow():
        raise _expired()


@router.post("/uploads/{upload_id}/parts")
def sign_parts(
    upload_id: uuid.UUID,
    body: PartsRequest,
    subject: Subject,
    db: Db,
    storage: Store,
    settings: Config,
):
    upload = load_scoped(db, Upload, upload_id, subject, "upload")
    load_project(db, upload.project_id, subject, roles=WRITERS, require_active=True)
    _require_open(upload)
    if any(n > upload.part_count for n in body.part_numbers):
        raise validation_failed("Part numbers must be between 1 and the upload's part count.")
    ttl = settings.upload_part_url_ttl_seconds
    expires_at = utcnow() + timedelta(seconds=ttl)
    urls = [
        {
            "partNumber": n,
            "url": storage.presign_upload_part(
                upload.staging_key,
                upload.multipart_upload_id,
                n,
                ttl,  # type: ignore[arg-type]
            ),
        }
        for n in body.part_numbers
    ]
    return {"urls": urls, "expiresAt": ser.iso(expires_at)}


def _reject(db: Session, upload: Upload, code: str, *, cleanup: bool = False) -> None:
    upload.status = "rejected"
    upload.rejection_code = code
    upload.multipart_upload_id = None
    upload.updated_at = utcnow()
    if cleanup:
        _queue_cleanup(db, upload)
    db.commit()


@router.post("/uploads/{upload_id}/complete", status_code=202)
def complete_upload(
    upload_id: uuid.UUID,
    subject: Subject,
    db: Db,
    storage: Store,
    body: EmptyBody | None = None,
):
    scoped = load_scoped(db, Upload, upload_id, subject, "upload")
    load_project(db, scoped.project_id, subject, roles=WRITERS, lock=True, require_active=True)
    upload = lock_row(db, Upload, scoped.id)
    if upload.status in ("uploaded_unverified", "verified"):
        return {"upload": ser.upload(upload)}  # idempotent replay
    _require_open(upload)
    multipart_id: str = upload.multipart_upload_id  # type: ignore[assignment]

    try:
        # A previous attempt may have completed the multipart upload and then failed
        # before the database commit; resume from the assembled staging object.
        assembled = storage.head_object(upload.staging_key)
    except StorageError:
        raise _storage_unavailable() from None
    if assembled is not None:
        return _freeze(db, storage, upload, assembled)

    try:
        listed = storage.list_parts(upload.staging_key, multipart_id)
    except Exception:
        logger.exception("list_parts failed")
        raise _storage_unavailable() from None
    by_number = {p.part_number: p for p in listed}
    expected = range(1, upload.part_count + 1)
    missing = [n for n in expected if n not in by_number]
    if missing:
        raise ApiError(
            409,
            "UPLOAD_INCOMPLETE",
            f"{len(missing)} of {upload.part_count} parts have not been uploaded.",
            retryable=True,
            suggested_action="Upload the missing parts, then complete again.",
        )
    parts = [by_number[n] for n in expected]
    if any(p.size > upload.part_size for p in parts):
        try:
            storage.abort_multipart_upload(upload.staging_key, multipart_id)
        except StorageError:
            logger.warning("multipart abort failed; lifecycle rules will reconcile")
        _reject(db, upload, "SIZE_MISMATCH")
        raise _size_mismatch()

    try:
        storage.complete_multipart_upload(upload.staging_key, multipart_id, parts)
        info = storage.head_object(upload.staging_key)
    except Exception:
        logger.exception("complete/head failed")
        raise _storage_unavailable() from None
    return _freeze(db, storage, upload, info)


def _freeze(db: Session, storage: Storage, upload: Upload, info: ObjectInfo | None) -> dict:
    """Verify size, copy to the immutable key, and enqueue inspection atomically."""
    if info is None:
        raise _storage_unavailable()
    if info.size != upload.expected_bytes:
        deleted = _delete_staging(storage, upload.staging_key)
        _reject(db, upload, "SIZE_MISMATCH", cleanup=not deleted)
        raise _size_mismatch()

    frozen_key = immutable_key(upload.project_id, upload.id)
    # The upload is still open (and locked), so nothing references this key yet: a
    # stored version can only be a copy whose transaction rolled back. Remove it
    # instead of keeping an untracked full copy beside the new one.
    try:
        storage.delete_all_versions(frozen_key)
    except StorageError:
        raise _storage_unavailable() from None
    try:
        version = storage.copy_if_match(upload.staging_key, frozen_key, info.etag)
    except PreconditionFailed:
        deleted = _delete_staging(storage, upload.staging_key)
        _reject(db, upload, "MODIFIED_DURING_COMPLETION", cleanup=not deleted)
        raise ApiError(
            409,
            "UPLOAD_MODIFIED",
            "The uploaded file changed while it was being completed.",
            suggested_action="Start a new upload for this file.",
        ) from None
    except StorageError:
        logger.exception("immutable copy failed")
        raise _storage_unavailable() from None

    upload.status = "uploaded_unverified"
    upload.immutable_key = frozen_key
    upload.storage_version = version
    upload.actual_bytes = info.size
    upload.multipart_upload_id = None
    upload.updated_at = utcnow()
    db.add(
        Job(
            id=uuid.uuid4(),
            project_id=upload.project_id,
            upload_id=upload.id,
            capability="cpu",
            stage="upload_inspection",
            state="queued",
            priority=5,
            input_manifest={
                "uploadId": str(upload.id),
                "projectId": str(upload.project_id),
                "key": frozen_key,
                "storageVersion": version,
                "mediaKind": upload.media_kind,
                "declaredContentType": upload.declared_content_type,
                "expectedBytes": upload.expected_bytes,
            },
        )
    )
    db.commit()
    # Only after commit, so a failed commit can be retried from the staging object.
    if not _delete_staging(storage, upload.staging_key):
        _queue_cleanup(db, upload)
        db.commit()
    db.refresh(upload)
    return {"upload": ser.upload(upload)}


def _size_mismatch() -> ApiError:
    return ApiError(
        422,
        "UPLOAD_SIZE_MISMATCH",
        "The uploaded file size does not match the declared size.",
        suggested_action="Start a new upload for this file.",
    )


def _delete_staging(storage: Storage, key: str) -> bool:
    """Delete every version of a staging object; False when storage refused.

    A staging copy left behind would keep a second full copy of the input, so a
    caller that gets False queues an ``upload_cleanup`` job to retry.
    """
    try:
        storage.delete_all_versions(key)
    except StorageError:
        logger.warning("staging delete failed; queueing a cleanup job")
        return False
    return True


@router.delete("/uploads/{upload_id}", status_code=204)
def delete_upload(upload_id: uuid.UUID, subject: Subject, db: Db, storage: Store):
    scoped = load_scoped(db, Upload, upload_id, subject, "upload")
    load_project(db, scoped.project_id, subject, roles=WRITERS, lock=True, require_active=True)
    upload = lock_row(db, Upload, scoped.id)
    if upload.status == "uploaded_unverified":
        raise invalid_state(
            "This upload is still being checked.", suggested_action="Try again in a moment."
        )
    if upload.status == "verified":
        _remove_verified(db, upload)
        return Response(status_code=204)
    if upload.status in ("aborted", "rejected") and upload.immutable_key is not None:
        # Removed or rejected but its bytes are still stored (the cleanup job may have
        # run out of attempts): queue cleanup again unless one is still pending.
        _queue_cleanup(db, upload)
        db.commit()
        return Response(status_code=204)
    if upload.status == "initiated":
        if upload.multipart_upload_id:
            try:
                storage.abort_multipart_upload(upload.staging_key, upload.multipart_upload_id)
            except StorageError:
                logger.warning("multipart abort failed; lifecycle rules will reconcile")
        upload.status = "aborted"
        upload.multipart_upload_id = None
        upload.updated_at = utcnow()
        db.commit()
    return Response(status_code=204)


def _retained(db: Session, upload: Upload) -> bool:
    """True when an active or succeeded run reads this upload.

    Those runs (and the scenes and retries built on them) read the original
    bytes, so the upload stays until the project or its retention policy removes it.
    A run that excludes supplementary photos from a capture set with a video never
    reads its photos (the worker drops them in validation), so it keeps none.
    """
    conditions = [
        ReconstructionRun.project_id == upload.project_id,
        ReconstructionRun.capture_set_id == CaptureSet.id,
        ReconstructionRun.status.in_((*ACTIVE_RUN_STATUSES, "succeeded")),
        CaptureSet.project_id == upload.project_id,
        or_(
            CaptureSet.inputs.contains([{"uploadId": str(upload.id)}]),
            CaptureSet.floor_plan_upload_id == upload.id,
        ),
    ]
    if upload.media_kind == "photo":
        conditions.append(
            or_(
                ReconstructionRun.include_supplementary_photos.is_(True),
                ~CaptureSet.inputs.contains([{"mediaKind": "video"}]),
            )
        )
    return bool(db.execute(select(exists().where(*conditions))).scalar_one())


def _remove_verified(db: Session, upload: Upload) -> None:
    """Mark a verified, unused upload removed and queue the deletion of its bytes.

    The upload keeps its immutable key (and its quota) until the fenced
    ``upload_cleanup`` job has deleted every stored version, so a failed delete
    is retried instead of leaving unaccounted objects behind.
    """
    if _retained(db, upload):
        raise invalid_state(
            "This file is used by a reconstruction and cannot be removed.",
            suggested_action="Delete the project to remove its capture files.",
        )
    upload.status = "aborted"
    upload.updated_at = utcnow()
    _queue_cleanup(db, upload)
    db.commit()


def _queue_cleanup(db: Session, upload: Upload) -> None:
    """Queue ``upload_cleanup``, which deletes the staging copy and, once the upload
    is removed or rejected, its frozen bytes. A cleanup still waiting to run reads
    the upload's current state, so it covers this request too; a running one may
    have read it before this change, so another is queued (both steps are idempotent).
    """
    pending = db.execute(
        select(
            exists().where(
                Job.project_id == upload.project_id,
                Job.upload_id == upload.id,
                Job.stage == "upload_cleanup",
                Job.state.in_(("queued", "retry_wait")),
            )
        )
    ).scalar_one()
    if pending:
        return
    db.add(
        Job(
            id=uuid.uuid4(),
            project_id=upload.project_id,
            upload_id=upload.id,
            capability="cpu",
            stage="upload_cleanup",
            state="queued",
            priority=5,
            input_manifest={
                "uploadId": str(upload.id),
                "projectId": str(upload.project_id),
                "key": upload.immutable_key,
                "stagingKey": upload.staging_key,
            },
        )
    )
