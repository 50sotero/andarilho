"""Upload intake limits (README section 2). Checked before any multipart upload exists."""

from __future__ import annotations

import math
import uuid

from sqlalchemy import and_, func, not_, or_, select
from sqlalchemy.orm import Session

from andarilho_api.errors import ApiError, validation_failed
from andarilho_api.models import Upload
from andarilho_api.settings import Settings

ALLOWED_CONTENT_TYPES: dict[str, frozenset[str]] = {
    "video": frozenset({"video/mp4", "video/quicktime"}),
    "photo": frozenset({"image/jpeg", "image/png"}),
    "floor_plan": frozenset({"image/png", "image/jpeg", "application/pdf"}),
}
# Uploads that still occupy quota.
COUNTED_STATUSES = ("initiated", "uploaded_unverified", "verified")


def too_large(message: str) -> ApiError:
    return ApiError(413, "PAYLOAD_TOO_LARGE", message, suggested_action="Upload a smaller file.")


def quota_exceeded(message: str) -> ApiError:
    return ApiError(
        413, "QUOTA_EXCEEDED", message, suggested_action="Remove unused uploads or files."
    )


def max_bytes_for(settings: Settings, media_kind: str) -> int:
    return {
        "video": settings.video_max_bytes,
        "photo": settings.upload_max_photo_bytes,
        "floor_plan": settings.upload_max_floor_plan_bytes,
    }[media_kind]


def check_upload(
    db: Session,
    settings: Settings,
    *,
    project_id: uuid.UUID,
    media_kind: str,
    content_type: str,
    size: int,
) -> tuple[int, int]:
    """Validate an upload request; return ``(part_size, part_count)``.

    The caller must hold the project row lock so concurrent requests cannot both
    pass the quota check.
    """
    if content_type not in ALLOWED_CONTENT_TYPES[media_kind]:
        raise validation_failed("This content type is not supported for this kind of media.")
    if size > max_bytes_for(settings, media_kind) or size > settings.upload_max_total_bytes:
        raise too_large("The file exceeds the size limit for this kind of media.")
    part_size = settings.upload_part_size
    part_count = max(1, math.ceil(size / part_size))
    if part_count > settings.upload_max_parts:
        raise too_large("The file needs more upload parts than allowed.")

    # Initiated uploads past their expiry no longer hold quota.
    holding = and_(
        Upload.project_id == project_id,
        or_(
            Upload.status.in_(COUNTED_STATUSES),
            # Rejected or removed uploads hold quota until the worker deletes their bytes.
            and_(Upload.status.in_(("rejected", "aborted")), Upload.immutable_key.is_not(None)),
        ),
        not_(and_(Upload.status == "initiated", Upload.expires_at <= func.now())),
    )
    used = db.execute(
        select(func.coalesce(func.sum(Upload.expected_bytes), 0)).where(holding)
    ).scalar_one()
    if int(used) + size > settings.upload_max_total_bytes:
        raise quota_exceeded("The project's total upload quota would be exceeded.")
    if media_kind == "photo":
        photos = db.execute(
            select(func.count()).where(holding, Upload.media_kind == "photo")
        ).scalar_one()
        if int(photos) >= settings.upload_max_photos:
            raise quota_exceeded("The project already has the maximum number of photos.")
    return part_size, part_count
