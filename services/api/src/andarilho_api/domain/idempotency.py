"""Idempotent submission records (README section 12, "Run submission").

The key scope is ``(project_id, operation, idempotency_key)``. The canonical
request hash covers the operation target and the validated body. Replaying the
same key and body returns the stored response; a different body or caller is a
``409 IDEMPOTENCY_KEY_REUSED``. Records expire after ``IDEMPOTENCY_TTL_SECONDS``.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from andarilho_api.domain.canonical import sha256_hex
from andarilho_api.errors import ApiError, precondition_required, validation_failed
from andarilho_api.models import IdempotencyRecord

_KEY = re.compile(r"^[A-Za-z0-9._:-]{8,255}$")


def require_key(value: str | None) -> str:
    if value is None or value == "":
        raise precondition_required("Idempotency-Key")
    if not _KEY.match(value):
        raise validation_failed(
            "Idempotency-Key must be 8-255 characters of letters, digits, '.', '_', ':' or '-'."
        )
    return value


def request_hash(operation: str, target: str, body: dict[str, Any]) -> str:
    return sha256_hex({"operation": operation, "target": target, "body": body})


def key_reused() -> ApiError:
    return ApiError(
        409,
        "IDEMPOTENCY_KEY_REUSED",
        "This Idempotency-Key was already used for a different request.",
        suggested_action="Use a new Idempotency-Key for a new request.",
    )


def find(
    db: Session,
    *,
    project_id: uuid.UUID,
    operation: str,
    key: str,
    subject: str,
    req_hash: str,
) -> IdempotencyRecord | None:
    """Return a replayable record, raise on reuse, or ``None`` when the key is fresh."""
    record = db.execute(
        select(IdempotencyRecord).where(
            IdempotencyRecord.project_id == project_id,
            IdempotencyRecord.operation == operation,
            IdempotencyRecord.idempotency_key == key,
        )
    ).scalar_one_or_none()
    if record is None:
        return None
    if record.expires_at <= datetime.now(UTC):
        db.delete(record)
        db.flush()
        return None
    if record.subject != subject or record.request_hash != req_hash:
        raise key_reused()
    return record


def store(
    db: Session,
    *,
    project_id: uuid.UUID,
    operation: str,
    key: str,
    subject: str,
    req_hash: str,
    status: int,
    body: dict[str, Any],
    ttl_seconds: int,
) -> None:
    db.add(
        IdempotencyRecord(
            project_id=project_id,
            operation=operation,
            idempotency_key=key,
            subject=subject,
            request_hash=req_hash,
            response_status=status,
            response_body=body,
            expires_at=datetime.now(UTC) + timedelta(seconds=ttl_seconds),
        )
    )
