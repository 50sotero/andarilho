"""Audit trail for publication, revocation, deletion and approval (README section 13)."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from andarilho_api.models import AuditEvent


def record(
    db: Session,
    *,
    project_id: uuid.UUID,
    actor: str,
    action: str,
    resource_type: str,
    resource_id: uuid.UUID | str,
    data: dict[str, Any] | None = None,
) -> None:
    """Add an audit row to the caller's transaction. Never put tokens or URLs in ``data``."""
    db.add(
        AuditEvent(
            project_id=project_id,
            actor_subject=actor,
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id),
            data=data or {},
        )
    )
