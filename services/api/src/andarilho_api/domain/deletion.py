"""Project deletion protocol, steps 1-2 (README section 13), in one transaction.

1. Mark the project ``deleting``, revoke every share and clear the published
   pointer so no new grants or mutations are accepted.
2. Cancel queued/retry_wait jobs; running jobs get ``cancel_requested_at`` and a
   bumped ``fencing_token`` so their heartbeats and fenced writes fail. Active
   runs get the run-level cancellation marker; idle ones are finalized and their
   reservations released.

Incomplete multipart uploads are marked ``aborted`` and returned so the caller
can abort them in storage after commit (best effort). A ``project_cleanup`` job
(capability ``cpu``, ``run_id`` NULL) is queued for the worker to reconcile and
delete everything under ``tenants/{project_id}/`` (steps 3-6).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from andarilho_api.domain import audit
from andarilho_api.domain.runs import cancel_jobs, finalize_if_idle
from andarilho_api.models import (
    ACTIVE_RUN_STATUSES,
    Job,
    Project,
    ReconstructionRun,
    ShareLink,
    Upload,
)
from andarilho_api.storage import tenant_prefix


def request_deletion(
    db: Session, project: Project, *, subject: str, now: datetime
) -> list[tuple[str, str]]:
    """Apply steps 1-2; return ``(staging_key, multipart_upload_id)`` pairs to abort."""
    project.lifecycle = "deleting"
    project.version += 1
    project.published_revision_id = None
    project.updated_at = now

    db.execute(
        update(ShareLink)
        .where(ShareLink.project_id == project.id, ShareLink.revoked_at.is_(None))
        .values(revoked_at=now)
        .execution_options(synchronize_session=False)
    )

    cancel_jobs(db, now=now, fence_running=True, project_id=project.id)
    runs = (
        db.execute(
            select(ReconstructionRun)
            .where(
                ReconstructionRun.project_id == project.id,
                ReconstructionRun.status.in_(ACTIVE_RUN_STATUSES),
            )
            .with_for_update()
        )
        .scalars()
        .all()
    )
    for run in runs:
        if run.cancel_requested_at is None:
            run.cancel_requested_at = now
        finalize_if_idle(db, run, now=now)

    to_abort: list[tuple[str, str]] = []
    uploads = (
        db.execute(
            select(Upload)
            .where(
                Upload.project_id == project.id,
                Upload.status.in_(("initiated", "uploaded_unverified")),
            )
            .with_for_update()
        )
        .scalars()
        .all()
    )
    for upload in uploads:
        if upload.status == "initiated" and upload.multipart_upload_id:
            to_abort.append((upload.staging_key, upload.multipart_upload_id))
        upload.status = "aborted"
        upload.multipart_upload_id = None

    db.add(
        Job(
            id=uuid.uuid4(),
            project_id=project.id,
            run_id=None,
            capability="cpu",
            stage="project_cleanup",
            state="queued",
            priority=10,
            input_manifest={
                "projectId": str(project.id),
                "prefix": tenant_prefix(project.id),
                "requestedAt": now.isoformat(),
            },
        )
    )
    audit.record(
        db,
        project_id=project.id,
        actor=subject,
        action="project.deletion_requested",
        resource_type="project",
        resource_id=project.id,
        data={"cancelledRuns": [str(r.id) for r in runs]},
    )
    return to_abort
