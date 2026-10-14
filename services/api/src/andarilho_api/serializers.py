"""Row -> camelCase JSON shapes from docs/api-contract.md."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from andarilho_api.models import (
    CaptureSet,
    Job,
    JobEvent,
    Project,
    ReconstructionRun,
    Scene,
    SceneRevision,
    ShareLink,
    TourImport,
    Upload,
)


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def sid(value: Any) -> str | None:
    return None if value is None else str(value)


def etag(version: int) -> str:
    return f'"{version}"'


def project(p: Project, role: str, published: uuid.UUID | None = None) -> dict[str, Any]:
    """``published`` is the live-derived revision (see domain.publication)."""
    return {
        "id": str(p.id),
        "title": p.title,
        "lifecycle": p.lifecycle,
        "version": p.version,
        "role": role,
        "activeDraftRevisionId": sid(p.active_draft_revision_id),
        "publishedRevisionId": sid(published),
        "createdAt": iso(p.created_at),
        "updatedAt": iso(p.updated_at),
    }


def upload(u: Upload) -> dict[str, Any]:
    return {
        "id": str(u.id),
        "projectId": str(u.project_id),
        "mediaKind": u.media_kind,
        "filename": u.display_filename,
        "contentType": u.declared_content_type,
        "expectedBytes": u.expected_bytes,
        "status": u.status,
        "partSize": u.part_size,
        "partCount": u.part_count,
        "rejectionCode": u.rejection_code,
        "metadata": u.media_metadata or {},
        "expiresAt": iso(u.expires_at),
        "createdAt": iso(u.created_at),
    }


def capture_set(c: CaptureSet) -> dict[str, Any]:
    return {
        "id": str(c.id),
        "projectId": str(c.project_id),
        "inputs": [
            {
                "uploadId": i["uploadId"],
                "mediaKind": i["mediaKind"],
                "sha256": i["sha256"],
                "bytes": i["bytes"],
            }
            for i in c.inputs
        ],
        "floorPlanUploadId": sid(c.floor_plan_upload_id),
        "createdAt": iso(c.created_at),
    }


def _text(value: Any, limit: int) -> str | None:
    return str(value)[:limit] if value else None


def safe_error(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Expose only the documented error fields of a worker-recorded error."""
    if not value:
        return None
    return {
        "code": str(value.get("code") or "PROCESSING_FAILED")[:64],
        "message": _text(value.get("message"), 500) or "Processing failed.",
        "retryable": bool(value.get("retryable", False)),
        "suggestedAction": _text(value.get("suggestedAction"), 500),
        "requestId": _text(value.get("requestId"), 64),
    }


def job_summary(j: Job) -> dict[str, Any]:
    return {"id": str(j.id), "stage": j.stage, "state": j.state}


def run(r: ReconstructionRun, jobs: list[Job]) -> dict[str, Any]:
    ordered = sorted(jobs, key=lambda j: (j.created_at, str(j.id)))
    current = next((j for j in ordered if j.state in ("queued", "running", "retry_wait")), None)
    if current is None and ordered:
        current = ordered[-1]
    return {
        "id": str(r.id),
        "captureSetId": str(r.capture_set_id),
        "quality": r.quality,
        "status": r.status,
        "cancelRequested": r.cancel_requested_at is not None,
        "error": safe_error(r.error),
        "currentJobId": sid(current.id) if current else None,
        "currentStage": current.stage if current else None,
        "jobs": [job_summary(j) for j in ordered],
        "createdAt": iso(r.created_at),
        "finishedAt": iso(r.finished_at),
    }


def job(j: Job) -> dict[str, Any]:
    return {
        "id": str(j.id),
        "runId": sid(j.run_id),
        "stage": j.stage,
        "state": j.state,
        "attempts": j.attempts,
        "error": safe_error(j.error),
        "createdAt": iso(j.created_at),
        "updatedAt": iso(j.updated_at),
    }


def job_event(e: JobEvent) -> dict[str, Any]:
    return {
        "id": e.id,
        "stage": e.stage,
        "state": e.state,
        "message": e.message,
        "progress": e.progress or {},
        "createdAt": iso(e.created_at),
    }


def scene_ref(s: Scene) -> dict[str, Any]:
    return {
        "id": str(s.id),
        "version": s.version,
        "currentDraftRevisionId": sid(s.current_draft_revision_id),
    }


def scene(s: Scene) -> dict[str, Any]:
    return {
        "id": str(s.id),
        "projectId": str(s.project_id),
        "runId": str(s.run_id),
        "version": s.version,
        "currentDraftRevisionId": sid(s.current_draft_revision_id),
    }


def _findings(value: list[Any]) -> list[dict[str, str]]:
    return [
        {"code": str(item.get("code", ""))[:64], "message": str(item.get("message", ""))[:500]}
        for item in value or []
        if isinstance(item, dict)
    ]


def revision(r: SceneRevision) -> dict[str, Any]:
    return {
        "id": str(r.id),
        "revision": r.revision,
        "layoutRevision": r.layout_revision,
        "layout": r.layout,
        "manifest": r.manifest,
        "validationState": r.validation_state,
        "validationFindings": _findings(r.validation_findings),
        "approvalState": r.approval_state,
        "approvedAt": iso(r.approved_at),
        "createdAt": iso(r.created_at),
    }


def revision_summary(r: SceneRevision) -> dict[str, Any]:
    return {
        "id": str(r.id),
        "revision": r.revision,
        "layoutRevision": r.layout_revision,
        "validationState": r.validation_state,
        "approvalState": r.approval_state,
        "createdAt": iso(r.created_at),
    }


def share(s: ShareLink) -> dict[str, Any]:
    return {
        "id": str(s.id),
        "sceneRevisionId": str(s.scene_revision_id),
        "expiresAt": iso(s.expires_at),
        "revokedAt": iso(s.revoked_at),
        "createdAt": iso(s.created_at),
    }


def tour_import(t: TourImport) -> dict[str, Any]:
    return {
        "id": str(t.id),
        "projectId": str(t.project_id),
        "kind": t.kind,
        "provider": t.provider,
        "modelId": t.model_id,
        "canonicalUrl": t.canonical_url,
        "listingUrl": t.listing_url,
        "createdAt": iso(t.created_at),
    }
