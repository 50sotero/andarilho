"""Scenes: review, layout revisions, approval, delivery, shares and public delivery."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from andarilho_api import serializers as ser
from andarilho_api.auth import current_subject
from andarilho_api.db import get_db
from andarilho_api.deps import check_if_match, get_settings_dep, get_storage, utcnow
from andarilho_api.domain import audit
from andarilho_api.domain.membership import OWNER, WRITERS, load_project, load_scoped, lock_row
from andarilho_api.domain.publication import published_revision_id
from andarilho_api.domain.revisions import (
    build_envelope,
    hash_token,
    manifest_sha256,
    new_share_token,
    validated_manifest,
)
from andarilho_api.errors import (
    ApiError,
    invalid_state,
    not_found,
    project_deleting,
    validation_failed,
    version_conflict,
)
from andarilho_api.models import (
    CaptureSet,
    Job,
    Project,
    ReconstructionRun,
    Scene,
    SceneRevision,
    ShareLink,
    Upload,
)
from andarilho_api.ratelimit import RateLimiter
from andarilho_api.schemas.api import LayoutSave, PublicDeliveryRequest, RevisionRef, ShareCreate
from andarilho_api.settings import Settings
from andarilho_api.storage import Storage

logger = logging.getLogger("andarilho_api.scenes")
INSPECTED_CONTENT_TYPES = {"pdf": "application/pdf", "png": "image/png", "jpeg": "image/jpeg"}

router = APIRouter(tags=["scenes"])
public_router = APIRouter(tags=["public"])

Subject = Annotated[str, Depends(current_subject)]
Db = Annotated[Session, Depends(get_db)]
Store = Annotated[Storage, Depends(get_storage)]
Config = Annotated[Settings, Depends(get_settings_dep)]


def _revision_of(db: Session, scene: Scene, revision_id: uuid.UUID) -> SceneRevision:
    revision = db.execute(
        select(SceneRevision).where(
            SceneRevision.id == revision_id,
            SceneRevision.scene_id == scene.id,
            SceneRevision.project_id == scene.project_id,
        )
    ).scalar_one_or_none()
    if revision is None:
        raise not_found("scene revision")
    return revision


def _floor_plan(db: Session, scene: Scene, store: Storage, ttl: int) -> dict | None:
    """A short-lived private URL for the run's verified floor plan (review only)."""
    upload = db.execute(
        select(Upload)
        .join(CaptureSet, CaptureSet.floor_plan_upload_id == Upload.id)
        .join(ReconstructionRun, ReconstructionRun.capture_set_id == CaptureSet.id)
        .where(
            ReconstructionRun.id == scene.run_id,
            ReconstructionRun.project_id == scene.project_id,
            Upload.project_id == scene.project_id,
            Upload.status == "verified",
        )
    ).scalar_one_or_none()
    if upload is None or upload.immutable_key is None:
        return None
    return {
        "uploadId": str(upload.id),
        # The inspected signature, not the client's declared type, decides how it renders.
        "contentType": INSPECTED_CONTENT_TYPES.get(
            (upload.media_metadata or {}).get("format"), upload.declared_content_type
        ),
        "url": store.presign_get(upload.immutable_key, ttl, upload.storage_version),
        "expiresAt": (utcnow() + timedelta(seconds=ttl)).isoformat(),
    }


@router.get("/scenes/{scene_id}")
def get_scene(
    scene_id: uuid.UUID, response: Response, subject: Subject, db: Db, store: Store, config: Config
):
    scene = load_scoped(db, Scene, scene_id, subject, "scene")
    # Draft layouts, findings and history are review material: owners/editors only.
    load_project(db, scene.project_id, subject, roles=WRITERS)
    revisions = (
        db.execute(
            select(SceneRevision)
            .where(SceneRevision.scene_id == scene.id, SceneRevision.project_id == scene.project_id)
            .order_by(SceneRevision.revision.desc())
        )
        .scalars()
        .all()
    )
    current = next((r for r in revisions if r.id == scene.current_draft_revision_id), None)
    response.headers["ETag"] = ser.etag(scene.version)
    return {
        "scene": ser.scene(scene),
        "revision": ser.revision(current) if current else None,
        "revisions": [ser.revision_summary(r) for r in revisions],
        # The original plan is private source material: owners/editors only.
        "floorPlan": _floor_plan(db, scene, store, config.delivery_url_ttl_seconds),
    }


def _lock_scene(db: Session, scene_id: uuid.UUID, subject: str, roles: tuple[str, ...]):
    scoped = load_scoped(db, Scene, scene_id, subject, "scene")
    project, role = load_project(
        db, scoped.project_id, subject, roles=roles, lock=True, require_active=True
    )
    return project, lock_row(db, Scene, scoped.id)


def _publication_changed(
    db: Session, revision_ids: list[uuid.UUID | None], *, skip: uuid.UUID | None = None
) -> None:
    """Bump the version of each scene holding one of these revisions (except ``skip``).

    Publishing is guarded by the scene ETag: a tab that read a scene before its
    revision was published or unpublished must get a conflict, not publish over it.
    """
    ids = {r for r in revision_ids if r is not None}
    if not ids:
        return
    holders = select(SceneRevision.scene_id).where(SceneRevision.id.in_(ids))
    for held in db.execute(
        select(Scene).where(Scene.id.in_(holders), Scene.id != skip).with_for_update()
    ).scalars():
        held.version += 1


@router.put("/scenes/{scene_id}/layout", status_code=202)
def save_layout(
    scene_id: uuid.UUID,
    body: LayoutSave,
    request: Request,
    response: Response,
    subject: Subject,
    db: Db,
):
    project, scene = _lock_scene(db, scene_id, subject, WRITERS)
    check_if_match(request, scene.version, "scene")
    if scene.current_draft_revision_id != body.base_revision_id:
        raise version_conflict("scene")
    base = _revision_of(db, scene, body.base_revision_id)
    next_number = db.execute(
        select(func.coalesce(func.max(SceneRevision.revision), 0)).where(
            SceneRevision.scene_id == scene.id
        )
    ).scalar_one()
    revision = SceneRevision(
        id=uuid.uuid4(),
        scene_id=scene.id,
        project_id=scene.project_id,
        run_id=scene.run_id,
        base_revision_id=base.id,
        revision=int(next_number) + 1,
        layout_revision=base.layout_revision + 1,
        layout=body.layout.model_dump(by_alias=True, mode="json"),
        manifest=None,
        validation_state="pending",
        approval_state="draft",
        created_by=subject,
    )
    db.add(revision)
    db.flush()
    scene.current_draft_revision_id = revision.id
    scene.version += 1
    scene.updated_at = utcnow()
    project.active_draft_revision_id = revision.id
    job = Job(
        id=uuid.uuid4(),
        project_id=scene.project_id,
        run_id=scene.run_id,
        scene_revision_id=revision.id,
        capability="cpu",
        stage="geometry_proposal",
        state="queued",
        input_manifest={
            "sceneRevisionId": str(revision.id),
            "sceneId": str(scene.id),
            "baseRevisionId": str(base.id),
            "layoutRevision": revision.layout_revision,
            "layoutSource": "user",
        },
    )
    db.add(job)
    db.commit()
    response.headers["ETag"] = ser.etag(scene.version)
    return {"sceneRevisionId": str(revision.id), "jobId": str(job.id), "version": scene.version}


@router.post("/scenes/{scene_id}/approve")
def approve_revision(
    scene_id: uuid.UUID,
    body: RevisionRef,
    request: Request,
    response: Response,
    subject: Subject,
    db: Db,
):
    _, scene = _lock_scene(db, scene_id, subject, WRITERS)
    check_if_match(request, scene.version, "scene")
    if scene.current_draft_revision_id != body.scene_revision_id:
        raise version_conflict("scene")
    revision = lock_row(db, SceneRevision, _revision_of(db, scene, body.scene_revision_id).id)
    if revision.approval_state != "approved":
        if revision.validation_state != "valid":
            raise invalid_state(
                "Only a validated revision can be approved.",
                "Wait for validation to finish or fix the reported findings.",
            )
        manifest = validated_manifest(revision)
        if manifest.navigation.mode != "approved_collision":
            raise invalid_state("This revision has no approvable collision geometry.")
        revision.approval_state = "approved"
        revision.approved_by = subject
        revision.approved_at = utcnow()
        scene.version += 1
        scene.updated_at = utcnow()
        audit.record(
            db,
            project_id=scene.project_id,
            actor=subject,
            action="scene_revision.approved",
            resource_type="scene_revision",
            resource_id=revision.id,
            data={"sceneId": str(scene.id), "revision": revision.revision},
        )
        db.commit()
    response.headers["ETag"] = ser.etag(scene.version)
    return {
        "sceneRevisionId": str(revision.id),
        "approvalState": "approved",
        "version": scene.version,
    }


@router.post("/scenes/{scene_id}/delivery")
def scene_delivery(
    scene_id: uuid.UUID,
    body: RevisionRef,
    subject: Subject,
    db: Db,
    storage: Store,
    settings: Config,
):
    scene = load_scoped(db, Scene, scene_id, subject, "scene")
    project, role = load_project(db, scene.project_id, subject)
    if project.lifecycle != "active":
        raise project_deleting()
    revision = _revision_of(db, scene, body.scene_revision_id)
    if revision.validation_state != "valid":
        raise invalid_state("This scene revision has not passed validation.")
    if role not in WRITERS and (
        revision.approval_state != "approved" or published_revision_id(db, project) != revision.id
    ):
        # Viewers walk only the revision currently published through a live share link;
        # drafts and formerly published revisions stay with the reviewers.
        raise not_found("scene revision")
    envelope = build_envelope(storage, settings, revision)
    if revision.manifest_sha256 is None:
        revision.manifest_sha256 = manifest_sha256(revision)
        db.commit()
    return envelope


@router.post("/scenes/{scene_id}/shares", status_code=201)
def create_share(
    scene_id: uuid.UUID,
    body: ShareCreate,
    request: Request,
    response: Response,
    subject: Subject,
    db: Db,
):
    project, scene = _lock_scene(db, scene_id, subject, OWNER)
    check_if_match(request, scene.version, "scene")
    revision = _revision_of(db, scene, body.scene_revision_id)
    if revision.approval_state != "approved":
        raise invalid_state(
            "Only an approved revision can be shared.", "Approve the revision first."
        )
    expires_at = body.expires_at
    if expires_at is not None:
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at <= utcnow():
            raise validation_failed("expiresAt must be in the future.")
    token, digest = new_share_token()
    share = ShareLink(
        id=uuid.uuid4(),
        project_id=scene.project_id,
        scene_revision_id=revision.id,
        token_hash=digest,
        created_by=subject,
        expires_at=expires_at,
    )
    # What viewers see is derived from live links: once the stored pointer's links
    # expire, an older live link's revision is the publication. Both stop being shown.
    effective = published_revision_id(db, project)
    stored = project.published_revision_id
    db.add(share)
    project.published_revision_id = revision.id
    project.updated_at = utcnow()
    # Publishing changes what the scene's viewers see: a tab still holding the old
    # ETag must get a conflict instead of silently publishing over this.
    scene.version += 1
    _publication_changed(db, [effective, stored], skip=scene.id)
    if revision.manifest_sha256 is None:
        revision.manifest_sha256 = manifest_sha256(revision)
    audit.record(
        db,
        project_id=scene.project_id,
        actor=subject,
        action="share.created",
        resource_type="share",
        resource_id=share.id,
        data={"sceneRevisionId": str(revision.id), "expiresAt": ser.iso(expires_at)},
    )
    db.commit()
    db.refresh(share)
    response.headers["ETag"] = ser.etag(scene.version)
    return {"share": ser.share(share), "token": token, "path": f"/s/{token}"}


@router.delete("/shares/{share_id}", status_code=204)
def revoke_share(share_id: uuid.UUID, subject: Subject, db: Db):
    scoped = load_scoped(db, ShareLink, share_id, subject, "share")
    project, _ = load_project(db, scoped.project_id, subject, roles=OWNER, lock=True)
    share = lock_row(db, ShareLink, scoped.id)
    if share.revoked_at is None:
        now = utcnow()
        # Viewers see the publication derived from live links, which differs from the
        # stored pointer once its links expire; compare that before and after.
        before = published_revision_id(db, project)
        share.revoked_at = now
        db.flush()
        if project.published_revision_id == share.scene_revision_id:
            # Point at the newest revision that still has a live link (this one if it
            # has others); only clear the pointer when no public link remains at all.
            live = (
                ShareLink.project_id == project.id,
                ShareLink.revoked_at.is_(None),
                or_(ShareLink.expires_at.is_(None), ShareLink.expires_at > now),
            )
            same = db.execute(
                select(func.count()).where(
                    *live, ShareLink.scene_revision_id == share.scene_revision_id
                )
            ).scalar_one()
            if same == 0:
                project.published_revision_id = db.execute(
                    select(ShareLink.scene_revision_id)
                    .where(*live)
                    .order_by(ShareLink.created_at.desc())
                    .limit(1)
                ).scalar_one_or_none()
                project.updated_at = now
        after = published_revision_id(db, project)
        if after != before:
            _publication_changed(db, [before, after])
        audit.record(
            db,
            project_id=project.id,
            actor=subject,
            action="share.revoked",
            resource_type="share",
            resource_id=share.id,
            data={"sceneRevisionId": str(share.scene_revision_id)},
        )
        db.commit()
    return Response(status_code=204)


def _client_key(request: Request, settings: Settings) -> str:
    if settings.public_delivery_trust_forwarded_for:
        # The nearest trusted hop (the web proxy) appends the visitor it observed;
        # earlier entries are client-supplied and must not choose the bucket.
        forwarded = request.headers.get("x-forwarded-for", "")
        nearest = forwarded.split(",")[-1].strip()
        if nearest:
            return nearest[:64]
    return request.client.host if request.client else "unknown"


@public_router.post("/public/scenes/delivery")
def public_delivery(
    body: PublicDeliveryRequest, request: Request, db: Db, storage: Store, settings: Config
):
    limiter: RateLimiter = request.app.state.public_rate_limiter
    retry_after = limiter.check(_client_key(request, settings))
    if retry_after is not None:
        raise ApiError(
            429,
            "RATE_LIMITED",
            "Too many requests.",
            retryable=True,
            suggested_action="Wait a moment and try again.",
            headers={"Retry-After": str(retry_after)},
        )
    unavailable = not_found("shared scene")
    row = db.execute(
        select(ShareLink, SceneRevision, Project)
        .join(
            SceneRevision,
            (SceneRevision.id == ShareLink.scene_revision_id)
            & (SceneRevision.project_id == ShareLink.project_id),
        )
        .join(Project, Project.id == ShareLink.project_id)
        .where(ShareLink.token_hash == hash_token(body.token))
    ).one_or_none()
    if row is None:
        raise unavailable
    share, revision, project = row
    now = utcnow()
    if (
        share.revoked_at is not None
        or (share.expires_at is not None and share.expires_at <= now)
        or project.lifecycle != "active"
        or revision.approval_state != "approved"
        or revision.validation_state != "valid"
        or revision.manifest is None
    ):
        raise unavailable
    try:
        return build_envelope(storage, settings, revision)
    except ApiError:
        raise unavailable from None
