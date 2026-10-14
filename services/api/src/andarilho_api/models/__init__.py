"""SQLAlchemy models. The API owns these tables and their Alembic migrations.

Enumerations are stored as text with CHECK constraints so that the worker (which
does not import this package) can use plain SQL against the same values.
Tenant relationships use composite foreign keys on ``(project_id, id)``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

PROJECT_LIFECYCLES = ("active", "deleting", "deleted")
MEMBER_ROLES = ("owner", "editor", "viewer")
MEDIA_KINDS = ("video", "photo", "floor_plan")
UPLOAD_STATUSES = (
    "initiated",
    "uploaded_unverified",
    "verified",
    "rejected",
    "aborted",
    "expired",
)
RUN_QUALITIES = ("preview", "standard", "synthetic")
RUN_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")
ACTIVE_RUN_STATUSES = ("queued", "running")
CAPABILITIES = ("cpu", "gpu")
# Canonical user-visible pipeline stages (README section 6), in order.
RUN_STAGES = (
    "validating",
    "preparing",
    "camera_reconstruction",
    "training",
    "normalizing",
    "geometry_proposal",
    "packaging",
    "scene_validation",
)
# Internal maintenance stages that are not part of a reconstruction run.
MAINTENANCE_STAGES = ("upload_inspection", "project_cleanup")
JOB_STAGES = RUN_STAGES + MAINTENANCE_STAGES
JOB_STATES = ("queued", "running", "retry_wait", "succeeded", "failed", "cancelled")
TERMINAL_JOB_STATES = ("succeeded", "failed", "cancelled")
VALIDATION_STATES = ("pending", "valid", "invalid")
APPROVAL_STATES = ("draft", "approved")
LEDGER_STATES = ("reserved", "released", "settled")


def _in(column: str, values: tuple[str, ...]) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({quoted})"


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _created() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def _updated() -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )
    type_annotation_map = {dict[str, Any]: JSONB, list[Any]: JSONB}


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = _uuid_pk()
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    owner_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    lifecycle: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    active_draft_revision_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    published_revision_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    cleanup_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _updated()

    __table_args__ = (
        CheckConstraint(_in("lifecycle", PROJECT_LIFECYCLES), name="lifecycle"),
        CheckConstraint("length(title) > 0", name="title_not_empty"),
        Index("ix_projects_owner_created", "owner_subject", "created_at", "id"),
        # Draft/published pointers must reference revisions of this same project.
        ForeignKeyConstraint(
            ["id", "active_draft_revision_id"],
            ["scene_revisions.project_id", "scene_revisions.id"],
            name="fk_projects_active_draft_revision",
            use_alter=True,
        ),
        ForeignKeyConstraint(
            ["id", "published_revision_id"],
            ["scene_revisions.project_id", "scene_revisions.id"],
            name="fk_projects_published_revision",
            use_alter=True,
        ),
    )


class ProjectMember(Base):
    __tablename__ = "project_members"

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = _created()

    __table_args__ = (
        UniqueConstraint("project_id", "subject", name="uq_project_members_project_subject"),
        CheckConstraint(_in("role", MEMBER_ROLES), name="role"),
        Index("ix_project_members_subject", "subject"),
    )


class Upload(Base):
    __tablename__ = "uploads"

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    media_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    display_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    declared_content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    expected_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    staging_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    immutable_key: Mapped[str | None] = mapped_column(String(512), unique=True)
    storage_version: Mapped[str | None] = mapped_column(String(255))
    multipart_upload_id: Mapped[str | None] = mapped_column(String(1024))
    part_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    part_count: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_bytes: Mapped[int | None] = mapped_column(BigInteger)
    sha256: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="initiated")
    rejection_code: Mapped[str | None] = mapped_column(String(64))
    media_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _updated()

    __table_args__ = (
        UniqueConstraint("project_id", "id", name="uq_uploads_project_id_id"),
        CheckConstraint(_in("media_kind", MEDIA_KINDS), name="media_kind"),
        CheckConstraint(_in("status", UPLOAD_STATUSES), name="status"),
        CheckConstraint("expected_bytes > 0", name="expected_bytes_positive"),
        CheckConstraint("actual_bytes IS NULL OR actual_bytes > 0", name="actual_bytes_positive"),
        CheckConstraint("part_size > 0 AND part_count > 0", name="parts_positive"),
        CheckConstraint(
            "status NOT IN ('uploaded_unverified', 'verified') OR immutable_key IS NOT NULL",
            name="frozen_has_immutable_key",
        ),
        CheckConstraint("status <> 'verified' OR sha256 IS NOT NULL", name="verified_has_hash"),
        Index("ix_uploads_project_status", "project_id", "status"),
    )


class CaptureSet(Base):
    __tablename__ = "capture_sets"

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    # Immutable snapshot: [{uploadId, mediaKind, key, storageVersion, sha256, bytes, metadata}]
    inputs: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)
    floor_plan_upload_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    capture_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = _created()

    __table_args__ = (
        UniqueConstraint("project_id", "id", name="uq_capture_sets_project_id_id"),
        ForeignKeyConstraint(
            ["project_id", "floor_plan_upload_id"],
            ["uploads.project_id", "uploads.id"],
            name="fk_capture_sets_floor_plan_upload",
        ),
    )


class ReconstructionRun(Base):
    __tablename__ = "reconstruction_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    capture_set_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    retry_of_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    recipe_name: Mapped[str] = mapped_column(String(64), nullable=False)
    recipe_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    quality: Mapped[str] = mapped_column(String(16), nullable=False)
    include_supplementary_photos: Mapped[bool] = mapped_column(nullable=False, default=True)
    resource_budget: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _updated()

    __table_args__ = (
        UniqueConstraint("project_id", "id", name="uq_reconstruction_runs_project_id_id"),
        CheckConstraint(_in("quality", RUN_QUALITIES), name="quality"),
        CheckConstraint(_in("status", RUN_STATUSES), name="status"),
        ForeignKeyConstraint(
            ["project_id", "capture_set_id"],
            ["capture_sets.project_id", "capture_sets.id"],
            name="fk_reconstruction_runs_capture_set",
        ),
        # At most one active expensive run per project in the first release.
        Index(
            "uq_reconstruction_runs_one_active",
            "project_id",
            unique=True,
            postgresql_where=text(f"status IN {ACTIVE_RUN_STATUSES!r}"),
        ),
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    upload_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    scene_revision_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    capability: Mapped[str] = mapped_column(String(16), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    lease_owner: Mapped[str | None] = mapped_column(String(255))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fencing_token: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    input_manifest: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    result_manifest: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _updated()

    __table_args__ = (
        CheckConstraint(_in("capability", CAPABILITIES), name="capability"),
        CheckConstraint(_in("stage", JOB_STAGES), name="stage"),
        CheckConstraint(_in("state", JOB_STATES), name="state"),
        CheckConstraint("attempts >= 0 AND max_attempts >= 1", name="attempts"),
        CheckConstraint(
            f"NOT ({_in('stage', RUN_STAGES)}) OR run_id IS NOT NULL", name="run_stage_has_run"
        ),
        CheckConstraint(
            "stage <> 'upload_inspection' OR upload_id IS NOT NULL", name="inspection_has_upload"
        ),
        CheckConstraint(
            "state <> 'running' OR (lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="running_has_lease",
        ),
        ForeignKeyConstraint(
            ["project_id", "run_id"],
            ["reconstruction_runs.project_id", "reconstruction_runs.id"],
            name="fk_jobs_run",
        ),
        ForeignKeyConstraint(
            ["project_id", "upload_id"], ["uploads.project_id", "uploads.id"], name="fk_jobs_upload"
        ),
        ForeignKeyConstraint(
            ["project_id", "scene_revision_id"],
            ["scene_revisions.project_id", "scene_revisions.id"],
            name="fk_jobs_scene_revision",
        ),
        # A stage is inserted at most once per run (or per layout revision): this
        # makes next-stage insertion races fail instead of duplicating work.
        Index(
            "uq_jobs_run_stage",
            "run_id",
            "stage",
            unique=True,
            postgresql_where=text("run_id IS NOT NULL AND scene_revision_id IS NULL"),
        ),
        Index(
            "uq_jobs_revision_stage",
            "scene_revision_id",
            "stage",
            unique=True,
            postgresql_where=text("scene_revision_id IS NOT NULL"),
        ),
        Index(
            "ix_jobs_claim",
            "capability",
            "priority",
            "created_at",
            postgresql_where=text("state = 'queued'"),
        ),
        Index("ix_jobs_lease", "lease_expires_at", postgresql_where=text("state = 'running'")),
    )


class JobEvent(Base):
    __tablename__ = "job_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    progress: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = _created()

    __table_args__ = (Index("ix_job_events_job_id_id", "job_id", "id"),)


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    object_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    storage_version: Mapped[str | None] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    provenance: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    validation_state: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    created_at: Mapped[datetime] = _created()

    __table_args__ = (
        UniqueConstraint("project_id", "id", name="uq_artifacts_project_id_id"),
        CheckConstraint("bytes > 0", name="bytes_positive"),
        CheckConstraint(_in("validation_state", VALIDATION_STATES), name="validation_state"),
        CheckConstraint("object_key LIKE 'tenants/' || project_id || '/%'", name="key_in_tenant"),
        ForeignKeyConstraint(
            ["project_id", "run_id"],
            ["reconstruction_runs.project_id", "reconstruction_runs.id"],
            name="fk_artifacts_run",
        ),
        Index("ix_artifacts_run_kind", "run_id", "kind"),
    )


class Scene(Base):
    __tablename__ = "scenes"

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    current_draft_revision_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _updated()

    __table_args__ = (
        UniqueConstraint("project_id", "id", name="uq_scenes_project_id_id"),
        UniqueConstraint("run_id", name="uq_scenes_run_id"),
        ForeignKeyConstraint(
            ["project_id", "run_id"],
            ["reconstruction_runs.project_id", "reconstruction_runs.id"],
            name="fk_scenes_run",
        ),
        ForeignKeyConstraint(
            ["project_id", "current_draft_revision_id"],
            ["scene_revisions.project_id", "scene_revisions.id"],
            name="fk_scenes_current_draft_revision",
            use_alter=True,
        ),
    )


class SceneRevision(Base):
    __tablename__ = "scene_revisions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    scene_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    base_revision_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    layout_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    layout: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    manifest: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    manifest_artifact_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    manifest_sha256: Mapped[str | None] = mapped_column(String(64))
    validation_state: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    validation_findings: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    approval_state: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")
    approved_by: Mapped[str | None] = mapped_column(String(255))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = _created()

    __table_args__ = (
        UniqueConstraint("project_id", "id", name="uq_scene_revisions_project_id_id"),
        UniqueConstraint("scene_id", "revision", name="uq_scene_revisions_scene_revision"),
        CheckConstraint(_in("validation_state", VALIDATION_STATES), name="validation_state"),
        CheckConstraint(_in("approval_state", APPROVAL_STATES), name="approval_state"),
        CheckConstraint(
            "approval_state <> 'approved' OR (validation_state = 'valid' "
            "AND manifest IS NOT NULL AND approved_by IS NOT NULL AND approved_at IS NOT NULL)",
            name="approval_gate",
        ),
        ForeignKeyConstraint(
            ["project_id", "scene_id"], ["scenes.project_id", "scenes.id"], name="fk_rev_scene"
        ),
        ForeignKeyConstraint(
            ["project_id", "run_id"],
            ["reconstruction_runs.project_id", "reconstruction_runs.id"],
            name="fk_rev_run",
        ),
        ForeignKeyConstraint(
            ["project_id", "manifest_artifact_id"],
            ["artifacts.project_id", "artifacts.id"],
            name="fk_rev_manifest_artifact",
        ),
    )


class ShareLink(Base):
    __tablename__ = "share_links"

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    scene_revision_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    token_hash: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False, unique=True)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created()

    __table_args__ = (
        ForeignKeyConstraint(
            ["project_id", "scene_revision_id"],
            ["scene_revisions.project_id", "scene_revisions.id"],
            name="fk_share_links_revision",
        ),
    )


class UsageLedger(Base):
    __tablename__ = "usage_ledger"

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    reserved_gpu_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    reserved_cpu_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_gpu_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    actual_cpu_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    storage_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="reserved")
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created()

    __table_args__ = (
        UniqueConstraint("run_id", name="uq_usage_ledger_run_id"),
        CheckConstraint(_in("state", LEDGER_STATES), name="state"),
        CheckConstraint(
            "reserved_gpu_seconds >= 0 AND reserved_cpu_seconds >= 0 "
            "AND actual_gpu_seconds >= 0 AND actual_cpu_seconds >= 0 AND storage_bytes >= 0",
            name="non_negative",
        ),
        ForeignKeyConstraint(
            ["project_id", "run_id"],
            ["reconstruction_runs.project_id", "reconstruction_runs.id"],
            name="fk_usage_ledger_run",
        ),
    )


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    response_status: Mapped[int] = mapped_column(Integer, nullable=False)
    response_body: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _created()
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "project_id", "operation", "idempotency_key", name="uq_idempotency_scope"
        ),
    )


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # No FK: audit records outlive project metadata removal.
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    actor_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(32), nullable=False)
    resource_id: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = _created()

    __table_args__ = (Index("ix_audit_events_project", "project_id", "id"),)
