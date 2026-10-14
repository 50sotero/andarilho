"""Row factories for tests (direct DB inserts where the worker would act)."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from andarilho_api.domain.canonical import sha256_hex
from andarilho_api.models import (
    CaptureSet,
    Job,
    ReconstructionRun,
    Scene,
    SceneRevision,
    Upload,
    UsageLedger,
)

REPO = Path(__file__).resolve().parents[3]
LAYOUT_FIXTURE = (
    REPO / "packages" / "scene-contract" / "fixtures" / "synthetic-two-room.layout.json"
)


def layout() -> dict[str, Any]:
    return json.loads(LAYOUT_FIXTURE.read_text(encoding="utf-8"))


def create_project(client, auth, sub: str = "user-a", title: str = "Home") -> dict[str, Any]:
    response = client.post("/v1/projects", json={"title": title}, headers=auth(sub))
    assert response.status_code == 201, response.text
    return response.json()


def verified_upload(
    db: Session, project_id: str, *, kind: str = "video", sub: str = "user-a", size: int = 1000
) -> str:
    upload_id = uuid.uuid4()
    content_type = {"video": "video/mp4", "photo": "image/jpeg", "floor_plan": "application/pdf"}
    db.add(
        Upload(
            id=upload_id,
            project_id=uuid.UUID(project_id),
            created_by=sub,
            media_kind=kind,
            display_filename=f"{kind}.bin",
            declared_content_type=content_type[kind],
            expected_bytes=size,
            actual_bytes=size,
            staging_key=f"tenants/{project_id}/staging/{upload_id}",
            immutable_key=f"tenants/{project_id}/originals/{upload_id}",
            part_size=16 * 1024 * 1024,
            part_count=1,
            status="verified",
            sha256=hashlib.sha256(str(upload_id).encode()).hexdigest(),
            media_metadata={"durationSeconds": 12.5} if kind == "video" else {},
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )
    )
    db.commit()
    return str(upload_id)


def capture_set(db: Session, project_id: str, sub: str = "user-a") -> str:
    cs_id = uuid.uuid4()
    db.add(
        CaptureSet(
            id=cs_id,
            project_id=uuid.UUID(project_id),
            created_by=sub,
            inputs=[],
            content_sha256=sha256_hex([]),
        )
    )
    db.commit()
    return str(cs_id)


def manifest(
    project_id: str,
    scene_id: str,
    revision_id: str,
    run_id: str,
    cs_id: str,
    *,
    mode: str = "approved_collision",
    assets: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    assets = assets or {}

    def asset(name: str, fmt: str) -> dict[str, Any]:
        data = assets.get(name, name.encode())
        return {
            "key": f"tenants/{project_id}/scenes/{revision_id}/{name}.{fmt}",
            "format": fmt,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    return {
        "schemaVersion": "1.0",
        "sceneId": scene_id,
        "sceneRevisionId": revision_id,
        "revision": 1,
        "runId": run_id,
        "renderMode": "gaussian",
        "units": "m",
        "upAxis": "+Y",
        "cameraForward": "-Z",
        "scaleStatus": "estimated",
        "worldFromDecodedRender": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "geometrySpace": "world",
        "bounds": {"min": [-5, 0, -4], "max": [5, 3, 4]},
        "spawn": {"feet": [2, 0, 0], "yawRadians": 0, "floorId": "floor-0"},
        "floors": [{"id": "floor-0", "elevation": 0}],
        "assets": {
            "render": asset("scene", "ply"),
            "collision": asset("collision", "glb"),
            "poster": asset("poster", "webp"),
        },
        "navigation": {
            "mode": mode,
            "layoutRevision": 1,
            "approved": False,
            "playerRadius": 0.25,
            "playerHeight": 1.7,
            "eyeHeight": 1.6,
        },
        "quality": {
            "registeredViews": 0,
            "selectedViews": 0,
            "disconnectedComponents": 1,
            "warnings": [],
        },
        "provenance": {
            "recipeDigest": "synthetic",
            "captureSetId": cs_id,
            "geometrySource": "synthetic_fixture",
        },
    }


def scene(
    db: Session,
    project_id: str,
    *,
    sub: str = "user-a",
    validation: str = "valid",
    mode: str = "approved_collision",
    with_manifest: bool = True,
) -> dict[str, str]:
    """A succeeded synthetic run with one scene and one draft revision."""
    cs_id = capture_set(db, project_id, sub)
    pid = uuid.UUID(project_id)
    run_id, scene_id, rev_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db.add(
        ReconstructionRun(
            id=run_id,
            project_id=pid,
            capture_set_id=uuid.UUID(cs_id),
            created_by=sub,
            recipe_name="synthetic",
            recipe_digest="digest",
            quality="synthetic",
            resource_budget={},
            status="succeeded",
            finished_at=datetime.now(UTC),
        )
    )
    db.flush()
    db.add(Scene(id=scene_id, project_id=pid, run_id=run_id))
    db.flush()
    doc = (
        manifest(project_id, str(scene_id), str(rev_id), str(run_id), cs_id, mode=mode)
        if with_manifest
        else None
    )
    db.add(
        SceneRevision(
            id=rev_id,
            scene_id=scene_id,
            project_id=pid,
            run_id=run_id,
            revision=1,
            layout_revision=1,
            layout=layout(),
            manifest=doc,
            manifest_sha256=sha256_hex(doc) if doc else None,
            validation_state=validation,
            created_by=sub,
        )
    )
    db.flush()
    db.get(Scene, scene_id).current_draft_revision_id = rev_id  # type: ignore[union-attr]
    from andarilho_api.models import Project

    db.get(Project, pid).active_draft_revision_id = rev_id  # type: ignore[union-attr]
    db.commit()
    return {"sceneId": str(scene_id), "revisionId": str(rev_id), "runId": str(run_id)}


def active_run(
    db: Session, project_id: str, *, job_states: tuple[str, ...] = ("queued",), sub: str = "user-a"
) -> dict[str, Any]:
    """A running run whose jobs are in the given states (one job per stage)."""
    cs_id = capture_set(db, project_id, sub)
    pid = uuid.UUID(project_id)
    run_id = uuid.uuid4()
    db.add(
        ReconstructionRun(
            id=run_id,
            project_id=pid,
            capture_set_id=uuid.UUID(cs_id),
            created_by=sub,
            recipe_name="preview",
            recipe_digest="digest",
            quality="preview",
            resource_budget={},
            status="running",
        )
    )
    db.flush()
    db.add(
        UsageLedger(project_id=pid, run_id=run_id, reserved_gpu_seconds=10, reserved_cpu_seconds=10)
    )
    stages = ["validating", "preparing", "camera_reconstruction", "training"]
    job_ids = []
    for stage, state in zip(stages, job_states, strict=False):
        job_id = uuid.uuid4()
        running = state == "running"
        db.add(
            Job(
                id=job_id,
                project_id=pid,
                run_id=run_id,
                capability="cpu",
                stage=stage,
                state=state,
                lease_owner="worker-1" if running else None,
                lease_expires_at=datetime.now(UTC) + timedelta(seconds=90) if running else None,
                fencing_token=1 if running else 0,
            )
        )
        job_ids.append(str(job_id))
    db.commit()
    return {"runId": str(run_id), "jobIds": job_ids, "captureSetId": cs_id}
