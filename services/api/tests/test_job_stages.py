"""The jobs table accepts exactly the stages the models define, and no other stage."""

from __future__ import annotations

import uuid

import pytest
from factories import capture_set, verified_upload
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from andarilho_api.models import (
    JOB_STAGES,
    LEGACY_JOB_STAGES,
    MAINTENANCE_STAGES,
    RUN_STAGES,
    Job,
    Project,
    ReconstructionRun,
)


def _project(db) -> uuid.UUID:
    project = Project(id=uuid.uuid4(), title="Home", owner_subject="user-a")
    db.add(project)
    db.commit()
    return project.id


def _run(db, project_id: uuid.UUID) -> uuid.UUID:
    run = ReconstructionRun(
        id=uuid.uuid4(),
        project_id=project_id,
        capture_set_id=uuid.UUID(capture_set(db, str(project_id))),
        created_by="user-a",
        recipe_name="preview-v1",
        recipe_digest="digest",
        quality="preview",
        resource_budget={},
        status="running",
    )
    db.add(run)
    db.commit()
    return run.id


def test_artifact_cleanup_is_a_maintenance_stage(db):
    assert "artifact_cleanup" in MAINTENANCE_STAGES
    project_id = _project(db)
    # The worker queues it with no run and no revision; the constraint must accept that.
    db.add(Job(project_id=project_id, capability="cpu", stage="artifact_cleanup", state="queued"))
    db.commit()
    stored = db.execute(select(Job).where(Job.project_id == project_id)).scalar_one()
    assert stored.stage == "artifact_cleanup" and stored.run_id is None


def test_unknown_stages_are_rejected(db):
    project_id = _project(db)
    db.add(Job(project_id=project_id, capability="cpu", stage="artifact_sweep", state="queued"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_texturing_is_the_run_stage_after_geometry_proposal():
    assert RUN_STAGES.index("texturing") == RUN_STAGES.index("geometry_proposal") + 1
    assert RUN_STAGES.index("texturing") < RUN_STAGES.index("packaging")
    assert "training" not in RUN_STAGES
    assert LEGACY_JOB_STAGES == ("training",)
    assert set(LEGACY_JOB_STAGES) <= set(JOB_STAGES)


def test_every_stage_the_models_define_is_accepted(db):
    project_id = _project(db)
    run_id = _run(db, project_id)
    upload_id = uuid.UUID(verified_upload(db, str(project_id)))
    for stage in JOB_STAGES:
        in_run = stage in RUN_STAGES or stage in LEGACY_JOB_STAGES
        db.add(
            Job(
                project_id=project_id,
                run_id=run_id if in_run else None,
                upload_id=upload_id if stage.startswith("upload_") else None,
                capability="cpu",
                stage=stage,
                state="queued",
            )
        )
    db.commit()
    stored = db.execute(select(Job.stage).where(Job.project_id == project_id)).scalars().all()
    assert sorted(stored) == sorted(JOB_STAGES)


def test_a_run_stage_needs_its_run(db):
    project_id = _project(db)
    db.add(Job(project_id=project_id, capability="cpu", stage="texturing", state="queued"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
