"""The jobs table accepts the maintenance stages the worker queues, and no other stage."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from andarilho_api.models import MAINTENANCE_STAGES, Job, Project


def _project(db) -> uuid.UUID:
    project = Project(id=uuid.uuid4(), title="Home", owner_subject="user-a")
    db.add(project)
    db.commit()
    return project.id


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
