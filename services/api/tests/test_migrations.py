"""Migration 0006: texturing joins the run stages; retired training rows stay valid.

These tests move the shared test database down to 0005 and back to head. Each restores
head in a ``finally`` so that later tests always see the current schema. No session is
open while a migration runs.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from factories import active_run
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from andarilho_api.models import Job, Project

API_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _alembic(database_url: str) -> Iterator[Config]:
    config = Config(str(API_ROOT / "alembic.ini"))
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = database_url
    try:
        yield config
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous


def _project(db: Session) -> uuid.UUID:
    project = Project(id=uuid.uuid4(), title="Home", owner_subject="user-a")
    db.add(project)
    db.commit()
    return project.id


def _training_run(db: Session, project_id: uuid.UUID) -> uuid.UUID:
    """A run whose validating job succeeded and whose historical training job is recorded."""
    run_id = uuid.UUID(active_run(db, str(project_id), job_states=("succeeded",))["runId"])
    db.add(
        Job(
            project_id=project_id,
            run_id=run_id,
            capability="gpu",
            stage="training",
            state="succeeded",
            fencing_token=0,
        )
    )
    db.commit()
    return run_id


def _texturing(project_id: uuid.UUID, run_id: uuid.UUID) -> Job:
    return Job(
        project_id=project_id,
        run_id=run_id,
        capability="cpu",
        stage="texturing",
        state="queued",
        fencing_token=0,
    )


def _stages(engine) -> list[str]:
    with engine.connect() as conn:
        return sorted(conn.execute(text("SELECT stage FROM jobs")).scalars().all())


def test_training_rows_survive_the_round_trip_and_texturing_is_accepted_after_upgrade(
    database_url, engine
):
    with Session(engine) as db:
        project_id = _project(db)
        run_id = _training_run(db, project_id)

    with _alembic(database_url) as config:
        command.downgrade(config, "0005")
        try:
            with Session(engine) as db:
                # 0005 knows no texturing stage, so the new job is refused.
                db.add(_texturing(project_id, run_id))
                with pytest.raises(IntegrityError):
                    db.commit()
                db.rollback()
        finally:
            command.upgrade(config, "head")

    with Session(engine) as db:
        # The historical row came through both migrations, and texturing now inserts.
        db.add(_texturing(project_id, run_id))
        db.commit()
    assert _stages(engine) == ["texturing", "training", "validating"]


def test_downgrade_removes_texturing_jobs_and_keeps_training(database_url, engine):
    with Session(engine) as db:
        project_id = _project(db)
        run_id = _training_run(db, project_id)
        db.add(_texturing(project_id, run_id))
        db.commit()

    with _alembic(database_url) as config:
        command.downgrade(config, "0005")
        try:
            assert _stages(engine) == ["training", "validating"]
        finally:
            command.upgrade(config, "head")

    assert _stages(engine) == ["training", "validating"]
