"""Migrations 0006 and 0007: texturing, then scene_analysis, join the run stages; 0008
adds the multipart reconciliation table.

Retired training rows stay valid throughout. These tests move the shared test database
down (to 0005, 0006 or 0007) and back to head. Each restores
head in a ``finally`` so that later tests always see the current schema. No session is
open while a migration runs.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from factories import active_run
from sqlalchemy import inspect, text
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
    """A finished run of the splat pipeline: its validating and training jobs succeeded."""
    run_id = uuid.UUID(active_run(db, str(project_id), job_states=("succeeded",))["runId"])
    db.execute(
        text(
            "UPDATE reconstruction_runs SET status = 'succeeded', finished_at = now() "
            "WHERE id = :id"
        ),
        {"id": run_id},
    )
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


def _job(project_id: uuid.UUID, run_id: uuid.UUID, stage: str, state: str = "queued") -> Job:
    running = state == "running"  # a running job holds a lease
    return Job(
        project_id=project_id,
        run_id=run_id,
        capability="cpu",
        stage=stage,
        state=state,
        fencing_token=1 if running else 0,
        lease_owner="worker-1" if running else None,
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=90) if running else None,
    )


def _texturing(project_id: uuid.UUID, run_id: uuid.UUID, state: str = "queued") -> Job:
    return _job(project_id, run_id, "texturing", state)


def _revision(engine) -> str:
    with engine.connect() as conn:
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()


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
        db.add(_texturing(project_id, run_id, "succeeded"))
        db.commit()

    with _alembic(database_url) as config:
        command.downgrade(config, "0005")
        try:
            assert _stages(engine) == ["training", "validating"]
        finally:
            command.upgrade(config, "head")

    assert _stages(engine) == ["training", "validating"]


def test_scene_analysis_is_refused_before_0007_and_accepted_after(database_url, engine):
    with Session(engine) as db:
        project_id = _project(db)
        run_id = _training_run(db, project_id)

    with _alembic(database_url) as config:
        command.downgrade(config, "0006")
        try:
            with Session(engine) as db:
                db.add(_job(project_id, run_id, "scene_analysis"))
                with pytest.raises(IntegrityError):
                    db.commit()
                db.rollback()
        finally:
            command.upgrade(config, "head")

    with Session(engine) as db:
        db.add(_job(project_id, run_id, "scene_analysis"))
        db.commit()
    assert _stages(engine) == ["scene_analysis", "training", "validating"]


def test_downgrade_to_0006_removes_scene_analysis_jobs_only(database_url, engine):
    with Session(engine) as db:
        project_id = _project(db)
        run_id = _training_run(db, project_id)
        db.add(_job(project_id, run_id, "scene_analysis", "succeeded"))
        db.add(_texturing(project_id, run_id, "succeeded"))
        db.commit()

    with _alembic(database_url) as config:
        command.downgrade(config, "0006")
        try:
            assert _stages(engine) == ["texturing", "training", "validating"]
        finally:
            command.upgrade(config, "head")


@pytest.mark.parametrize(
    ("stage", "start", "target"),
    [("scene_analysis", "0007", "0006"), ("texturing", "0006", "0005")],
)
def test_downgrade_refuses_while_a_run_is_active_before_it_reaches_the_removed_stage(
    database_url, engine, stage, start, target
):
    # The run has no job of the stage yet, but its pinned recipe will need one. Each
    # migration's own guard is reached from its own revision.
    with _alembic(database_url) as config:
        try:
            if start != _revision(engine):
                command.downgrade(config, start)
            with Session(engine) as db:
                project_id = _project(db)
                active_run(db, str(project_id), job_states=("succeeded", "queued"))
            with pytest.raises(RuntimeError, match=rf"1 run\(s\) and 0 {stage} job"):
                command.downgrade(config, target)
            assert _revision(engine) == start
            assert _stages(engine) == ["preparing", "validating"]
        finally:
            command.upgrade(config, "head")


@pytest.mark.parametrize(
    ("stage", "target", "state"),
    [
        ("scene_analysis", "0006", "queued"),
        ("scene_analysis", "0006", "running"),
        ("texturing", "0005", "retry_wait"),
    ],
)
def test_downgrade_refuses_while_a_job_of_the_removed_stage_can_still_run(
    database_url, engine, stage, target, state
):
    # A layout save's job runs on a finished run; deleting it would strand its revision.
    with Session(engine) as db:
        project_id = _project(db)
        run_id = _training_run(db, project_id)
        db.add(_job(project_id, run_id, stage, state))
        db.commit()

    head = _revision(engine)
    with _alembic(database_url) as config:
        try:
            with pytest.raises(RuntimeError, match=f"{stage} job"):
                command.downgrade(config, target)
            # Nothing was rolled back: the schema and the job are as they were.
            assert _revision(engine) == head
            assert stage in _stages(engine)
        finally:
            command.upgrade(config, "head")


def test_0008_adds_the_multipart_reconciliation_table_and_its_downgrade_drops_it(
    database_url, engine
):
    with _alembic(database_url) as config:
        try:
            command.downgrade(config, "0007")
            assert "multipart_reconciliations" not in inspect(engine).get_table_names()
        finally:
            command.upgrade(config, "head")
    assert "multipart_reconciliations" in inspect(engine).get_table_names()
