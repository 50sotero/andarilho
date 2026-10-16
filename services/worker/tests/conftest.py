"""Test fixtures: a dedicated Postgres database migrated with the API's real Alembic history.

The schema comes from running the API's migrations as a subprocess (a test-time
dependency on the repository layout, never a package import). Set
TEST_DATABASE_URL to override the default local Docker database.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg.types.json import Jsonb

from andarilho_worker.artifacts import LocalObjectStore
from andarilho_worker.db import Database, libpq_url
from andarilho_worker.recipes import recipe_for
from andarilho_worker.settings import Settings

REPO = Path(__file__).resolve().parents[3]
DEFAULT_URL = "postgresql+psycopg://andarilho:local-password@localhost:55432/andarilho_worker_test"
TABLES = (
    "job_events, jobs, share_links, artifacts, usage_ledger, idempotency_records, "
    "project_members, audit_events, uploads, capture_sets, reconstruction_runs, scenes, "
    "scene_revisions, projects, multipart_reconciliations"
)


def _test_url() -> str:
    return os.environ.get("TEST_DATABASE_URL", DEFAULT_URL)


def _ensure_database(url: str) -> None:
    libpq = libpq_url(url)
    base, _, name = libpq.rpartition("/")
    name = name.split("?", 1)[0]
    with psycopg.connect(f"{base}/postgres", autocommit=True, connect_timeout=5) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{name}"')


@pytest.fixture(scope="session")
def database_url() -> str:
    url = _test_url()
    try:
        _ensure_database(url)
    except psycopg.OperationalError as exc:
        pytest.skip(f"test database unavailable: {exc}")
    env = {**os.environ, "DATABASE_URL": url}
    api = REPO / "services" / "api"
    cmd = ["uv", "run", "--project", str(api), "alembic", "-c", str(api / "alembic.ini"),
           "upgrade", "head"]  # fmt: skip
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600, check=False)
    if proc.returncode != 0:
        pytest.fail(f"API migrations failed:\n{proc.stdout}\n{proc.stderr}")
    return url


@pytest.fixture
def db(database_url: str) -> Database:
    database = Database(database_url)
    with database.transaction() as conn:
        conn.execute(f"TRUNCATE {TABLES} CASCADE")
    return database


@pytest.fixture
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(tmp_path / "objects")


@pytest.fixture
def settings(database_url: str, tmp_path: Path) -> Settings:
    return Settings(
        database_url=database_url,
        worker_capability="cpu",
        worker_scratch_dir=tmp_path / "scratch",
        worker_id="worker-test",
        heartbeat_seconds=0.2,
        retry_base_seconds=0.0,
        cleanup_reconcile_horizon_seconds=0,
    )


# ---------------------------------------------------------------------------
# Seeding helpers (plain SQL mirroring what the API inserts)
# ---------------------------------------------------------------------------


class Seed:
    def __init__(self, db: Database) -> None:
        self.db = db

    def project(self, lifecycle: str = "active") -> str:
        pid = str(uuid.uuid4())
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO projects (id, title, owner_subject, lifecycle, version) "
                "VALUES (%s, 'Test', 'user-1', %s, 1)",
                (pid, lifecycle),
            )
        return pid

    def capture_set(self, project_id: str, inputs: list[dict[str, Any]] | None = None) -> str:
        cid = str(uuid.uuid4())
        inputs = inputs or []
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO capture_sets (id, project_id, created_by, inputs, content_sha256) "
                "VALUES (%s, %s, 'user-1', %s, %s)",
                (
                    cid,
                    project_id,
                    Jsonb(inputs),
                    hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
                ),
            )
        return cid

    def run(
        self,
        project_id: str,
        *,
        quality: str = "synthetic",
        first_stage: str = "validating",
        capability: str = "cpu",
        max_attempts: int = 3,
        capture_set_id: str | None = None,
    ) -> tuple[str, str]:
        recipe = recipe_for(quality)
        capture = capture_set_id or self.capture_set(project_id)
        run_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
        manifest = {
            "runId": run_id,
            "projectId": project_id,
            "captureSetId": capture,
            "quality": quality,
            "includeSupplementaryPhotos": True,
            "recipe": recipe.manifest_entry(),
        }
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO reconstruction_runs (id, project_id, capture_set_id, created_by,
                    recipe_name, recipe_digest, quality, include_supplementary_photos,
                    resource_budget, status)
                VALUES (%s, %s, %s, 'user-1', %s, %s, %s, true, %s, 'queued')
                """,
                (run_id, project_id, capture, recipe.name, recipe.digest, quality,
                 Jsonb({"maxSeconds": 3600})),
            )  # fmt: skip
            conn.execute(
                """
                INSERT INTO usage_ledger (id, project_id, run_id, reserved_gpu_seconds,
                    reserved_cpu_seconds, actual_gpu_seconds, actual_cpu_seconds, storage_bytes,
                    state)
                VALUES (%s, %s, %s, 3600, 3600, 0, 0, 0, 'reserved')
                """,
                (str(uuid.uuid4()), project_id, run_id),
            )
            self._job(conn, job_id, project_id, run_id, first_stage, capability, manifest,
                      max_attempts=max_attempts)  # fmt: skip
        return run_id, job_id

    def _job(self, conn, job_id, project_id, run_id, stage, capability, manifest, *,
             max_attempts=3, upload_id=None, scene_revision_id=None, priority=0) -> None:  # fmt: skip
        conn.execute(
            """
            INSERT INTO jobs (id, project_id, run_id, upload_id, scene_revision_id, capability,
                stage, state, priority, attempts, max_attempts, fencing_token, available_at,
                input_manifest)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'queued', %s, 0, %s, 0, now(), %s)
            """,
            (job_id, project_id, run_id, upload_id, scene_revision_id, capability, stage,
             priority, max_attempts, Jsonb(manifest)),
        )  # fmt: skip

    def job(self, project_id: str, stage: str, manifest: dict[str, Any] | None = None, **kw) -> str:
        job_id = str(uuid.uuid4())
        with self.db.transaction() as conn:
            self._job(conn, job_id, project_id, kw.pop("run_id", None), stage,
                      kw.pop("capability", "cpu"), manifest or {}, **kw)  # fmt: skip
        return job_id

    def one(self, sql: str, *params: Any) -> dict[str, Any] | None:
        with self.db.transaction() as conn:
            return conn.execute(sql, params).fetchone()

    def all(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        with self.db.transaction() as conn:
            return conn.execute(sql, params).fetchall()

    def exec(self, sql: str, *params: Any) -> None:
        with self.db.transaction() as conn:
            conn.execute(sql, params)


@pytest.fixture
def seed(db: Database) -> Seed:
    return Seed(db)


@pytest.fixture
def python_exe() -> str:
    return sys.executable
