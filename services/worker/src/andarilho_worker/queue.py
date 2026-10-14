"""PostgreSQL job queue: claim, heartbeat, fenced writes, terminal transitions, reaper.

Every write made on behalf of a running attempt is *fenced*: it only applies
while ``(id, lease_owner, fencing_token, state='running')`` still matches and,
for progress and success, while the job, its run and its project are not
cancelled/deleting. Terminal transitions lock rows in a fixed order
(project -> run -> job) so they serialize with API cancellation and deletion.
"""

from __future__ import annotations

import logging
import random
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from andarilho_worker.db import Database
from andarilho_worker.errors import Cancelled, LeaseLost, error_body

log = logging.getLogger(__name__)

# README section 6, verbatim CTE plus the run/project liveness predicates.
CLAIM_SQL = """
WITH candidate AS (
  SELECT id
  FROM jobs
  WHERE state = 'queued'
    AND capability = %(worker_capability)s
    AND available_at <= now()
    AND cancel_requested_at IS NULL
    AND NOT EXISTS (
      SELECT 1 FROM reconstruction_runs r
      WHERE r.id = jobs.run_id
        AND (r.cancel_requested_at IS NOT NULL OR r.status = 'cancelled')
    )
    AND EXISTS (
      SELECT 1 FROM projects p
      WHERE p.id = jobs.project_id
        AND (p.lifecycle = 'active'
             OR (jobs.stage = 'project_cleanup' AND p.lifecycle = 'deleting'))
    )
  ORDER BY priority DESC, created_at, id
  FOR UPDATE SKIP LOCKED
  LIMIT 1
)
UPDATE jobs AS j
SET state = 'running',
    lease_owner = %(worker_id)s,
    lease_expires_at = now() + interval '90 seconds',
    fencing_token = j.fencing_token + 1,
    attempts = j.attempts + 1,
    updated_at = now()
FROM candidate
WHERE j.id = candidate.id
RETURNING j.*;
"""

FENCE = """
    j.id = %(job_id)s
    AND j.fencing_token = %(fence)s
    AND j.lease_owner = %(owner)s
    AND j.state = 'running'
"""

LIVE = """
    AND j.cancel_requested_at IS NULL
    AND NOT EXISTS (
      SELECT 1 FROM reconstruction_runs r
      WHERE r.id = j.run_id AND r.cancel_requested_at IS NOT NULL
    )
    AND EXISTS (
      SELECT 1 FROM projects p
      WHERE p.id = j.project_id
        AND (p.lifecycle = 'active'
             OR (j.stage = 'project_cleanup' AND p.lifecycle = 'deleting'))
    )
"""

# The same "should this job stop" predicate, for jobs aliased j (reaper/sweeper).
CANCEL_COND = """
    (j.cancel_requested_at IS NOT NULL
     OR EXISTS (SELECT 1 FROM reconstruction_runs r
                WHERE r.id = j.run_id AND r.cancel_requested_at IS NOT NULL)
     OR NOT EXISTS (SELECT 1 FROM projects p
                    WHERE p.id = j.project_id
                      AND (p.lifecycle = 'active'
                           OR (j.stage = 'project_cleanup' AND p.lifecycle = 'deleting'))))
"""


@dataclass(frozen=True)
class ClaimedJob:
    id: str
    project_id: str
    run_id: str | None
    upload_id: str | None
    scene_revision_id: str | None
    capability: str
    stage: str
    attempts: int
    max_attempts: int
    fencing_token: int
    lease_owner: str
    priority: int
    input_manifest: Mapping[str, Any]
    lease_expires_at: datetime | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> ClaimedJob:
        def s(v: Any) -> str | None:
            return None if v is None else str(v)

        return cls(
            id=str(row["id"]),
            project_id=str(row["project_id"]),
            run_id=s(row["run_id"]),
            upload_id=s(row["upload_id"]),
            scene_revision_id=s(row["scene_revision_id"]),
            capability=row["capability"],
            stage=row["stage"],
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
            fencing_token=int(row["fencing_token"]),
            lease_owner=row["lease_owner"],
            priority=int(row["priority"]),
            input_manifest=dict(row["input_manifest"] or {}),
            lease_expires_at=row.get("lease_expires_at"),
        )

    @property
    def run_level(self) -> bool:
        """True for reconstruction-run stages (not layout-save or maintenance jobs)."""
        return self.run_id is not None and self.scene_revision_id is None

    def fence_params(self) -> dict[str, Any]:
        return {"job_id": self.id, "fence": self.fencing_token, "owner": self.lease_owner}


@dataclass(frozen=True)
class NextJob:
    stage: str
    capability: str
    input_manifest: Mapping[str, Any]


def _event(conn: psycopg.Connection, job_id: str, stage: str, state: str, message: str,
           progress: Mapping[str, Any] | None = None) -> None:  # fmt: skip
    conn.execute(
        "INSERT INTO job_events (job_id, stage, state, message, progress) "
        "VALUES (%s, %s, %s, %s, %s)",
        (job_id, stage, state, message[:500], Jsonb(dict(progress or {}))),
    )


# ---------------------------------------------------------------------------
# Claim / heartbeat / fenced progress
# ---------------------------------------------------------------------------


def claim(db: Database, worker_id: str, capability: str) -> ClaimedJob | None:
    with db.transaction() as conn:
        row = conn.execute(
            CLAIM_SQL, {"worker_capability": capability, "worker_id": worker_id}
        ).fetchone()
        if row is None:
            return None
        job = ClaimedJob.from_row(row)
        if job.run_level:
            conn.execute(
                """
                UPDATE reconstruction_runs SET status = 'running', updated_at = now()
                WHERE id = (SELECT id FROM reconstruction_runs
                            WHERE id = %s AND status = 'queued' FOR UPDATE SKIP LOCKED)
                """,
                (job.run_id,),
            )
        _event(conn, job.id, job.stage, "running", f"Started attempt {job.attempts}.")
    log.info("claimed job %s stage=%s fence=%s", job.id, job.stage, job.fencing_token)
    return job


def heartbeat(db: Database, job: ClaimedJob) -> bool:
    """Extend the lease. False means the worker must stop (lost lease or cancellation)."""
    with db.transaction() as conn:
        row = conn.execute(
            f"""
            UPDATE jobs AS j
            SET lease_expires_at = now() + interval '90 seconds', updated_at = now()
            WHERE {FENCE} AND j.lease_expires_at > now() {LIVE}
            RETURNING j.id
            """,
            job.fence_params(),
        ).fetchone()
    return row is not None


def fence_status(db: Database, job: ClaimedJob) -> str:
    """'ok', 'cancelled' (fence holds but stop was requested) or 'lost'."""
    with db.transaction() as conn:
        held = conn.execute(
            f"SELECT j.id, (j.lease_expires_at > now()) AS valid FROM jobs j WHERE {FENCE}",
            job.fence_params(),
        ).fetchone()
        if held is None:
            return "lost"
        live = conn.execute(
            f"SELECT j.id FROM jobs j WHERE {FENCE} {LIVE}", job.fence_params()
        ).fetchone()
    if live is None:
        return "cancelled"
    return "ok" if held["valid"] else "lost"


def record_event(
    db: Database,
    job: ClaimedJob,
    message: str,
    progress: Mapping[str, Any] | None = None,
) -> bool:
    """Insert a progress event only if the fence still holds and nothing is cancelled."""
    with db.transaction() as conn:
        cur = conn.execute(
            f"""
            INSERT INTO job_events (job_id, stage, state, message, progress)
            SELECT j.id, j.stage, j.state, %(message)s, %(progress)s
            FROM jobs j WHERE {FENCE} {LIVE}
            """,
            {
                **job.fence_params(),
                "message": message[:500],
                "progress": Jsonb(dict(progress or {})),
            },
        )
        return cur.rowcount == 1


def fenced_execute(db: Database, job: ClaimedJob, fn: Callable[[psycopg.Connection], Any]) -> Any:
    """Run ``fn(conn)`` in a transaction that first locks and verifies the live fence."""
    with db.transaction() as conn:
        _lock(conn, job, require_live=True)
        return fn(conn)


# ---------------------------------------------------------------------------
# Terminal transitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Locked:
    row: Mapping[str, Any]
    stop_requested: bool
    lease_valid: bool


def _lock(conn: psycopg.Connection, job: ClaimedJob, *, require_live: bool) -> _Locked:
    project = conn.execute(
        "SELECT lifecycle FROM projects WHERE id = %s FOR SHARE", (job.project_id,)
    ).fetchone()
    run = None
    if job.run_id:
        run = conn.execute(
            "SELECT cancel_requested_at, status FROM reconstruction_runs WHERE id = %s FOR UPDATE",
            (job.run_id,),
        ).fetchone()
    row = conn.execute(
        f"SELECT j.*, (j.lease_expires_at > now()) AS lease_valid FROM jobs j WHERE {FENCE} "
        "FOR UPDATE",
        job.fence_params(),
    ).fetchone()
    if row is None:
        raise LeaseLost(job.id)
    lifecycle_ok = project is not None and (
        project["lifecycle"] == "active"
        or (job.stage == "project_cleanup" and project["lifecycle"] == "deleting")
    )
    stop = (
        row["cancel_requested_at"] is not None
        or (run is not None and run["cancel_requested_at"] is not None)
        or not lifecycle_ok
    )
    locked = _Locked(row, stop, bool(row["lease_valid"]))
    if require_live and stop:
        raise Cancelled(job.id)
    return locked


def _clear_lease_sql(state: str) -> str:
    return f"""
        UPDATE jobs SET state = '{state}', lease_owner = NULL, lease_expires_at = NULL,
               updated_at = now(), finished_at = now()
    """


def add_usage(conn: psycopg.Connection, job: ClaimedJob, seconds: float) -> None:
    if not job.run_id or seconds <= 0:
        return
    column = "actual_gpu_seconds" if job.capability == "gpu" else "actual_cpu_seconds"
    conn.execute(
        f"UPDATE usage_ledger SET {column} = {column} + %s WHERE run_id = %s",  # noqa: S608
        (int(round(seconds)) or 1, job.run_id),
    )


def release_ledger(conn: psycopg.Connection, run_id: str) -> bool:
    """reserved -> released, exactly once (conditional update)."""
    row = conn.execute(
        "UPDATE usage_ledger SET state = 'released', settled_at = now() "
        "WHERE run_id = %s AND state = 'reserved' RETURNING id",
        (run_id,),
    ).fetchone()
    return row is not None


def settle_ledger(conn: psycopg.Connection, run_id: str) -> bool:
    row = conn.execute(
        "UPDATE usage_ledger SET state = 'settled', settled_at = now() "
        "WHERE run_id = %s AND state = 'reserved' RETURNING id",
        (run_id,),
    ).fetchone()
    return row is not None


def finalize_run_if_cancelled(conn: psycopg.Connection, run_id: str) -> bool:
    """When a cancelled run has no active jobs left, mark it cancelled and release budget."""
    row = conn.execute(
        """
        UPDATE reconstruction_runs AS r
        SET status = 'cancelled', finished_at = now(), updated_at = now()
        WHERE r.id = %(run)s
          AND r.status IN ('queued', 'running')
          AND (r.cancel_requested_at IS NOT NULL
               OR EXISTS (SELECT 1 FROM projects p
                          WHERE p.id = r.project_id AND p.lifecycle <> 'active'))
          AND NOT EXISTS (
            SELECT 1 FROM jobs j
            WHERE j.run_id = r.id AND j.scene_revision_id IS NULL
              AND j.state IN ('queued', 'running', 'retry_wait'))
        RETURNING r.id
        """,
        {"run": run_id},
    ).fetchone()
    if row is None:
        return False
    released = release_ledger(conn, run_id)
    log.info("run %s cancelled (ledger released=%s)", run_id, released)
    return True


def fail_run(conn: psycopg.Connection, run_id: str, error: Mapping[str, Any]) -> bool:
    row = conn.execute(
        """
        UPDATE reconstruction_runs SET status = 'failed', error = %s, finished_at = now(),
               updated_at = now()
        WHERE id = %s AND status IN ('queued', 'running') RETURNING id
        """,
        (Jsonb(dict(error)), run_id),
    ).fetchone()
    if row is None:
        return False
    conn.execute(
        "UPDATE jobs SET state = 'cancelled', finished_at = now(), updated_at = now() "
        "WHERE run_id = %s AND scene_revision_id IS NULL AND state IN ('queued', 'retry_wait')",
        (run_id,),
    )
    release_ledger(conn, run_id)
    return True


def _insert_next(conn: psycopg.Connection, job: ClaimedJob, nxt: NextJob) -> bool:
    try:
        with conn.transaction():  # savepoint: a duplicate must not abort the completion
            conn.execute(
                """
                INSERT INTO jobs (id, project_id, run_id, scene_revision_id, capability, stage,
                                  state, priority, attempts, max_attempts, fencing_token,
                                  available_at, input_manifest)
                VALUES (%s, %s, %s, %s, %s, %s, 'queued', %s, 0, %s, 0, now(), %s)
                """,
                (
                    str(uuid.uuid4()),
                    job.project_id,
                    job.run_id,
                    job.scene_revision_id,
                    nxt.capability,
                    nxt.stage,
                    job.priority,
                    job.max_attempts,
                    Jsonb(dict(nxt.input_manifest)),
                ),
            )
    except psycopg.errors.UniqueViolation:
        log.warning("next stage %s for job %s already exists", nxt.stage, job.id)
        return False
    return True


CommitHook = Callable[[psycopg.Connection, ClaimedJob], Any]


def complete_job(
    db: Database,
    job: ClaimedJob,
    *,
    result_manifest: Mapping[str, Any],
    next_job: NextJob | None,
    commit: CommitHook | None = None,
    usage_seconds: float = 0.0,
    finish_run: bool = False,
) -> None:
    """Mark the job succeeded and insert the next stage in ONE fenced transaction.

    Raises :class:`Cancelled` (nothing written) when cancellation won the race,
    and :class:`LeaseLost` when the fence moved or the lease already expired.
    """
    with db.transaction() as conn:
        locked = _lock(conn, job, require_live=True)
        if not locked.lease_valid:
            raise LeaseLost(job.id)
        if commit is not None:
            commit(conn, job)
        updated = conn.execute(
            _clear_lease_sql("succeeded")
            + ", result_manifest = %s, error = NULL WHERE id = %s RETURNING id",
            (Jsonb(dict(result_manifest)), job.id),
        ).fetchone()
        if updated is not None:
            _event(conn, job.id, job.stage, "succeeded", "Stage completed.")
        elif job.stage != "project_cleanup":
            raise LeaseLost(job.id)
        add_usage(conn, job, usage_seconds)
        if next_job is not None:
            _insert_next(conn, job, next_job)
        elif finish_run and job.run_level:
            conn.execute(
                "UPDATE reconstruction_runs SET status = 'succeeded', finished_at = now(), "
                "updated_at = now() WHERE id = %s AND status IN ('queued', 'running')",
                (job.run_id,),
            )
            settle_ledger(conn, job.run_id)  # type: ignore[arg-type]


def cancel_job(db: Database, job: ClaimedJob, *, usage_seconds: float = 0.0) -> bool:
    """Acknowledge cancellation after the process group has been stopped (fenced)."""
    with db.transaction() as conn:
        try:
            _lock(conn, job, require_live=False)
        except LeaseLost:
            return False
        conn.execute(_clear_lease_sql("cancelled") + " WHERE id = %s", (job.id,))
        _event(conn, job.id, job.stage, "cancelled", "Cancelled.")
        add_usage(conn, job, usage_seconds)
        if job.run_id:
            finalize_run_if_cancelled(conn, job.run_id)
    return True


def fail_job(
    db: Database,
    job: ClaimedJob,
    error: Mapping[str, Any],
    *,
    on_fail: CommitHook | None = None,
    usage_seconds: float = 0.0,
) -> str:
    """Permanent failure. Returns the resulting job state ('failed' or 'cancelled')."""
    with db.transaction() as conn:
        locked = _lock(conn, job, require_live=False)
        if locked.stop_requested:
            conn.execute(_clear_lease_sql("cancelled") + " WHERE id = %s", (job.id,))
            _event(conn, job.id, job.stage, "cancelled", "Cancelled.")
            add_usage(conn, job, usage_seconds)
            if job.run_id:
                finalize_run_if_cancelled(conn, job.run_id)
            return "cancelled"
        if on_fail is not None:
            on_fail(conn, job)
        conn.execute(
            _clear_lease_sql("failed") + ", error = %s WHERE id = %s", (Jsonb(dict(error)), job.id)
        )
        _event(conn, job.id, job.stage, "failed", str(error.get("message", "Failed.")))
        add_usage(conn, job, usage_seconds)
        if job.run_level:
            fail_run(conn, job.run_id, error)  # type: ignore[arg-type]
    return "failed"


def backoff_seconds(attempt: int, *, base: float = 30.0, cap: float = 900.0,
                    rng: random.Random | None = None) -> float:  # fmt: skip
    """Exponential backoff with full jitter on top of the base delay."""
    rng = rng or random.Random()  # noqa: S311 - jitter, not cryptography
    delay = min(cap, base * (2 ** max(0, attempt - 1)))
    return delay + rng.uniform(0, min(base, delay))


def retry_job(
    db: Database,
    job: ClaimedJob,
    error: Mapping[str, Any],
    *,
    delay_seconds: float,
    usage_seconds: float = 0.0,
) -> str:
    """Transient failure: retry_wait with backoff, or failed when attempts are exhausted."""
    with db.transaction() as conn:
        locked = _lock(conn, job, require_live=False)
        if locked.stop_requested:
            conn.execute(_clear_lease_sql("cancelled") + " WHERE id = %s", (job.id,))
            _event(conn, job.id, job.stage, "cancelled", "Cancelled.")
            add_usage(conn, job, usage_seconds)
            if job.run_id:
                finalize_run_if_cancelled(conn, job.run_id)
            return "cancelled"
        add_usage(conn, job, usage_seconds)
        if locked.row["attempts"] >= locked.row["max_attempts"]:
            final = {**error, "retryable": False}
            conn.execute(
                _clear_lease_sql("failed") + ", error = %s WHERE id = %s",
                (Jsonb(final), job.id),
            )
            _event(conn, job.id, job.stage, "failed", "Retries exhausted.")
            if job.run_level:
                fail_run(conn, job.run_id, final)  # type: ignore[arg-type]
            return "failed"
        conn.execute(
            """
            UPDATE jobs SET state = 'retry_wait', lease_owner = NULL, lease_expires_at = NULL,
                   available_at = now() + make_interval(secs => %s), error = %s, updated_at = now()
            WHERE id = %s
            """,
            (float(delay_seconds), Jsonb(dict(error)), job.id),
        )
        _event(conn, job.id, job.stage, "retry_wait", "Retrying after a temporary problem.")
    return "retry_wait"


def defer_job(
    db: Database, job: ClaimedJob, *, delay_seconds: float, patch: Mapping[str, Any] | None = None
) -> bool:
    """Re-queue later without consuming an attempt (cleanup waiting on other workers)."""
    with db.transaction() as conn:
        try:
            _lock(conn, job, require_live=False)
        except LeaseLost:
            return False
        conn.execute(
            """
            UPDATE jobs SET state = 'retry_wait', lease_owner = NULL, lease_expires_at = NULL,
                   attempts = GREATEST(attempts - 1, 0),
                   available_at = now() + make_interval(secs => %s),
                   input_manifest = input_manifest || %s, updated_at = now()
            WHERE id = %s
            """,
            (float(delay_seconds), Jsonb(dict(patch or {})), job.id),
        )
    return True


# ---------------------------------------------------------------------------
# Maintenance: reaper, retry promotion, cancellation sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReapOutcome:
    job_id: str
    run_id: str | None
    state: str


def reap_expired(db: Database) -> list[ReapOutcome]:
    """Recover expired leases: bump the fence and mark cancelled / failed / queued."""
    lost = error_body("WORKER_LOST")
    with db.transaction() as conn:
        rows = conn.execute(
            f"""
            WITH expired AS (
              SELECT j.id,
                     CASE WHEN {CANCEL_COND} THEN 'cancelled'
                          WHEN j.attempts >= j.max_attempts THEN 'failed'
                          ELSE 'queued' END AS outcome
              FROM jobs j
              WHERE j.state = 'running' AND j.lease_expires_at < now()
              FOR UPDATE OF j SKIP LOCKED
            )
            UPDATE jobs AS j
            SET fencing_token = j.fencing_token + 1,
                state = e.outcome,
                lease_owner = NULL,
                lease_expires_at = NULL,
                available_at = CASE WHEN e.outcome = 'queued' THEN now() ELSE j.available_at END,
                finished_at = CASE WHEN e.outcome = 'queued' THEN NULL ELSE now() END,
                error = CASE WHEN e.outcome = 'failed' THEN %(err)s ELSE j.error END,
                updated_at = now()
            FROM expired AS e
            WHERE j.id = e.id
            RETURNING j.id, j.run_id, j.scene_revision_id, j.stage, j.state
            """,
            {"err": Jsonb(lost)},
        ).fetchall()
        outcomes = []
        for row in rows:
            job_id, run_id = str(row["id"]), (str(row["run_id"]) if row["run_id"] else None)
            message = {
                "queued": "Worker lease expired; re-queued.",
                "failed": "Worker lease expired too many times.",
                "cancelled": "Cancelled after the worker stopped.",
            }[row["state"]]
            _event(conn, job_id, row["stage"], row["state"], message)
            if run_id and row["state"] == "failed" and row["scene_revision_id"] is None:
                fail_run(conn, run_id, lost)
            elif run_id and row["state"] == "cancelled":
                finalize_run_if_cancelled(conn, run_id)
            outcomes.append(ReapOutcome(job_id, run_id, row["state"]))
    for outcome in outcomes:
        log.warning("reaped job %s -> %s", outcome.job_id, outcome.state)
    return outcomes


def promote_retry_wait(db: Database) -> int:
    with db.transaction() as conn:
        cur = conn.execute(
            f"""
            UPDATE jobs AS j SET state = 'queued', updated_at = now()
            WHERE j.state = 'retry_wait' AND j.available_at <= now() AND NOT {CANCEL_COND}
            """
        )
        return cur.rowcount


def sweep_cancelled(db: Database) -> int:
    """Cancel queued/retry_wait jobs whose job, run or project was cancelled/deleted."""
    with db.transaction() as conn:
        rows = conn.execute(
            f"""
            UPDATE jobs AS j SET state = 'cancelled', finished_at = now(), updated_at = now()
            WHERE j.state IN ('queued', 'retry_wait') AND {CANCEL_COND}
            RETURNING j.id, j.run_id, j.stage
            """
        ).fetchall()
        runs = {str(r["run_id"]) for r in rows if r["run_id"]}
        for row in rows:
            _event(conn, str(row["id"]), row["stage"], "cancelled", "Cancelled.")
        # Also finalize runs whose last running job was already acknowledged elsewhere.
        stale = conn.execute(
            """
            SELECT r.id FROM reconstruction_runs r
            WHERE r.status IN ('queued', 'running') AND r.cancel_requested_at IS NOT NULL
            """
        ).fetchall()
        runs.update(str(r["id"]) for r in stale)
        for run_id in sorted(runs):
            finalize_run_if_cancelled(conn, run_id)
        return len(rows)


def maintenance(db: Database) -> dict[str, int]:
    reaped = reap_expired(db)
    return {
        "reaped": len(reaped),
        "cancelled": sweep_cancelled(db),
        "promoted": promote_retry_wait(db),
    }
