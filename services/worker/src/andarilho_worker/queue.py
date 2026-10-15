"""PostgreSQL job queue: claim, heartbeat, fenced writes, terminal transitions, reaper.

Every write made on behalf of a running attempt is *fenced*: it only applies
while ``(id, lease_owner, fencing_token, state='running')`` still matches and,
for transitions and fenced writes, while the lease has not expired; for progress
and success, also while the job, its run and its project are not
cancelled/deleting. Terminal transitions, including the reaper's, lock rows in a
fixed order (project -> run -> job) so they serialize with API cancellation and
deletion.
"""

from __future__ import annotations

import logging
import random
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from andarilho_worker.artifacts import attempt_prefix, scene_prefix
from andarilho_worker.db import Database
from andarilho_worker.errors import Cancelled, LeaseLost, error_body

log = logging.getLogger(__name__)
# Storage cleanup waits this long so that a worker which lost its lease (but may still
# be finishing a write) has stopped before its objects are removed.
ARTIFACT_CLEANUP_DELAY_SECONDS = 300
# A cleanup that used up its attempts on temporary failures (a storage outage, lost
# workers) is queued again, waiting twice as long each time up to a day, so an outage
# longer than the retry window cannot leave its objects until the project is deleted.
# It gives up after this many requeues (about four days).
ARTIFACT_CLEANUP_REQUEUE_SECONDS = 3600
ARTIFACT_CLEANUP_REQUEUE_CAP_SECONDS = 24 * 3600
ARTIFACT_CLEANUP_MAX_REQUEUES = 8
# A project deletion that used up its attempts is queued again on the same schedule, and
# it never gives up: the owner asked for the data to go, and nothing else would retry it
# (the web hides the delete control once a project is being deleted).
PROJECT_CLEANUP_PRIORITY = 10

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
    started_at = now(),
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
    """Insert a progress event only if the fence and lease hold and nothing is cancelled."""
    with db.transaction() as conn:
        cur = conn.execute(
            f"""
            INSERT INTO job_events (job_id, stage, state, message, progress)
            SELECT j.id, j.stage, j.state, %(message)s, %(progress)s
            FROM jobs j WHERE {FENCE} AND j.lease_expires_at > now() {LIVE}
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
    # An expired lease belongs to the reaper even before it moves the fence: a worker
    # that could not renew in time writes nothing, and the reaper settles the job.
    if row is None or not row["lease_valid"]:
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
    locked = _Locked(row, stop)
    if require_live and stop:
        raise Cancelled(job.id)
    return locked


def _clear_lease_sql(state: str) -> str:
    return f"""
        UPDATE jobs SET state = '{state}', lease_owner = NULL, lease_expires_at = NULL,
               updated_at = now(), finished_at = now()
    """


def add_usage(conn: psycopg.Connection, job: ClaimedJob, seconds: float) -> None:
    # Only reconstruction stages charge the run ledger; layout revisions run after
    # it has settled and are tracked by per-stage metrics instead.
    if not job.run_level or seconds <= 0:
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


def queue_artifact_cleanup(
    conn: psycopg.Connection,
    project_id: str,
    prefixes: Sequence[str],
    *,
    reason: str,
    scene_revision_id: str | None = None,
    delay_seconds: float = ARTIFACT_CLEANUP_DELAY_SECONDS,
    requeues: int = 0,
) -> None:
    """Queue ``artifact_cleanup`` to delete storage prefixes that no pipeline will complete.

    The job has no run, revision or upload, so run budgets, the one-stage-per-run index
    and the next-stage chain ignore it. It becomes claimable after ``delay_seconds``
    (``ARTIFACT_CLEANUP_DELAY_SECONDS`` unless it is a requeue). A request identical to a
    queued or retrying one is already covered and is not queued again. The check is not
    serialized with a concurrent enqueue: a lock here would sit outside the project ->
    run -> job order and could deadlock with fenced transitions. A race can therefore
    queue a duplicate, which costs one more idempotent delete. Prefixes outside the
    project's tenant directory are a programming error.
    """
    if not prefixes:
        return
    tenant = f"tenants/{project_id}/"
    for prefix in prefixes:
        if not (prefix.startswith(tenant) and prefix.endswith("/")):
            raise ValueError(f"artifact prefix {prefix!r} is not a directory of the project")
    unique = sorted(set(prefixes))
    pending = conn.execute(
        """
        SELECT 1 FROM jobs
        WHERE project_id = %s AND stage = 'artifact_cleanup'
          AND state IN ('queued', 'retry_wait')
          AND input_manifest -> 'prefixes' = %s::jsonb
        LIMIT 1
        """,
        (project_id, Jsonb(unique)),
    ).fetchone()
    if pending is not None:
        return
    manifest: dict[str, Any] = {"projectId": project_id, "prefixes": unique, "reason": reason}
    if scene_revision_id is not None:
        manifest["sceneRevisionId"] = scene_revision_id
    if requeues:
        manifest["requeues"] = requeues
    conn.execute(
        """
        INSERT INTO jobs (id, project_id, run_id, upload_id, scene_revision_id, capability, stage,
                          state, priority, attempts, max_attempts, fencing_token, available_at,
                          input_manifest)
        VALUES (%s, %s, NULL, NULL, NULL, 'cpu', 'artifact_cleanup', 'queued', 5, 0, 3, 0,
                now() + make_interval(secs => %s), %s)
        """,
        (
            str(uuid.uuid4()),
            project_id,
            float(delay_seconds),
            Jsonb(manifest),
        ),
    )


def requeue_artifact_cleanup(
    conn: psycopg.Connection, project_id: str, manifest: Mapping[str, Any]
) -> None:
    """Queue a cleanup again, later, after it used up its attempts on temporary failures.

    Only the paths that exhaust retries call this (retry_job and the reaper); a refused
    request fails permanently and is never requeued. A malformed request is dropped
    rather than raised, because this runs inside the failing job's transition.
    """
    prefixes = manifest.get("prefixes")
    requeues = int(manifest.get("requeues") or 0)
    if manifest.get("projectId") != project_id or not isinstance(prefixes, list):
        return
    if requeues >= ARTIFACT_CLEANUP_MAX_REQUEUES:
        log.error("artifact cleanup gave up after %s requeues: %s", requeues, prefixes)
        return
    delay = min(
        ARTIFACT_CLEANUP_REQUEUE_CAP_SECONDS, ARTIFACT_CLEANUP_REQUEUE_SECONDS * 2**requeues
    )
    revision = manifest.get("sceneRevisionId")
    try:
        queue_artifact_cleanup(
            conn,
            project_id,
            prefixes,
            reason=str(manifest.get("reason") or "requeued"),
            scene_revision_id=str(revision) if revision else None,
            delay_seconds=delay,
            requeues=requeues + 1,
        )
    except (ValueError, TypeError, AttributeError):
        log.error("artifact cleanup request is malformed; not requeued: %s", prefixes)


def _queue_scene_cleanup(conn: psycopg.Connection, project_id: str, revision_id: str) -> None:
    """Delete the packaged files of a revision that will never be validated."""
    queue_artifact_cleanup(
        conn,
        project_id,
        [scene_prefix(project_id, revision_id)],
        reason="revision_invalid",
        scene_revision_id=revision_id,
    )


def _queue_leftovers(conn: psycopg.Connection, job: ClaimedJob, leftovers: Sequence[str]) -> None:
    """The job has ended, so no later attempt sweeps the prefixes this one could not delete."""
    queue_artifact_cleanup(conn, job.project_id, leftovers, reason="delete_failed")


# Findings on the revision a run's own pipeline created when the run ends before
# that revision is validated (the API writes the same cancellation finding).
RUN_FAILED_FINDING = {
    "code": "RUN_FAILED",
    "message": "Processing failed before this scene was ready. Retry the run to try again.",
}
RUN_CANCELLED_FINDING = {
    "code": "RUN_CANCELLED",
    "message": "Processing was cancelled before this scene was ready.",
}


def fail_run_revision(conn: psycopg.Connection, run_id: str, finding: Mapping[str, str]) -> None:
    """The run's first revision (created in geometry_proposal) stops pending.

    Otherwise the scene would wait forever for a validation that never runs. Its
    packaged files are then unreferenced and queued for cleanup. The check reads the
    revision afterwards because a stage's own failure hook may already have made it
    invalid (for example scene_validation), which leaves nothing to update here. So a
    revision that was already invalid before this call is queued too. A repeat is
    harmless: the cleanup deletes the same prefix again, and an identical request still
    queued is not added twice.
    """
    conn.execute(
        "UPDATE scene_revisions SET validation_state = 'invalid', validation_findings = %s "
        "WHERE run_id = %s AND revision = 1 AND validation_state = 'pending'",
        (Jsonb([dict(finding)]), run_id),
    )
    invalid = conn.execute(
        "SELECT id, project_id FROM scene_revisions "
        "WHERE run_id = %s AND revision = 1 AND validation_state = 'invalid'",
        (run_id,),
    ).fetchall()
    for row in invalid:
        _queue_scene_cleanup(conn, str(row["project_id"]), str(row["id"]))


def queue_run_outputs(conn: psycopg.Connection, run_id: str) -> None:
    """Queue what a run's succeeded stages wrote, once the run has ended unfinished.

    A failed or cancelled run never had a valid scene (scene_validation succeeds the
    run in the same transaction), so no layout save can read these outputs, and a retry
    is a new run with its own prefixes. Without this the frames and datasets the
    finished stages wrote would stay until the project is deleted. Each job's current
    fence is the one it committed under; earlier fences were swept or queued already.
    """
    rows = conn.execute(
        "SELECT id, project_id, stage, fencing_token FROM jobs "
        "WHERE run_id = %s AND scene_revision_id IS NULL AND state = 'succeeded' "
        "AND fencing_token > 0",
        (run_id,),
    ).fetchall()
    by_project: dict[str, list[str]] = {}
    for row in rows:
        project_id = str(row["project_id"])
        by_project.setdefault(project_id, []).append(
            attempt_prefix(project_id, run_id, row["stage"], str(row["id"]), row["fencing_token"])
        )
    for project_id, prefixes in by_project.items():
        queue_artifact_cleanup(conn, project_id, prefixes, reason="run_ended")


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
    fail_run_revision(conn, run_id, RUN_CANCELLED_FINDING)
    queue_run_outputs(conn, run_id)
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
    # A sibling claimed before may have written under an earlier fence, and it is never
    # claimed again, so its attempts are queued for deletion as sweep_cancelled does.
    siblings = conn.execute(
        "UPDATE jobs SET state = 'cancelled', finished_at = now(), updated_at = now() "
        "WHERE run_id = %s AND scene_revision_id IS NULL AND state IN ('queued', 'retry_wait') "
        "RETURNING id, project_id, run_id, stage, fencing_token",
        (run_id,),
    ).fetchall()
    _queue_unclaimed_attempts(conn, siblings)
    fail_run_revision(conn, run_id, RUN_FAILED_FINDING)
    queue_run_outputs(conn, run_id)
    release_ledger(conn, run_id)
    return True


def fail_revision(conn: psycopg.Connection, scene_revision_id: str | None) -> None:
    """A layout revision whose job failed for good stops pending: mark it invalid.

    Otherwise the editor would show "validating" forever. The finding is safe to
    show and tells the user how to recover (saving again queues a new pipeline).
    """
    if scene_revision_id is None:
        return
    finding = {
        "code": "LAYOUT_PROCESSING_FAILED",
        "message": "Processing this layout failed. Save the layout again to retry.",
    }
    conn.execute(
        "UPDATE scene_revisions SET validation_state = 'invalid', validation_findings = %s "
        "WHERE id = %s AND validation_state = 'pending'",
        (Jsonb([finding]), scene_revision_id),
    )
    # Read back as well: a stage's failure hook may have invalidated it first, and an
    # invalid revision never becomes valid, so its packaged files are unreferenced.
    row = conn.execute(
        "SELECT project_id FROM scene_revisions WHERE id = %s AND validation_state = 'invalid'",
        (scene_revision_id,),
    ).fetchone()
    if row is not None:
        _queue_scene_cleanup(conn, str(row["project_id"]), scene_revision_id)


def fail_upload(conn: psycopg.Connection, project_id: str, upload_id: str | None) -> None:
    """An inspection that failed for good rejects its upload instead of stranding it.

    An ``uploaded_unverified`` upload holds quota, cannot be deleted and is polled
    by the capture page, so it must not outlive its last inspection attempt. The
    stored bytes are then removed by ``upload_cleanup``, which releases the quota.
    """
    if upload_id is None:
        return
    row = conn.execute(
        """
        UPDATE uploads SET status = 'rejected', rejection_code = 'INSPECTION_FAILED',
               media_metadata = media_metadata || %s, updated_at = now()
        WHERE id = %s AND project_id = %s AND status = 'uploaded_unverified'
        RETURNING immutable_key
        """,
        (Jsonb({"rejectionReason": "the file could not be checked"}), upload_id, project_id),
    ).fetchone()
    if row is None or row["immutable_key"] is None:
        return
    conn.execute(
        """
        INSERT INTO jobs (id, project_id, upload_id, capability, stage, state, priority,
                          attempts, max_attempts, fencing_token, available_at, input_manifest)
        VALUES (%s, %s, %s, 'cpu', 'upload_cleanup', 'queued', 5, 0, 3, 0, now(), %s)
        """,
        (str(uuid.uuid4()), project_id, upload_id,
         Jsonb({"uploadId": upload_id, "projectId": project_id, "key": row["immutable_key"]})),
    )  # fmt: skip


def requeue_project_cleanup(
    conn: psycopg.Connection, project_id: str, manifest: Mapping[str, Any]
) -> None:
    """Queue a project's deletion again, later, after it used up its attempts.

    Only the paths that exhaust retries call this (retry_job and the reaper). The wait
    doubles from ``ARTIFACT_CLEANUP_REQUEUE_SECONDS`` up to a day and the deletion never
    gives up. The new job keeps the request's manifest (its request time and first pass),
    so the reconcile horizon does not start again. Nothing is queued once the project is
    no longer being deleted, or while another cleanup of it is still pending.
    """
    project = conn.execute("SELECT lifecycle FROM projects WHERE id = %s", (project_id,)).fetchone()
    if project is None or project["lifecycle"] != "deleting":
        return
    pending = conn.execute(
        "SELECT 1 FROM jobs WHERE project_id = %s AND stage = 'project_cleanup' "
        "AND state IN ('queued', 'retry_wait', 'running') LIMIT 1",
        (project_id,),
    ).fetchone()
    if pending is not None:
        return
    requeues = int(manifest.get("requeues") or 0)
    delay = _requeue_delay(requeues)
    log.warning("project %s cleanup ran out of attempts; queued again in %ss", project_id, delay)
    conn.execute(
        """
        INSERT INTO jobs (id, project_id, run_id, upload_id, scene_revision_id, capability, stage,
                          state, priority, attempts, max_attempts, fencing_token, available_at,
                          input_manifest)
        VALUES (%s, %s, NULL, NULL, NULL, 'cpu', 'project_cleanup', 'queued', %s, 0, 3, 0,
                now() + make_interval(secs => %s), %s)
        """,
        (
            str(uuid.uuid4()),
            project_id,
            PROJECT_CLEANUP_PRIORITY,
            float(delay),
            Jsonb({**manifest, "requeues": requeues + 1}),
        ),
    )


def requeue_upload_cleanup(
    conn: psycopg.Connection, project_id: str, upload_id: str | None, manifest: Mapping[str, Any]
) -> None:
    """Queue an upload's cleanup again, later, after it used up its attempts.

    The upload keeps its immutable key, and with it its quota, until the cleanup has
    deleted its bytes, and nothing in the capture page can ask for that again. So, as for
    a project's deletion, only the paths that exhaust retries call this, the wait doubles
    up to a day and the cleanup never gives up. Nothing is queued once the project is no
    longer active (its deletion removes the bytes), when the upload is gone or still
    uploading, or while another cleanup of it is pending.
    """
    if upload_id is None:
        return
    upload = conn.execute(
        "SELECT u.status FROM uploads u JOIN projects p ON p.id = u.project_id "
        "WHERE u.id = %s AND u.project_id = %s AND p.lifecycle = 'active'",
        (upload_id, project_id),
    ).fetchone()
    if upload is None or upload["status"] == "initiated":
        return
    pending = conn.execute(
        "SELECT 1 FROM jobs WHERE upload_id = %s AND stage = 'upload_cleanup' "
        "AND state IN ('queued', 'retry_wait', 'running') LIMIT 1",
        (upload_id,),
    ).fetchone()
    if pending is not None:
        return
    requeues = int(manifest.get("requeues") or 0)
    delay = _requeue_delay(requeues)
    log.warning("upload %s cleanup ran out of attempts; queued again in %ss", upload_id, delay)
    conn.execute(
        """
        INSERT INTO jobs (id, project_id, upload_id, capability, stage, state, priority,
                          attempts, max_attempts, fencing_token, available_at, input_manifest)
        VALUES (%s, %s, %s, 'cpu', 'upload_cleanup', 'queued', 5, 0, 3, 0,
                now() + make_interval(secs => %s), %s)
        """,
        (str(uuid.uuid4()), project_id, upload_id, float(delay),
         Jsonb({**manifest, "requeues": requeues + 1})),
    )  # fmt: skip


def _requeue_delay(requeues: int) -> float:
    """A requeued cleanup's wait: doubling from an hour, up to a day."""
    return min(
        ARTIFACT_CLEANUP_REQUEUE_CAP_SECONDS,
        ARTIFACT_CLEANUP_REQUEUE_SECONDS * 2 ** min(requeues, 16),
    )


def fail_subject(conn: psycopg.Connection, job: ClaimedJob) -> None:
    """Finalize what a non-run job was working on once it failed for good."""
    if job.stage == "upload_inspection":
        fail_upload(conn, job.project_id, job.upload_id)
    else:
        fail_revision(conn, job.scene_revision_id)


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
    leftovers: Sequence[str] = (),
) -> None:
    """Mark the job succeeded and insert the next stage in ONE fenced transaction.

    ``leftovers`` (here and in the other transitions) are attempt prefixes the worker
    could not delete; they are queued for ``artifact_cleanup`` when the job ends.
    Raises :class:`Cancelled` (nothing written) when cancellation won the race,
    and :class:`LeaseLost` when the fence moved or the lease already expired.
    """
    with db.transaction() as conn:
        _lock(conn, job, require_live=True)
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
        _queue_leftovers(conn, job, leftovers)
        if next_job is not None:
            _insert_next(conn, job, next_job)
        elif finish_run and job.run_level:
            conn.execute(
                "UPDATE reconstruction_runs SET status = 'succeeded', finished_at = now(), "
                "updated_at = now() WHERE id = %s AND status IN ('queued', 'running')",
                (job.run_id,),
            )
            settle_ledger(conn, job.run_id)  # type: ignore[arg-type]


def cancel_job(
    db: Database, job: ClaimedJob, *, usage_seconds: float = 0.0, leftovers: Sequence[str] = ()
) -> bool:
    """Acknowledge cancellation after the process group has been stopped (fenced)."""
    with db.transaction() as conn:
        try:
            _lock(conn, job, require_live=False)
        except LeaseLost:
            return False
        conn.execute(_clear_lease_sql("cancelled") + " WHERE id = %s", (job.id,))
        _event(conn, job.id, job.stage, "cancelled", "Cancelled.")
        add_usage(conn, job, usage_seconds)
        _queue_leftovers(conn, job, leftovers)
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
    leftovers: Sequence[str] = (),
) -> str:
    """Permanent failure. Returns the resulting job state ('failed' or 'cancelled')."""
    with db.transaction() as conn:
        locked = _lock(conn, job, require_live=False)
        _queue_leftovers(conn, job, leftovers)
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
        else:
            fail_subject(conn, job)
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
    leftovers: Sequence[str] = (),
) -> str:
    """Transient failure: retry_wait with backoff, or failed when attempts are exhausted.

    A retried job's next attempt sweeps ``leftovers`` again, and a job cancelled before
    that attempt has all its fences queued, so only the terminal outcomes queue them.
    """
    with db.transaction() as conn:
        locked = _lock(conn, job, require_live=False)
        if locked.stop_requested:
            conn.execute(_clear_lease_sql("cancelled") + " WHERE id = %s", (job.id,))
            _event(conn, job.id, job.stage, "cancelled", "Cancelled.")
            add_usage(conn, job, usage_seconds)
            _queue_leftovers(conn, job, leftovers)
            if job.run_id:
                finalize_run_if_cancelled(conn, job.run_id)
            return "cancelled"
        add_usage(conn, job, usage_seconds)
        if locked.row["attempts"] >= locked.row["max_attempts"]:
            _queue_leftovers(conn, job, leftovers)
            final = {**error, "retryable": False}
            conn.execute(
                _clear_lease_sql("failed") + ", error = %s WHERE id = %s",
                (Jsonb(final), job.id),
            )
            _event(conn, job.id, job.stage, "failed", "Retries exhausted.")
            if job.run_level:
                fail_run(conn, job.run_id, final)  # type: ignore[arg-type]
            else:
                fail_subject(conn, job)
            if job.stage == "artifact_cleanup":
                requeue_artifact_cleanup(conn, job.project_id, job.input_manifest)
            elif job.stage == "project_cleanup":
                requeue_project_cleanup(conn, job.project_id, locked.row["input_manifest"])
            elif job.stage == "upload_cleanup":
                manifest = locked.row["input_manifest"] or {}
                requeue_upload_cleanup(conn, job.project_id, job.upload_id, manifest)
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
    """Recover expired leases: bump the fence and mark cancelled / failed / queued.

    Each job is reaped in its own transaction, which locks the job's project, then its
    run, then the job: the order of every other transition and of the API's
    cancellation, so the reaper never holds a job while it waits for that job's run.
    """
    with db.transaction() as conn:
        expired = conn.execute(
            "SELECT id, project_id, run_id FROM jobs "
            "WHERE state = 'running' AND lease_expires_at < now() ORDER BY lease_expires_at, id"
        ).fetchall()
    outcomes = []
    for job in expired:
        with db.transaction() as conn:
            outcome = _reap(conn, job)
        if outcome is not None:
            outcomes.append(outcome)
    for outcome in outcomes:
        log.warning("reaped job %s -> %s", outcome.job_id, outcome.state)
    return outcomes


def _reap(conn: psycopg.Connection, job: Mapping[str, Any]) -> ReapOutcome | None:
    """Reap one job found expired, or None when it was renewed or settled since."""
    job_id, project_id = str(job["id"]), str(job["project_id"])
    run_id = str(job["run_id"]) if job["run_id"] else None
    conn.execute("SELECT 1 FROM projects WHERE id = %s FOR SHARE", (project_id,))
    if run_id:
        conn.execute("SELECT 1 FROM reconstruction_runs WHERE id = %s FOR UPDATE", (run_id,))
    lost = error_body("WORKER_LOST")
    row = conn.execute(
        f"""
        WITH expired AS (
          SELECT j.id, j.capability,
                 CASE WHEN {CANCEL_COND} THEN 'cancelled'
                      WHEN j.attempts >= j.max_attempts THEN 'failed'
                      ELSE 'queued' END AS outcome,
                 -- The lost attempt ran at least until its last heartbeat, which
                 -- set updated_at (the lease then ran on for up to 90 s with no
                 -- worker behind it, so that tail is not charged).
                 GREATEST(1, CEIL(EXTRACT(EPOCH FROM
                   (LEAST(j.updated_at, j.lease_expires_at)
                    - COALESCE(j.started_at, j.lease_expires_at)))))
                   AS charge_seconds
          FROM jobs j
          WHERE j.id = %(id)s AND j.state = 'running' AND j.lease_expires_at < now()
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
        RETURNING j.id, j.project_id, j.run_id, j.upload_id, j.scene_revision_id, j.stage,
                  j.state, j.fencing_token, j.input_manifest,
                  e.capability, e.charge_seconds
        """,
        {"err": Jsonb(lost), "id": job_id},
    ).fetchone()
    if row is None:
        return None
    message = {
        "queued": "Worker lease expired; re-queued.",
        "failed": "Worker lease expired too many times.",
        "cancelled": "Cancelled after the worker stopped.",
    }[row["state"]]
    _event(conn, job_id, row["stage"], row["state"], message)
    if run_id and row["scene_revision_id"] is None:
        # Charge the lost attempt so worker losses cannot multiply the run budget
        # (the stale worker records nothing once its lease is gone).
        gpu = row["capability"] == "gpu"
        column = "actual_gpu_seconds" if gpu else "actual_cpu_seconds"
        conn.execute(
            f"UPDATE usage_ledger SET {column} = {column} + %s WHERE run_id = %s",  # noqa: S608
            (int(row["charge_seconds"]), run_id),
        )
    if run_id and row["state"] in ("failed", "cancelled"):
        # The lost attempt wrote its outputs under the fence the bump just
        # replaced, and its sweep of the earlier fences may have failed. A job
        # that ended here is never claimed again, so nothing would sweep them. A
        # re-queued job is swept by its next attempt, or by the cancellation
        # paths if it is cancelled before that attempt starts.
        attempts = [
            attempt_prefix(project_id, run_id, row["stage"], job_id, fence)
            for fence in range(1, int(row["fencing_token"]))
        ]
        queue_artifact_cleanup(conn, project_id, attempts, reason="attempt_lost")
    if run_id and row["state"] == "failed" and row["scene_revision_id"] is None:
        fail_run(conn, run_id, lost)
    elif row["state"] == "failed" and row["stage"] == "upload_inspection":
        upload_id = str(row["upload_id"]) if row["upload_id"] else None
        fail_upload(conn, str(row["project_id"]), upload_id)
    elif row["state"] == "failed" and row["scene_revision_id"] is not None:
        fail_revision(conn, str(row["scene_revision_id"]))
    elif row["state"] == "failed" and row["stage"] == "artifact_cleanup":
        requeue_artifact_cleanup(conn, project_id, row["input_manifest"] or {})
    elif row["state"] == "failed" and row["stage"] == "project_cleanup":
        requeue_project_cleanup(conn, project_id, row["input_manifest"] or {})
    elif row["state"] == "failed" and row["stage"] == "upload_cleanup":
        upload_id = str(row["upload_id"]) if row["upload_id"] else None
        requeue_upload_cleanup(conn, project_id, upload_id, row["input_manifest"] or {})
    elif run_id and row["state"] == "cancelled":
        finalize_run_if_cancelled(conn, run_id)
    return ReapOutcome(job_id, run_id, row["state"])


def promote_retry_wait(db: Database) -> int:
    with db.transaction() as conn:
        cur = conn.execute(
            f"""
            UPDATE jobs AS j SET state = 'queued', updated_at = now()
            WHERE j.state = 'retry_wait' AND j.available_at <= now() AND NOT {CANCEL_COND}
            """
        )
        return cur.rowcount


def _queue_unclaimed_attempts(conn: psycopg.Connection, rows: Sequence[Mapping[str, Any]]) -> None:
    """Queue the outputs of every fence that a cancelled run job may have written under.

    A job cancelled while queued or retrying will not be claimed again, so no later
    attempt sweeps what its earlier fences left. Its fence counts every claim and every
    reaper bump, so fences 1..fencing_token cover them all. Jobs that were never claimed
    (fence 0) and rows without a run have nothing to clean.
    """
    by_project: dict[str, list[str]] = {}
    for row in rows:
        if row["run_id"] is None or int(row["fencing_token"]) < 1:
            continue
        project_id, run_id, job_id = str(row["project_id"]), str(row["run_id"]), str(row["id"])
        by_project.setdefault(project_id, []).extend(
            attempt_prefix(project_id, run_id, row["stage"], job_id, fence)
            for fence in range(1, int(row["fencing_token"]) + 1)
        )
    for project_id, prefixes in by_project.items():
        queue_artifact_cleanup(conn, project_id, prefixes, reason="cancelled_before_claim")


def sweep_cancelled(db: Database) -> int:
    """Cancel queued/retry_wait jobs whose job, run or project was cancelled/deleted."""
    with db.transaction() as conn:
        rows = conn.execute(
            f"""
            UPDATE jobs AS j SET state = 'cancelled', finished_at = now(), updated_at = now()
            WHERE j.state IN ('queued', 'retry_wait') AND {CANCEL_COND}
            RETURNING j.id, j.project_id, j.run_id, j.stage, j.fencing_token
            """
        ).fetchall()
        runs = {str(r["run_id"]) for r in rows if r["run_id"]}
        for row in rows:
            _event(conn, str(row["id"]), row["stage"], "cancelled", "Cancelled.")
        _queue_unclaimed_attempts(conn, rows)
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
