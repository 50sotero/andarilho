from __future__ import annotations

import threading
import uuid

import psycopg
import pytest
from psycopg.types.json import Jsonb

from andarilho_worker import queue
from andarilho_worker.errors import Cancelled, LeaseLost, error_body
from andarilho_worker.queue import NextJob


def expire(seed, job_id: str) -> None:
    seed.exec(
        "UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id = %s", job_id
    )


def first_revision(seed, project_id: str, run_id: str) -> str:
    """The scene and revision 1 a run's geometry_proposal stage creates (still pending)."""
    scene_id, revision_id = str(uuid.uuid4()), str(uuid.uuid4())
    seed.exec(
        "INSERT INTO scenes (id, project_id, run_id, version) VALUES (%s, %s, %s, 1)",
        scene_id, project_id, run_id,
    )  # fmt: skip
    seed.exec(
        "INSERT INTO scene_revisions (id, scene_id, project_id, run_id, revision, layout_revision, "
        "layout, validation_state, approval_state, created_by) "
        "VALUES (%s, %s, %s, %s, 1, 1, %s, 'pending', 'draft', 'user-1')",
        revision_id, scene_id, project_id, run_id, Jsonb({}),
    )  # fmt: skip
    return revision_id


def revision_findings(seed, revision_id: str) -> tuple[str, list[str]]:
    row = seed.one(
        "SELECT validation_state, validation_findings FROM scene_revisions WHERE id = %s",
        revision_id,
    )
    return row["validation_state"], [f["code"] for f in row["validation_findings"] or []]


def test_competing_workers_only_one_wins(db, seed):
    pid = seed.project()
    seed.run(pid)
    barrier = threading.Barrier(2)
    results = []

    def worker(name: str) -> None:
        barrier.wait()
        results.append(queue.claim(db, name, "cpu"))

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    claimed = [r for r in results if r is not None]
    assert len(claimed) == 1
    job = claimed[0]
    assert job.fencing_token == 1 and job.attempts == 1
    row = seed.one("SELECT state, lease_owner, lease_expires_at > now() AS leased FROM jobs")
    assert row["state"] == "running" and row["leased"]
    assert seed.one("SELECT status FROM reconstruction_runs")["status"] == "running"


def test_claim_respects_capability_and_cancellation(db, seed):
    pid = seed.project()
    run_id, _ = seed.run(pid, first_stage="training", capability="gpu", quality="preview")
    assert queue.claim(db, "w", "cpu") is None
    seed.exec("UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", run_id)
    assert queue.claim(db, "w", "gpu") is None


def test_claim_skips_deleting_project_but_allows_cleanup(db, seed):
    pid = seed.project(lifecycle="deleting")
    seed.run(pid)
    cleanup = seed.job(pid, "project_cleanup", {"projectId": pid}, priority=10)
    job = queue.claim(db, "w", "cpu")
    assert job is not None and job.id == cleanup
    assert queue.claim(db, "w", "cpu") is None


def test_heartbeat_extends_and_stale_fence_rejected(db, seed):
    pid = seed.project()
    seed.run(pid)
    job = queue.claim(db, "w1", "cpu")
    assert queue.heartbeat(db, job)
    assert queue.record_event(db, job, "progress", {"pct": 10})
    expire(seed, job.id)
    assert not queue.heartbeat(db, job)  # expired lease cannot be renewed
    outcomes = queue.reap_expired(db)
    assert [o.state for o in outcomes] == ["queued"]
    row = seed.one("SELECT state, fencing_token FROM jobs WHERE id = %s", job.id)
    assert row["state"] == "queued" and row["fencing_token"] == 2
    # Every write by the stale worker is now rejected.
    assert not queue.record_event(db, job, "late progress")
    assert queue.fence_status(db, job) == "lost"
    with pytest.raises(LeaseLost):
        queue.complete_job(db, job, result_manifest={}, next_job=None)
    with pytest.raises(LeaseLost):
        queue.fail_job(db, job, error_body("INVALID_MEDIA"))
    assert not queue.cancel_job(db, job)
    assert seed.one("SELECT state FROM jobs WHERE id = %s", job.id)["state"] == "queued"
    # A new claim gets a fresh fence and the old token stays dead.
    job2 = queue.claim(db, "w2", "cpu")
    assert job2.fencing_token == 3
    assert not queue.heartbeat(db, job)


def test_an_expired_lease_refuses_every_write_before_the_reaper_runs(db, seed):
    pid = seed.project()
    run_id, _ = seed.run(pid)
    job = queue.claim(db, "w1", "cpu")
    expire(seed, job.id)  # the fence still matches: no reaper has run
    ran = []
    with pytest.raises(LeaseLost):
        queue.fenced_execute(db, job, ran.append)
    assert ran == []
    assert not queue.record_event(db, job, "late progress")
    with pytest.raises(LeaseLost):
        queue.complete_job(db, job, result_manifest={}, next_job=None)
    with pytest.raises(LeaseLost):
        queue.fail_job(db, job, error_body("INVALID_MEDIA"))
    with pytest.raises(LeaseLost):
        queue.retry_job(db, job, error_body("TRANSIENT_FAILURE"), delay_seconds=0)
    assert not queue.defer_job(db, job, delay_seconds=60)
    assert not queue.cancel_job(db, job)
    row = seed.one("SELECT state, fencing_token, error FROM jobs WHERE id = %s", job.id)
    assert row["state"] == "running" and row["fencing_token"] == 1 and row["error"] is None
    run = seed.one("SELECT status FROM reconstruction_runs WHERE id = %s", run_id)
    assert run["status"] == "running"
    # The job is the reaper's to settle.
    assert [o.state for o in queue.reap_expired(db)] == ["queued"]


def test_the_reaper_waits_for_the_run_before_it_locks_the_job(db, seed):
    pid = seed.project()
    run_id, _ = seed.run(pid, max_attempts=1)
    job = queue.claim(db, "w", "cpu")
    expire(seed, job.id)
    reaped = []
    reaper = threading.Thread(target=lambda: reaped.extend(queue.reap_expired(db)))
    with db.transaction() as conn:
        # The API's cancellation: the project, then the run, then the run's jobs.
        conn.execute("SELECT 1 FROM projects WHERE id = %s FOR UPDATE", (pid,))
        conn.execute("SELECT 1 FROM reconstruction_runs WHERE id = %s FOR UPDATE", (run_id,))
        reaper.start()
        reaper.join(timeout=1.0)
        assert reaper.is_alive()  # waiting for the run, holding no lock on the job
        conn.execute(
            "UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", (run_id,)
        )
        conn.execute("UPDATE jobs SET cancel_requested_at = now() WHERE id = %s", (job.id,))
    reaper.join(timeout=10.0)
    assert [o.state for o in reaped] == ["cancelled"]
    assert seed.one("SELECT status FROM reconstruction_runs")["status"] == "cancelled"


def initiated_upload(seed, project_id: str, expires_in: str) -> str:
    upload_id = str(uuid.uuid4())
    seed.exec(
        "INSERT INTO uploads (id, project_id, created_by, media_kind, display_filename, "
        "declared_content_type, expected_bytes, staging_key, part_size, part_count, status, "
        "expires_at) VALUES (%s, %s, 'user-1', 'video', 'walk.mp4', 'video/mp4', 10, %s, "
        "16777216, 1, 'initiated', now() + %s::interval)",
        upload_id, project_id, f"tenants/{project_id}/staging/{upload_id}", expires_in,
    )  # fmt: skip
    return upload_id


def test_maintenance_expires_abandoned_uploads_and_deletes_them_a_day_later(db, seed):
    pid = seed.project()
    live = initiated_upload(seed, pid, "1 hour")
    lapsed = initiated_upload(seed, pid, "-1 minute")
    old = initiated_upload(seed, pid, "-2 days")
    referenced = initiated_upload(seed, pid, "-2 days")
    seed.job(pid, "upload_cleanup", {"uploadId": referenced, "projectId": pid},
             upload_id=referenced)  # fmt: skip

    stats = queue.maintenance(db)
    assert (stats["uploadsExpired"], stats["uploadsDeleted"]) == (3, 1)
    statuses = {str(r["id"]): r["status"] for r in seed.all("SELECT id, status FROM uploads")}
    assert statuses == {live: "initiated", lapsed: "expired", referenced: "expired"}
    assert old not in statuses
    # Nothing is left to do on the next pass until the lapsed one is a day old.
    assert queue.expire_uploads(db) == (0, 0)


def idempotency_record(
    seed,
    project_id: str,
    expires_in: str,
    body: dict | None = None,
    operation: str = "uploads.create",
) -> str:
    key = str(uuid.uuid4())
    seed.exec(
        "INSERT INTO idempotency_records (id, project_id, operation, idempotency_key, subject, "
        "request_hash, response_status, response_body, expires_at) VALUES (%s, %s, "
        "%s, %s, 'user-1', %s, 201, %s, now() + %s::interval)",
        str(uuid.uuid4()), project_id, operation, key, "0" * 64, Jsonb(body or {}), expires_in,
    )  # fmt: skip
    return key


def test_an_expired_upload_is_kept_while_its_creation_can_be_replayed(db, seed):
    # With IDEMPOTENCY_TTL_SECONDS above a day, a retry of the creation would replay a
    # response naming the upload, so the row must outlive that record, not just the day.
    pid = seed.project()
    upload = initiated_upload(seed, pid, "-2 days")
    created = {"upload": {"id": upload}, "partSize": 1, "partCount": 1}
    idempotency_record(seed, pid, "1 hour", created)
    # Records that cannot replay it do not keep it: an expired one, another operation's.
    idempotency_record(seed, pid, "-1 minute", created)
    idempotency_record(seed, pid, "1 hour", created, operation="runs.create")

    assert queue.expire_uploads(db) == (1, 0)
    assert seed.one("SELECT status FROM uploads WHERE id = %s", upload)["status"] == "expired"

    seed.exec(
        "UPDATE idempotency_records SET expires_at = now() WHERE operation = 'uploads.create'"
    )
    assert queue.expire_uploads(db) == (0, 1)
    assert seed.one("SELECT id FROM uploads WHERE id = %s", upload) is None


def test_maintenance_purges_expired_idempotency_records_a_batch_at_a_time(db, seed):
    # Clients send a fresh key with every request, so no lookup ever finds these again.
    pid = seed.project()
    live = idempotency_record(seed, pid, "1 hour")
    for _ in range(3):
        idempotency_record(seed, pid, "-1 minute")
    assert queue.purge_idempotency_records(db, batch=2) == 2
    assert queue.maintenance(db)["idempotencyPurged"] == 1
    keys = [
        r["idempotency_key"] for r in seed.all("SELECT idempotency_key FROM idempotency_records")
    ]
    assert keys == [live]


def test_reaper_fails_when_attempts_exhausted(db, seed):
    pid = seed.project()
    run_id, _ = seed.run(pid, max_attempts=1)
    job = queue.claim(db, "w", "cpu")
    expire(seed, job.id)
    assert [o.state for o in queue.reap_expired(db)] == ["failed"]
    run = seed.one("SELECT status, error FROM reconstruction_runs WHERE id = %s", run_id)
    assert run["status"] == "failed" and run["error"]["code"] == "WORKER_LOST"
    assert seed.one("SELECT state FROM usage_ledger")["state"] == "released"


def test_reaper_cancels_when_run_cancelled_and_releases_once(db, seed):
    pid = seed.project()
    run_id, _ = seed.run(pid)
    job = queue.claim(db, "w", "cpu")
    seed.exec("UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", run_id)
    seed.exec("UPDATE jobs SET cancel_requested_at = now() WHERE id = %s", job.id)
    expire(seed, job.id)
    assert [o.state for o in queue.reap_expired(db)] == ["cancelled"]
    assert seed.one("SELECT status FROM reconstruction_runs")["status"] == "cancelled"
    assert seed.one("SELECT state FROM usage_ledger")["state"] == "released"
    with db.transaction() as conn:
        assert not queue.release_ledger(conn, run_id)  # exactly once
        assert not queue.finalize_run_if_cancelled(conn, run_id)


def test_cancellation_while_queued_and_retry_wait(db, seed):
    pid = seed.project()
    run_id, job_id = seed.run(pid)
    seed.exec("UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", run_id)
    queue.sweep_cancelled(db)
    assert seed.one("SELECT state FROM jobs WHERE id = %s", job_id)["state"] == "cancelled"
    assert seed.one("SELECT status FROM reconstruction_runs")["status"] == "cancelled"

    pid2 = seed.project()
    run2, job2 = seed.run(pid2)
    job = queue.claim(db, "w", "cpu")
    assert job.id == job2
    state = queue.retry_job(db, job, error_body("TRANSIENT_FAILURE"), delay_seconds=3600)
    assert state == "retry_wait"
    seed.exec("UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", run2)
    assert queue.promote_retry_wait(db) == 0
    queue.sweep_cancelled(db)
    assert seed.one("SELECT state FROM jobs WHERE id = %s", job2)["state"] == "cancelled"
    ledger = seed.one("SELECT state FROM usage_ledger WHERE run_id = %s", run2)
    assert ledger["state"] == "released"


def test_cancellation_while_running_is_acknowledged(db, seed):
    pid = seed.project()
    run_id, _ = seed.run(pid)
    job = queue.claim(db, "w", "cpu")
    seed.exec("UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", run_id)
    seed.exec("UPDATE jobs SET cancel_requested_at = now() WHERE id = %s", job.id)
    assert not queue.heartbeat(db, job)
    assert queue.fence_status(db, job) == "cancelled"
    assert not queue.record_event(db, job, "progress after cancel")
    nxt = NextJob("preparing", "cpu", {})
    with pytest.raises(Cancelled):
        queue.complete_job(db, job, result_manifest={}, next_job=nxt)
    assert seed.one("SELECT count(*) AS n FROM jobs")["n"] == 1  # no next stage inserted
    assert queue.cancel_job(db, job)
    assert seed.one("SELECT state FROM jobs")["state"] == "cancelled"
    assert seed.one("SELECT status FROM reconstruction_runs")["status"] == "cancelled"
    assert seed.one("SELECT state FROM usage_ledger")["state"] == "released"


def test_failure_after_cancel_becomes_cancelled(db, seed):
    pid = seed.project()
    run_id, _ = seed.run(pid)
    job = queue.claim(db, "w", "cpu")
    seed.exec("UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", run_id)
    assert queue.fail_job(db, job, error_body("INVALID_MEDIA")) == "cancelled"
    assert seed.one("SELECT status FROM reconstruction_runs")["status"] == "cancelled"


def test_retry_backoff_then_exhaustion(db, seed):
    pid = seed.project()
    run_id, job_id = seed.run(pid, max_attempts=2)
    job = queue.claim(db, "w", "cpu")
    assert (
        queue.retry_job(db, job, error_body("TRANSIENT_FAILURE"), delay_seconds=0) == "retry_wait"
    )
    row = seed.one("SELECT state, lease_owner FROM jobs WHERE id = %s", job_id)
    assert row["state"] == "retry_wait" and row["lease_owner"] is None
    assert queue.claim(db, "w", "cpu") is None  # not queued until promoted
    assert queue.promote_retry_wait(db) == 1
    job = queue.claim(db, "w", "cpu")
    assert job.attempts == 2
    assert queue.retry_job(db, job, error_body("TRANSIENT_FAILURE"), delay_seconds=0) == "failed"
    run = seed.one("SELECT status, error FROM reconstruction_runs WHERE id = %s", run_id)
    assert run["status"] == "failed" and run["error"]["retryable"] is False
    assert seed.one("SELECT state FROM usage_ledger")["state"] == "released"


def test_backoff_grows_with_jitter():
    import random

    rng = random.Random(1)
    delays = [queue.backoff_seconds(a, base=10, cap=100, rng=rng) for a in (1, 2, 3, 6)]
    assert 10 <= delays[0] <= 20
    assert 20 <= delays[1] <= 30
    assert 40 <= delays[2] <= 50
    assert 100 <= delays[3] <= 110


def test_complete_inserts_next_stage_once(db, seed):
    pid = seed.project()
    run_id, job_id = seed.run(pid)
    job = queue.claim(db, "w", "cpu")
    # Simulate a racing duplicate: the next stage already exists.
    seed.job(pid, "preparing", {"dup": True}, run_id=run_id)
    queue.complete_job(
        db, job, result_manifest={"ok": True}, next_job=NextJob("preparing", "cpu", {"x": 1}),
        usage_seconds=5,
    )  # fmt: skip
    rows = seed.all(
        "SELECT stage, input_manifest FROM jobs WHERE run_id = %s ORDER BY stage", run_id
    )
    assert [r["stage"] for r in rows] == ["preparing", "validating"]
    assert rows[0]["input_manifest"] == {"dup": True}
    assert seed.one("SELECT state FROM jobs WHERE id = %s", job_id)["state"] == "succeeded"
    assert seed.one("SELECT actual_cpu_seconds FROM usage_ledger")["actual_cpu_seconds"] == 5
    # A direct duplicate insert fails on the unique index.
    with pytest.raises(psycopg.errors.UniqueViolation):
        seed.job(pid, "preparing", {}, run_id=run_id)


def test_final_stage_settles_ledger(db, seed):
    pid = seed.project()
    run_id, _ = seed.run(pid, first_stage="scene_validation")
    job = queue.claim(db, "w", "cpu")
    queue.complete_job(db, job, result_manifest={}, next_job=None, finish_run=True, usage_seconds=3)
    assert seed.one("SELECT status FROM reconstruction_runs")["status"] == "succeeded"
    ledger = seed.one("SELECT state, actual_cpu_seconds FROM usage_ledger")
    assert ledger["state"] == "settled" and ledger["actual_cpu_seconds"] == 3


def test_project_deleting_blocks_commit(db, seed):
    pid = seed.project()
    seed.run(pid)
    job = queue.claim(db, "w", "cpu")
    seed.exec("UPDATE projects SET lifecycle = 'deleting' WHERE id = %s", pid)
    assert not queue.heartbeat(db, job)
    with pytest.raises(Cancelled):
        queue.complete_job(db, job, result_manifest={}, next_job=NextJob("preparing", "cpu", {}))
    assert seed.one("SELECT count(*) AS n FROM jobs")["n"] == 1


def test_reaper_charges_lost_attempts_to_the_run_budget(db, seed):
    pid = seed.project()
    seed.run(pid, max_attempts=3)
    job = queue.claim(db, "w", "cpu")
    # The attempt ran for ten minutes before its last lease lapsed.
    seed.exec(
        "UPDATE jobs SET started_at = now() - interval '11 minutes', "
        "lease_expires_at = now() - interval '1 minute' WHERE id = %s",
        job.id,
    )
    assert [o.state for o in queue.reap_expired(db)] == ["queued"]
    used = seed.one("SELECT actual_cpu_seconds FROM usage_ledger")["actual_cpu_seconds"]
    assert 595 <= used <= 605


def test_reaper_charges_a_lost_attempt_only_until_its_last_heartbeat(db, seed):
    pid = seed.project()
    seed.run(pid, max_attempts=3)
    job = queue.claim(db, "w", "cpu")
    # Five minutes of work, a heartbeat, then the worker died at once: its lease
    # ran on for another 90 s with nothing behind it.
    seed.exec(
        "UPDATE jobs SET started_at = now() - interval '10 minutes', "
        "updated_at = now() - interval '5 minutes', "
        "lease_expires_at = now() - interval '210 seconds' WHERE id = %s",
        job.id,
    )
    assert [o.state for o in queue.reap_expired(db)] == ["queued"]
    used = seed.one("SELECT actual_cpu_seconds FROM usage_ledger")["actual_cpu_seconds"]
    assert 295 <= used <= 305


def test_a_failed_run_stops_its_first_revision_pending(db, seed):
    pid = seed.project()
    run_id, _ = seed.run(pid, first_stage="packaging")
    revision_id = first_revision(seed, pid, run_id)
    job = queue.claim(db, "w", "cpu")
    assert queue.fail_job(db, job, error_body("INVALID_MEDIA")) == "failed"
    assert revision_findings(seed, revision_id) == ("invalid", ["RUN_FAILED"])

    pid2 = seed.project()
    run2, _ = seed.run(pid2, first_stage="packaging", max_attempts=1)
    revision2 = first_revision(seed, pid2, run2)
    job = queue.claim(db, "w", "cpu")
    assert queue.retry_job(db, job, error_body("TRANSIENT_FAILURE"), delay_seconds=0) == "failed"
    assert revision_findings(seed, revision2) == ("invalid", ["RUN_FAILED"])


def test_a_cancelled_run_stops_its_first_revision_pending(db, seed):
    pid = seed.project()
    run_id, _ = seed.run(pid, first_stage="packaging")
    revision_id = first_revision(seed, pid, run_id)
    job = queue.claim(db, "w", "cpu")
    seed.exec("UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", run_id)
    seed.exec("UPDATE jobs SET cancel_requested_at = now() WHERE id = %s", job.id)
    assert queue.cancel_job(db, job)
    assert seed.one("SELECT status FROM reconstruction_runs")["status"] == "cancelled"
    assert revision_findings(seed, revision_id) == ("invalid", ["RUN_CANCELLED"])


def cleanup_jobs(seed, project_id: str) -> list[dict]:
    return seed.all(
        "SELECT * FROM jobs WHERE project_id = %s AND stage = 'artifact_cleanup' ORDER BY id",
        project_id,
    )


def seconds_until_available(seed, job_id: str) -> float:
    row = seed.one(
        "SELECT EXTRACT(EPOCH FROM available_at - now()) AS s FROM jobs WHERE id = %s", job_id
    )
    return float(row["s"])


def test_reaper_queues_the_lost_attempt_when_it_fails_a_run_job(db, seed):
    pid = seed.project()
    run_id, job_id = seed.run(pid, max_attempts=1)
    job = queue.claim(db, "w", "cpu")  # fence 1
    expire(seed, job.id)
    assert [o.state for o in queue.reap_expired(db)] == ["failed"]
    [cleanup] = cleanup_jobs(seed, pid)
    assert cleanup["state"] == "queued" and cleanup["run_id"] is None
    assert cleanup["scene_revision_id"] is None and cleanup["upload_id"] is None
    assert cleanup["input_manifest"] == {
        "projectId": pid,
        "prefixes": [f"tenants/{pid}/runs/{run_id}/attempts/validating/{job_id}/1/"],
        "reason": "attempt_lost",
    }
    assert 290 < seconds_until_available(seed, cleanup["id"]) <= 300


def test_reaper_queues_cleanup_for_cancelled_attempts_but_not_requeued_ones(db, seed):
    pid = seed.project()
    run_id, job_id = seed.run(pid)
    job = queue.claim(db, "w", "cpu")
    seed.exec("UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", run_id)
    seed.exec("UPDATE jobs SET cancel_requested_at = now() WHERE id = %s", job.id)
    expire(seed, job.id)
    assert [o.state for o in queue.reap_expired(db)] == ["cancelled"]
    [cleanup] = cleanup_jobs(seed, pid)
    assert cleanup["input_manifest"]["prefixes"] == [
        f"tenants/{pid}/runs/{run_id}/attempts/validating/{job_id}/1/"
    ]

    # A lost attempt that is re-queued is swept by its successor, not queued here.
    pid2 = seed.project()
    _, job2 = seed.run(pid2, max_attempts=3)
    job = queue.claim(db, "w", "cpu")
    assert job.id == job2
    expire(seed, job.id)
    assert [o.state for o in queue.reap_expired(db)] == ["queued"]
    assert cleanup_jobs(seed, pid2) == []


def test_reaper_queues_every_fence_a_job_ran_under_when_it_ends_it(db, seed):
    # A lost attempt's sweep of the earlier fences may have failed before it was lost.
    pid = seed.project()
    run_id, job_id = seed.run(pid, max_attempts=2)
    for outcome in ("queued", "failed"):
        job = queue.claim(db, "w", "cpu")  # fences 1 and 3; each reaper bump adds one
        expire(seed, job.id)
        assert [o.state for o in queue.reap_expired(db)] == [outcome]
    [cleanup] = cleanup_jobs(seed, pid)
    attempts = f"tenants/{pid}/runs/{run_id}/attempts/validating/{job_id}/"
    assert cleanup["input_manifest"]["prefixes"] == [f"{attempts}{fence}/" for fence in (1, 2, 3)]


@pytest.mark.parametrize("ending", ["failed", "cancelled"])
def test_a_run_that_ends_unfinished_queues_what_its_finished_stages_wrote(db, seed, ending):
    pid = seed.project()
    run_id, validating = seed.run(pid)
    job = queue.claim(db, "w", "cpu")  # validating succeeds at fence 1
    queue.complete_job(db, job, result_manifest={}, next_job=NextJob("preparing", "cpu", {}))
    job = queue.claim(db, "w", "cpu")
    assert job.stage == "preparing"
    if ending == "failed":
        assert queue.fail_job(db, job, error_body("INVALID_MEDIA")) == "failed"
    else:
        seed.exec(
            "UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", run_id
        )
        assert queue.cancel_job(db, job)
    assert seed.one("SELECT status FROM reconstruction_runs")["status"] == ending
    [cleanup] = cleanup_jobs(seed, pid)
    assert cleanup["input_manifest"] == {
        "projectId": pid,
        "prefixes": [f"tenants/{pid}/runs/{run_id}/attempts/validating/{validating}/1/"],
        "reason": "run_ended",
    }


def test_failing_a_run_queues_its_first_revision_scene_prefix(db, seed):
    pid = seed.project()
    run_id, _ = seed.run(pid, first_stage="packaging")
    revision_id = first_revision(seed, pid, run_id)
    job = queue.claim(db, "w", "cpu")
    assert queue.fail_job(db, job, error_body("INVALID_MEDIA")) == "failed"
    [cleanup] = cleanup_jobs(seed, pid)
    assert cleanup["run_id"] is None and cleanup["scene_revision_id"] is None
    assert cleanup["input_manifest"] == {
        "projectId": pid,
        "prefixes": [f"tenants/{pid}/scenes/{revision_id}/"],
        "reason": "revision_invalid",
        "sceneRevisionId": revision_id,
    }


def test_a_stage_that_invalidates_its_revision_before_failing_still_queues_cleanup(db, seed):
    pid = seed.project()
    run_id, _ = seed.run(pid, first_stage="scene_validation")
    revision_id = first_revision(seed, pid, run_id)

    def invalidate(conn, _job):  # what scene_validation's failure hook does first
        conn.execute(
            "UPDATE scene_revisions SET validation_state = 'invalid', validation_findings = %s "
            "WHERE id = %s",
            (Jsonb([{"code": "MANIFEST", "message": "manifest missing"}]), revision_id),
        )

    job = queue.claim(db, "w", "cpu")
    body = error_body("SCENE_VALIDATION_FAILED")
    assert queue.fail_job(db, job, body, on_fail=invalidate) == "failed"
    assert revision_findings(seed, revision_id) == ("invalid", ["MANIFEST"])  # not overwritten
    [cleanup] = cleanup_jobs(seed, pid)
    assert cleanup["input_manifest"]["prefixes"] == [f"tenants/{pid}/scenes/{revision_id}/"]


def test_failing_a_layout_revision_queues_its_scene_prefix(db, seed):
    pid = seed.project()
    run_id, _ = seed.run(pid, first_stage="validating")
    seed.exec("DELETE FROM jobs WHERE run_id = %s", run_id)  # only the layout job below
    revision_id = first_revision(seed, pid, run_id)
    seed.job(
        pid, "packaging", {"sceneRevisionId": revision_id}, run_id=run_id,
        scene_revision_id=revision_id,
    )  # fmt: skip
    job = queue.claim(db, "w", "cpu")
    assert job.scene_revision_id == revision_id
    assert queue.fail_job(db, job, error_body("INVALID_MEDIA")) == "failed"
    assert revision_findings(seed, revision_id)[0] == "invalid"
    [cleanup] = cleanup_jobs(seed, pid)
    assert cleanup["input_manifest"]["sceneRevisionId"] == revision_id
    assert cleanup["input_manifest"]["prefixes"] == [f"tenants/{pid}/scenes/{revision_id}/"]


def test_failing_a_run_queues_the_attempts_of_its_cancelled_siblings(db, seed):
    pid = seed.project()
    run_id, _ = seed.run(pid, max_attempts=1)
    job = queue.claim(db, "w", "cpu")  # validating, fence 1
    requeued = seed.job(pid, "preparing", {}, run_id=run_id)  # claimed twice, then re-queued
    seed.exec("UPDATE jobs SET fencing_token = 2 WHERE id = %s", requeued)
    retrying = seed.job(pid, "camera_reconstruction", {}, run_id=run_id)
    seed.exec("UPDATE jobs SET state = 'retry_wait', fencing_token = 1 WHERE id = %s", retrying)
    never_claimed = seed.job(pid, "training", {}, run_id=run_id)

    assert queue.fail_job(db, job, error_body("INVALID_MEDIA")) == "failed"
    for sibling in (requeued, retrying, never_claimed):
        assert seed.one("SELECT state FROM jobs WHERE id = %s", sibling)["state"] == "cancelled"
    [cleanup] = cleanup_jobs(seed, pid)
    assert cleanup["input_manifest"] == {
        "projectId": pid,
        "prefixes": [
            f"tenants/{pid}/runs/{run_id}/attempts/camera_reconstruction/{retrying}/1/",
            f"tenants/{pid}/runs/{run_id}/attempts/preparing/{requeued}/1/",
            f"tenants/{pid}/runs/{run_id}/attempts/preparing/{requeued}/2/",
        ],
        "reason": "cancelled_before_claim",
    }


def test_identical_cleanup_requests_are_queued_once(db, seed):
    pid = seed.project()
    scenes = f"tenants/{pid}/scenes/{uuid.uuid4()}/"
    attempts = f"tenants/{pid}/runs/{uuid.uuid4()}/attempts/packaging/{uuid.uuid4()}/1/"
    with db.transaction() as conn:
        queue.queue_artifact_cleanup(conn, pid, [attempts, scenes], reason="attempt_lost")
        queue.queue_artifact_cleanup(conn, pid, [scenes, attempts, scenes], reason="attempt_lost")
        queue.queue_artifact_cleanup(conn, pid, [], reason="attempt_lost")  # a no-op
    assert len(cleanup_jobs(seed, pid)) == 1

    with db.transaction() as conn:
        queue.queue_artifact_cleanup(conn, pid, [scenes], reason="revision_invalid")
    assert len(cleanup_jobs(seed, pid)) == 2

    # Once the earlier request has finished, the same prefixes are queued again.
    seed.exec("UPDATE jobs SET state = 'succeeded' WHERE stage = 'artifact_cleanup'")
    with db.transaction() as conn:
        queue.queue_artifact_cleanup(conn, pid, [scenes, attempts], reason="attempt_lost")
    assert len(cleanup_jobs(seed, pid)) == 3


def test_cleanup_prefixes_must_be_directories_of_the_project(db, seed):
    pid = seed.project()
    other = seed.project()
    with db.transaction() as conn:
        with pytest.raises(ValueError):
            queue.queue_artifact_cleanup(
                conn, pid, [f"tenants/{other}/scenes/{uuid.uuid4()}/"], reason="x"
            )
        with pytest.raises(ValueError):
            queue.queue_artifact_cleanup(conn, pid, [f"tenants/{pid}/scenes/x"], reason="x")
    assert cleanup_jobs(seed, pid) == []


def test_cancelling_a_requeued_job_queues_every_fence_it_may_have_written(db, seed):
    pid = seed.project()
    run_id, job_id = seed.run(pid, max_attempts=3)
    queue.claim(db, "w", "cpu")  # fence 1
    expire(seed, job_id)
    assert [o.state for o in queue.reap_expired(db)] == ["queued"]  # fence 2, nothing queued
    assert cleanup_jobs(seed, pid) == []
    seed.exec("UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", run_id)
    assert queue.sweep_cancelled(db) == 1  # cancelled before any worker claims it again

    [cleanup] = cleanup_jobs(seed, pid)
    assert cleanup["run_id"] is None
    assert cleanup["input_manifest"] == {
        "projectId": pid,
        "prefixes": [
            f"tenants/{pid}/runs/{run_id}/attempts/validating/{job_id}/1/",
            f"tenants/{pid}/runs/{run_id}/attempts/validating/{job_id}/2/",
        ],
        "reason": "cancelled_before_claim",
    }
    assert 290 < seconds_until_available(seed, cleanup["id"]) <= 300


def test_cancelling_a_job_that_never_ran_queues_no_cleanup(db, seed):
    pid = seed.project()
    run_id, _ = seed.run(pid)
    seed.exec("UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", run_id)
    assert queue.sweep_cancelled(db) == 1
    assert cleanup_jobs(seed, pid) == []


def test_a_retrying_cleanup_request_is_not_queued_again(db, seed):
    pid = seed.project()
    prefix = f"tenants/{pid}/scenes/{uuid.uuid4()}/"
    with db.transaction() as conn:
        queue.queue_artifact_cleanup(conn, pid, [prefix], reason="revision_invalid")
    seed.exec("UPDATE jobs SET state = 'retry_wait' WHERE stage = 'artifact_cleanup'")
    with db.transaction() as conn:
        queue.queue_artifact_cleanup(conn, pid, [prefix], reason="revision_invalid")
    assert len(cleanup_jobs(seed, pid)) == 1
