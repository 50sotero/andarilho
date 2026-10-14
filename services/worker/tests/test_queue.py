from __future__ import annotations

import threading

import psycopg
import pytest

from andarilho_worker import queue
from andarilho_worker.errors import Cancelled, LeaseLost, error_body
from andarilho_worker.queue import NextJob


def expire(seed, job_id: str) -> None:
    seed.exec(
        "UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id = %s", job_id
    )


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
