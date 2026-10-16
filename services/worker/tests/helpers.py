"""Helpers shared by the worker's pipeline-level tests."""

from __future__ import annotations

from andarilho_worker import queue
from andarilho_worker.db import Database
from andarilho_worker.supervisor import Supervisor


def drain(db: Database, sup: Supervisor, *, limit: int = 30) -> list[tuple[str, str]]:
    """Run every claimable job through the supervisor; return ``(stage, outcome)`` in order."""
    done = []
    for _ in range(limit):
        queue.promote_retry_wait(db)
        job = queue.claim(db, "worker-test", "cpu")
        if job is None:
            break
        done.append((job.stage, sup.execute(job)))
    return done
