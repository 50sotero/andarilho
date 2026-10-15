"""Worker entry point: ``python -m andarilho_worker``.

Loop: run maintenance (reaper, cancellation sweep, retry promotion, checks of
aborted multipart uploads), claim jobs for the advertised capability and
supervise up to ``WORKER_MAX_CONCURRENCY`` of them in threads. SIGTERM/SIGINT
stop claiming; running jobs get a short grace period, then their process groups
are terminated and the jobs are abandoned without further writes, left
recoverable through their leases.

How GPU stages are gated (infra notes)
--------------------------------------
* Every job row carries a ``capability`` (``cpu``/``gpu``) copied from the pinned
  recipe when the API inserts the first job and when the worker inserts each
  next stage. ``camera_reconstruction`` is the only ``gpu`` stage in the
  ``preview``/``standard`` recipes; the ``synthetic`` recipe has no GPU stage.
* A worker only claims jobs whose capability equals ``WORKER_CAPABILITY``
  (claim SQL filters on it), so CPU workers never pick up GPU stages and a run
  simply waits in ``queued`` at the first GPU stage until a GPU worker exists.
* Readiness (``python -m andarilho_worker --check``, also run at start-up)
  verifies the database, the object store and the advertised capability. A
  ``gpu`` worker must find ``nvidia-smi`` (and it must succeed) and ``colmap``;
  otherwise it refuses to start rather than claiming GPU jobs it cannot run.
* Run one GPU worker per GPU (``WORKER_MAX_CONCURRENCY=1``) and scale workers
  from the number of queued ``gpu`` jobs. The GPU image is
  ``infra/Dockerfile.worker-gpu``; the toolchain must be validated in that exact
  image (a working ``nvidia-smi`` does not prove that COLMAP was built with CUDA).
* Budgets: each stage has a pinned timeout (capped by ``RUN_MAX_SECONDS``), the
  scratch directory has a byte budget, and a stage exceeding either fails with
  ``BUDGET_EXHAUSTED`` instead of retrying. Actual seconds are added to
  ``usage_ledger.actual_gpu_seconds``/``actual_cpu_seconds`` on every terminal
  transition; the reservation is settled on success and released exactly once
  on cancellation/failure.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import psycopg

from andarilho_worker import multipart, queue
from andarilho_worker.artifacts import LocalObjectStore, ObjectStore, S3ObjectStore
from andarilho_worker.db import Database
from andarilho_worker.runner import Redactor
from andarilho_worker.settings import Settings
from andarilho_worker.supervisor import Supervisor

log = logging.getLogger("andarilho_worker")
GPU_TOOLS = ("nvidia-smi", "colmap")


class RedactingFilter(logging.Filter):
    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self.redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.redactor(record.getMessage())
        record.args = None
        return True


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    handler.addFilter(RedactingFilter(Redactor(settings.secrets())))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)


def capability_problems(capability: str) -> list[str]:
    if capability != "gpu":
        return []
    problems = [f"{tool} not found" for tool in GPU_TOOLS if shutil.which(tool) is None]
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            subprocess.run(  # noqa: S603 - fixed argv, shell=False
                [smi, "-L"], check=True, capture_output=True, timeout=20
            )
        except (subprocess.SubprocessError, OSError):
            problems.append("nvidia-smi failed")
    return problems


def build_store(settings: Settings) -> ObjectStore:
    if settings.local_store_dir is not None:
        return LocalObjectStore(settings.local_store_dir)
    return S3ObjectStore.from_settings(settings)


def readiness(settings: Settings, db: Database, store: ObjectStore) -> list[str]:
    problems = capability_problems(settings.worker_capability)
    try:
        db.ping()
    except psycopg.Error:
        problems.append("database unreachable")
    try:
        store.list_keys("tenants/readiness-probe/")
    except Exception:  # any storage failure means not ready
        problems.append("object store unreachable")
    try:
        if shutil.disk_usage(settings.worker_scratch_dir).free < 1024**3:
            problems.append("less than 1 GiB free scratch space")
    except OSError:
        problems.append("scratch directory unavailable")
    return problems


class Worker:
    def __init__(self, settings: Settings, db: Database, store: ObjectStore) -> None:
        self.settings = settings
        self.db = db
        self.store = store
        self.supervisor = Supervisor(db, store, settings)
        self.stopping = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=settings.worker_max_concurrency)
        self.active: set[Future] = set()

    def stop(self, *_args: object) -> None:
        log.info("shutdown requested; no new claims")
        self.stopping.set()

    def run_forever(self) -> None:
        last_maintenance = 0.0
        while not self.stopping.is_set():
            now = time.monotonic()
            try:
                if now - last_maintenance >= self.settings.reaper_seconds:
                    last_maintenance = now
                    stats = queue.maintenance(self.db)
                    stats |= multipart.reconcile(
                        self.db,
                        self.store,
                        horizon_seconds=self.settings.cleanup_reconcile_horizon_seconds,
                    )
                    if any(stats.values()):
                        log.info("maintenance %s", stats)
                self.active = {f for f in self.active if not f.done()}
                claimed = False
                while len(self.active) < self.settings.worker_max_concurrency:
                    job = queue.claim(
                        self.db, self.settings.worker_id, self.settings.worker_capability
                    )
                    if job is None:
                        break
                    claimed = True
                    self.active.add(self.pool.submit(self.supervisor.execute, job))
            except psycopg.Error as exc:
                log.warning("queue error: %s", type(exc).__name__)
                claimed = False
            self.stopping.wait(0.2 if claimed else self.settings.poll_seconds)
        self.drain()

    def drain(self, grace_seconds: float = 20.0) -> None:
        """Give running jobs a short grace period, then stop their process groups.

        Abandoned jobs write nothing more; their leases expire and the reaper
        re-queues them (or cancels them if their run was cancelled meanwhile).
        """
        deadline = time.monotonic() + grace_seconds
        while any(not f.done() for f in self.active) and time.monotonic() < deadline:
            time.sleep(0.2)
        self.supervisor.abandon_all()
        self.pool.shutdown(wait=True, cancel_futures=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="andarilho_worker")
    parser.add_argument("--check", action="store_true", help="run readiness checks and exit")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    configure_logging(settings)
    settings.worker_scratch_dir.mkdir(parents=True, exist_ok=True)
    db = Database(settings.database_url)
    store = build_store(settings)
    problems = readiness(settings, db, store)
    if problems:
        for problem in problems:
            log.error("not ready: %s", problem)
        return 1
    if args.check:
        log.info("ready (capability=%s)", settings.worker_capability)
        return 0
    worker = Worker(settings, db, store)
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    log.info(
        "worker %s started (capability=%s, concurrency=%d)",
        settings.worker_id,
        settings.worker_capability,
        settings.worker_max_concurrency,
    )
    worker.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
