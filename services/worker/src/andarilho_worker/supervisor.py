"""Executes one claimed job: heartbeat, cancellation, stage call, fenced commit.

The supervisor owns leases, budgets, subprocess termination and committing
results; stage functions only compute outputs (README section 8).
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg

from andarilho_worker import queue
from andarilho_worker.artifacts import ObjectStore
from andarilho_worker.db import Database
from andarilho_worker.errors import (
    Cancelled,
    Deferred,
    LeaseLost,
    StageError,
    TransientError,
    error_body,
)
from andarilho_worker.metrics import measure
from andarilho_worker.queue import ClaimedJob, NextJob
from andarilho_worker.recipes import Recipe, recipe_for
from andarilho_worker.runner import (
    ProcessRunner,
    ProcessTimeout,
    Redactor,
    RunnerError,
    ScratchBudgetExceeded,
)
from andarilho_worker.settings import Settings
from andarilho_worker.stages import STAGES
from andarilho_worker.stages.base import (
    CancellationToken,
    JobInfo,
    Stage,
    StageContext,
    StageResult,
    StageServices,
)

log = logging.getLogger(__name__)
MAINTENANCE_TIMEOUT = 1800


class Heartbeat:
    """Renews the lease on its own thread, independent of subprocess output."""

    def __init__(self, db: Database, job: ClaimedJob, token: CancellationToken,
                 interval: float, lease_seconds: float) -> None:  # fmt: skip
        self.db, self.job, self.token = db, job, token
        self.interval, self.lease_seconds = interval, lease_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"heartbeat-{job.id}", daemon=True)
        self.last_ok = time.monotonic()

    def start(self) -> Heartbeat:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def beat(self) -> None:
        try:
            if queue.heartbeat(self.db, self.job):
                self.last_ok = time.monotonic()
                return
            status = queue.fence_status(self.db, self.job)
            if status == "cancelled":
                self.token.cancel()
            elif status == "lost":
                self.token.lose_lease()
        except psycopg.Error:
            log.warning("heartbeat failed for job %s", self.job.id)
            if time.monotonic() - self.last_ok >= self.lease_seconds:
                self.token.lose_lease()  # our lease has certainly expired: stop now

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.beat()
            if self.token.lease_lost:
                return


class Supervisor:
    def __init__(
        self,
        db: Database,
        store: ObjectStore,
        settings: Settings,
        *,
        runner: ProcessRunner | None = None,
        stages: Mapping[str, Stage] | None = None,
        delay_fn: Callable[[int], float] | None = None,
    ) -> None:
        self.db = db
        self.store = store
        self.settings = settings
        self.runner = runner or ProcessRunner(redactor=Redactor(settings.secrets()))
        self.stages = dict(stages or STAGES)
        self.tokens: dict[str, CancellationToken] = {}
        self._tokens_lock = threading.Lock()
        self.delay_fn = delay_fn or (
            lambda attempt: queue.backoff_seconds(
                attempt, base=settings.retry_base_seconds, cap=settings.retry_max_seconds
            )
        )

    def abandon_all(self) -> None:
        """Stop every running stage (process groups are killed) without further writes."""
        with self._tokens_lock:
            for token in self.tokens.values():
                token.lose_lease()

    # -- helpers ---------------------------------------------------------------
    def _read(self, fn: Callable[[Any], Any]) -> Any:
        with self.db.transaction() as conn:
            return fn(conn)

    def _remaining_run_seconds(self, job: ClaimedJob) -> float:
        row = self._read(
            lambda c: c.execute(
                """
                SELECT (r.resource_budget ->> 'maxSeconds')::float AS max_seconds,
                       COALESCE(l.actual_cpu_seconds + l.actual_gpu_seconds, 0) AS used
                FROM reconstruction_runs r
                LEFT JOIN usage_ledger l ON l.run_id = r.id
                WHERE r.id = %s
                """,
                (job.run_id,),
            ).fetchone()
        )
        ceiling = float(self.settings.run_max_seconds)
        if row is None:
            return ceiling
        if row["max_seconds"] is not None:
            ceiling = min(ceiling, float(row["max_seconds"]))
        return ceiling - float(row["used"])

    def _recipe(self, job: ClaimedJob) -> Recipe | None:
        if not job.run_id:
            return None
        run = self._read(
            lambda c: c.execute(
                "SELECT quality, recipe_name, recipe_digest FROM reconstruction_runs WHERE id = %s",
                (job.run_id,),
            ).fetchone()
        )
        recipe = recipe_for(run["quality"])
        declared = (job.input_manifest.get("recipe") or {}).get("stages")
        pinned = [{"stage": s.stage, "capability": s.capability} for s in recipe.stages]
        if (
            declared
            and [{"stage": d.get("stage"), "capability": d.get("capability")} for d in declared]
            != pinned
        ):
            # The worker's pinned recipe is authoritative; never follow an unknown stage list.
            log.warning(
                "run %s declares a stage list that differs from the pinned recipe", job.run_id
            )
        if run["recipe_digest"] != recipe.digest:
            log.warning(
                "run %s recipe digest %s differs from the worker's pinned %s",
                job.run_id,
                run["recipe_digest"],
                recipe.digest,
            )
        return recipe

    def _next(self, job: ClaimedJob, recipe: Recipe | None, result: StageResult) -> NextJob | None:
        if recipe is None or not job.run_id:
            return None
        spec = recipe.next_stage(job.stage, layout_pipeline=job.scene_revision_id is not None)
        if spec is None:
            return None
        prior = dict(job.input_manifest)
        outputs = {**dict(prior.get("outputs", {})), **dict(result.outputs)}
        stage_results = {
            **dict(prior.get("stageResults", {})),
            job.stage: {"resultManifestKey": result.result_manifest_key,
                        "resultSha256": result.result_sha256},
        }  # fmt: skip
        manifest = {**prior, "outputs": outputs, "stageResults": stage_results}
        return NextJob(spec.stage, spec.capability, manifest)

    # -- execution -------------------------------------------------------------
    def execute(self, job: ClaimedJob) -> str:
        stage_fn = self.stages.get(job.stage)
        recipe = None
        try:
            recipe = self._recipe(job)
        except KeyError:
            return self._fail(job, error_body("INTERNAL_ERROR"), None, 0.0)
        if stage_fn is None:
            return self._fail(job, error_body("INTERNAL_ERROR"), None, 0.0)
        timeout = MAINTENANCE_TIMEOUT
        if recipe is not None and job.stage in recipe.stage_names():
            timeout = recipe.spec(job.stage).timeout_seconds
        timeout = min(timeout, self.settings.run_max_seconds)
        if job.run_id:
            # The run budget is a ceiling for the whole run, not per stage: subtract
            # the seconds every earlier stage and attempt already consumed.
            remaining = self._remaining_run_seconds(job)
            if remaining <= 0:
                return self._fail(job, error_body("BUDGET_EXHAUSTED"), None, 0.0)
            timeout = min(timeout, remaining)
        deadline = datetime.now(UTC) + timedelta(seconds=timeout)
        token = CancellationToken(deadline)
        with self._tokens_lock:
            self.tokens[job.id] = token
        scratch = Path(self.settings.worker_scratch_dir) / f"{job.id}-{job.fencing_token}"
        scratch.mkdir(parents=True, exist_ok=True)
        heartbeat = Heartbeat(
            self.db, job, token, self.settings.heartbeat_seconds, self.settings.lease_seconds
        ).start()

        def progress(message: str, data: Mapping[str, Any]) -> bool:
            ok = queue.record_event(self.db, job, message, data)
            if not ok:
                heartbeat.beat()
            return ok

        services = StageServices(
            store=self.store,
            runner=self.runner,
            settings=self.settings,
            db=self.db,
            progress=progress,
            fenced=lambda fn: queue.fenced_execute(self.db, job, fn),
            read=self._read,
        )
        ctx = StageContext(
            job_id=job.id,
            run_id=job.run_id or "",
            stage=job.stage,
            attempt=job.attempts,
            fencing_token=job.fencing_token,
            scratch=scratch,
            recipe_digest=recipe.digest if recipe else "",
            recipe=recipe.as_dict() if recipe else {},
            input_manifest=job.input_manifest,
            deadline=deadline,
            max_scratch_bytes=self.settings.max_scratch_bytes,
            cancellation=token,
            job=JobInfo(
                job.project_id, job.run_id, job.upload_id, job.scene_revision_id, job.capability
            ),  # fmt: skip
            services=services,
        )
        started = time.monotonic()
        with measure(job.stage, job.id, scratch) as metrics:
            try:
                outcome = self._run_stage(job, recipe, stage_fn, ctx, token, started)
            finally:
                heartbeat.stop()
                shutil.rmtree(scratch, ignore_errors=True)
                with self._tokens_lock:
                    self.tokens.pop(job.id, None)
            metrics.outcome = outcome
        return outcome

    def _run_stage(self, job, recipe, stage_fn, ctx, token, started) -> str:
        def used() -> float:
            return time.monotonic() - started

        try:
            result = stage_fn(ctx)
            token.raise_if_cancelled()
            queue.complete_job(
                self.db,
                job,
                result_manifest={
                    "resultManifestKey": result.result_manifest_key,
                    "resultSha256": result.result_sha256,
                    **dict(result.manifest),
                },
                next_job=self._next(job, recipe, result),
                commit=result.commit,
                usage_seconds=used(),
                finish_run=result.finish_run,
            )
            return "succeeded"
        except LeaseLost:
            log.warning("job %s lost its lease; stopping without writes", job.id)
            return "lost"
        except Cancelled:
            if token.lease_lost:
                return "lost"
            return "cancelled" if queue.cancel_job(self.db, job, usage_seconds=used()) else "lost"
        except Deferred as deferred:
            queue.defer_job(
                self.db, job, delay_seconds=deferred.delay_seconds, patch=deferred.manifest_patch
            )
            return "deferred"
        except TransientError as exc:
            return self._retry(job, exc.body(), used())
        except StageError as exc:
            return self._fail(job, exc.body(), exc.on_fail, used())
        except (ProcessTimeout, ScratchBudgetExceeded):
            return self._fail(job, error_body("BUDGET_EXHAUSTED"), None, used())
        except (RunnerError, psycopg.OperationalError, OSError) as exc:
            log.warning("job %s transient failure: %s", job.id, type(exc).__name__)
            return self._retry(job, error_body("TRANSIENT_FAILURE", retryable=True), used())
        except Exception:
            log.exception("job %s failed unexpectedly", job.id)
            return self._retry(job, error_body("INTERNAL_ERROR", retryable=True), used())

    def _retry(self, job: ClaimedJob, body: dict[str, Any], used: float) -> str:
        try:
            return queue.retry_job(
                self.db, job, body, delay_seconds=self.delay_fn(job.attempts), usage_seconds=used
            )
        except LeaseLost:
            return "lost"

    def _fail(self, job: ClaimedJob, body: dict[str, Any], on_fail, used: float) -> str:
        try:
            return queue.fail_job(self.db, job, body, on_fail=on_fail, usage_seconds=used)
        except LeaseLost:
            return "lost"
