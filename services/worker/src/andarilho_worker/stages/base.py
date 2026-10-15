"""Worker orchestration contract (README section 8).

One function per stage takes a :class:`StageContext` and returns a
:class:`StageResult`. Stage functions never publish a project directly: any
database change they need is returned as a ``commit`` hook that the supervisor
runs inside the fenced terminal transaction (or, for idempotent preparatory
rows, executed through :meth:`StageServices.fenced`).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from andarilho_worker.artifacts import ObjectStore, attempt_prefix
from andarilho_worker.errors import Cancelled, CommitHook, LeaseLost, StageError
from andarilho_worker.runner import ProcessRunner, RunResult


class Cancellation(Protocol):
    def raise_if_cancelled(self) -> None: ...


class CancellationToken:
    """Set by the heartbeat thread (or the supervisor); polled by stages and the runner."""

    def __init__(self, deadline: datetime | None = None) -> None:
        self._cancelled = threading.Event()
        self._lost = threading.Event()
        self.deadline = deadline

    def cancel(self) -> None:
        self._cancelled.set()

    def lose_lease(self) -> None:
        self._lost.set()

    @property
    def lease_lost(self) -> bool:
        return self._lost.is_set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def raise_if_cancelled(self) -> None:
        if self._lost.is_set():
            raise LeaseLost("lease lost")
        if self._cancelled.is_set():
            raise Cancelled("cancellation requested")
        if self.deadline is not None and datetime.now(self.deadline.tzinfo) > self.deadline:
            raise StageError("BUDGET_EXHAUSTED")


@dataclass(frozen=True)
class JobInfo:
    project_id: str
    run_id: str | None
    upload_id: str | None
    scene_revision_id: str | None
    capability: str


@dataclass
class StageServices:
    store: ObjectStore
    runner: ProcessRunner
    settings: Any
    db: Any = None  # andarilho_worker.db.Database
    progress: Callable[[str, Mapping[str, Any]], bool] = lambda message, progress: True
    fenced: Callable[[Callable[[Any], Any]], Any] | None = None
    read: Callable[[Callable[[Any], Any]], Any] | None = None
    # The scene analysis's model client (``complete_json``). None builds one from the
    # settings; the smoke runner passes one that records the request or replays a reply.
    vision: Any = None


@dataclass(frozen=True)
class StageContext:
    job_id: str
    run_id: str
    stage: str
    attempt: int
    fencing_token: int
    scratch: Path
    recipe_digest: str
    recipe: Mapping[str, object]
    input_manifest: Mapping[str, object]
    deadline: datetime
    max_scratch_bytes: int
    cancellation: Cancellation
    # Andarilho extensions to the README contract.
    job: JobInfo = field(default=None)  # type: ignore[assignment]
    services: StageServices = field(default=None)  # type: ignore[assignment]

    # -- helpers ------------------------------------------------------------
    @property
    def project_id(self) -> str:
        return self.job.project_id

    @property
    def store(self) -> ObjectStore:
        return self.services.store

    @property
    def params(self) -> Mapping[str, Any]:
        return self.recipe.get("params", {})  # type: ignore[return-value]

    def attempt_key(self, name: str) -> str:
        prefix = attempt_prefix(
            self.project_id, self.run_id, self.stage, self.job_id, self.fencing_token
        )
        return prefix + name

    def outputs(self) -> Mapping[str, Any]:
        return self.input_manifest.get("outputs", {})  # type: ignore[return-value]

    def output(self, name: str) -> Any:
        value = self.outputs().get(name)
        if value is None:
            raise StageError("MISSING_INPUT", details={"missing": name})
        return value

    def progress(self, message: str, **progress: Any) -> None:
        self.cancellation.raise_if_cancelled()
        self.services.progress(message, progress)

    def remaining_seconds(self) -> float:
        return max(1.0, (self.deadline - datetime.now(self.deadline.tzinfo)).total_seconds())

    def run(self, argv: list[str], *, timeout: float | None = None, **kw: Any) -> RunResult:
        return self.services.runner.run(
            argv,
            timeout=min(timeout or self.remaining_seconds(), self.remaining_seconds()),
            check_cancelled=self.cancellation.raise_if_cancelled,
            scratch=self.scratch,
            max_scratch_bytes=self.max_scratch_bytes,
            **kw,
        )

    def query(self, fn: Callable[[Any], Any]) -> Any:
        if self.services.read is None:
            raise RuntimeError("no database reader configured")
        return self.services.read(fn)

    def fenced(self, fn: Callable[[Any], Any]) -> Any:
        if self.services.fenced is None:
            raise RuntimeError("no fenced writer configured")
        return self.services.fenced(fn)


@dataclass(frozen=True)
class StageResult:
    result_manifest_key: str
    result_sha256: str
    manifest: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    commit: CommitHook | None = None
    finish_run: bool = False


class Stage(Protocol):
    def __call__(self, context: StageContext) -> StageResult: ...


def finish(
    ctx: StageContext,
    manifest: Mapping[str, Any],
    *,
    outputs: Mapping[str, Any] | None = None,
    commit: CommitHook | None = None,
    finish_run: bool = False,
    store_manifest: bool = True,
) -> StageResult:
    """Write the result manifest to the attempt path and build the StageResult."""
    payload = {
        "stage": ctx.stage,
        "jobId": ctx.job_id,
        "attempt": ctx.attempt,
        "fencingToken": ctx.fencing_token,
        "recipeDigest": ctx.recipe_digest,
        **manifest,
        "outputs": dict(outputs or {}),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    sha = hashlib.sha256(blob).hexdigest()
    key = ""
    if store_manifest and ctx.run_id:
        key = ctx.attempt_key("result.json")
        ctx.store.put_bytes(key, blob, "application/json")
    return StageResult(key, sha, payload, dict(outputs or {}), commit, finish_run)
