"""Per-stage timing and byte counters (logged as structured key=value lines)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from andarilho_worker.runner import directory_bytes

log = logging.getLogger("andarilho_worker.metrics")


@dataclass
class StageMetrics:
    stage: str
    job_id: str
    started: float = field(default_factory=time.monotonic)
    seconds: float = 0.0
    scratch_peak_bytes: int = 0
    outcome: str = "running"

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "jobId": self.job_id,
            "seconds": round(self.seconds, 3),
            "scratchPeakBytes": self.scratch_peak_bytes,
            "outcome": self.outcome,
        }


@contextmanager
def measure(stage: str, job_id: str, scratch: Path | None = None) -> Iterator[StageMetrics]:
    metrics = StageMetrics(stage, job_id)
    try:
        yield metrics
    finally:
        metrics.seconds = time.monotonic() - metrics.started
        if scratch is not None and scratch.exists():
            metrics.scratch_peak_bytes = directory_bytes(scratch)
        log.info(
            "stage_metrics stage=%s job=%s outcome=%s seconds=%.3f scratch_bytes=%d",
            stage,
            job_id,
            metrics.outcome,
            metrics.seconds,
            metrics.scratch_peak_bytes,
        )
