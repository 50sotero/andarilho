"""Stable, user-safe error codes and the exception types stages raise.

Public messages never contain filesystem paths, subprocess output, SQL or
signed URLs. Details for operators go to the worker log (already redacted).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

# code -> (safe message, suggested action)
CATALOG: dict[str, tuple[str, str]] = {
    "INSUFFICIENT_OVERLAP": (
        "Too few views overlapped to reconstruct the place reliably.",
        "Record again, moving slowly and keeping each view overlapping the previous one.",
    ),
    "DISCONNECTED_CAPTURE": (
        "Parts of the capture could not be connected to each other.",
        "Record a slow continuous walk through every doorway between rooms.",
    ),
    "CAMERA_REGISTRATION_FAILED": (
        "Camera positions could not be estimated from this capture.",
        "Record again in good light with textured surfaces in view.",
    ),
    "CAPTURE_TOO_BLURRY": (
        "Most frames were too blurry to use.",
        "Move the camera more slowly and avoid low light.",
    ),
    "INSUFFICIENT_INPUTS": (
        "The capture set does not contain enough usable media.",
        "Add a walkthrough video or at least 30 overlapping photos.",
    ),
    "INVALID_MEDIA": (
        "A file could not be decoded as the declared media type.",
        "Upload the original file exported from the camera or phone.",
    ),
    "UNSUPPORTED_MEDIA": (
        "A file type or codec is not supported.",
        "Use MP4/MOV (H.264/HEVC) video, JPEG/PNG photos, or a PNG/JPEG/PDF floor plan.",
    ),
    "INPUT_NOT_VERIFIED": (
        "An input has not been verified or changed after verification.",
        "Create a new capture set from verified uploads.",
    ),
    "BUDGET_EXHAUSTED": (
        "Processing exceeded its time or resource budget.",
        "Try a shorter capture or the preview quality.",
    ),
    "GPU_REQUIRED": (
        "This stage needs a GPU worker, which is not available.",
        "Try again later.",
    ),
    "INVALID_LAYOUT": (
        "The layout has invalid geometry.",
        "Fix the highlighted walls, floors or obstacles and save again.",
    ),
    "SCENE_VALIDATION_FAILED": (
        "The packaged scene did not pass validation.",
        "Adjust the layout so the spawn point is clear and on a floor, then save again.",
    ),
    "TRAINING_FAILED": (
        "The visual representation could not be trained.",
        "Retry later or record the place again.",
    ),
    "EXPORT_INVALID": (
        "The exported scene was incomplete.",
        "Retry the run.",
    ),
    "ARTIFACT_CONFLICT": (
        "A scene file already exists with different content.",
        "Retry the run to create a new revision.",
    ),
    "MISSING_INPUT": (
        "A required intermediate result is missing.",
        "Retry the run.",
    ),
    "WORKER_LOST": (
        "Processing stopped unexpectedly too many times.",
        "Retry the run.",
    ),
    "TRANSIENT_FAILURE": (
        "A temporary infrastructure problem interrupted processing.",
        "Retry the run.",
    ),
    "INTERNAL_ERROR": (
        "Processing failed because of an internal error.",
        "Retry the run. Contact support if it keeps failing.",
    ),
    "CLEANUP_PENDING": (
        "Project deletion is waiting for running work to stop.",
        "No action needed; cleanup retries automatically.",
    ),
}


def error_body(code: str, message: str | None = None, *, retryable: bool = False) -> dict[str, Any]:
    default_message, action = CATALOG.get(code, CATALOG["INTERNAL_ERROR"])
    return {
        "code": code,
        "message": message or default_message,
        "retryable": retryable,
        "suggestedAction": action,
    }


CommitHook = Callable[[Any, Any], None]
"""``hook(connection, job)`` executed inside the fenced terminal transaction."""


class StageError(Exception):
    """Permanent failure: the job (and its run) fail with a stable code."""

    retryable = False

    def __init__(
        self,
        code: str,
        message: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
        on_fail: CommitHook | None = None,
    ) -> None:
        self.code = code
        self.public_message = message or CATALOG.get(code, CATALOG["INTERNAL_ERROR"])[0]
        self.details = dict(details or {})
        self.on_fail = on_fail
        super().__init__(f"{code}: {self.public_message}")

    def body(self) -> dict[str, Any]:
        body = error_body(self.code, self.public_message, retryable=self.retryable)
        if self.details:
            body["details"] = self.details
        return body


class TransientError(StageError):
    """Infrastructure failure: retry with exponential backoff and jitter."""

    retryable = True

    def __init__(self, code: str = "TRANSIENT_FAILURE", message: str | None = None, **kw: Any):
        super().__init__(code, message, **kw)


@dataclass
class Deferred(Exception):  # noqa: N818 - control-flow signal, not an error
    """Re-queue later without consuming an attempt (used by project cleanup)."""

    delay_seconds: float
    reason: str = "CLEANUP_PENDING"
    manifest_patch: Mapping[str, Any] | None = None


class Cancelled(Exception):  # noqa: N818
    """Cancellation was requested for the job, its run or its project."""


class LeaseLost(Exception):  # noqa: N818
    """The lease/fence moved on: this worker must stop without writing anything."""
