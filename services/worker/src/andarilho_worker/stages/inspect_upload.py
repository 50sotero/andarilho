"""``upload_inspection``: verify one frozen upload (README section 7, steps 5-7).

Reads the exact immutable key/version, checks the file signature against the
declared media kind (MP4/MOV ``ftyp``, JPEG, PNG, PDF), rejects any
SVG/HTML/archive/executable content, decodes images and probes videos with
FFmpeg/FFprobe when available, walks PDF page trees in a bounded subprocess
(rejecting encrypted, empty or oversized PDFs), computes the SHA-256 of the
complete object and marks
the upload ``verified`` or ``rejected``. The final update is conditional on
``status = 'uploaded_unverified'`` and an active project, inside the fenced
completion transaction.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from psycopg.types.json import Jsonb

from andarilho_worker.errors import StageError, TransientError
from andarilho_worker.media import (
    ALLOWED,
    MAX_IMAGE_PIXELS,
    jpeg_dimensions,
    parse_ffprobe,
    png_dimensions,
    sniff,
)
from andarilho_worker.recipes import render_argv
from andarilho_worker.runner import ProcessFailed, ProcessTimeout, tool_available
from andarilho_worker.stages.base import StageContext, StageResult, finish

MAX_INSPECT_BYTES = 8 * 1024**3
PDF_MAX_BYTES = 64 * 1024**2
# Wall-clock bound for one image decode or PDF walk (normally well under a second).
CONTENT_CHECK_SECONDS = 120


class Rejected(Exception):  # noqa: N818
    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


def _load(conn: Any, ctx: StageContext) -> dict[str, Any] | None:
    return conn.execute(
        "SELECT id, project_id, media_kind, immutable_key, storage_version, status, "
        "expected_bytes FROM uploads WHERE id = %s AND project_id = %s",
        (ctx.job.upload_id, ctx.project_id),
    ).fetchone()


def _inspect_pdf(ctx: StageContext, local: Any) -> dict[str, Any]:
    argv = render_argv("pdf_inspect", python=sys.executable, input=local)
    try:
        result = ctx.run(argv, timeout=CONTENT_CHECK_SECONDS)
        report = json.loads(result.stdout)
    except (ProcessFailed, ProcessTimeout, ValueError) as exc:
        raise Rejected("UNSUPPORTED_MEDIA", "PDF could not be inspected") from exc
    if not report.get("ok"):
        raise Rejected("UNSUPPORTED_MEDIA", str(report.get("reason") or "PDF is not supported"))
    return {"pages": int(report["pages"]), "hasJavaScript": bool(report["hasJavaScript"])}


def _inspect(ctx: StageContext, upload: dict[str, Any]) -> dict[str, Any]:
    key, version = upload["immutable_key"], upload["storage_version"]
    head = ctx.store.head(key, version)
    if head is None:
        # Nothing left to verify: either a previous attempt already deleted these
        # rejected bytes and its database commit was lost, or the object vanished.
        # Finalizing as rejected keeps the upload from staying unverified forever.
        raise Rejected("INVALID_MEDIA", "stored object is missing")
    if head.size <= 0 or head.size > MAX_INSPECT_BYTES:
        raise Rejected("INVALID_MEDIA", "size out of range")
    if head.size != upload["expected_bytes"]:
        raise Rejected("INVALID_MEDIA", "size differs from the declared size")
    local = ctx.scratch / "object"
    sha256 = ctx.store.download(key, local, version)
    actual = local.stat().st_size
    if actual != head.size:
        raise TransientError()  # storage returned a different length; retry
    with local.open("rb") as fh:
        head_bytes = fh.read(4096)
    found = sniff(head_bytes)
    kind = upload["media_kind"]
    if found.kind == "forbidden":
        raise Rejected("UNSUPPORTED_MEDIA", f"forbidden content ({found.detail})")
    if found.kind not in ALLOWED[kind]:
        raise Rejected("UNSUPPORTED_MEDIA", f"signature {found.kind} not allowed for {kind}")
    meta: dict[str, Any] = {"format": found.kind, "signature": found.detail or found.kind}
    if found.kind in ("jpeg", "png"):
        with local.open("rb") as fh:
            blob = fh.read(min(actual, 4 * 1024 * 1024))
        dims = png_dimensions(blob) if found.kind == "png" else jpeg_dimensions(blob)
        if dims is None or dims[0] <= 0 or dims[1] <= 0:
            raise Rejected("INVALID_MEDIA", "image dimensions unreadable")
        if dims[0] * dims[1] > MAX_IMAGE_PIXELS:
            raise Rejected("INVALID_MEDIA", "image has too many pixels")
        meta.update(width=dims[0], height=dims[1])
        if tool_available("ffmpeg"):
            # A valid header says nothing about the pixels: decode them now, so a
            # truncated file is rejected before it can be picked for a run.
            try:
                ctx.run(render_argv("ffmpeg_decode_image", input=local),
                        timeout=CONTENT_CHECK_SECONDS)  # fmt: skip
            except (ProcessFailed, ProcessTimeout) as exc:
                raise Rejected("INVALID_MEDIA", "image could not be decoded") from exc
            meta["decoded"] = True
        else:
            meta["decoded"] = False  # preparation decodes it with FFmpeg before use
    elif found.kind == "pdf":
        if actual > PDF_MAX_BYTES:
            raise Rejected("INVALID_MEDIA", "PDF too large")
        meta.update(_inspect_pdf(ctx, local))
    else:  # mp4 / mov
        if tool_available("ffprobe"):
            try:
                result = ctx.run(render_argv("ffprobe", input=local), timeout=120)
                probe = parse_ffprobe(result.stdout)
            except (ProcessFailed, ProcessTimeout, ValueError) as exc:
                raise Rejected("INVALID_MEDIA", "video could not be probed") from exc
            if probe["codec"] not in ("h264", "hevc"):
                raise Rejected("UNSUPPORTED_MEDIA", "unsupported video codec")
            meta.update(probe, probed=True)
        else:
            meta["probed"] = False  # preparation probes again with FFprobe before use
    return {"sha256": sha256, "bytes": actual, "metadata": meta}


def run(ctx: StageContext) -> StageResult:
    upload = ctx.query(lambda conn: _load(conn, ctx))
    if upload is None or upload["immutable_key"] is None:
        raise StageError("MISSING_INPUT")
    if upload["status"] != "uploaded_unverified":
        return finish(ctx, {"verdict": "skipped", "status": upload["status"]}, store_manifest=False)
    upload_id = str(upload["id"])
    try:
        outcome = _inspect(ctx, upload)
    except Rejected as rejected:
        code, reason = rejected.code, rejected.reason

        def reject(conn: Any, job: Any) -> None:
            row = conn.execute(
                """
                UPDATE uploads SET status = 'rejected', rejection_code = %s, updated_at = now(),
                       media_metadata = media_metadata || %s
                WHERE id = %s AND status = 'uploaded_unverified'
                  AND EXISTS (SELECT 1 FROM projects p WHERE p.id = uploads.project_id
                              AND p.lifecycle = 'active')
                RETURNING immutable_key
                """,
                (code, Jsonb({"rejectionReason": reason}), upload_id),
            ).fetchone()
            if row is None:
                return
            # Rejected bytes keep counting toward the project quota until they are
            # gone: delete every version, then clear the key that releases the quota.
            ctx.store.delete_prefix(row["immutable_key"])
            conn.execute(
                "UPDATE uploads SET immutable_key = NULL, storage_version = NULL WHERE id = %s",
                (upload_id,),
            )

        return finish(
            ctx, {"verdict": "rejected", "code": code}, commit=reject, store_manifest=False
        )

    def verify(conn: Any, job: Any) -> None:
        row = conn.execute(
            """
            UPDATE uploads SET status = 'verified', sha256 = %s, actual_bytes = %s,
                   media_metadata = %s, updated_at = now()
            WHERE id = %s AND status = 'uploaded_unverified'
              AND immutable_key = %s AND storage_version IS NOT DISTINCT FROM %s
              AND EXISTS (SELECT 1 FROM projects p WHERE p.id = uploads.project_id
                          AND p.lifecycle = 'active')
            RETURNING id
            """,
            (
                outcome["sha256"],
                outcome["bytes"],
                Jsonb(outcome["metadata"]),
                upload_id,
                upload["immutable_key"],
                upload["storage_version"],
            ),
        ).fetchone()
        if row is None:
            raise StageError("INPUT_NOT_VERIFIED", "The upload changed during inspection.")

    return finish(
        ctx,
        {"verdict": "verified", "sha256": outcome["sha256"], "bytes": outcome["bytes"]},
        commit=verify,
        store_manifest=False,
    )
