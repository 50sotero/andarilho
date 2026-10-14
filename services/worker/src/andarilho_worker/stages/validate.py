"""``validating``: confirm the capture set references verified, immutable inputs."""

from __future__ import annotations

from typing import Any

from andarilho_worker.errors import StageError
from andarilho_worker.stages.base import StageContext, StageResult, finish

MIN_PHOTOS_WITHOUT_VIDEO = 30


def _load(conn: Any, ctx: StageContext) -> dict[str, Any]:
    run = conn.execute(
        "SELECT capture_set_id, quality, include_supplementary_photos FROM reconstruction_runs "
        "WHERE id = %s AND project_id = %s",
        (ctx.run_id, ctx.project_id),
    ).fetchone()
    if run is None:
        raise StageError("MISSING_INPUT")
    capture = conn.execute(
        "SELECT id, inputs, floor_plan_upload_id FROM capture_sets "
        "WHERE id = %s AND project_id = %s",
        (run["capture_set_id"], ctx.project_id),
    ).fetchone()
    if capture is None:
        raise StageError("MISSING_INPUT")
    ids = [str(i.get("uploadId")) for i in capture["inputs"]]
    uploads = {}
    if ids:
        rows = conn.execute(
            "SELECT id, status, immutable_key, storage_version, sha256, actual_bytes, media_kind "
            "FROM uploads WHERE project_id = %s AND id = ANY(%s::uuid[])",
            (ctx.project_id, ids),
        ).fetchall()
        uploads = {str(r["id"]): r for r in rows}
    return {"run": run, "capture": capture, "uploads": uploads}


def run(ctx: StageContext) -> StageResult:
    data = ctx.query(lambda conn: _load(conn, ctx))
    capture, uploads, run_row = data["capture"], data["uploads"], data["run"]
    inputs = list(capture["inputs"])
    synthetic = run_row["quality"] == "synthetic"
    params = ctx.params
    verified: list[dict[str, Any]] = []
    total_bytes = 0
    for item in inputs:
        ctx.cancellation.raise_if_cancelled()
        upload = uploads.get(str(item.get("uploadId")))
        if (
            upload is None
            or upload["status"] != "verified"
            or upload["immutable_key"] != item.get("key")
            or (upload["storage_version"] or None) != (item.get("storageVersion") or None)
            or upload["sha256"] != item.get("sha256")
            or upload["media_kind"] != item.get("mediaKind")
        ):
            raise StageError("INPUT_NOT_VERIFIED", details={"uploadId": str(item.get("uploadId"))})
        if not str(item["key"]).startswith(f"tenants/{ctx.project_id}/"):
            raise StageError("INPUT_NOT_VERIFIED", details={"uploadId": str(item.get("uploadId"))})
        head = ctx.store.head(item["key"], item.get("storageVersion"))
        if head is None or head.size != int(item.get("bytes") or upload["actual_bytes"] or -1):
            raise StageError("INPUT_NOT_VERIFIED", details={"uploadId": str(item.get("uploadId"))})
        total_bytes += head.size
        verified.append(
            {
                "uploadId": str(item["uploadId"]),
                "mediaKind": item["mediaKind"],
                "key": item["key"],
                "storageVersion": item.get("storageVersion"),
                "sha256": item["sha256"],
                "bytes": head.size,
                "metadata": item.get("metadata") or {},
            }
        )
    videos = [i for i in verified if i["mediaKind"] == "video"]
    photos = [i for i in verified if i["mediaKind"] == "photo"]
    if not run_row["include_supplementary_photos"] and videos:
        photos = []
    if total_bytes > ctx.services.settings.upload_max_total_bytes:
        raise StageError("BUDGET_EXHAUSTED", "The capture set exceeds the size limit.")
    if not synthetic:
        if len(videos) > params.get("maxVideos", 8) or len(photos) > params.get("maxPhotos", 400):
            raise StageError("BUDGET_EXHAUSTED", "The capture set has too many files.")
        if not videos and len(photos) < MIN_PHOTOS_WITHOUT_VIDEO:
            raise StageError("INSUFFICIENT_INPUTS")
    ctx.progress("Inputs verified.", videos=len(videos), photos=len(photos))
    selected = videos + photos
    return finish(
        ctx,
        {"counts": {"videos": len(videos), "photos": len(photos)}, "totalBytes": total_bytes},
        outputs={
            "captureSetId": str(capture["id"]),
            "quality": run_row["quality"],
            "inputs": selected,
        },
    )
