"""``upload_cleanup``: delete the stored bytes an upload no longer needs.

The staging copy is never read once an upload is frozen or rejected. The API
deletes it right away and queues this job when that delete fails, so this job
deletes it first.

The API marks a verified, unused upload ``aborted`` and queues this job; the
queue does the same for an upload rejected after its inspection failed for good. The
upload keeps its immutable key, and with it its quota, until every stored
version is gone; deleting first and clearing the key afterwards makes a retry
after a lost commit harmless.
"""

from __future__ import annotations

from typing import Any

from andarilho_worker.stages.base import StageContext, StageResult, finish


def run(ctx: StageContext) -> StageResult:
    upload = ctx.query(
        lambda conn: conn.execute(
            "SELECT status, immutable_key, staging_key FROM uploads "
            "WHERE id = %s AND project_id = %s",
            (ctx.job.upload_id, ctx.project_id),
        ).fetchone()
    )
    if upload is None or upload["status"] == "initiated":  # still uploading into staging
        return finish(ctx, {"deleted": False}, store_manifest=False)
    staging = upload["staging_key"] or ""
    staged = staging.startswith(f"tenants/{ctx.project_id}/staging/")
    if staged:
        ctx.store.delete_prefix(staging)
    removable = ("aborted", "rejected")  # removed by the user, or rejected by inspection
    if upload["status"] not in removable or upload["immutable_key"] is None:
        return finish(ctx, {"deleted": False, "stagingDeleted": staged}, store_manifest=False)
    key = upload["immutable_key"]
    ctx.store.delete_prefix(key)

    def release(conn: Any, job: Any) -> None:
        conn.execute(
            "UPDATE uploads SET immutable_key = NULL, storage_version = NULL, updated_at = now() "
            "WHERE id = %s AND status IN ('aborted', 'rejected') AND immutable_key = %s",
            (ctx.job.upload_id, key),
        )

    return finish(
        ctx, {"deleted": True, "stagingDeleted": staged}, commit=release, store_manifest=False
    )
