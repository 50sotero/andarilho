"""``upload_cleanup``: delete the stored bytes of an upload the user removed.

The API marks a verified, unused upload ``aborted`` and queues this job. The
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
            "SELECT status, immutable_key FROM uploads WHERE id = %s AND project_id = %s",
            (ctx.job.upload_id, ctx.project_id),
        ).fetchone()
    )
    if upload is None or upload["status"] != "aborted" or upload["immutable_key"] is None:
        return finish(ctx, {"deleted": False}, store_manifest=False)
    key = upload["immutable_key"]
    ctx.store.delete_prefix(key)

    def release(conn: Any, job: Any) -> None:
        conn.execute(
            "UPDATE uploads SET immutable_key = NULL, storage_version = NULL, updated_at = now() "
            "WHERE id = %s AND status = 'aborted' AND immutable_key = %s",
            (ctx.job.upload_id, key),
        )

    return finish(ctx, {"deleted": True}, commit=release, store_manifest=False)
