"""``upload_cleanup``: delete the stored bytes an upload no longer needs.

The staging copy is never read once an upload is frozen or rejected. The API
deletes it right away and queues this job when that delete fails, so this job
deletes it first.

The API marks a verified, unused upload ``aborted`` and queues this job; the
queue does the same for an upload rejected after its inspection failed for good. The
upload keeps its immutable key, and with it its quota, until every stored
version is gone; deleting first and clearing the key afterwards makes a retry
after a lost commit harmless.

An upload removed or rejected while it was still being uploaded keeps its multipart
upload ID. The API aborted that multipart upload, but S3 lets a part in flight land
after an abort, and a part upload has no time limit of its own. So this job aborts it
again, then checks for a late part every few hours, aborting again whenever it finds
one, until the bucket's abort-incomplete-multipart rule has run
(:func:`~andarilho_worker.artifacts.multipart_expiry`). It forgets the ID only after
that, once a check made a whole reconciliation horizon after the last abort, before any
new one, finds no part left. The quota does not wait for any of this: it is released
with the bytes.
"""

from __future__ import annotations

import time
from typing import Any

from andarilho_worker.artifacts import MULTIPART_RECHECK_SECONDS, multipart_expiry
from andarilho_worker.errors import Deferred
from andarilho_worker.stages.base import StageContext, StageResult, finish


def _settle_multipart(ctx: StageContext, key: str, upload_id: str, initiated: float) -> None:
    """Return once the aborted upload is verified to hold no part and can no longer get
    one; defer until then."""
    now = time.time()
    settings = ctx.services.settings
    horizon = float(settings.cleanup_reconcile_horizon_seconds)
    last = ctx.input_manifest.get("multipartAbortedAt")
    if last is not None:
        quiet_left = horizon - (now - float(last))
        if quiet_left > 0:  # an early pass: aborting again would only push the check back
            raise Deferred(max(1.0, quiet_left), "CLEANUP_PENDING")
        if ctx.store.count_multipart_parts(key, upload_id) == 0:
            expiry = multipart_expiry(initiated, settings.multipart_abort_days)
            if now >= expiry:
                return
            # A part upload still running could land until the bucket's rule ends the
            # upload: keep the ID and look again.
            raise Deferred(
                max(1.0, min(expiry - now, MULTIPART_RECHECK_SECONDS)), "CLEANUP_PENDING"
            )
    ctx.store.abort_multipart_upload(key, upload_id)
    raise Deferred(max(1.0, horizon), "CLEANUP_PENDING", manifest_patch={"multipartAbortedAt": now})


def run(ctx: StageContext) -> StageResult:
    upload = ctx.query(
        lambda conn: conn.execute(
            "SELECT status, immutable_key, staging_key, multipart_upload_id, "
            "extract(epoch FROM created_at)::float8 AS created FROM uploads "
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
    if upload["status"] not in removable:
        return finish(ctx, {"deleted": False, "stagingDeleted": staged}, store_manifest=False)
    key = upload["immutable_key"]
    if key is not None:
        ctx.store.delete_prefix(key)

        def release(conn: Any) -> None:
            conn.execute(
                "UPDATE uploads SET immutable_key = NULL, storage_version = NULL, "
                "updated_at = now() "
                "WHERE id = %s AND status IN ('aborted', 'rejected') AND immutable_key = %s",
                (ctx.job.upload_id, key),
            )

        ctx.fenced(release)  # before any wait below: the quota is free once the bytes are gone
    upload_id = upload["multipart_upload_id"]
    if upload_id and staged:
        # The row is created as the multipart upload is initiated (it may be a moment
        # earlier: multipart_expiry allows for far more).
        _settle_multipart(ctx, staging, str(upload_id), float(upload["created"]))

    def forget(conn: Any, job: Any) -> None:
        conn.execute(
            "UPDATE uploads SET multipart_upload_id = NULL, updated_at = now() "
            "WHERE id = %s AND multipart_upload_id = %s",
            (ctx.job.upload_id, upload_id),
        )

    return finish(
        ctx,
        {
            "deleted": key is not None,
            "stagingDeleted": staged,
            "multipartVerified": bool(upload_id),
        },
        commit=forget if upload_id and staged else None,
        store_manifest=False,
    )
