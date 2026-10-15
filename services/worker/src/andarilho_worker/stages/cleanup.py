"""``project_cleanup``: README section 13, deletion protocol steps 3-6.

3. Wait until no other job of the project is running (any lease, expired or not:
   the reaper must first bump the fence and cancel it) and cancel anything still
   queued. Otherwise defer and retry later without consuming an attempt.
4. Delete every object and object version under ``tenants/{project_id}/`` and abort
   its multipart uploads: those still in progress under the prefix, those the upload
   rows record (the API aborts an upload but keeps its ID) and those an earlier pass
   saw, whose IDs the job keeps in its manifest.
5. Reconcile again after the write-grant horizon
   (``cleanup_reconcile_horizon_seconds``). A part in flight can still land after an
   abort, so a pass only completes the deletion when it comes at least one horizon
   after the last abort and finds, before aborting anything, the prefix empty, no
   multipart upload in progress under it and no part stored for any upload it knows
   of. Otherwise it aborts every known upload again and waits out another horizon.
   The upload rows, and with them the IDs, are deleted only on completion.
6. Apply the metadata policy below and record an audit event, in the fenced
   completion transaction.

Metadata policy (documented here and in the final report):

* deleted: jobs and job events (including this job), share links, attached
  tours (``tour_imports``), project members, idempotency records, scene
  revisions, scenes, artifacts, uploads;
* retained but scrubbed for usage accounting: ``reconstruction_runs``
  (``created_by`` -> ``'deleted'``, ``error`` cleared), ``usage_ledger`` rows
  (reservations still ``reserved`` are released) and ``capture_sets``
  (``inputs`` -> ``[]``, metadata cleared, ``created_by`` -> ``'deleted'``);
* the ``projects`` row stays as a tombstone: title ``'Deleted project'``,
  ``owner_subject`` ``'deleted'``, draft/published pointers cleared,
  ``lifecycle='deleted'``, ``cleanup_completed_at``/``deleted_at`` set;
* ``audit_events`` are retained (identifiers only) and a
  ``project.cleanup_completed`` event is added.
"""

from __future__ import annotations

import time
from typing import Any

from psycopg.types.json import Jsonb

from andarilho_worker.artifacts import project_prefix
from andarilho_worker.errors import Deferred, StageError
from andarilho_worker.stages.base import StageContext, StageResult, finish

BUSY_RETRY_SECONDS = 60.0


def _busy(conn: Any, ctx: StageContext) -> int:
    conn.execute(
        """
        UPDATE jobs SET state = 'cancelled', finished_at = now(), updated_at = now(),
               cancel_requested_at = COALESCE(cancel_requested_at, now())
        WHERE project_id = %s AND id <> %s AND state IN ('queued', 'retry_wait')
        """,
        (ctx.project_id, ctx.job_id),
    )
    row = conn.execute(
        "SELECT count(*) AS n FROM jobs WHERE project_id = %s AND id <> %s AND state = 'running'",
        (ctx.project_id, ctx.job_id),
    ).fetchone()
    return int(row["n"])


def _recorded_uploads(conn: Any, project_id: str) -> list[tuple[str, str]]:
    """The multipart uploads the project's upload rows still name."""
    rows = conn.execute(
        "SELECT staging_key, multipart_upload_id FROM uploads "
        "WHERE project_id = %s AND multipart_upload_id IS NOT NULL",
        (project_id,),
    ).fetchall()
    return [(str(r["staging_key"]), str(r["multipart_upload_id"])) for r in rows]


def scrub(conn: Any, project_id: str, job_id: str, stats: dict[str, Any]) -> None:
    p = (project_id,)
    conn.execute(
        "UPDATE projects SET active_draft_revision_id = NULL, published_revision_id = NULL "
        "WHERE id = %s",
        p,
    )
    conn.execute("DELETE FROM jobs WHERE project_id = %s", p)  # job_events cascade
    conn.execute("DELETE FROM share_links WHERE project_id = %s", p)
    conn.execute("DELETE FROM tour_imports WHERE project_id = %s", p)
    conn.execute("UPDATE scenes SET current_draft_revision_id = NULL WHERE project_id = %s", p)
    conn.execute("DELETE FROM scene_revisions WHERE project_id = %s", p)
    conn.execute("DELETE FROM artifacts WHERE project_id = %s", p)
    conn.execute("DELETE FROM scenes WHERE project_id = %s", p)
    conn.execute(
        "UPDATE capture_sets SET inputs = '[]'::jsonb, capture_metadata = '{}'::jsonb, "
        "floor_plan_upload_id = NULL, created_by = 'deleted' WHERE project_id = %s",
        p,
    )
    conn.execute("DELETE FROM uploads WHERE project_id = %s", p)
    conn.execute(
        "UPDATE reconstruction_runs SET created_by = 'deleted', error = NULL, "
        "status = CASE WHEN status IN ('queued', 'running') THEN 'cancelled' ELSE status END "
        "WHERE project_id = %s",
        p,
    )
    conn.execute(
        "UPDATE usage_ledger SET state = 'released', settled_at = now() "
        "WHERE project_id = %s AND state = 'reserved'",
        p,
    )
    conn.execute("DELETE FROM project_members WHERE project_id = %s", p)
    conn.execute("DELETE FROM idempotency_records WHERE project_id = %s", p)
    conn.execute(
        """
        UPDATE projects SET lifecycle = 'deleted', title = 'Deleted project',
               owner_subject = 'deleted', cleanup_completed_at = now(),
               deleted_at = COALESCE(deleted_at, now()), updated_at = now(),
               version = version + 1
        WHERE id = %s AND lifecycle = 'deleting'
        """,
        p,
    )
    conn.execute(
        "INSERT INTO audit_events (project_id, actor_subject, action, resource_type, resource_id, "
        "data) VALUES (%s, 'system:worker', 'project.cleanup_completed', 'project', %s, %s)",
        (project_id, project_id, Jsonb({"jobId": job_id, **stats})),
    )


def run(ctx: StageContext) -> StageResult:
    busy = ctx.fenced(lambda conn: _busy(conn, ctx))
    if busy:
        raise Deferred(BUSY_RETRY_SECONDS, "CLEANUP_PENDING")
    prefix = project_prefix(ctx.project_id)
    deleted = ctx.store.delete_prefix(prefix)
    # Every upload seen so far: an aborted one is no longer listed, and one the API never
    # recorded has no row, so the job keeps their IDs itself until they are verified.
    listed = ctx.store.list_multipart_uploads(prefix)
    recorded = ctx.query(lambda conn: _recorded_uploads(conn, ctx.project_id))
    seen = [tuple(u) for u in ctx.input_manifest.get("multipartUploads") or []]
    uploads = sorted(
        (str(key), str(uid))
        for key, uid in {*seen, *listed, *recorded}
        if str(key).startswith(prefix)
    )
    now = time.time()
    passes = int(ctx.input_manifest.get("passes", 0)) + 1
    first = float(ctx.input_manifest.get("firstPassAt") or now)
    horizon = float(ctx.services.settings.cleanup_reconcile_horizon_seconds)
    last_abort = ctx.input_manifest.get("multipartAbortedAt")
    totals = {
        "passes": passes,
        "firstPassAt": first,
        "objectsDeleted": int(ctx.input_manifest.get("objectsDeleted", 0)) + deleted,
        "multipartAborted": int(ctx.input_manifest.get("multipartAborted", 0)),
        "multipartUploads": [list(u) for u in uploads],
        "multipartAbortedAt": last_abort,
    }
    # A part that was in flight can land after an abort, and a check that follows an abort
    # at once would miss it. So the deciding check runs before any abort of this pass,
    # and only once a whole horizon has passed since the last abort.
    since_abort = now - float(last_abort) if last_abort is not None else None
    quiet_left = horizon - since_abort if since_abort is not None else 0.0
    recent = bool(uploads) and since_abort is not None and quiet_left > 0
    settled = not uploads or (since_abort is not None and not recent)
    if passes >= 2 and now - first >= horizon and settled:
        parts = sum(ctx.store.count_multipart_parts(key, upload_id) for key, upload_id in uploads)
        if not (parts or listed or ctx.store.list_keys(prefix)):
            if ctx.fenced(lambda conn: _busy(conn, ctx)):
                raise Deferred(BUSY_RETRY_SECONDS, "CLEANUP_PENDING", manifest_patch=totals)
            return _complete(ctx, totals, verified=len(uploads))
    delay = max(1.0, horizon - (now - first)) if passes < 2 else BUSY_RETRY_SECONDS
    if recent and not listed:
        # An early pass: aborting again would only move the quiet interval further away.
        raise Deferred(max(delay, quiet_left, 1.0), "CLEANUP_PENDING", manifest_patch=totals)
    if uploads:
        for key, upload_id in uploads:
            ctx.store.abort_multipart_upload(key, upload_id)
        totals["multipartAborted"] += len(listed)
        totals["multipartAbortedAt"] = now
        delay = max(delay, horizon, 1.0)
    raise Deferred(delay, "CLEANUP_PENDING", manifest_patch=totals)


def _complete(ctx: StageContext, totals: dict[str, Any], *, verified: int) -> StageResult:
    # The audit event keeps counts only, not the uploads' keys and IDs.
    stats = {k: v for k, v in totals.items() if k not in ("multipartUploads", "multipartAbortedAt")}
    stats["multipartVerified"] = verified

    def commit(conn: Any, job: Any) -> None:
        lifecycle = conn.execute(
            "SELECT lifecycle FROM projects WHERE id = %s", (ctx.project_id,)
        ).fetchone()
        if lifecycle is None or lifecycle["lifecycle"] != "deleting":
            raise StageError("INTERNAL_ERROR", "The project is not being deleted.")
        scrub(conn, ctx.project_id, ctx.job_id, stats)

    return finish(ctx, {"cleanup": stats}, commit=commit, store_manifest=False)
