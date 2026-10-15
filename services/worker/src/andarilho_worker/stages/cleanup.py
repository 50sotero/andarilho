"""``project_cleanup``: README section 13, deletion protocol steps 3-6.

3. Wait until no other job of the project is running (any lease, expired or not:
   the reaper must first bump the fence and cancel it) and cancel anything still
   queued. Otherwise defer and retry later without consuming an attempt.
4. Delete every object and object version under ``tenants/{project_id}/`` and abort
   its multipart uploads: those still in progress under the prefix, and those the
   upload rows record (the API aborts an upload but keeps its ID). A part in flight
   can still land after an abort, so every pass aborts each of them again.
5. Reconcile again after the write-grant horizon
   (``cleanup_reconcile_horizon_seconds``): the job defers itself after the first
   pass and only completes when a later pass finds the prefix empty, no multipart
   upload in progress under it, and no part stored for any upload it knows of. The
   upload rows, and with them the IDs, are deleted only then.
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
    uploads = sorted({*seen, *listed, *recorded})
    uploads = [(str(key), str(uid)) for key, uid in uploads if str(key).startswith(prefix)]
    for key, upload_id in uploads:
        ctx.store.abort_multipart_upload(key, upload_id)
    aborted = len(listed)
    passes = int(ctx.input_manifest.get("passes", 0)) + 1
    first = float(ctx.input_manifest.get("firstPassAt") or time.time())
    horizon = float(ctx.services.settings.cleanup_reconcile_horizon_seconds)
    totals = {
        "passes": passes,
        "firstPassAt": first,
        "objectsDeleted": int(ctx.input_manifest.get("objectsDeleted", 0)) + deleted,
        "multipartAborted": int(ctx.input_manifest.get("multipartAborted", 0)) + aborted,
        "multipartUploads": [list(u) for u in uploads],
    }
    waited = time.time() - first
    if passes < 2 or waited < horizon:
        raise Deferred(max(1.0, horizon - waited), "CLEANUP_PENDING", manifest_patch=totals)
    remaining = ctx.store.list_keys(prefix)
    # A part that landed after this pass's abort keeps its upload's storage: the next
    # pass aborts it again. Only an empty ListParts for every upload proves it is gone.
    parts = sum(ctx.store.count_multipart_parts(key, upload_id) for key, upload_id in uploads)
    if remaining or parts or ctx.store.list_multipart_uploads(prefix):
        raise Deferred(BUSY_RETRY_SECONDS, "CLEANUP_PENDING", manifest_patch=totals)
    if ctx.fenced(lambda conn: _busy(conn, ctx)):
        raise Deferred(BUSY_RETRY_SECONDS, "CLEANUP_PENDING", manifest_patch=totals)

    def commit(conn: Any, job: Any) -> None:
        lifecycle = conn.execute(
            "SELECT lifecycle FROM projects WHERE id = %s", (ctx.project_id,)
        ).fetchone()
        if lifecycle is None or lifecycle["lifecycle"] != "deleting":
            raise StageError("INTERNAL_ERROR", "The project is not being deleted.")
        scrub(conn, ctx.project_id, ctx.job_id, stats)

    # The audit event keeps counts only, not the uploads' keys and IDs.
    stats = {k: v for k, v in totals.items() if k != "multipartUploads"}
    stats["multipartVerified"] = len(uploads)
    return finish(ctx, {"cleanup": stats}, commit=commit, store_manifest=False)
