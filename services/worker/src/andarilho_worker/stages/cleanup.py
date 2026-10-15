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
   abort, so a pass only goes on when it comes at least one horizon after the last
   abort and finds, before aborting anything, the prefix empty, no multipart upload in
   progress under it and no part stored for any upload it knows of. Otherwise it
   aborts every known upload again and waits out another horizon.
   A part upload has no time limit of its own, so the job keeps every upload's ID until
   the bucket's abort-incomplete-multipart rule has ended it
   (:func:`~andarilho_worker.artifacts.multipart_expiry`), checking again every few
   hours and aborting again whenever a part landed. The project's data does not wait
   for that: the first pass to find nothing left applies the metadata policy below at
   once, but keeps this job, which then holds only the uploads' keys, IDs and
   initiation times, and leaves the project ``deleting``.
6. Apply the metadata policy below and record an audit event, in the fenced
   completion transaction of a pass that finds nothing left once every known upload
   has expired.

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
from collections.abc import Iterable, Sequence
from typing import Any

from psycopg.types.json import Jsonb

from andarilho_worker.artifacts import MULTIPART_RECHECK_SECONDS, multipart_expiry, project_prefix
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


def _recorded_uploads(conn: Any, project_id: str) -> list[tuple[str, str, float]]:
    """The multipart uploads the project's upload rows still name, with the rows'
    creation times: a row is created as its upload is initiated."""
    rows = conn.execute(
        "SELECT staging_key, multipart_upload_id, "
        "extract(epoch FROM created_at)::float8 AS created FROM uploads "
        "WHERE project_id = %s AND multipart_upload_id IS NOT NULL",
        (project_id,),
    ).fetchall()
    return [
        (str(r["staging_key"]), str(r["multipart_upload_id"]), float(r["created"])) for r in rows
    ]


def _known_uploads(
    prefix: str, now: float, *sources: Iterable[Sequence[Any]]
) -> dict[tuple[str, str], float]:
    """Every ``(key, upload id)`` under ``prefix`` that ``sources`` name, with the latest
    initiation time given for it. An entry with no time counts as initiated ``now``."""
    known: dict[tuple[str, str], float] = {}
    for source in sources:
        for entry in source:
            upload = (str(entry[0]), str(entry[1]))
            if not upload[0].startswith(prefix):
                continue
            initiated = float(entry[2]) if len(entry) > 2 else now
            known[upload] = max(initiated, known.get(upload, initiated))
    return known


def scrub(
    conn: Any, project_id: str, job_id: str, stats: dict[str, Any], *, keep_job: bool = False
) -> None:
    """Apply the metadata policy. With ``keep_job``, this job stays and the project stays
    ``deleting``, with no audit event yet: the job still watches multipart uploads."""
    p = (project_id,)
    conn.execute(
        "UPDATE projects SET active_draft_revision_id = NULL, published_revision_id = NULL "
        "WHERE id = %s",
        p,
    )
    conn.execute(  # job_events cascade
        "DELETE FROM jobs WHERE project_id = %s AND (NOT %s OR id <> %s)",
        (project_id, keep_job, job_id),
    )
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
        UPDATE projects SET title = 'Deleted project', owner_subject = 'deleted',
               deleted_at = COALESCE(deleted_at, now()), updated_at = now(),
               version = version + 1
        WHERE id = %s AND lifecycle = 'deleting'
        """,
        p,
    )
    if keep_job:
        return
    conn.execute(
        "UPDATE projects SET lifecycle = 'deleted', cleanup_completed_at = now() "
        "WHERE id = %s AND lifecycle = 'deleting'",
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
    now = time.time()
    # Every upload seen so far: an aborted one is no longer listed, and one the API never
    # recorded has no row (nor has any once the data is scrubbed), so the job keeps their
    # IDs and initiation times itself until they have expired.
    listed = ctx.store.list_multipart_uploads(prefix)
    recorded = ctx.query(lambda conn: _recorded_uploads(conn, ctx.project_id))
    seen = ctx.input_manifest.get("multipartUploads") or []
    known = _known_uploads(prefix, now, seen, listed, recorded)
    uploads = sorted(known)
    settings = ctx.services.settings
    passes = int(ctx.input_manifest.get("passes", 0)) + 1
    first = float(ctx.input_manifest.get("firstPassAt") or now)
    horizon = float(settings.cleanup_reconcile_horizon_seconds)
    last_abort = ctx.input_manifest.get("multipartAbortedAt")
    totals = {
        "passes": passes,
        "firstPassAt": first,
        "objectsDeleted": int(ctx.input_manifest.get("objectsDeleted", 0)) + deleted,
        "multipartAborted": int(ctx.input_manifest.get("multipartAborted", 0)),
        "multipartUploads": [[key, upload_id, known[key, upload_id]] for key, upload_id in uploads],
        "multipartAbortedAt": last_abort,
        "scrubbedAt": ctx.input_manifest.get("scrubbedAt"),
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
            days = settings.multipart_abort_days
            expiry = max((multipart_expiry(t, days) for t in known.values()), default=0.0)
            if now >= expiry:
                return _complete(ctx, totals, verified=len(uploads))
            # A part upload still running could land until the bucket's rule ends its
            # upload. The project's data need not wait for that.
            if totals["scrubbedAt"] is None:
                totals["scrubbedAt"] = now
                ctx.fenced(lambda conn: _scrub_keeping_job(conn, ctx, totals))
            delay = max(1.0, min(expiry - now, MULTIPART_RECHECK_SECONDS))
            raise Deferred(delay, "CLEANUP_PENDING", manifest_patch=totals)
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


def _scrub_keeping_job(conn: Any, ctx: StageContext, totals: dict[str, Any]) -> None:
    lifecycle = conn.execute(
        "SELECT lifecycle FROM projects WHERE id = %s", (ctx.project_id,)
    ).fetchone()
    if lifecycle is None or lifecycle["lifecycle"] != "deleting":
        raise StageError("INTERNAL_ERROR", "The project is not being deleted.")
    # Recorded with the scrub, which deletes the upload rows: the job's own record then
    # holds the only copy of their IDs.
    conn.execute(
        "UPDATE jobs SET input_manifest = input_manifest || %s WHERE id = %s",
        (Jsonb(totals), ctx.job_id),
    )
    scrub(conn, ctx.project_id, ctx.job_id, {}, keep_job=True)


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
