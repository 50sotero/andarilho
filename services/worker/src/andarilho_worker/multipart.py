"""Keep checking aborted multipart uploads for parts that landed late (README section 7).

S3 lets a part upload that was in flight during an ``AbortMultipartUpload`` succeed
afterwards, and asks for the abort to be repeated until ``ListParts`` comes back empty.
A part upload has no time limit of its own (its URL only has to be used before it
expires), so no check is ever final. Every multipart upload the app stops tracking is
therefore recorded in ``multipart_reconciliations``: one removed or rejected mid-upload
(``upload_cleanup``), one of a deleted project (``project_cleanup``) and one that expired
(:func:`andarilho_worker.queue.expire_uploads`). Maintenance checks each entry for good.
An upload never aborted here, or one holding a part, is aborted and checked again a
reconciliation horizon later; a check that finds nothing moves the next one further
out, doubling from an hour up to a week. An expired upload keeps its ID, and with it
its quota, until such a check: that check clears it. The horizon covers every part
upload the app supports (:func:`andarilho_worker.settings.reconcile_horizon`); a part that
lands later anyway charges its upload's quota again, through its ID, until a check finds
it gone.
A completion can join the parts into the staging object and fail before freezing the
upload, which may then expire with nothing else to delete that object: the first clean
check after an abort deletes it too, unless an upload still takes parts into that key.
Entries are never deleted: one costs a row and a ``ListParts`` call a week.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

from andarilho_worker.artifacts import ObjectStore
from andarilho_worker.db import Database

log = logging.getLogger(__name__)

RECHECK_FIRST_SECONDS = 3600.0
RECHECK_MAX_SECONDS = 7 * 86400.0
# A check that failed (a storage outage) is tried again after this long.
ERROR_RETRY_SECONDS = 300.0
# Entries checked per maintenance pass at most, so one pass stays short.
BATCH = 50
# The API's staging keys, tenants/{project}/staging/{upload}: deleting one as a prefix
# removes only that object, as an ID of fixed length is never another's prefix.
_UUID = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
STAGING_KEY = re.compile(rf"^tenants/{_UUID}/staging/{_UUID}$")


def record(conn: Any, uploads: Iterable[tuple[str, str]], *, aborted_at: float | None) -> None:
    """Record uploads to keep checking, in the caller's transaction.

    ``aborted_at`` (seconds since the epoch) is when the caller last aborted them, after
    which it found no part a whole horizon later; their first check here then comes
    ``RECHECK_FIRST_SECONDS`` later. With ``None`` the next maintenance pass aborts them.
    An upload already recorded keeps its entry.
    """
    for key, upload_id in uploads:
        conn.execute(
            """
            INSERT INTO multipart_reconciliations
                   (object_key, upload_id, last_aborted_at, next_check_at)
            VALUES (%s, %s, to_timestamp(%s),
                    CASE WHEN %s::float8 IS NULL THEN now()
                         ELSE now() + make_interval(secs => %s) END)
            ON CONFLICT (object_key, upload_id) DO NOTHING
            """,
            (key, upload_id, aborted_at, aborted_at, RECHECK_FIRST_SECONDS),
        )


def _abandoned_staging(conn: Any, key: str) -> bool:
    """Whether ``key`` is a staging object no upload may still complete into."""
    if not STAGING_KEY.match(key):
        return False
    taking = conn.execute(
        "SELECT 1 FROM uploads WHERE staging_key = %s AND status = 'initiated'", (key,)
    ).fetchone()
    return taking is None


def clean_delay(clean_checks: int) -> float:
    """The wait after the ``clean_checks``-th check in a row that found no part."""
    doublings = min(max(clean_checks - 1, 0), 30)
    return min(RECHECK_MAX_SECONDS, RECHECK_FIRST_SECONDS * 2.0**doublings)


def reconcile(
    db: Database, store: ObjectStore, *, horizon_seconds: float, batch: int = BATCH
) -> dict[str, int]:
    """Check the entries that are due, each in its own short transaction."""
    stats = {"multipartChecked": 0, "multipartAborted": 0, "multipartFailed": 0}
    done: list[int] = []  # an entry is checked at most once per pass
    for _ in range(batch):
        with db.transaction() as conn:
            row = conn.execute(
                """
                SELECT id, object_key, upload_id, last_aborted_at, clean_checks
                FROM multipart_reconciliations
                WHERE next_check_at <= now() AND NOT id = ANY(%s)
                ORDER BY next_check_at LIMIT 1 FOR UPDATE SKIP LOCKED
                """,
                (done,),
            ).fetchone()
            if row is None:
                break
            done.append(int(row["id"]))
            key, upload_id = str(row["object_key"]), str(row["upload_id"])
            try:
                abort = row["last_aborted_at"] is None
                if not abort:
                    abort = store.count_multipart_parts(key, upload_id) > 0
                if abort:
                    store.abort_multipart_upload(key, upload_id)
                elif int(row["clean_checks"]) == 0 and _abandoned_staging(conn, key):
                    # Before the quota is released below: the object holds the whole file.
                    store.delete_prefix(key)
            except Exception as exc:  # any storage failure: try this entry again later
                log.warning("multipart check failed: %s", type(exc).__name__)
                stats["multipartFailed"] += 1
                conn.execute(
                    "UPDATE multipart_reconciliations "
                    "SET next_check_at = now() + make_interval(secs => %s) WHERE id = %s",
                    (ERROR_RETRY_SECONDS, row["id"]),
                )
                continue
            stats["multipartChecked"] += 1
            if abort:
                # The next check must come a whole horizon after this abort: a part in
                # flight during it can still land.
                stats["multipartAborted"] += 1
                if row["last_aborted_at"] is not None:
                    # A part landed after the upload's quota was released: charge it
                    # again until a check finds the upload empty.
                    conn.execute(
                        "UPDATE uploads SET multipart_upload_id = %s, updated_at = now() "
                        "WHERE staging_key = %s AND multipart_upload_id IS NULL "
                        "AND status IN ('expired', 'aborted', 'rejected')",
                        (upload_id, key),
                    )
                conn.execute(
                    """
                    UPDATE multipart_reconciliations
                    SET aborts = aborts + 1, clean_checks = 0, last_aborted_at = now(),
                        last_checked_at = now(),
                        next_check_at = now() + make_interval(secs => %s)
                    WHERE id = %s
                    """,
                    (max(1.0, float(horizon_seconds)), row["id"]),
                )
            else:
                # Its storage is free: the upload no longer holds quota.
                conn.execute(
                    "UPDATE uploads SET multipart_upload_id = NULL, updated_at = now() "
                    "WHERE staging_key = %s AND multipart_upload_id = %s "
                    "AND status IN ('expired', 'aborted', 'rejected')",
                    (key, upload_id),
                )
                conn.execute(
                    """
                    UPDATE multipart_reconciliations
                    SET clean_checks = clean_checks + 1, last_checked_at = now(),
                        next_check_at = now() + make_interval(secs => %s)
                    WHERE id = %s
                    """,
                    (clean_delay(int(row["clean_checks"]) + 1), row["id"]),
                )
    return stats
