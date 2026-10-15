"""Aborted multipart uploads are checked for late parts for good (andarilho_worker.multipart)."""

from __future__ import annotations

import time
import uuid

from andarilho_worker import queue
from andarilho_worker.artifacts import LocalObjectStore
from andarilho_worker.multipart import (
    ERROR_RETRY_SECONDS,
    RECHECK_FIRST_SECONDS,
    RECHECK_MAX_SECONDS,
    clean_delay,
    reconcile,
    record,
)

KEY = "tenants/p1/staging/u1"


def _record(db, upload_id: str = "mp-1", *, aborted_at: float | None) -> None:
    with db.transaction() as conn:
        record(conn, [(KEY, upload_id)], aborted_at=aborted_at)


def _entry(seed, upload_id: str = "mp-1") -> dict:
    return seed.one(
        "SELECT aborts, clean_checks, last_aborted_at, last_checked_at, "
        "extract(epoch FROM next_check_at - now())::float8 AS wait "
        "FROM multipart_reconciliations WHERE upload_id = %s",
        upload_id,
    )


def _due(seed) -> None:
    seed.exec("UPDATE multipart_reconciliations SET next_check_at = now()")


def test_clean_checks_double_the_wait_from_an_hour_up_to_a_week():
    assert clean_delay(1) == RECHECK_FIRST_SECONDS
    assert clean_delay(2) == 2 * RECHECK_FIRST_SECONDS
    assert clean_delay(4) == 8 * RECHECK_FIRST_SECONDS
    assert clean_delay(9) == RECHECK_MAX_SECONDS == 7 * 86400
    assert clean_delay(10_000) == RECHECK_MAX_SECONDS


def test_a_settled_upload_is_checked_an_hour_later_then_less_and_less_often(db, seed, store):
    _record(db, aborted_at=time.time() - 900)
    entry = _entry(seed)
    assert RECHECK_FIRST_SECONDS - 60 < entry["wait"] <= RECHECK_FIRST_SECONDS
    assert reconcile(db, store, horizon_seconds=900)["multipartChecked"] == 0  # not due yet

    waits = []
    for _ in range(10):
        _due(seed)
        assert reconcile(db, store, horizon_seconds=900) == {
            "multipartChecked": 1, "multipartAborted": 0, "multipartFailed": 0,
        }  # fmt: skip
        waits.append(round(_entry(seed)["wait"] / RECHECK_FIRST_SECONDS))
    assert waits == [1, 2, 4, 8, 16, 32, 64, 128, 168, 168]  # hours, capped at a week
    entry = _entry(seed)
    assert entry["clean_checks"] == 10 and entry["aborts"] == 0  # and the entry stays


def test_a_part_that_landed_late_is_aborted_and_checked_again_a_horizon_later(db, seed, store):
    _record(db, aborted_at=time.time() - 900)
    for _ in range(3):  # a few clean checks first
        _due(seed)
        reconcile(db, store, horizon_seconds=900)
    store.stored_parts["mp-1"] = 1  # a slow part upload lands days later
    _due(seed)

    assert reconcile(db, store, horizon_seconds=900)["multipartAborted"] == 1
    assert store.stored_parts == {}
    entry = _entry(seed)
    assert entry["aborts"] == 1 and entry["clean_checks"] == 0
    assert 840 < entry["wait"] <= 900  # a part in flight during the abort can still land
    _due(seed)
    reconcile(db, store, horizon_seconds=900)
    assert RECHECK_FIRST_SECONDS - 60 < _entry(seed)["wait"] <= RECHECK_FIRST_SECONDS


def test_an_upload_never_aborted_here_is_aborted_on_its_first_check(db, seed, store):
    # An expired upload: nobody aborted it, and it may hold no part yet.
    store.multipart_uploads["mp-1"] = KEY
    _record(db, aborted_at=None)
    assert _entry(seed)["wait"] <= 0  # due at once

    assert reconcile(db, store, horizon_seconds=900)["multipartAborted"] == 1
    assert store.multipart_uploads == {}
    entry = _entry(seed)
    assert entry["aborts"] == 1 and entry["last_aborted_at"] is not None


def test_recording_an_upload_twice_keeps_its_entry(db, seed, store):
    _record(db, aborted_at=None)
    reconcile(db, store, horizon_seconds=900)
    _record(db, aborted_at=time.time())
    assert seed.one("SELECT count(*) AS n FROM multipart_reconciliations")["n"] == 1
    assert _entry(seed)["aborts"] == 1


class _FlakyStore(LocalObjectStore):
    """Storage that fails every call about one upload."""

    def count_multipart_parts(self, key: str, upload_id: str) -> int:
        if upload_id == "mp-down":
            raise OSError("storage unavailable")
        return super().count_multipart_parts(key, upload_id)


def test_a_failed_check_is_tried_again_later_and_does_not_stop_the_others(db, seed, tmp_path):
    store = _FlakyStore(tmp_path / "objects")
    _record(db, "mp-down", aborted_at=time.time() - 900)
    _record(db, "mp-up", aborted_at=time.time() - 900)
    _due(seed)

    assert reconcile(db, store, horizon_seconds=900) == {
        "multipartChecked": 1, "multipartAborted": 0, "multipartFailed": 1,
    }  # fmt: skip
    down = _entry(seed, "mp-down")
    assert ERROR_RETRY_SECONDS - 60 < down["wait"] <= ERROR_RETRY_SECONDS
    assert down["last_checked_at"] is None and down["clean_checks"] == 0
    assert _entry(seed, "mp-up")["clean_checks"] == 1


def test_a_pass_checks_at_most_a_batch_of_entries(db, seed, store):
    for n in range(5):
        _record(db, f"mp-{n}", aborted_at=None)
    assert reconcile(db, store, horizon_seconds=900, batch=3)["multipartAborted"] == 3
    assert reconcile(db, store, horizon_seconds=900, batch=10)["multipartChecked"] == 2


def _expired_upload(seed, pid: str, status: str = "initiated") -> tuple[str, str]:
    """An upload two days past its expiry, still holding its multipart upload ``mp-old``."""
    uid = str(uuid.uuid4())
    key = f"tenants/{pid}/staging/{uid}"
    seed.exec(
        "INSERT INTO uploads (id, project_id, created_by, media_kind, display_filename, "
        "declared_content_type, expected_bytes, staging_key, multipart_upload_id, part_size, "
        "part_count, status, expires_at) VALUES (%s, %s, 'user-1', 'video', 'walk.mp4', "
        "'video/mp4', 10, %s, 'mp-old', 16777216, 1, %s, now() - interval '2 days')",
        uid, pid, key, status,
    )  # fmt: skip
    return uid, key


def _multipart_id(seed, uid: str) -> str | None:
    row = seed.one("SELECT multipart_upload_id FROM uploads WHERE id = %s", uid)
    return row["multipart_upload_id"]


def test_an_expired_upload_is_aborted_at_once_and_keeps_its_quota_until_a_clean_check(
    db, seed, store
):
    pid = seed.project()
    uid, key = _expired_upload(seed, pid)
    store.multipart_uploads["mp-old"] = key
    store.stored_parts["mp-old"] = 3

    # Marked expired and handed over at once; the row keeps its ID, and so its quota.
    assert queue.expire_uploads(db) == (1, 0)
    entry = seed.one("SELECT object_key, last_aborted_at FROM multipart_reconciliations")
    assert entry == {"object_key": key, "last_aborted_at": None}
    assert reconcile(db, store, horizon_seconds=900)["multipartAborted"] == 1
    assert store.multipart_uploads == {} and store.stored_parts == {}
    assert _multipart_id(seed, uid) == "mp-old"  # only a check a horizon later decides
    store.stored_parts["mp-old"] = 1  # a part that was in flight lands after the abort
    _due(seed)
    assert reconcile(db, store, horizon_seconds=900)["multipartAborted"] == 1
    assert _multipart_id(seed, uid) == "mp-old"
    _due(seed)
    assert reconcile(db, store, horizon_seconds=900)["multipartAborted"] == 0
    assert _multipart_id(seed, uid) is None  # nothing left: the quota is free

    # A day past its expiry, with its ID cleared, the row goes; the checks go on.
    assert queue.expire_uploads(db) == (0, 1)
    assert seed.one("SELECT count(*) AS n FROM multipart_reconciliations")["n"] == 1


def test_an_upload_already_marked_expired_is_handed_over_too(db, seed, store):
    pid = seed.project()
    uid, key = _expired_upload(seed, pid, status="expired")
    assert queue.expire_uploads(db) == (0, 0)  # kept: it still holds its ID
    entry = seed.one("SELECT object_key, upload_id FROM multipart_reconciliations")
    assert entry == {"object_key": key, "upload_id": "mp-old"}
    assert queue.expire_uploads(db) == (0, 0)  # recorded once
    assert seed.one("SELECT count(*) AS n FROM multipart_reconciliations")["n"] == 1
