"""Multipart upload flow against moto S3."""

from __future__ import annotations

import uuid
from urllib.parse import urlparse

import pytest
from conftest import BUCKET
from factories import create_project, verified_upload
from sqlalchemy import func, select

from andarilho_api.models import Job, ReconstructionRun, Upload
from andarilho_api.storage import StorageError

MIB = 1024 * 1024


def _init(client, auth, project_id, *, size=1000, kind="video", ctype="video/mp4", sub="user-a"):
    return client.post(
        f"/v1/projects/{project_id}/uploads",
        json={
            "filename": "C:\\Users\\me\\walk.mp4",
            "bytes": size,
            "mediaKind": kind,
            "contentType": ctype,
        },
        headers=auth(sub),
    )


def _put_part(s3, db, upload_id: str, part_number: int, data: bytes) -> None:
    row = db.get(Upload, uuid.UUID(upload_id))
    db.refresh(row)
    s3.upload_part(
        Bucket=BUCKET,
        Key=row.staging_key,
        UploadId=row.multipart_upload_id,
        PartNumber=part_number,
        Body=data,
    )


def test_init_validates_type_and_size(client, auth, settings):
    project = create_project(client, auth)
    pid = project["id"]
    bad_type = _init(client, auth, pid, ctype="image/svg+xml", kind="photo")
    assert bad_type.status_code == 422
    assert bad_type.json()["error"]["code"] == "VALIDATION_FAILED"
    assert _init(client, auth, pid, ctype="video/webm").status_code == 422
    assert _init(client, auth, pid, kind="scan", ctype="video/mp4").status_code == 422

    big_plan = _init(
        client, auth, pid, kind="floor_plan", ctype="application/pdf", size=20 * MIB + 1
    )
    assert big_plan.status_code == 413
    assert big_plan.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"
    too_big = _init(client, auth, pid, size=settings.upload_max_total_bytes + 1)
    assert too_big.status_code == 413

    ok = _init(client, auth, pid, size=settings.upload_max_total_bytes - 10)
    assert ok.status_code == 201
    quota = _init(client, auth, pid, size=100, kind="photo", ctype="image/jpeg")
    assert quota.status_code == 413
    assert quota.json()["error"]["code"] == "QUOTA_EXCEEDED"


def test_completing_on_a_versioned_bucket_leaves_no_staging_bytes(client, auth, db, s3):
    # Production buckets are versioned (bootstrap-storage.py): a plain delete of the
    # staging key would only add a delete marker and keep a full copy of the input.
    s3.put_bucket_versioning(Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"})
    pid = create_project(client, auth)["id"]
    upload_id = _init(client, auth, pid, size=1000).json()["upload"]["id"]
    _put_part(s3, db, upload_id, 1, b"v" * 1000)
    done = client.post(f"/v1/uploads/{upload_id}/complete", json={}, headers=auth())
    assert done.status_code == 202, done.text
    row = db.get(Upload, uuid.UUID(upload_id))
    db.refresh(row)
    listed = s3.list_object_versions(Bucket=BUCKET, Prefix=row.staging_key)
    assert listed.get("Versions", []) == [] and listed.get("DeleteMarkers", []) == []
    assert s3.get_object(Bucket=BUCKET, Key=row.immutable_key)["Body"].read() == b"v" * 1000


def test_completing_again_replaces_a_copy_from_a_rolled_back_attempt(client, auth, db, s3):
    s3.put_bucket_versioning(Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"})
    pid = create_project(client, auth)["id"]
    upload_id = _init(client, auth, pid, size=1000).json()["upload"]["id"]
    _put_part(s3, db, upload_id, 1, b"v" * 1000)
    # An earlier completion made the frozen copy, then its transaction rolled back.
    frozen = f"tenants/{pid}/originals/{upload_id}"
    s3.put_object(Bucket=BUCKET, Key=frozen, Body=b"v" * 1000)
    done = client.post(f"/v1/uploads/{upload_id}/complete", json={}, headers=auth())
    assert done.status_code == 202, done.text
    row = db.get(Upload, uuid.UUID(upload_id))
    db.refresh(row)
    assert row.immutable_key == frozen
    # Only the copy the database records remains.
    versions = s3.list_object_versions(Bucket=BUCKET, Prefix=frozen).get("Versions", [])
    assert [v["VersionId"] for v in versions] == [row.storage_version]


def _refuse_deletes(app, monkeypatch) -> None:
    """Storage refuses to delete staging copies (other deletes still work)."""
    delete = app.state.storage.delete_all_versions

    def refuse(key: str) -> None:
        if "/staging/" in key:
            raise StorageError("delete refused")
        delete(key)

    monkeypatch.setattr(app.state.storage, "delete_all_versions", refuse)


def _cleanup_jobs(db, upload_id: str) -> list[Job]:
    query = select(Job).where(Job.upload_id == uuid.UUID(upload_id), Job.stage == "upload_cleanup")
    return list(db.scalars(query))


def test_a_failed_staging_delete_after_freezing_queues_a_cleanup(
    client, auth, db, s3, app, monkeypatch
):
    pid = create_project(client, auth)["id"]
    upload_id = _init(client, auth, pid, size=1000).json()["upload"]["id"]
    _put_part(s3, db, upload_id, 1, b"v" * 1000)
    _refuse_deletes(app, monkeypatch)
    done = client.post(f"/v1/uploads/{upload_id}/complete", json={}, headers=auth())
    assert done.status_code == 202, done.text
    row = db.get(Upload, uuid.UUID(upload_id))
    db.refresh(row)
    assert row.status == "uploaded_unverified"
    [job] = _cleanup_jobs(db, upload_id)
    assert job.state == "queued" and job.input_manifest["stagingKey"] == row.staging_key


def test_a_failed_staging_delete_on_rejection_queues_a_cleanup(
    client, auth, db, s3, app, monkeypatch
):
    pid = create_project(client, auth)["id"]
    upload_id = _init(client, auth, pid, size=2000).json()["upload"]["id"]
    _put_part(s3, db, upload_id, 1, b"x" * 1500)
    _refuse_deletes(app, monkeypatch)
    response = client.post(f"/v1/uploads/{upload_id}/complete", json={}, headers=auth())
    assert response.status_code == 422
    row = db.get(Upload, uuid.UUID(upload_id))
    db.refresh(row)
    assert row.status == "rejected" and row.immutable_key is None
    [job] = _cleanup_jobs(db, upload_id)
    assert job.state == "queued" and job.input_manifest["stagingKey"] == row.staging_key


def test_intake_policy_publishes_the_deployment_limits(client, auth, app):
    assert client.get("/v1/intake-policy").status_code == 401
    app.state.settings.upload_max_total_bytes = 3 * 1024 * MIB
    app.state.settings.upload_max_photos = 120
    policy = client.get("/v1/intake-policy", headers=auth()).json()
    assert policy == {
        "maxTotalBytes": 3 * 1024 * MIB,
        "maxVideoBytes": 3 * 1024 * MIB,  # no separate video cap: the total applies
        "maxPhotoBytes": 50 * MIB,
        "maxFloorPlanBytes": 20 * MIB,
        "maxPhotos": 120,
        "minPhotosWithoutVideo": 30,  # the worker's rule, from recipes.json
    }


def test_no_upload_is_larger_than_one_storage_copy(client, auth, app):
    app.state.settings.upload_max_total_bytes = 8 * 1024 * MIB
    policy = client.get("/v1/intake-policy", headers=auth()).json()
    assert policy["maxVideoBytes"] == 5 * 1024 * MIB
    pid = create_project(client, auth)["id"]
    assert _init(client, auth, pid, size=5 * 1024 * MIB + 1).status_code == 413
    assert _init(client, auth, pid, size=5 * 1024 * MIB).status_code == 201


def test_creating_an_upload_again_with_its_key_replays_it(client, auth, db, app):
    # Room for exactly one such file: a replay must not be refused for its own quota.
    app.state.settings.upload_max_total_bytes = 1000
    pid = create_project(client, auth)["id"]
    body = {"filename": "walk.mp4", "bytes": 1000, "mediaKind": "video",
            "contentType": "video/mp4"}  # fmt: skip
    url = f"/v1/projects/{pid}/uploads"
    first = client.post(url, json=body, headers=auth(**{"Idempotency-Key": "create-walk-1"}))
    assert first.status_code == 201, first.text
    # The response was lost; the browser retries with the same key.
    again = client.post(url, json=body, headers=auth(**{"Idempotency-Key": "create-walk-1"}))
    assert again.status_code == 201 and again.json() == first.json()
    assert db.execute(select(func.count()).select_from(Upload)).scalar_one() == 1
    # The same key for a different file is refused; without a key nothing is replayed.
    other = client.post(
        url, json={**body, "bytes": 999}, headers=auth(**{"Idempotency-Key": "create-walk-1"})
    )
    assert other.status_code == 409
    assert other.json()["error"]["code"] == "IDEMPOTENCY_KEY_REUSED"
    assert client.post(url, json=body, headers=auth()).status_code == 413  # quota is held


def test_photo_count_limit(client, auth, db, app):
    app.state.settings.upload_max_photos = 2
    pid = create_project(client, auth)["id"]
    for _ in range(2):
        assert _init(client, auth, pid, kind="photo", ctype="image/jpeg").status_code == 201
    response = _init(client, auth, pid, kind="photo", ctype="image/jpeg")
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "QUOTA_EXCEEDED"


def test_full_upload_flow_with_immutable_copy(client, auth, db, s3):
    pid = create_project(client, auth)["id"]
    created = _init(client, auth, pid, size=1000)
    assert created.status_code == 201
    body = created.json()
    upload = body["upload"]
    assert body["partSize"] == 16 * MIB and body["partCount"] == 1
    assert upload["filename"] == "walk.mp4"
    assert upload["status"] == "initiated"
    upload_id = upload["id"]

    parts = client.post(f"/v1/uploads/{upload_id}/parts", json={"partNumbers": [1]}, headers=auth())
    assert parts.status_code == 200
    url = parts.json()["urls"][0]["url"]
    assert urlparse(url).netloc == "localhost:9000"
    assert "uploadId=" in url and "partNumber=1" in url
    assert parts.json()["expiresAt"].endswith("Z")

    incomplete = client.post(f"/v1/uploads/{upload_id}/complete", json={}, headers=auth())
    assert incomplete.status_code == 409
    assert incomplete.json()["error"]["code"] == "UPLOAD_INCOMPLETE"

    _put_part(s3, db, upload_id, 1, b"v" * 1000)
    status = client.get(f"/v1/uploads/{upload_id}", headers=auth()).json()
    assert [p["partNumber"] for p in status["completedParts"]] == [1]
    assert status["completedParts"][0]["size"] == 1000

    done = client.post(f"/v1/uploads/{upload_id}/complete", json={}, headers=auth())
    assert done.status_code == 202, done.text
    assert done.json()["upload"]["status"] == "uploaded_unverified"

    row = db.get(Upload, uuid.UUID(upload_id))
    db.refresh(row)
    assert row.immutable_key == f"tenants/{pid}/originals/{upload_id}"
    assert row.staging_key == f"tenants/{pid}/staging/{upload_id}"
    assert row.actual_bytes == 1000
    assert row.multipart_upload_id is None
    frozen = s3.get_object(Bucket=BUCKET, Key=row.immutable_key)["Body"].read()
    assert frozen == b"v" * 1000
    with pytest.raises(s3.exceptions.ClientError):
        s3.head_object(Bucket=BUCKET, Key=row.staging_key)
    job = db.execute(select(Job).where(Job.upload_id == row.id)).scalar_one()
    assert (job.stage, job.capability, job.state) == ("upload_inspection", "cpu", "queued")
    assert job.input_manifest["key"] == row.immutable_key

    # Completing again is an idempotent replay; aborting a frozen upload is refused.
    again = client.post(f"/v1/uploads/{upload_id}/complete", json={}, headers=auth())
    assert again.status_code == 202
    assert db.execute(select(Job).where(Job.upload_id == row.id)).scalars().all().__len__() == 1
    assert client.delete(f"/v1/uploads/{upload_id}", headers=auth()).status_code == 409
    # No more part URLs once frozen.
    assert (
        client.post(
            f"/v1/uploads/{upload_id}/parts", json={"partNumbers": [1]}, headers=auth()
        ).status_code
        == 409
    )


def test_multipart_upload_with_two_parts(client, auth, db, s3, app):
    app.state.settings.upload_part_size = 5 * MIB
    pid = create_project(client, auth)["id"]
    size = 5 * MIB + 10
    body = _init(client, auth, pid, size=size).json()
    assert body["partCount"] == 2
    upload_id = body["upload"]["id"]
    _put_part(s3, db, upload_id, 1, b"a" * (5 * MIB))
    _put_part(s3, db, upload_id, 2, b"b" * 10)
    done = client.post(f"/v1/uploads/{upload_id}/complete", headers=auth())
    assert done.status_code == 202, done.text
    row = db.get(Upload, uuid.UUID(upload_id))
    db.refresh(row)
    head = s3.head_object(Bucket=BUCKET, Key=row.immutable_key)
    assert head["ContentLength"] == size


def test_size_mismatch_rejected_and_deleted(client, auth, db, s3):
    pid = create_project(client, auth)["id"]
    upload_id = _init(client, auth, pid, size=2000).json()["upload"]["id"]
    _put_part(s3, db, upload_id, 1, b"x" * 1500)
    response = client.post(f"/v1/uploads/{upload_id}/complete", json={}, headers=auth())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UPLOAD_SIZE_MISMATCH"
    row = db.get(Upload, uuid.UUID(upload_id))
    db.refresh(row)
    assert row.status == "rejected" and row.rejection_code == "SIZE_MISMATCH"
    assert row.immutable_key is None
    listed = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"tenants/{pid}/")
    assert listed.get("KeyCount", 0) == 0
    assert db.execute(select(Job)).first() is None


def test_parts_validation(client, auth):
    pid = create_project(client, auth)["id"]
    upload_id = _init(client, auth, pid).json()["upload"]["id"]
    url = f"/v1/uploads/{upload_id}/parts"
    assert client.post(url, json={"partNumbers": [2]}, headers=auth()).status_code == 422
    assert client.post(url, json={"partNumbers": []}, headers=auth()).status_code == 422
    assert (
        client.post(url, json={"partNumbers": list(range(1, 22))}, headers=auth()).status_code
        == 422
    )
    assert client.post(url, json={"partNumbers": [1, 1]}, headers=auth()).status_code == 422


def test_abort_upload_and_release_quota(client, auth, db, settings):
    pid = create_project(client, auth)["id"]
    upload_id = _init(client, auth, pid, size=settings.upload_max_total_bytes).json()["upload"][
        "id"
    ]
    assert _init(client, auth, pid, size=10).status_code == 413
    assert client.delete(f"/v1/uploads/{upload_id}", headers=auth()).status_code == 204
    assert client.delete(f"/v1/uploads/{upload_id}", headers=auth()).status_code == 204
    row = db.get(Upload, uuid.UUID(upload_id))
    db.refresh(row)
    assert row.status == "aborted"
    assert row.multipart_upload_id is not None
    # A part in flight can land after the abort, so a cleanup checks none is left.
    [job] = _cleanup_jobs(db, upload_id)
    assert job.state == "queued" and job.input_manifest["stagingKey"] == row.staging_key
    assert _init(client, auth, pid, size=10).status_code == 201


def test_rejected_upload_holds_quota_until_its_object_is_deleted(client, auth, db, settings):
    pid = create_project(client, auth)["id"]
    upload_id = _init(client, auth, pid, size=settings.upload_max_total_bytes).json()["upload"][
        "id"
    ]
    row = db.get(Upload, uuid.UUID(upload_id))
    row.status, row.immutable_key = "rejected", f"tenants/{pid}/originals/{upload_id}"
    db.commit()
    assert _init(client, auth, pid, size=10).status_code == 413
    row.immutable_key = None  # the worker deleted the stored object
    db.commit()
    assert _init(client, auth, pid, size=10).status_code == 201


def test_upload_tenant_isolation(client, auth, db):
    pid = create_project(client, auth)["id"]
    upload_id = _init(client, auth, pid).json()["upload"]["id"]
    other = auth("user-b")
    assert client.get(f"/v1/uploads/{upload_id}", headers=other).status_code == 404
    assert (
        client.post(
            f"/v1/uploads/{upload_id}/parts", json={"partNumbers": [1]}, headers=other
        ).status_code
        == 404
    )
    assert client.post(f"/v1/uploads/{upload_id}/complete", headers=other).status_code == 404
    assert client.delete(f"/v1/uploads/{upload_id}", headers=other).status_code == 404
    assert _init(client, auth, pid, sub="user-b").status_code == 404


def test_capture_set_requires_verified_uploads(client, auth, db):
    pid = create_project(client, auth)["id"]
    pending = _init(client, auth, pid).json()["upload"]["id"]
    video = verified_upload(db, pid, kind="video")
    photo = verified_upload(db, pid, kind="photo")
    plan = verified_upload(db, pid, kind="floor_plan")
    url = f"/v1/projects/{pid}/capture-sets"

    r = client.post(url, json={"uploadIds": [video, pending]}, headers=auth())
    assert r.status_code == 422
    other_pid = create_project(client, auth, title="Other")["id"]
    foreign = verified_upload(db, other_pid)
    assert client.post(url, json={"uploadIds": [foreign]}, headers=auth()).status_code == 422
    assert client.post(url, json={"uploadIds": [plan]}, headers=auth()).status_code == 422
    assert (
        client.post(
            url, json={"uploadIds": [video], "floorPlanUploadId": photo}, headers=auth()
        ).status_code
        == 422
    )
    assert client.post(url, json={"uploadIds": []}, headers=auth()).status_code == 422

    ok = client.post(
        url,
        json={
            "uploadIds": [photo, video],
            "floorPlanUploadId": plan,
            "metadata": {"note": "ground floor"},
        },
        headers=auth(),
    )
    assert ok.status_code == 201, ok.text
    body = ok.json()
    assert body["floorPlanUploadId"] == plan
    assert [i["uploadId"] for i in body["inputs"]] == sorted([photo, video, plan])
    assert set(body["inputs"][0]) == {"uploadId", "mediaKind", "sha256", "bytes"}
    # Not visible to other users.
    assert client.post(url, json={"uploadIds": [video]}, headers=auth("user-b")).status_code == 404


def test_expired_upload_rejects_parts_and_frees_quota(client, auth, db, settings):
    from datetime import UTC, datetime, timedelta

    pid = create_project(client, auth)["id"]
    body = _init(client, auth, pid, size=settings.upload_max_total_bytes).json()
    upload_id = body["upload"]["id"]
    row = db.get(Upload, uuid.UUID(upload_id))
    row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()
    response = client.post(
        f"/v1/uploads/{upload_id}/parts", json={"partNumbers": [1]}, headers=auth()
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "UPLOAD_EXPIRED"
    assert _init(client, auth, pid, size=10).status_code == 201


def test_expired_uploads_are_refused_as_expired_and_no_longer_listed(client, auth, db):
    from datetime import UTC, datetime, timedelta

    pid = create_project(client, auth)["id"]
    lapsed, marked, live = (_init(client, auth, pid, size=10).json()["upload"]["id"]
                            for _ in range(3))  # fmt: skip
    db.get(Upload, uuid.UUID(lapsed)).expires_at = datetime.now(UTC) - timedelta(seconds=1)
    # The worker's maintenance marks an upload past its expiry this way.
    db.get(Upload, uuid.UUID(marked)).status = "expired"
    db.commit()
    for upload_id in (lapsed, marked):
        for path, body in (("parts", {"partNumbers": [1]}), ("complete", {})):
            response = client.post(f"/v1/uploads/{upload_id}/{path}", json=body, headers=auth())
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "UPLOAD_EXPIRED"
    # Neither can be resumed, so the project no longer lists them.
    detail = client.get(f"/v1/projects/{pid}", headers=auth()).json()
    assert [u["id"] for u in detail["uploads"]] == [live]


def test_removed_and_rejected_uploads_are_no_longer_listed(client, auth, db):
    pid = create_project(client, auth)["id"]
    removed, rejected, live = (_init(client, auth, pid, size=10).json()["upload"]["id"]
                               for _ in range(3))  # fmt: skip
    # As a removal and a failed inspection leave them, before and after their cleanup.
    db.get(Upload, uuid.UUID(removed)).status = "aborted"
    db.get(Upload, uuid.UUID(rejected)).status = "rejected"
    db.commit()
    detail = client.get(f"/v1/projects/{pid}", headers=auth()).json()
    assert [u["id"] for u in detail["uploads"]] == [live]
    # Each is still readable on its own, so a client can show why a file went away.
    for upload_id, status in ((removed, "aborted"), (rejected, "rejected")):
        response = client.get(f"/v1/uploads/{upload_id}", headers=auth())
        assert response.status_code == 200
        assert response.json()["upload"]["status"] == status


def _run(client, auth, pid, capture_set_id, key):
    return client.post(
        f"/v1/projects/{pid}/runs",
        json={"captureSetId": capture_set_id, "quality": "preview"},
        headers=auth(**{"Idempotency-Key": key}),
    )


def test_remove_unused_verified_upload(client, auth, db, settings):
    pid = create_project(client, auth)["id"]
    video = verified_upload(db, pid, size=settings.upload_max_total_bytes)
    assert _init(client, auth, pid, size=10).status_code == 413
    cs = client.post(
        f"/v1/projects/{pid}/capture-sets", json={"uploadIds": [video]}, headers=auth()
    )
    assert cs.status_code == 201
    assert client.delete(f"/v1/uploads/{video}", headers=auth()).status_code == 204
    row = db.get(Upload, uuid.UUID(video))
    db.refresh(row)
    assert row.status == "aborted" and row.immutable_key is not None
    job = db.execute(select(Job).where(Job.upload_id == row.id)).scalar_one()
    assert job.stage == "upload_cleanup" and job.input_manifest["key"] == row.immutable_key
    # Quota is released once the worker has deleted the bytes, not before.
    assert _init(client, auth, pid, size=10).status_code == 413
    row.immutable_key = None
    db.commit()
    assert _init(client, auth, pid, size=10).status_code == 201
    # A capture set made before the removal can no longer start a run.
    started = _run(client, auth, pid, cs.json()["id"], "key-removed-1")
    assert started.status_code == 409
    assert started.json()["error"]["code"] == "INVALID_STATE"


def test_upload_backing_a_run_cannot_be_removed(client, auth, db):
    pid = create_project(client, auth)["id"]
    video = verified_upload(db, pid)
    cs = client.post(
        f"/v1/projects/{pid}/capture-sets", json={"uploadIds": [video]}, headers=auth()
    )
    assert _run(client, auth, pid, cs.json()["id"], "key-backing-1").status_code == 202
    response = client.delete(f"/v1/uploads/{video}", headers=auth())
    assert response.status_code == 409
    row = db.get(Upload, uuid.UUID(video))
    db.refresh(row)
    assert row.status == "verified"


@pytest.mark.parametrize("include_photos", [False, True])
def test_photos_a_succeeded_run_never_read_can_be_removed(client, auth, db, include_photos):
    pid = create_project(client, auth)["id"]
    video = verified_upload(db, pid)
    photo = verified_upload(db, pid, kind="photo")
    cs = client.post(
        f"/v1/projects/{pid}/capture-sets", json={"uploadIds": [video, photo]}, headers=auth()
    )
    started = client.post(
        f"/v1/projects/{pid}/runs",
        json={
            "captureSetId": cs.json()["id"],
            "quality": "preview",
            "includeSupplementaryPhotos": include_photos,
        },
        headers=auth(**{"Idempotency-Key": f"key-photos-{include_photos}"}),
    )
    assert started.status_code == 202, started.text
    run = db.get(ReconstructionRun, uuid.UUID(started.json()["runId"]))
    run.status = "succeeded"
    db.commit()
    # The video backs the scene either way.
    assert client.delete(f"/v1/uploads/{video}", headers=auth()).status_code == 409
    # The worker left the photo out of a video run without supplementary photos.
    expected = 409 if include_photos else 204
    assert client.delete(f"/v1/uploads/{photo}", headers=auth()).status_code == expected


@pytest.mark.parametrize("include_photos", [False, True])
def test_retrying_after_removing_a_photo_the_run_never_read(client, auth, db, include_photos):
    pid = create_project(client, auth)["id"]
    video = verified_upload(db, pid)
    photo = verified_upload(db, pid, kind="photo")
    cs = client.post(
        f"/v1/projects/{pid}/capture-sets", json={"uploadIds": [video, photo]}, headers=auth()
    )
    started = client.post(
        f"/v1/projects/{pid}/runs",
        json={
            "captureSetId": cs.json()["id"],
            "quality": "preview",
            "includeSupplementaryPhotos": include_photos,
        },
        headers=auth(**{"Idempotency-Key": f"key-start-{include_photos}"}),
    )
    run = db.get(ReconstructionRun, uuid.UUID(started.json()["runId"]))
    run.status = "failed"
    db.commit()
    # A failed run keeps none of its inputs.
    assert client.delete(f"/v1/uploads/{photo}", headers=auth()).status_code == 204
    retry = client.post(
        f"/v1/runs/{run.id}/retry",
        json={},
        headers=auth(**{"Idempotency-Key": f"key-retry-{include_photos}"}),
    )
    # Only a run that reads the photo needs it back.
    assert retry.status_code == (409 if include_photos else 202), retry.text


def test_deleting_a_removed_upload_requeues_failed_cleanup(client, auth, db):
    pid = create_project(client, auth)["id"]
    video = verified_upload(db, pid)
    assert client.delete(f"/v1/uploads/{video}", headers=auth()).status_code == 204
    # A pending cleanup is not duplicated.
    assert client.delete(f"/v1/uploads/{video}", headers=auth()).status_code == 204
    jobs = select(Job).where(Job.upload_id == uuid.UUID(video), Job.stage == "upload_cleanup")
    first = db.execute(jobs).scalar_one()
    first.state = "failed"  # e.g. storage outage exhausted its attempts
    db.commit()
    assert client.delete(f"/v1/uploads/{video}", headers=auth()).status_code == 204
    states = sorted(j.state for j in db.execute(jobs).scalars())
    assert states == ["failed", "queued"]
