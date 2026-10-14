"""Multipart upload flow against moto S3."""

from __future__ import annotations

import uuid
from urllib.parse import urlparse

import pytest
from conftest import BUCKET
from factories import create_project, verified_upload
from sqlalchemy import select

from andarilho_api.models import Job, Upload

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
