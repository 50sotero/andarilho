"""Project deletion protocol steps 1-2 and post-deletion behaviour."""

from __future__ import annotations

import uuid

from factories import active_run, create_project, layout, scene, verified_upload
from sqlalchemy import select

from andarilho_api.models import (
    AuditEvent,
    Job,
    Project,
    ReconstructionRun,
    ShareLink,
    Upload,
    UsageLedger,
)


def test_delete_project_lifecycle(client, auth, db, s3):
    pid = create_project(client, auth)["id"]
    ids = scene(db, pid)
    client.post(
        f"/v1/scenes/{ids['sceneId']}/approve",
        json={"sceneRevisionId": ids["revisionId"]},
        headers=auth(**{"If-Match": '"1"'}),
    )
    token = client.post(
        f"/v1/scenes/{ids['sceneId']}/shares",
        json={"sceneRevisionId": ids["revisionId"]},
        headers=auth(**{"If-Match": '"2"'}),
    ).json()["token"]
    running = active_run(db, pid, job_states=("succeeded", "running", "queued"))
    pending_upload = client.post(
        f"/v1/projects/{pid}/uploads",
        json={"filename": "a.mp4", "bytes": 100, "mediaKind": "video", "contentType": "video/mp4"},
        headers=auth(),
    ).json()["upload"]["id"]
    video = verified_upload(db, pid)

    # Only the owner may delete; strangers get 404.
    assert client.delete(f"/v1/projects/{pid}", headers=auth("user-b")).status_code == 404
    response = client.delete(f"/v1/projects/{pid}", headers=auth())
    assert response.status_code == 202
    assert response.json() == {"projectId": pid, "lifecycle": "deleting"}
    db.expire_all()

    project = db.get(Project, uuid.UUID(pid))
    assert project.lifecycle == "deleting" and project.published_revision_id is None
    assert all(s.revoked_at is not None for s in db.execute(select(ShareLink)).scalars())
    jobs = {
        j.stage: j
        for j in db.execute(select(Job).where(Job.run_id == uuid.UUID(running["runId"]))).scalars()
    }
    assert jobs["validating"].state == "succeeded"
    assert jobs["preparing"].state == "running"
    assert jobs["preparing"].cancel_requested_at is not None
    assert jobs["preparing"].fencing_token == 2  # stale worker writes now fail
    assert jobs["camera_reconstruction"].state == "cancelled"
    run = db.get(ReconstructionRun, uuid.UUID(running["runId"]))
    assert run.cancel_requested_at is not None
    ledger = db.execute(select(UsageLedger).where(UsageLedger.run_id == run.id)).scalar_one()
    assert ledger.state == "reserved"  # released after the running job is confirmed stopped
    pending = db.get(Upload, uuid.UUID(pending_upload))
    # The ID stays so the cleanup can abort it again and check no late part was stored.
    assert pending.status == "aborted" and pending.multipart_upload_id is not None
    assert db.get(Upload, uuid.UUID(video)).status == "verified"
    cleanup = db.execute(select(Job).where(Job.stage == "project_cleanup")).scalar_one()
    assert cleanup.run_id is None and cleanup.capability == "cpu" and cleanup.state == "queued"
    assert cleanup.input_manifest["prefix"] == f"tenants/{pid}/"
    assert db.execute(
        select(AuditEvent).where(AuditEvent.action == "project.deletion_requested")
    ).scalar_one()

    # Public share stops working.
    assert client.post("/v1/public/scenes/delivery", json={"token": token}).status_code == 404

    # Mutations are rejected; reads still show the lifecycle.
    detail = client.get(f"/v1/projects/{pid}", headers=auth())
    assert detail.json()["project"]["lifecycle"] == "deleting"
    checks = [
        client.patch(
            f"/v1/projects/{pid}", json={"title": "x"}, headers=auth(**{"If-Match": '"2"'})
        ),
        client.post(
            f"/v1/projects/{pid}/uploads",
            json={
                "filename": "a.mp4",
                "bytes": 1,
                "mediaKind": "video",
                "contentType": "video/mp4",
            },
            headers=auth(),
        ),
        client.post(
            f"/v1/projects/{pid}/capture-sets", json={"uploadIds": [video]}, headers=auth()
        ),
        client.post(
            f"/v1/projects/{pid}/runs",
            json={"captureSetId": running["captureSetId"], "quality": "preview"},
            headers=auth(**{"Idempotency-Key": "key-after-delete"}),
        ),
        client.put(
            f"/v1/scenes/{ids['sceneId']}/layout",
            json={"baseRevisionId": ids["revisionId"], "layout": layout()},
            headers=auth(**{"If-Match": '"2"'}),
        ),
        client.post(
            f"/v1/scenes/{ids['sceneId']}/shares",
            json={"sceneRevisionId": ids["revisionId"]},
            headers=auth(**{"If-Match": '"2"'}),
        ),
        client.post(
            f"/v1/scenes/{ids['sceneId']}/delivery",
            json={"sceneRevisionId": ids["revisionId"]},
            headers=auth(),
        ),
    ]
    for r in checks:
        assert r.status_code == 409, r.text
        assert r.json()["error"]["code"] == "PROJECT_DELETING"

    # Deleting again is idempotent and does not queue a second cleanup.
    assert client.delete(f"/v1/projects/{pid}", headers=auth()).status_code == 202
    assert len(db.execute(select(Job).where(Job.stage == "project_cleanup")).all()) == 1


def test_delete_idle_run_releases_reservation(client, auth, db):
    pid = create_project(client, auth)["id"]
    info = active_run(db, pid, job_states=("queued",))
    assert client.delete(f"/v1/projects/{pid}", headers=auth()).status_code == 202
    db.expire_all()
    run = db.get(ReconstructionRun, uuid.UUID(info["runId"]))
    assert run.status == "cancelled"
    ledger = db.execute(select(UsageLedger).where(UsageLedger.run_id == run.id)).scalar_one()
    assert ledger.state == "released"


def test_repeated_delete_requeues_failed_cleanup(client, auth, db):
    pid = create_project(client, auth)["id"]
    assert client.delete(f"/v1/projects/{pid}", headers=auth()).status_code == 202
    cleanups = select(Job).where(Job.stage == "project_cleanup")
    first = db.execute(cleanups).scalar_one()
    first.state = "failed"  # e.g. attempts exhausted during a storage outage
    db.commit()
    assert client.delete(f"/v1/projects/{pid}", headers=auth()).status_code == 202
    db.expire_all()
    jobs = sorted(db.execute(cleanups).scalars(), key=lambda j: j.state)
    assert [j.state for j in jobs] == ["failed", "queued"]
    # The original request time is kept, so the cleanup horizon does not move.
    assert jobs[1].input_manifest["requestedAt"] == jobs[0].input_manifest["requestedAt"]
    assert client.delete(f"/v1/projects/{pid}", headers=auth()).status_code == 202
    assert len(db.execute(cleanups).all()) == 2
