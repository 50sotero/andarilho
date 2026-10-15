"""Scene review: layout save, approval, delivery, shares, public delivery."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

import pytest
from factories import create_project, layout, scene, verified_upload
from sqlalchemy import select

from andarilho_api.domain.canonical import sha256_hex
from andarilho_api.models import (
    AuditEvent,
    CaptureSet,
    Job,
    Project,
    ProjectMember,
    ReconstructionRun,
    Scene,
    SceneRevision,
    ShareLink,
    Upload,
)


@pytest.fixture
def setup(client, auth, db):
    pid = create_project(client, auth)["id"]
    ids = scene(db, pid)
    return {"pid": pid, **ids}


def _get_scene(client, auth, scene_id, sub="user-a"):
    return client.get(f"/v1/scenes/{scene_id}", headers=auth(sub))


def _approve(client, auth, s, version=1):
    return client.post(
        f"/v1/scenes/{s['sceneId']}/approve",
        json={"sceneRevisionId": s["revisionId"]},
        headers=auth(**{"If-Match": f'"{version}"'}),
    )


def _share(client, auth, s, version, **extra):
    return client.post(
        f"/v1/scenes/{s['sceneId']}/shares",
        json={"sceneRevisionId": s["revisionId"], **extra},
        headers=auth(**{"If-Match": f'"{version}"'}),
    )


def test_get_scene(client, auth, setup):
    response = _get_scene(client, auth, setup["sceneId"])
    assert response.status_code == 200
    assert response.headers["ETag"] == '"1"'
    body = response.json()
    assert body["scene"]["currentDraftRevisionId"] == setup["revisionId"]
    assert body["revision"]["validationState"] == "valid"
    assert body["revision"]["layout"]["schemaVersion"] == "1.0"
    assert len(body["revisions"]) == 1
    detail = client.get(f"/v1/projects/{setup['pid']}", headers=auth()).json()
    assert detail["scene"] == {
        "id": setup["sceneId"],
        "version": 1,
        "currentDraftRevisionId": setup["revisionId"],
    }


def test_layout_save(client, auth, db, setup):
    url = f"/v1/scenes/{setup['sceneId']}/layout"
    body = {"baseRevisionId": setup["revisionId"], "layout": layout()}
    assert client.put(url, json=body, headers=auth()).status_code == 428
    assert client.put(url, json=body, headers=auth(**{"If-Match": '"9"'})).status_code == 409

    bad = layout()
    bad["walls"][0]["thickness"] = -1
    invalid = client.put(url, json={**body, "layout": bad}, headers=auth(**{"If-Match": '"1"'}))
    assert invalid.status_code == 422
    unknown = layout() | {"shellCommand": "rm -rf /"}
    assert (
        client.put(
            url, json={**body, "layout": unknown}, headers=auth(**{"If-Match": '"1"'})
        ).status_code
        == 422
    )

    response = client.put(url, json=body, headers=auth(**{"If-Match": '"1"'}))
    assert response.status_code == 202, response.text
    assert response.headers["ETag"] == '"2"'
    result = response.json()
    assert result["version"] == 2
    revision = db.get(SceneRevision, uuid.UUID(result["sceneRevisionId"]))
    assert (revision.revision, revision.layout_revision) == (2, 2)
    assert revision.validation_state == "pending" and revision.manifest is None
    assert str(revision.base_revision_id) == setup["revisionId"]
    job = db.get(Job, uuid.UUID(result["jobId"]))
    assert job.stage == "geometry_proposal" and str(job.run_id) == setup["runId"]
    assert str(job.scene_revision_id) == result["sceneRevisionId"]
    assert job.input_manifest["sceneRevisionId"] == result["sceneRevisionId"]
    assert job.input_manifest["layoutSource"] == "user"
    project = db.get(Project, uuid.UUID(setup["pid"]))
    db.refresh(project)
    assert str(project.active_draft_revision_id) == result["sceneRevisionId"]

    # The old base is no longer current: saving on top of it conflicts.
    stale = client.put(url, json=body, headers=auth(**{"If-Match": '"2"'}))
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "VERSION_CONFLICT"
    # Viewers cannot save.
    from andarilho_api.models import ProjectMember

    db.add(ProjectMember(project_id=uuid.UUID(setup["pid"]), subject="viewer", role="viewer"))
    db.commit()
    assert (
        client.put(
            url,
            json={**body, "baseRevisionId": result["sceneRevisionId"]},
            headers=auth("viewer", **{"If-Match": '"2"'}),
        ).status_code
        == 403
    )


def test_approve_rules(client, auth, db):
    pid = create_project(client, auth)["id"]
    pending = scene(db, pid, validation="pending")
    response = _approve(client, auth, pending)
    assert response.status_code == 409 and response.json()["error"]["code"] == "INVALID_STATE"

    pid2 = create_project(client, auth, title="b")["id"]
    inspection = scene(db, pid2, mode="inspection_only")
    assert _approve(client, auth, inspection).status_code == 409


def test_approve_share_and_public_delivery(client, auth, db, setup):
    assert _approve(client, auth, setup, version=7).status_code == 409
    approved = _approve(client, auth, setup)
    assert approved.status_code == 200
    assert approved.json() == {
        "sceneRevisionId": setup["revisionId"],
        "approvalState": "approved",
        "version": 2,
    }
    assert db.execute(
        select(AuditEvent).where(AuditEvent.action == "scene_revision.approved")
    ).scalar_one()

    # Owner-only and If-Match protected.
    assert (
        client.post(
            f"/v1/scenes/{setup['sceneId']}/shares",
            json={"sceneRevisionId": setup["revisionId"]},
            headers=auth(),
        ).status_code
        == 428
    )
    past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    assert _share(client, auth, setup, 2, expiresAt=past).status_code == 422

    created = _share(client, auth, setup, 2)
    assert created.status_code == 201, created.text
    body = created.json()
    token = body["token"]
    assert len(token) >= 43 and body["path"] == f"/s/{token}"
    share = db.get(ShareLink, uuid.UUID(body["share"]["id"]))
    assert share.token_hash != token.encode() and len(share.token_hash) == 32
    project = db.get(Project, uuid.UUID(setup["pid"]))
    db.refresh(project)
    assert str(project.published_revision_id) == setup["revisionId"]
    audit_rows = db.execute(select(AuditEvent.data)).scalars().all()
    assert all(token not in str(row) for row in audit_rows)

    public = client.post("/v1/public/scenes/delivery", json={"token": token})
    assert public.status_code == 200, public.text
    envelope = public.json()
    assert set(envelope["assetUrls"]) <= {"render", "collision", "poster"}
    assert set(envelope["assetUrls"]) == {"render", "collision", "poster"}
    assert envelope["sceneRevisionId"] == setup["revisionId"]
    assert envelope["manifestSha256"] == sha256_hex(envelope["manifest"])
    assert public.headers["Cache-Control"] == "private, no-store"
    for asset in envelope["assetUrls"].values():
        parsed = urlparse(asset["url"])
        assert parsed.netloc == "localhost:9000"
        assert f"/tenants/{setup['pid']}/scenes/{setup['revisionId']}/" in parsed.path
        assert "originals" not in parsed.path and "staging" not in parsed.path
    assert token not in public.text

    # Revocation: same 404 as an unknown token.
    unknown = client.post("/v1/public/scenes/delivery", json={"token": "x" * 43})
    assert unknown.status_code == 404
    assert client.delete(f"/v1/shares/{share.id}", headers=auth("user-b")).status_code == 404
    assert client.delete(f"/v1/shares/{share.id}", headers=auth()).status_code == 204
    revoked = client.post("/v1/public/scenes/delivery", json={"token": token})
    assert revoked.status_code == 404
    assert revoked.json()["error"] == unknown.json()["error"] | {
        "requestId": revoked.json()["error"]["requestId"]
    }
    db.refresh(project)
    assert project.published_revision_id is None


def test_expired_share(client, auth, db, setup):
    _approve(client, auth, setup)
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    body = _share(client, auth, setup, 2, expiresAt=future).json()
    assert (
        client.post("/v1/public/scenes/delivery", json={"token": body["token"]}).status_code == 200
    )
    share = db.get(ShareLink, uuid.UUID(body["share"]["id"]))
    share.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()
    assert (
        client.post("/v1/public/scenes/delivery", json={"token": body["token"]}).status_code == 404
    )


def test_share_requires_approved_revision(client, auth, setup):
    response = _share(client, auth, setup, 1)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INVALID_STATE"


def test_member_delivery(client, auth, db, setup):
    url = f"/v1/scenes/{setup['sceneId']}/delivery"
    response = client.post(url, json={"sceneRevisionId": setup["revisionId"]}, headers=auth())
    assert response.status_code == 200
    envelope = response.json()
    assert set(envelope) == {
        "sceneRevisionId",
        "manifestSha256",
        "expiresAt",
        "manifest",
        "assetUrls",
    }
    assert envelope["expiresAt"].endswith("Z")
    other = client.post(url, json={"sceneRevisionId": setup["revisionId"]}, headers=auth("user-b"))
    assert other.status_code == 404
    wrong_rev = client.post(url, json={"sceneRevisionId": str(uuid.uuid4())}, headers=auth())
    assert wrong_rev.status_code == 404


def test_delivery_rejects_foreign_asset_keys(client, auth, db, setup):
    revision = db.get(SceneRevision, uuid.UUID(setup["revisionId"]))
    doc = dict(revision.manifest)
    other_project = str(uuid.uuid4())
    assets = {k: dict(v) for k, v in doc["assets"].items()}
    for asset in assets.values():
        asset["key"] = asset["key"].replace(setup["pid"], other_project)
    revision.manifest = {**doc, "assets": assets}
    db.commit()
    response = client.post(
        f"/v1/scenes/{setup['sceneId']}/delivery",
        json={"sceneRevisionId": setup["revisionId"]},
        headers=auth(),
    )
    assert response.status_code == 409
    assert other_project not in response.text


def test_public_rate_limit(client, auth, app):
    app.state.public_rate_limiter.limit = 3
    for _ in range(3):
        assert (
            client.post("/v1/public/scenes/delivery", json={"token": "y" * 43}).status_code == 404
        )
    limited = client.post("/v1/public/scenes/delivery", json={"token": "y" * 43})
    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "RATE_LIMITED"
    assert int(limited.headers["Retry-After"]) >= 1


def test_scene_isolation(client, auth, setup):
    other = auth("user-b", **{"If-Match": '"1"'})
    sid = setup["sceneId"]
    assert client.get(f"/v1/scenes/{sid}", headers=other).status_code == 404
    assert (
        client.put(
            f"/v1/scenes/{sid}/layout",
            json={"baseRevisionId": setup["revisionId"], "layout": layout()},
            headers=other,
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/v1/scenes/{sid}/approve",
            json={"sceneRevisionId": setup["revisionId"]},
            headers=other,
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/v1/scenes/{sid}/shares", json={"sceneRevisionId": setup["revisionId"]}, headers=other
        ).status_code
        == 404
    )


def test_public_rate_limit_keys_on_nearest_forwarded_hop(settings):
    from starlette.requests import Request

    from andarilho_api.routes.scenes import _client_key

    def request(forwarded: str) -> Request:
        headers = [(b"x-forwarded-for", forwarded.encode())]
        return Request({"type": "http", "headers": headers, "client": ("10.0.0.5", 1)})

    trusted = settings.model_copy(update={"public_delivery_trust_forwarded_for": True})
    # Visitors behind the web proxy get their own buckets; a spoofed prefix is ignored.
    assert _client_key(request("203.0.113.9"), trusted) == "203.0.113.9"
    assert _client_key(request("6.6.6.6, 203.0.113.9"), trusted) == "203.0.113.9"
    assert _client_key(request("203.0.113.9"), settings) == "10.0.0.5"


def test_floor_plan_is_private_to_owners_and_editors(client, auth, db, setup):
    assert _get_scene(client, auth, setup["sceneId"]).json()["floorPlan"] is None
    plan_id = verified_upload(db, setup["pid"], kind="floor_plan")
    scene = db.get(Scene, uuid.UUID(setup["sceneId"]))
    run = db.get(ReconstructionRun, scene.run_id)
    db.get(CaptureSet, run.capture_set_id).floor_plan_upload_id = uuid.UUID(plan_id)
    db.add(ProjectMember(project_id=uuid.UUID(setup["pid"]), subject="viewer-v", role="viewer"))
    db.commit()

    plan = _get_scene(client, auth, setup["sceneId"]).json()["floorPlan"]
    assert plan["uploadId"] == plan_id and plan["contentType"] == "application/pdf"
    assert f"originals/{plan_id}" in plan["url"] and plan["expiresAt"]
    # A PNG declared as a PDF is labelled by its inspected signature.
    upload = db.get(Upload, uuid.UUID(plan_id))
    upload.media_metadata = {"format": "png", "width": 10, "height": 10}
    db.commit()
    assert _get_scene(client, auth, setup["sceneId"]).json()["floorPlan"]["contentType"] == (
        "image/png"
    )
    # Viewers cannot read draft review material at all (layout, findings, plan).
    assert _get_scene(client, auth, setup["sceneId"], sub="viewer-v").status_code == 403


def test_viewers_only_receive_the_live_published_revision(client, auth, db, setup):
    db.add(ProjectMember(project_id=uuid.UUID(setup["pid"]), subject="viewer-v", role="viewer"))
    db.commit()
    url = f"/v1/scenes/{setup['sceneId']}/delivery"
    body = {"sceneRevisionId": setup["revisionId"]}
    assert client.post(url, json=body, headers=auth("viewer-v")).status_code == 404
    assert _approve(client, auth, setup).status_code == 200
    # Approved but not published: still review material.
    assert client.post(url, json=body, headers=auth("viewer-v")).status_code == 404
    assert _share(client, auth, setup, 2).status_code == 201
    assert client.post(url, json=body, headers=auth("viewer-v")).status_code == 200
    share = db.execute(select(ShareLink)).scalar_one()
    share.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()
    assert client.post(url, json=body, headers=auth("viewer-v")).status_code == 404
    assert client.post(url, json=body, headers=auth()).status_code == 200  # owners still can


def test_publishing_invalidates_the_scene_etag(client, auth, setup):
    assert _approve(client, auth, setup).status_code == 200
    # Two owner tabs loaded the scene at version 2; the first one publishes.
    first = _share(client, auth, setup, 2)
    assert first.status_code == 201 and first.headers["ETag"] == '"3"'
    assert _get_scene(client, auth, setup["sceneId"]).headers["ETag"] == '"3"'
    # The second tab's stale ETag no longer passes, so it cannot publish over it.
    stale = _share(client, auth, setup, 2)
    assert stale.status_code == 409
    assert _share(client, auth, setup, 3).status_code == 201


def test_unpublishing_by_revocation_invalidates_the_scene_etag(client, auth, setup):
    assert _approve(client, auth, setup).status_code == 200
    first = _share(client, auth, setup, 2).json()["share"]["id"]
    second = _share(client, auth, setup, 3).json()["share"]["id"]
    # Revoking one of two links keeps the revision published: the ETag stands.
    assert client.delete(f"/v1/shares/{first}", headers=auth()).status_code == 204
    assert _get_scene(client, auth, setup["sceneId"]).headers["ETag"] == '"4"'
    # Revoking the last link unpublishes it.
    assert client.delete(f"/v1/shares/{second}", headers=auth()).status_code == 204
    assert _get_scene(client, auth, setup["sceneId"]).headers["ETag"] == '"5"'
    # A tab that loaded the scene while it was published cannot publish on that state.
    assert _share(client, auth, setup, 4).status_code == 409
    assert _share(client, auth, setup, 5).status_code == 201


def test_publishing_another_scene_invalidates_the_previously_published_one(client, auth, db, setup):
    assert _approve(client, auth, setup).status_code == 200
    assert _share(client, auth, setup, 2).status_code == 201
    newer = {"pid": setup["pid"], **scene(db, setup["pid"])}
    assert _approve(client, auth, newer).status_code == 200
    assert _share(client, auth, newer, 2).status_code == 201
    # The first scene's revision is no longer the published one.
    assert _get_scene(client, auth, setup["sceneId"]).headers["ETag"] == '"4"'


def _expire_shares_of(db, revision_id: str) -> None:
    for share in db.execute(
        select(ShareLink).where(ShareLink.scene_revision_id == uuid.UUID(revision_id))
    ).scalars():
        share.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()


def _publish_then_expire_a_newer_scene(client, auth, db, setup) -> dict:
    """Scene A published, then scene B published over it and B's link expired.

    The stored pointer still names B, but viewers see A again through its live link.
    """
    assert _approve(client, auth, setup).status_code == 200
    assert _share(client, auth, setup, 2).status_code == 201  # A at version 3
    newer = {"pid": setup["pid"], **scene(db, setup["pid"])}
    assert _approve(client, auth, newer).status_code == 200
    assert _share(client, auth, newer, 2).status_code == 201  # A bumped to 4
    _expire_shares_of(db, newer["revisionId"])
    project = client.get(f"/v1/projects/{setup['pid']}", headers=auth()).json()["project"]
    assert project["publishedRevisionId"] == setup["revisionId"]
    assert _get_scene(client, auth, setup["sceneId"]).headers["ETag"] == '"4"'
    return newer


def test_publishing_bumps_the_scene_viewers_see_when_the_pointer_expired(client, auth, db, setup):
    _publish_then_expire_a_newer_scene(client, auth, db, setup)
    third = {"pid": setup["pid"], **scene(db, setup["pid"])}
    assert _approve(client, auth, third).status_code == 200
    assert _share(client, auth, third, 2).status_code == 201
    # Scene A was what viewers saw, and now it is not: a tab holding its ETag must conflict.
    assert _get_scene(client, auth, setup["sceneId"]).headers["ETag"] == '"5"'
    assert _share(client, auth, setup, 4).status_code == 409


def test_revoking_the_link_viewers_see_bumps_its_scene_when_the_pointer_expired(
    client, auth, db, setup
):
    _publish_then_expire_a_newer_scene(client, auth, db, setup)
    live = db.execute(
        select(ShareLink).where(ShareLink.scene_revision_id == uuid.UUID(setup["revisionId"]))
    ).scalar_one()
    assert client.delete(f"/v1/shares/{live.id}", headers=auth()).status_code == 204
    # The stored pointer named the expired scene, but scene A is what stopped being shown.
    assert _get_scene(client, auth, setup["sceneId"]).headers["ETag"] == '"5"'
    project = client.get(f"/v1/projects/{setup['pid']}", headers=auth()).json()["project"]
    assert project["publishedRevisionId"] is None


def test_revoking_newest_share_keeps_older_live_link_published(client, auth, db, setup):
    assert _approve(client, auth, setup).status_code == 200
    assert _share(client, auth, setup, 2).status_code == 201
    old = db.get(SceneRevision, uuid.UUID(setup["revisionId"]))
    newer = SceneRevision(
        id=uuid.uuid4(),
        scene_id=old.scene_id,
        project_id=old.project_id,
        run_id=old.run_id,
        revision=old.revision + 1,
        layout_revision=old.layout_revision + 1,
        layout=old.layout,
        manifest=old.manifest,
        validation_state="valid",
        approval_state="approved",
        approved_by="user-a",
        approved_at=datetime.now(UTC),
        created_by="user-a",
    )
    db.add(newer)
    db.flush()
    newest = ShareLink(
        project_id=old.project_id,
        scene_revision_id=newer.id,
        token_hash=b"n" * 32,
        created_by="user-a",
        created_at=datetime.now(UTC) + timedelta(seconds=5),
    )
    db.add(newest)
    db.get(Project, old.project_id).published_revision_id = newer.id
    db.commit()

    assert client.delete(f"/v1/shares/{newest.id}", headers=auth()).status_code == 204
    project = db.get(Project, old.project_id)
    db.refresh(project)
    # The older revision's link still works, so the project stays published on it.
    assert project.published_revision_id == old.id


def test_viewer_project_detail_uses_the_published_scene(client, auth, db, setup):
    db.add(ProjectMember(project_id=uuid.UUID(setup["pid"]), subject="viewer-v", role="viewer"))
    db.commit()
    url = f"/v1/projects/{setup['pid']}"
    assert client.get(url, headers=auth("viewer-v")).json()["scene"] is None  # nothing published
    assert _approve(client, auth, setup).status_code == 200
    assert _share(client, auth, setup, 2).status_code == 201
    newer = scene(db, setup["pid"])  # a newer draft scene becomes the active draft
    assert client.get(url, headers=auth()).json()["scene"]["id"] == newer["sceneId"]
    assert client.get(url, headers=auth("viewer-v")).json()["scene"]["id"] == setup["sceneId"]


def test_publication_ends_when_the_last_share_expires(client, auth, db, setup):
    assert _approve(client, auth, setup).status_code == 200
    expires = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    assert _share(client, auth, setup, 2, expiresAt=expires).status_code == 201
    url = f"/v1/projects/{setup['pid']}"
    assert (
        client.get(url, headers=auth()).json()["project"]["publishedRevisionId"]
        == (setup["revisionId"])
    )
    share = db.execute(select(ShareLink)).scalar_one()
    share.expires_at = datetime.now(UTC) - timedelta(seconds=1)  # expired naturally
    db.commit()
    assert client.get(url, headers=auth()).json()["project"]["publishedRevisionId"] is None
    listed = client.get("/v1/projects", headers=auth()).json()["items"][0]
    assert listed["publishedRevisionId"] is None
