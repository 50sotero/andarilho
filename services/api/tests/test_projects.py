"""Projects: CRUD, pagination, If-Match, error envelope, health."""

from __future__ import annotations

import uuid

from factories import create_project
from sqlalchemy import select

from andarilho_api.models import AuditEvent, ProjectMember


def test_create_and_get_project(client, auth, db):
    response = client.post("/v1/projects", json={"title": "  Casa  "}, headers=auth())
    assert response.status_code == 201
    project = response.json()
    assert response.headers["ETag"] == '"1"'
    assert response.headers["Cache-Control"] == "private, no-store"
    assert project["title"] == "Casa"
    assert project["role"] == "owner"
    assert project["lifecycle"] == "active"
    assert project["activeDraftRevisionId"] is None
    assert set(project) == {
        "id",
        "title",
        "lifecycle",
        "version",
        "role",
        "activeDraftRevisionId",
        "publishedRevisionId",
        "createdAt",
        "updatedAt",
    }
    member = db.execute(select(ProjectMember)).scalar_one()
    assert (member.subject, member.role) == ("user-a", "owner")
    assert db.execute(select(AuditEvent.action)).scalar_one() == "project.created"

    detail = client.get(f"/v1/projects/{project['id']}", headers=auth())
    assert detail.status_code == 200
    assert detail.headers["ETag"] == '"1"'
    body = detail.json()
    assert set(body) == {"project", "uploads", "captureSets", "runs", "scene", "shares"}
    assert body["scene"] is None


def test_validation_errors_do_not_echo_input(client, auth):
    evil = "'; DROP TABLE projects; --" + "x" * 300
    response = client.post("/v1/projects", json={"title": evil}, headers=auth())
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_FAILED"
    assert "DROP TABLE" not in response.text
    assert set(error) == {"code", "message", "retryable", "suggestedAction", "requestId"}
    assert error["requestId"] == response.headers["X-Request-ID"]

    response = client.post(
        "/v1/projects", json={"title": "ok", "<script>alert(1)</script>": 1}, headers=auth()
    )
    assert response.status_code == 422
    assert "script" not in response.text
    assert "unknown field" in response.json()["error"]["message"]

    response = client.post(
        "/v1/projects",
        content=b"{not json SELECT * FROM pg_user",
        headers={**auth(), "Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "pg_user" not in response.text


def test_request_id_header_policy(client, auth):
    good = client.get("/healthz", headers={"X-Request-ID": "abc12345-req"})
    assert good.headers["X-Request-ID"] == "abc12345-req"
    bad = client.get("/healthz", headers={"X-Request-ID": 'x" onload=alert(1) ' * 3})
    assert bad.headers["X-Request-ID"] != "x"
    assert "alert" not in bad.headers["X-Request-ID"]


def test_unknown_route_and_bad_uuid(client, auth):
    response = client.get("/v1/nope", headers=auth())
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
    response = client.get("/v1/projects/not-a-uuid", headers=auth())
    assert response.status_code == 422


def test_unhandled_errors_are_generic(app, auth, monkeypatch):
    from fastapi.testclient import TestClient

    import andarilho_api.routes.projects as projects_module

    def boom(*args, **kwargs):
        raise RuntimeError("psycopg.errors.UndefinedTable: SELECT secret FROM /etc/passwd")

    monkeypatch.setattr(projects_module, "_project_detail", boom)
    with TestClient(app, raise_server_exceptions=False) as client:
        created = create_project(client, auth)
        response = client.get(f"/v1/projects/{created['id']}", headers=auth())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL"
    assert "passwd" not in response.text and "SELECT" not in response.text
    assert response.json()["error"]["requestId"]


def test_pagination(client, auth):
    ids = [create_project(client, auth, title=f"P{i}")["id"] for i in range(5)]
    create_project(client, auth, sub="user-b", title="other")
    seen: list[str] = []
    cursor = None
    for _ in range(5):
        params = {"limit": 2, **({"cursor": cursor} if cursor else {})}
        page = client.get("/v1/projects", params=params, headers=auth()).json()
        seen.extend(p["id"] for p in page["items"])
        cursor = page["nextCursor"]
        if cursor is None:
            break
    assert seen == list(reversed(ids))
    assert client.get("/v1/projects", params={"limit": 101}, headers=auth()).status_code == 422
    bad = client.get("/v1/projects", params={"cursor": "%%%garbage"}, headers=auth())
    assert bad.status_code == 422


def test_patch_requires_if_match(client, auth):
    project = create_project(client, auth)
    url = f"/v1/projects/{project['id']}"
    missing = client.patch(url, json={"title": "New"}, headers=auth())
    assert missing.status_code == 428
    assert missing.json()["error"]["code"] == "PRECONDITION_REQUIRED"
    stale = client.patch(url, json={"title": "New"}, headers=auth(**{"If-Match": '"7"'}))
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "VERSION_CONFLICT"
    ok = client.patch(url, json={"title": "New"}, headers=auth(**{"If-Match": '"1"'}))
    assert ok.status_code == 200
    assert ok.json()["title"] == "New"
    assert ok.json()["version"] == 2
    assert ok.headers["ETag"] == '"2"'
    again = client.patch(url, json={"title": "Newer"}, headers=auth(**{"If-Match": '"1"'}))
    assert again.status_code == 409


def test_viewer_cannot_patch_and_non_member_gets_404(client, auth, db):
    project = create_project(client, auth)
    db.add(ProjectMember(project_id=uuid.UUID(project["id"]), subject="viewer-v", role="viewer"))
    db.commit()
    url = f"/v1/projects/{project['id']}"
    assert client.get(url, headers=auth("viewer-v")).json()["project"]["role"] == "viewer"
    response = client.patch(url, json={"title": "x"}, headers=auth("viewer-v", **{"If-Match": "1"}))
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN"
    assert client.delete(url, headers=auth("viewer-v")).status_code == 403
    for method in ("get", "delete"):
        response = getattr(client, method)(url, headers=auth("user-b"))
        assert response.status_code == 404
    response = client.patch(url, json={"title": "x"}, headers=auth("user-b", **{"If-Match": "1"}))
    assert response.status_code == 404
    missing = client.get(f"/v1/projects/{uuid.uuid4()}", headers=auth())
    assert missing.status_code == 404
    assert missing.json()["error"]["message"] == response.json()["error"]["message"]


def test_health(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["checks"] == {"database": "ok", "storage": "ok"}
