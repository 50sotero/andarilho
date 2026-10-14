"""CPU end-to-end tests through the real supervisor, queue and Postgres schema."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import threading
import time
import uuid

import pytest
from psycopg.types.json import Jsonb

from andarilho_worker import queue
from andarilho_worker.stages import STAGES
from andarilho_worker.stages import inspect_upload as inspect_mod
from andarilho_worker.supervisor import Supervisor
from andarilho_worker.validation import manifest_sha256


def drain(db, sup, *, limit: int = 30) -> list[tuple[str, str]]:
    done = []
    for _ in range(limit):
        queue.promote_retry_wait(db)
        job = queue.claim(db, "worker-test", "cpu")
        if job is None:
            break
        done.append((job.stage, sup.execute(job)))
    return done


@pytest.fixture
def sup(db, store, settings):
    return Supervisor(db, store, settings, delay_fn=lambda attempt: 0.0)


def test_synthetic_recipe_end_to_end(db, seed, store, sup):
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    done = drain(db, sup)
    assert done == [
        ("validating", "succeeded"),
        ("preparing", "succeeded"),
        ("normalizing", "succeeded"),
        ("geometry_proposal", "succeeded"),
        ("packaging", "succeeded"),
        ("scene_validation", "succeeded"),
    ]
    run = seed.one("SELECT status FROM reconstruction_runs WHERE id = %s", run_id)
    assert run["status"] == "succeeded"
    ledger = seed.one("SELECT state, actual_cpu_seconds FROM usage_ledger")
    assert ledger["state"] == "settled" and ledger["actual_cpu_seconds"] >= 1
    rev = seed.one("SELECT * FROM scene_revisions")
    assert rev["validation_state"] == "valid" and rev["revision"] == 1
    manifest = rev["manifest"]
    assert manifest["sceneRevisionId"] == str(rev["id"])
    assert manifest["sceneId"] == str(rev["scene_id"]) and manifest["runId"] == run_id
    assert manifest["navigation"]["mode"] == "approved_collision"
    assert manifest["provenance"]["geometrySource"] == "synthetic_fixture"
    assert rev["manifest_sha256"] == manifest_sha256(manifest)
    prefix = f"tenants/{pid}/scenes/{rev['id']}/"
    assert set(store.list_keys(prefix)) == {
        prefix + "scene.ply",
        prefix + "collision.glb",
        prefix + "manifest.json",
    }
    stored = json.loads(store.get_bytes(prefix + "manifest.json"))
    assert stored == manifest
    project = seed.one("SELECT active_draft_revision_id FROM projects WHERE id = %s", pid)
    scene = seed.one("SELECT current_draft_revision_id FROM scenes")
    assert project["active_draft_revision_id"] == rev["id"] == scene["current_draft_revision_id"]
    arts = seed.all("SELECT kind, validation_state FROM artifacts ORDER BY kind")
    assert [a["kind"] for a in arts] == ["scene_collision", "scene_manifest", "scene_render"]
    assert rev["manifest_artifact_id"] is not None
    # Every stage carried earlier outputs forward and wrote an immutable attempt manifest.
    last = seed.one("SELECT input_manifest FROM jobs WHERE stage = 'scene_validation'")
    assert {"captureSetId", "renderKey", "collisionKey", "manifestKey"} <= set(
        last["input_manifest"]["outputs"]
    )
    assert any(
        "/attempts/packaging/1/result.json" in k for k in store.list_keys(f"tenants/{pid}/runs/")
    )

    # --- user layout save (API-style), then the layout pipeline --------------
    layout = copy.deepcopy(rev["layout"])
    layout["spawn"]["feet"] = [-3.0, 0.0, 2.0]
    rev2 = str(uuid.uuid4())
    seed.exec(
        "INSERT INTO scene_revisions (id, scene_id, project_id, run_id, base_revision_id, revision, "
        "layout_revision, layout, validation_state, approval_state, created_by) "
        "VALUES (%s, %s, %s, %s, %s, 2, 2, %s, 'pending', 'draft', 'user-1')",
        rev2, rev["scene_id"], pid, run_id, rev["id"], Jsonb(layout),
    )  # fmt: skip
    seed.exec("UPDATE scenes SET current_draft_revision_id = %s", rev2)
    seed.exec("UPDATE projects SET active_draft_revision_id = %s WHERE id = %s", rev2, pid)
    seed.job(
        pid, "geometry_proposal",
        {"sceneRevisionId": rev2, "sceneId": str(rev["scene_id"]), "baseRevisionId": str(rev["id"]),
         "layoutRevision": 2, "layoutSource": "user"},
        run_id=run_id, scene_revision_id=rev2,
    )  # fmt: skip
    done = drain(db, sup)
    assert [d[1] for d in done] == ["succeeded"] * 3
    second = seed.one("SELECT * FROM scene_revisions WHERE id = %s", rev2)
    assert second["validation_state"] == "valid"
    m2 = second["manifest"]
    assert m2["navigation"]["mode"] == "approved_collision"
    assert m2["provenance"]["geometrySource"] == "user_confirmed_trace"
    assert m2["spawn"]["feet"] == [-3, 0, 2]
    assert m2["assets"]["render"]["sha256"] == manifest["assets"]["render"]["sha256"]
    assert m2["assets"]["render"]["key"].startswith(f"tenants/{pid}/scenes/{rev2}/")
    assert seed.one("SELECT status FROM reconstruction_runs")["status"] == "succeeded"
    assert seed.one("SELECT current_draft_revision_id AS c FROM scenes")["c"] == second["id"]

    # --- an invalid layout save is rejected with findings ---------------------
    bad = copy.deepcopy(layout)
    bad["spawn"]["feet"] = [0.0, 0.0, 1.5]  # inside the divider wall
    rev3 = str(uuid.uuid4())
    seed.exec(
        "INSERT INTO scene_revisions (id, scene_id, project_id, run_id, base_revision_id, revision, "
        "layout_revision, layout, validation_state, approval_state, created_by) "
        "VALUES (%s, %s, %s, %s, %s, 3, 3, %s, 'pending', 'draft', 'user-1')",
        rev3, rev["scene_id"], pid, run_id, rev2, Jsonb(bad),
    )  # fmt: skip
    seed.job(pid, "geometry_proposal", {"sceneRevisionId": rev3, "layoutSource": "user"},
             run_id=run_id, scene_revision_id=rev3)  # fmt: skip
    done = drain(db, sup)
    assert done == [("geometry_proposal", "succeeded"), ("packaging", "succeeded"),
                    ("scene_validation", "failed")]  # fmt: skip
    third = seed.one(
        "SELECT validation_state, validation_findings FROM scene_revisions WHERE id = %s", rev3
    )
    assert third["validation_state"] == "invalid"
    assert {f["code"] for f in third["validation_findings"]} & {"SPAWN_BLOCKED"}
    assert seed.one("SELECT status FROM reconstruction_runs")["status"] == "succeeded"


def test_invalid_layout_rejected_at_geometry_stage(db, seed, sup):
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    drain(db, sup)
    rev = seed.one("SELECT * FROM scene_revisions")
    layout = copy.deepcopy(rev["layout"])
    layout["walls"][0]["height"] = 99
    rev2 = str(uuid.uuid4())
    seed.exec(
        "INSERT INTO scene_revisions (id, scene_id, project_id, run_id, revision, layout_revision, "
        "layout, validation_state, approval_state, created_by) "
        "VALUES (%s, %s, %s, %s, 2, 2, %s, 'pending', 'draft', 'user-1')",
        rev2, rev["scene_id"], pid, run_id, Jsonb(layout),
    )  # fmt: skip
    seed.job(pid, "geometry_proposal", {"sceneRevisionId": rev2, "layoutSource": "user"},
             run_id=run_id, scene_revision_id=rev2)  # fmt: skip
    assert drain(db, sup) == [("geometry_proposal", "failed")]
    row = seed.one(
        "SELECT validation_state, validation_findings FROM scene_revisions WHERE id = %s", rev2
    )
    assert row["validation_state"] == "invalid" and row["validation_findings"]
    job = seed.one("SELECT error FROM jobs WHERE scene_revision_id = %s", rev2)
    assert job["error"]["code"] == "INVALID_LAYOUT" and job["error"]["suggestedAction"]


def _sleeping_stage(ctx):
    ctx.run([sys.executable, "-c", "import time; time.sleep(60)"], timeout=60)
    raise AssertionError("should have been stopped")


def test_running_cancellation_stops_process_and_cancels_run(db, seed, store, settings):
    sup = Supervisor(db, store, settings, stages={**STAGES, "validating": _sleeping_stage})
    pid = seed.project()
    run_id, _ = seed.run(pid)
    job = queue.claim(db, "worker-test", "cpu")

    def cancel_soon():
        time.sleep(1.0)
        seed.exec(
            "UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", run_id
        )
        seed.exec("UPDATE jobs SET cancel_requested_at = now() WHERE id = %s", job.id)

    threading.Thread(target=cancel_soon, daemon=True).start()
    started = time.monotonic()
    assert sup.execute(job) == "cancelled"
    assert time.monotonic() - started < 20
    assert seed.one("SELECT state FROM jobs")["state"] == "cancelled"
    assert seed.one("SELECT status FROM reconstruction_runs")["status"] == "cancelled"
    assert seed.one("SELECT state FROM usage_ledger")["state"] == "released"
    assert seed.one("SELECT count(*) AS n FROM jobs")["n"] == 1


def test_fence_bump_stops_stale_worker_without_writes(db, seed, store, settings):
    sup = Supervisor(db, store, settings, stages={**STAGES, "validating": _sleeping_stage})
    pid = seed.project()
    seed.run(pid)
    job = queue.claim(db, "worker-test", "cpu")

    def bump():
        time.sleep(1.0)
        seed.exec("UPDATE jobs SET fencing_token = fencing_token + 1, state = 'queued', "
                  "lease_owner = NULL, lease_expires_at = NULL WHERE id = %s", job.id)  # fmt: skip

    threading.Thread(target=bump, daemon=True).start()
    assert sup.execute(job) == "lost"
    row = seed.one("SELECT state, fencing_token FROM jobs")
    assert row["state"] == "queued" and row["fencing_token"] == 2


def _transient_stage(ctx):
    from andarilho_worker.errors import TransientError

    raise TransientError()


def test_transient_failures_exhaust_retries(db, seed, store, settings):
    sup = Supervisor(db, store, settings, stages={**STAGES, "validating": _transient_stage},
                     delay_fn=lambda attempt: 0.0)  # fmt: skip
    pid = seed.project()
    run_id, _ = seed.run(pid, max_attempts=3)
    done = drain(db, sup)
    assert [d[1] for d in done] == ["retry_wait", "retry_wait", "failed"]
    run = seed.one("SELECT status, error FROM reconstruction_runs")
    assert run["status"] == "failed" and run["error"]["code"] == "TRANSIENT_FAILURE"
    assert seed.one("SELECT state FROM usage_ledger")["state"] == "released"


# ---------------------------------------------------------------------------
# Upload inspection
# ---------------------------------------------------------------------------

JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xc0\x00\x11\x08\x00\x10\x00\x20\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00" + b"\x00" * 64 + b"\xff\xd9"
)
PNG = (
    b"\x89PNG\r\n\x1a\n"
    + b"\x00\x00\x00\rIHDR"
    + (64).to_bytes(4, "big")
    + (48).to_bytes(4, "big")
    + b"\x08\x02\x00\x00\x00"
    + b"\x00" * 32
)
PDF = b"%PDF-1.7\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n3 0 obj << /Type /Page >> endobj\n%%EOF\n"
ENCRYPTED_PDF = PDF.replace(b"%%EOF", b"trailer << /Encrypt 9 0 R >>\n%%EOF")
SVG = (
    b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
)
MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 100


def make_upload(seed, store, pid: str, data: bytes, kind: str, status="uploaded_unverified") -> str:
    uid = str(uuid.uuid4())
    key = f"tenants/{pid}/originals/{uid}"
    store.put_bytes(key, data)
    seed.exec(
        "INSERT INTO uploads (id, project_id, created_by, media_kind, display_filename, "
        "declared_content_type, expected_bytes, staging_key, immutable_key, part_size, part_count, "
        "status, expires_at) VALUES (%s, %s, 'user-1', %s, 'file', 'application/octet-stream', %s, "
        "%s, %s, 16777216, 1, %s, now() + interval '1 day')",
        uid, pid, kind, len(data), f"tenants/{pid}/staging/{uid}", key, status,
    )  # fmt: skip
    seed.job(pid, "upload_inspection", {"uploadId": uid, "projectId": pid, "key": key,
             "mediaKind": kind, "expectedBytes": len(data)}, upload_id=uid)  # fmt: skip
    return uid


@pytest.mark.parametrize(
    "data,kind,status,code",
    [
        (JPEG, "photo", "verified", None),
        (PNG, "photo", "verified", None),
        (PDF, "floor_plan", "verified", None),
        (MP4, "video", "verified", None),
        (PNG, "video", "rejected", "UNSUPPORTED_MEDIA"),
        (SVG, "floor_plan", "rejected", "UNSUPPORTED_MEDIA"),
        (ENCRYPTED_PDF, "floor_plan", "rejected", "UNSUPPORTED_MEDIA"),
        (b"PK\x03\x04" + b"\x00" * 50, "photo", "rejected", "UNSUPPORTED_MEDIA"),
        (b"<html><body>hi</body></html>", "photo", "rejected", "UNSUPPORTED_MEDIA"),
    ],
)
def test_upload_inspection(db, seed, store, sup, monkeypatch, data, kind, status, code):
    monkeypatch.setattr(inspect_mod, "tool_available", lambda name: False)
    pid = seed.project()
    uid = make_upload(seed, store, pid, data, kind)
    assert drain(db, sup) == [("upload_inspection", "succeeded")]
    row = seed.one(
        "SELECT status, sha256, rejection_code, media_metadata, actual_bytes FROM uploads WHERE id = %s",
        uid,
    )
    assert row["status"] == status
    assert row["rejection_code"] == code
    if status == "verified":
        assert row["sha256"] == hashlib.sha256(data).hexdigest()
        assert row["actual_bytes"] == len(data)
        assert row["media_metadata"]["format"] in ("jpeg", "png", "pdf", "mp4")
    if kind == "photo" and status == "verified":
        assert row["media_metadata"]["width"] in (32, 64)


def test_inspection_rejects_pdf_with_too_many_pages(db, seed, store, sup):
    pid = seed.project()
    many = b"%PDF-1.7\n" + b"<< /Type /Page >>\n" * 50 + b"%%EOF"
    uid = make_upload(seed, store, pid, many, "floor_plan")
    drain(db, sup)
    assert seed.one("SELECT status FROM uploads WHERE id = %s", uid)["status"] == "rejected"


def test_inspection_verify_is_conditional(db, seed, store, settings):
    def racing(ctx):
        result = inspect_mod.run(ctx)
        seed.exec("UPDATE uploads SET status = 'aborted' WHERE id = %s", ctx.job.upload_id)
        return result

    sup = Supervisor(db, store, settings, stages={**STAGES, "upload_inspection": racing})
    pid = seed.project()
    uid = make_upload(seed, store, pid, JPEG, "photo")
    assert drain(db, sup) == [("upload_inspection", "failed")]
    row = seed.one("SELECT status, sha256 FROM uploads WHERE id = %s", uid)
    assert row["status"] == "aborted" and row["sha256"] is None


def test_inspection_skips_already_decided_upload(db, seed, store, sup):
    pid = seed.project()
    uid = make_upload(seed, store, pid, JPEG, "photo", status="rejected")
    assert drain(db, sup) == [("upload_inspection", "succeeded")]
    assert seed.one("SELECT status FROM uploads WHERE id = %s", uid)["status"] == "rejected"


# ---------------------------------------------------------------------------
# Project cleanup
# ---------------------------------------------------------------------------


def test_project_cleanup_deletes_objects_and_tombstones(db, seed, store, sup):
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    drain(db, sup)
    other = seed.project()
    store.put_bytes(f"tenants/{other}/keep.bin", b"x")
    store.multipart_uploads["mp-1"] = f"tenants/{pid}/staging/abc"
    assert store.list_keys(f"tenants/{pid}/")
    seed.exec("UPDATE projects SET lifecycle = 'deleting' WHERE id = %s", pid)
    seed.job(pid, "project_cleanup", {"projectId": pid, "prefix": f"tenants/{pid}/"}, priority=10)
    done = drain(db, sup)
    assert done == [("project_cleanup", "deferred")]  # first pass, waiting for the horizon
    assert store.list_keys(f"tenants/{pid}/") == []
    assert store.multipart_uploads == {}
    job = seed.one(
        "SELECT state, attempts, input_manifest FROM jobs WHERE stage = 'project_cleanup'"
    )
    assert job["state"] == "retry_wait" and job["attempts"] == 0
    assert job["input_manifest"]["passes"] == 1
    seed.exec("UPDATE jobs SET available_at = now() WHERE stage = 'project_cleanup'")
    assert drain(db, sup) == [("project_cleanup", "succeeded")]
    project = seed.one("SELECT * FROM projects WHERE id = %s", pid)
    assert project["lifecycle"] == "deleted" and project["cleanup_completed_at"] is not None
    assert project["title"] == "Deleted project" and project["active_draft_revision_id"] is None
    assert seed.one("SELECT count(*) AS n FROM jobs WHERE project_id = %s", pid)["n"] == 0
    assert (
        seed.one("SELECT count(*) AS n FROM scene_revisions WHERE project_id = %s", pid)["n"] == 0
    )
    assert seed.one("SELECT count(*) AS n FROM artifacts WHERE project_id = %s", pid)["n"] == 0
    audit = seed.one("SELECT action, data FROM audit_events WHERE project_id = %s", pid)
    assert audit["action"] == "project.cleanup_completed" and audit["data"]["passes"] == 2
    assert store.list_keys(f"tenants/{other}/") == [f"tenants/{other}/keep.bin"]
    assert (
        seed.one("SELECT created_by FROM reconstruction_runs WHERE id = %s", run_id)["created_by"]
        == "deleted"
    )


def test_project_cleanup_waits_for_running_jobs(db, seed, store, sup):
    pid = seed.project()
    seed.run(pid)
    running = queue.claim(db, "other-worker", "cpu")
    seed.exec("UPDATE projects SET lifecycle = 'deleting' WHERE id = %s", pid)
    seed.job(pid, "project_cleanup", {"projectId": pid}, priority=10)
    assert drain(db, sup) == [("project_cleanup", "deferred")]
    assert seed.one("SELECT lifecycle FROM projects")["lifecycle"] == "deleting"
    assert seed.one("SELECT state FROM jobs WHERE id = %s", running.id)["state"] == "running"


def test_worker_loop_processes_synthetic_run(db, seed, store, settings):
    from andarilho_worker.main import Worker

    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    worker = Worker(settings, db, store)
    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60
    status = None
    while time.monotonic() < deadline:
        status = seed.one("SELECT status FROM reconstruction_runs WHERE id = %s", run_id)["status"]
        if status in ("succeeded", "failed"):
            break
        time.sleep(0.2)
    worker.stop()
    thread.join(timeout=30)
    assert status == "succeeded"
    assert not thread.is_alive()
