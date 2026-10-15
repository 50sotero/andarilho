"""Run submission (idempotency, active-run guard), cancellation, retry, jobs."""

from __future__ import annotations

import uuid

import pytest
from factories import active_run, capture_set, create_project, layout
from sqlalchemy import select

from andarilho_api.domain.recipes import RECIPES
from andarilho_api.models import (
    IdempotencyRecord,
    Job,
    JobEvent,
    ReconstructionRun,
    Scene,
    SceneRevision,
    UsageLedger,
)


def _submit(client, auth, pid, body, key="key-00000001", sub="user-a"):
    return client.post(
        f"/v1/projects/{pid}/runs",
        json=body,
        headers=auth(sub, **({"Idempotency-Key": key} if key else {})),
    )


@pytest.fixture
def project(client, auth, db):
    pid = create_project(client, auth)["id"]
    return {"id": pid, "cs": capture_set(db, pid)}


def test_submit_run_creates_run_job_ledger_atomically(client, auth, db, project):
    body = {"captureSetId": project["cs"], "quality": "preview", "includeSupplementaryPhotos": True}
    response = _submit(client, auth, project["id"], body)
    assert response.status_code == 202, response.text
    result = response.json()
    assert result["state"] == "queued" and result["stage"] == "validating"
    run = db.get(ReconstructionRun, uuid.UUID(result["runId"]))
    assert run.recipe_name == RECIPES["preview"].name == "preview-v1"
    assert run.recipe_digest == RECIPES["preview"].digest
    assert run.resource_budget["maxSeconds"] == 3600
    job = db.get(Job, uuid.UUID(result["jobId"]))
    assert (job.stage, job.capability, job.state) == ("validating", "cpu", "queued")
    manifest = job.input_manifest
    assert manifest["captureSetId"] == project["cs"]
    assert manifest["quality"] == "preview"
    assert manifest["recipe"]["digest"] == run.recipe_digest
    stages = {s["stage"]: s["capability"] for s in manifest["recipe"]["stages"]}
    assert stages["camera_reconstruction"] == "gpu" and stages["training"] == "gpu"
    ledger = db.execute(select(UsageLedger)).scalar_one()
    assert ledger.state == "reserved" and ledger.reserved_gpu_seconds > 0

    detail = client.get(f"/v1/projects/{project['id']}", headers=auth()).json()
    assert detail["runs"][0]["currentStage"] == "validating"
    assert detail["runs"][0]["jobs"] == [
        {"id": result["jobId"], "stage": "validating", "state": "queued"}
    ]


def test_idempotent_replay_and_key_reuse(client, auth, db, project):
    body = {"captureSetId": project["cs"], "quality": "preview"}
    first = _submit(client, auth, project["id"], body)
    replay = _submit(client, auth, project["id"], body)
    assert replay.status_code == 202
    assert replay.json() == first.json()
    assert len(db.execute(select(ReconstructionRun)).all()) == 1

    reused = _submit(client, auth, project["id"], {**body, "quality": "standard"})
    assert reused.status_code == 409
    assert reused.json()["error"]["code"] == "IDEMPOTENCY_KEY_REUSED"

    second = _submit(client, auth, project["id"], body, key="key-00000002")
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "ACTIVE_RUN_EXISTS"
    assert db.execute(select(IdempotencyRecord)).scalar_one().response_status == 202


def test_partial_unique_index_catches_races(client, auth, db, project, monkeypatch):
    """Even if the pre-check is bypassed, the database index produces ACTIVE_RUN_EXISTS."""
    body = {"captureSetId": project["cs"], "quality": "preview"}
    assert _submit(client, auth, project["id"], body).status_code == 202

    import andarilho_api.domain.runs as runs_module
    import andarilho_api.routes.runs as routes_runs

    class _NoActive:
        def first(self):
            return None

    original = runs_module.submit_run

    def submit_without_precheck(db_, *args, **kwargs):
        orig_execute = db_.execute
        calls = {"n": 0}

        def execute(stmt, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return _NoActive()
            return orig_execute(stmt, *a, **k)

        db_.execute = execute
        try:
            return original(db_, *args, **kwargs)
        finally:
            db_.execute = orig_execute

    monkeypatch.setattr(routes_runs, "submit_run", submit_without_precheck)
    response = _submit(client, auth, project["id"], body, key="key-race-0001")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "ACTIVE_RUN_EXISTS"


def test_submission_validation(client, auth, project, app):
    pid = project["id"]
    body = {"captureSetId": project["cs"], "quality": "preview"}
    missing = _submit(client, auth, pid, body, key=None)
    assert missing.status_code == 428
    assert missing.json()["error"]["code"] == "PRECONDITION_REQUIRED"
    assert _submit(client, auth, pid, body, key="short").status_code == 422
    for extra in ({"recipe": "x"}, {"flags": "--shell"}, {"inputPath": "/etc/passwd"}):
        response = _submit(client, auth, pid, {**body, **extra})
        assert response.status_code == 422
        assert "passwd" not in response.text
    assert _submit(client, auth, pid, {**body, "quality": "ultra"}).status_code == 422
    unknown_cs = _submit(client, auth, pid, {**body, "captureSetId": str(uuid.uuid4())})
    assert unknown_cs.status_code == 404
    app.state.settings.allow_synthetic_recipe = False
    synthetic = _submit(client, auth, pid, {**body, "quality": "synthetic"}, key="key-synth-01")
    assert synthetic.status_code == 422


def test_synthetic_recipe_skips_gpu_stages(client, auth, db, project):
    body = {"captureSetId": project["cs"], "quality": "synthetic"}
    response = _submit(client, auth, project["id"], body)
    assert response.status_code == 202
    job = db.get(Job, uuid.UUID(response.json()["jobId"]))
    stages = [s["stage"] for s in job.input_manifest["recipe"]["stages"]]
    assert "training" not in stages and "camera_reconstruction" not in stages
    assert all(s["capability"] == "cpu" for s in job.input_manifest["recipe"]["stages"])


def _run_state(db, run_id):
    db.expire_all()
    run = db.get(ReconstructionRun, uuid.UUID(run_id))
    jobs = db.execute(select(Job).where(Job.run_id == run.id).order_by(Job.stage)).scalars().all()
    ledger = db.execute(select(UsageLedger).where(UsageLedger.run_id == run.id)).scalar_one()
    return run, {j.stage: j for j in jobs}, ledger


@pytest.mark.parametrize("state", ["queued", "retry_wait"])
def test_cancel_pending_jobs_finishes_run(client, auth, db, project, state):
    info = active_run(db, project["id"], job_states=("succeeded", state))
    response = client.post(f"/v1/runs/{info['runId']}/cancel", headers=auth())
    assert response.status_code == 202
    assert response.json() == {
        "runId": info["runId"],
        "status": "cancelled",
        "cancelRequested": True,
    }
    run, jobs, ledger = _run_state(db, info["runId"])
    assert run.status == "cancelled" and run.cancel_requested_at and run.finished_at
    assert jobs["validating"].state == "succeeded"
    assert jobs["preparing"].state == "cancelled"
    assert ledger.state == "released"
    # Cancel is idempotent; the ledger is released once.
    again = client.post(f"/v1/runs/{info['runId']}/cancel", headers=auth())
    assert again.status_code == 202 and again.json()["status"] == "cancelled"


def _first_revision(db, project_id: str, run_id: str) -> uuid.UUID:
    """The scene and pending revision 1 a run's geometry_proposal stage creates."""
    pid, rid = uuid.UUID(project_id), uuid.UUID(run_id)
    scene_id, rev_id = uuid.uuid4(), uuid.uuid4()
    db.add(Scene(id=scene_id, project_id=pid, run_id=rid))
    db.flush()
    db.add(
        SceneRevision(
            id=rev_id,
            scene_id=scene_id,
            project_id=pid,
            run_id=rid,
            revision=1,
            layout_revision=1,
            layout=layout(),
            validation_state="pending",
            created_by="user-a",
        )
    )
    db.commit()
    return rev_id


def test_cancelling_a_run_ends_its_unfinished_scene(client, auth, db, project):
    info = active_run(db, project["id"], job_states=("succeeded", "queued"))
    rev_id = _first_revision(db, project["id"], info["runId"])
    # While the run is processing, the project offers its scene (it validates soon).
    detail = client.get(f"/v1/projects/{project['id']}", headers=auth()).json()
    assert detail["scene"] is not None

    response = client.post(f"/v1/runs/{info['runId']}/cancel", headers=auth())
    assert response.status_code == 202 and response.json()["status"] == "cancelled"
    db.expire_all()
    revision = db.get(SceneRevision, rev_id)
    assert revision.validation_state == "invalid"
    assert [f["code"] for f in revision.validation_findings] == ["RUN_CANCELLED"]
    # A scene that will never get a revision is not offered for review or walking.
    detail = client.get(f"/v1/projects/{project['id']}", headers=auth()).json()
    assert detail["scene"] is None


def test_cancel_with_running_job_marks_and_waits(client, auth, db, project):
    info = active_run(db, project["id"], job_states=("succeeded", "running", "queued"))
    response = client.post(f"/v1/runs/{info['runId']}/cancel", headers=auth())
    assert response.status_code == 202
    assert response.json()["status"] == "running"
    assert response.json()["cancelRequested"] is True
    run, jobs, ledger = _run_state(db, info["runId"])
    assert run.status == "running" and run.cancel_requested_at is not None
    assert jobs["preparing"].state == "running"
    assert jobs["preparing"].cancel_requested_at is not None
    assert jobs["preparing"].fencing_token == 1  # the worker acknowledges with its token
    assert jobs["camera_reconstruction"].state == "cancelled"
    assert ledger.state == "reserved"  # released once the worker confirms termination


def test_cancel_finished_run_is_conflict(client, auth, db, project):
    info = active_run(db, project["id"], job_states=("succeeded",))
    run = db.get(ReconstructionRun, uuid.UUID(info["runId"]))
    run.status = "succeeded"
    db.commit()
    response = client.post(f"/v1/runs/{info['runId']}/cancel", headers=auth())
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INVALID_STATE"


def test_retry(client, auth, db, project):
    info = active_run(db, project["id"], job_states=("failed",))
    url = f"/v1/runs/{info['runId']}/retry"
    assert client.post(url, json={}, headers=auth()).status_code == 428
    early = client.post(url, json={}, headers=auth(**{"Idempotency-Key": "retry-key-1"}))
    assert early.status_code == 409  # still running
    run = db.get(ReconstructionRun, uuid.UUID(info["runId"]))
    run.status = "failed"
    db.commit()
    response = client.post(url, json={}, headers=auth(**{"Idempotency-Key": "retry-key-2"}))
    assert response.status_code == 202, response.text
    replay = client.post(url, json={}, headers=auth(**{"Idempotency-Key": "retry-key-2"}))
    assert replay.json() == response.json()
    new_run = db.get(ReconstructionRun, uuid.UUID(response.json()["runId"]))
    assert new_run.retry_of_run_id == run.id
    assert new_run.capture_set_id == run.capture_set_id and new_run.quality == run.quality
    assert (
        client.post(
            url, json={"quality": "standard"}, headers=auth(**{"Idempotency-Key": "retry-key-3"})
        ).status_code
        == 422
    )


def test_job_events_pagination(client, auth, db, project):
    info = active_run(db, project["id"], job_states=("running",))
    job_id = uuid.UUID(info["jobIds"][0])
    for i in range(5):
        db.add(
            JobEvent(
                job_id=job_id,
                stage="validating",
                state="running",
                message=f"step {i}",
                progress={"done": i, "total": 5},
            )
        )
    db.commit()
    page = client.get(f"/v1/jobs/{job_id}", params={"limit": 2}, headers=auth()).json()
    assert page["job"]["stage"] == "validating" and page["job"]["runId"] == info["runId"]
    assert [e["message"] for e in page["events"]] == ["step 0", "step 1"]
    cursor = page["nextCursor"]
    rest = client.get(
        f"/v1/jobs/{job_id}", params={"after": cursor, "limit": 10}, headers=auth()
    ).json()
    assert [e["message"] for e in rest["events"]] == ["step 2", "step 3", "step 4"]
    assert rest["nextCursor"] is None
    assert (
        client.get(f"/v1/jobs/{job_id}", params={"limit": 101}, headers=auth()).status_code == 422
    )


def test_run_and_job_isolation(client, auth, db, project):
    info = active_run(db, project["id"], job_states=("running",))
    other = auth("user-b")
    assert client.get(f"/v1/jobs/{info['jobIds'][0]}", headers=other).status_code == 404
    assert client.post(f"/v1/runs/{info['runId']}/cancel", headers=other).status_code == 404
    assert (
        client.post(
            f"/v1/runs/{info['runId']}/retry",
            json={},
            headers=auth("user-b", **{"Idempotency-Key": "k-000000001"}),
        ).status_code
        == 404
    )
    body = {"captureSetId": project["cs"], "quality": "preview"}
    assert _submit(client, auth, project["id"], body, sub="user-b").status_code == 404
