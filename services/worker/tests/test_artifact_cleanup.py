"""``artifact_cleanup``: the queued deletion of lost attempts and invalid scene revisions."""

from __future__ import annotations

import copy
import uuid

import pytest
from helpers import drain
from psycopg.types.json import Jsonb

from andarilho_worker import queue
from andarilho_worker.artifacts import FRAMES_DIRECTORY, LocalObjectStore, attempt_prefix
from andarilho_worker.errors import StageError, TransientError
from andarilho_worker.recipes import RETEXTURE_STAGES, RUN_STAGES
from andarilho_worker.stages import STAGES
from andarilho_worker.stages.base import finish
from andarilho_worker.supervisor import Supervisor


@pytest.fixture
def sup(db, store, settings):
    return Supervisor(db, store, settings, delay_fn=lambda attempt: 0.0)


def make_revision(
    seed, project_id: str, *, state: str = "invalid", run_status: str = "failed"
) -> dict[str, str]:
    """A run (failed unless ``run_status`` says otherwise) and the scene and revision 1 it
    created, with the given validation state."""
    run_id, scene_id, revision_id = (str(uuid.uuid4()) for _ in range(3))
    seed.exec(
        "INSERT INTO reconstruction_runs (id, project_id, capture_set_id, created_by, "
        "recipe_name, recipe_digest, quality, include_supplementary_photos, resource_budget, "
        "status) VALUES (%s, %s, %s, 'user-1', 'synthetic', 'sha256:test', 'synthetic', true, "
        "'{}'::jsonb, %s)",
        run_id, project_id, seed.capture_set(project_id), run_status,
    )  # fmt: skip
    seed.exec(
        "INSERT INTO scenes (id, project_id, run_id, version) VALUES (%s, %s, %s, 1)",
        scene_id, project_id, run_id,
    )  # fmt: skip
    seed.exec(
        "INSERT INTO scene_revisions (id, scene_id, project_id, run_id, revision, layout_revision, "
        "layout, validation_state, approval_state, created_by) "
        "VALUES (%s, %s, %s, %s, 1, 1, %s, %s, 'draft', 'user-1')",
        revision_id, scene_id, project_id, run_id, Jsonb({}), state,
    )  # fmt: skip
    return {"run": run_id, "scene": scene_id, "revision": revision_id}


def scene_of(project_id: str, rev: dict[str, str]) -> str:
    return f"tenants/{project_id}/scenes/{rev['revision']}/"


def attempt_of(
    project_id: str, run_id: str, job_id: str, fence: int = 1, stage: str = "packaging"
) -> str:
    return f"tenants/{project_id}/runs/{run_id}/attempts/{stage}/{job_id}/{fence}/"


def make_run_job(seed, project_id: str, run_id: str, state: str, stage: str = "packaging") -> str:
    """A ``stage`` job of ``run_id`` that is in ``state`` (a lease is held while running),
    claimed once, so its only attempt ran at fence 1."""
    job_id = seed.job(project_id, stage, {}, run_id=run_id)
    if state == "running":
        seed.exec(
            "UPDATE jobs SET state = 'running', lease_owner = 'worker-1', fencing_token = 1, "
            "lease_expires_at = now() + interval '90 seconds' WHERE id = %s",
            job_id,
        )
    else:
        seed.exec("UPDATE jobs SET state = %s, fencing_token = 1 WHERE id = %s", state, job_id)
    return job_id


def point_at(seed, project_id: str, rev: dict[str, str], how: str) -> None:
    """Make a project or scene pointer, or a share link, name the revision."""
    if how == "active_draft":
        seed.exec(
            "UPDATE projects SET active_draft_revision_id = %s WHERE id = %s",
            rev["revision"], project_id,
        )  # fmt: skip
    elif how == "published":
        seed.exec(
            "UPDATE projects SET published_revision_id = %s WHERE id = %s",
            rev["revision"], project_id,
        )  # fmt: skip
    elif how == "scene_draft":
        seed.exec(
            "UPDATE scenes SET current_draft_revision_id = %s WHERE id = %s",
            rev["revision"], rev["scene"],
        )  # fmt: skip
    elif how == "shared":
        seed.exec(
            "INSERT INTO share_links (id, project_id, scene_revision_id, token_hash, created_by) "
            "VALUES (%s, %s, %s, %s, 'user-1')",
            str(uuid.uuid4()), project_id, rev["revision"], b"\x01" * 32,
        )  # fmt: skip
    else:
        assert how is None, how


def enqueue(db, project_id: str, prefixes: list[str], **kw) -> None:
    with db.transaction() as conn:
        queue.queue_artifact_cleanup(conn, project_id, prefixes, **kw)


def run_due_cleanup(db, sup) -> str:
    """Make the queued cleanup claimable now and run it through the supervisor."""
    with db.transaction() as conn:
        conn.execute("UPDATE jobs SET available_at = now() WHERE stage = 'artifact_cleanup'")
    job = queue.claim(db, "worker-test", "cpu")
    assert job is not None and job.stage == "artifact_cleanup"
    assert job.run_id is None and job.scene_revision_id is None
    return sup.execute(job)


def run_all_due_cleanups(db, sup) -> list[str]:
    """Run every queued cleanup now; return their outcomes."""
    with db.transaction() as conn:
        conn.execute("UPDATE jobs SET available_at = now() WHERE stage = 'artifact_cleanup'")
    outcomes = []
    while (job := queue.claim(db, "worker-test", "cpu")) is not None:
        assert job.stage == "artifact_cleanup"
        outcomes.append(sup.execute(job))
    return outcomes


def cleanup_row(seed, project_id: str) -> dict:
    row = seed.one(
        "SELECT state, result_manifest, error FROM jobs "
        "WHERE project_id = %s AND stage = 'artifact_cleanup'",
        project_id,
    )
    assert row is not None
    return row


def test_the_job_waits_out_its_delay_before_it_can_run(db, seed, store, sup):
    pid = seed.project()
    prefix = f"tenants/{pid}/scenes/{uuid.uuid4()}/"
    store.put_bytes(prefix + "scene.ply", b"x")
    enqueue(db, pid, [prefix], reason="revision_invalid")
    assert queue.claim(db, "worker-test", "cpu") is None
    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(prefix) == []


def test_the_stage_deletes_every_object_under_its_prefixes_and_aborts_their_uploads(
    db, seed, store, sup
):
    pid = seed.project()
    rev = make_revision(seed, pid)
    scene = scene_of(pid, rev)
    job_id = make_run_job(seed, pid, rev["run"], "failed")
    attempt = attempt_of(pid, rev["run"], job_id)
    for key in (scene + "scene.ply", scene + "manifest.json", attempt + "partial.bin"):
        store.put_bytes(key, b"x")
    store.multipart_uploads["mp-scene"] = scene + "scene.ply"
    store.multipart_uploads["mp-attempt"] = attempt + "partial.bin"
    store.multipart_uploads["mp-other"] = f"tenants/{pid}/originals/keep"
    enqueue(db, pid, [attempt, scene], reason="revision_invalid")

    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(f"tenants/{pid}/") == []
    assert store.multipart_uploads == {"mp-other": f"tenants/{pid}/originals/keep"}
    row = cleanup_row(seed, pid)
    assert row["state"] == "succeeded" and row["error"] is None
    assert row["result_manifest"]["deleted"] == [attempt, scene]


@pytest.mark.parametrize(
    "how",
    [
        pytest.param(None, id="valid"),
        pytest.param("active_draft", id="valid-in-project-draft"),
        pytest.param("published", id="valid-published"),
        pytest.param("scene_draft", id="valid-in-scene-draft"),
        pytest.param("shared", id="valid-shared"),
    ],
)
def test_the_stage_keeps_the_scene_of_a_valid_revision(db, seed, store, sup, how):
    pid = seed.project()
    rev = make_revision(seed, pid, state="valid")
    point_at(seed, pid, rev, how)
    scene = scene_of(pid, rev)
    store.put_bytes(scene + "scene.ply", b"x")
    enqueue(db, pid, [scene], reason="revision_invalid")

    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(scene) == [scene + "scene.ply"]
    assert cleanup_row(seed, pid)["result_manifest"]["kept"] == [scene]


REFERENCES = [
    pytest.param("active_draft", id="project-draft"),
    pytest.param("published", id="published"),
    pytest.param("scene_draft", id="scene-draft"),
    pytest.param("shared", id="shared"),
]


@pytest.mark.parametrize("how", REFERENCES)
def test_the_stage_deletes_the_scene_of_an_invalid_revision_whatever_names_it(
    db, seed, store, sup, how
):
    # A failed layout save usually stays the draft. Nothing serves an invalid revision,
    # and moving the pointer on later queues no cleanup, so a reference keeps nothing.
    pid = seed.project()
    rev = make_revision(seed, pid, state="invalid")
    point_at(seed, pid, rev, how)
    scene = scene_of(pid, rev)
    store.put_bytes(scene + "model.glb", b"x")
    enqueue(db, pid, [scene], reason="revision_invalid")

    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(scene) == []
    assert cleanup_row(seed, pid)["result_manifest"]["deleted"] == [scene]


@pytest.mark.parametrize("how", REFERENCES)
def test_the_stage_keeps_the_scene_of_a_pending_revision_that_a_reference_names(
    db, seed, store, sup, how
):
    # A pending revision may still be processing, so a reference keeps its files.
    pid = seed.project()
    rev = make_revision(seed, pid, state="pending")
    point_at(seed, pid, rev, how)
    scene = scene_of(pid, rev)
    store.put_bytes(scene + "model.glb", b"x")
    enqueue(db, pid, [scene], reason="revision_invalid")

    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(scene) == [scene + "model.glb"]
    assert cleanup_row(seed, pid)["result_manifest"]["kept"] == [scene]


@pytest.mark.parametrize(
    "state",
    [pytest.param("invalid", id="invalid"), pytest.param("pending", id="pending")],
)
def test_the_stage_deletes_the_scene_of_an_unreferenced_revision_that_is_not_valid(
    db, seed, store, sup, state
):
    pid = seed.project()
    rev = make_revision(seed, pid, state=state)
    scene = scene_of(pid, rev)
    store.put_bytes(scene + "scene.ply", b"x")
    enqueue(db, pid, [scene], reason="revision_invalid")

    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(scene) == []
    assert cleanup_row(seed, pid)["result_manifest"]["deleted"] == [scene]


def test_a_reference_to_another_revision_does_not_keep_this_one(db, seed, store, sup):
    pid = seed.project()
    doomed = make_revision(seed, pid, state="pending")
    kept = make_revision(seed, pid, state="pending")
    point_at(seed, pid, kept, "active_draft")
    point_at(seed, pid, kept, "shared")
    for rev in (doomed, kept):
        store.put_bytes(scene_of(pid, rev) + "scene.ply", b"x")
    enqueue(db, pid, [scene_of(pid, doomed), scene_of(pid, kept)], reason="revision_invalid")

    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(scene_of(pid, doomed)) == []
    assert store.list_keys(scene_of(pid, kept)) == [scene_of(pid, kept) + "scene.ply"]
    row = cleanup_row(seed, pid)
    assert row["result_manifest"]["deleted"] == [scene_of(pid, doomed)]
    assert row["result_manifest"]["kept"] == [scene_of(pid, kept)]


def test_an_attempt_of_a_succeeded_job_is_kept_while_its_run_is_processing(db, seed, store, sup):
    # A later stage of the run may still read it.
    pid = seed.project()
    rev = make_revision(seed, pid, state="pending", run_status="running")
    job_id = make_run_job(seed, pid, rev["run"], "succeeded")
    attempt = attempt_of(pid, rev["run"], job_id)
    store.put_bytes(attempt + "out.bin", b"x")
    enqueue(db, pid, [attempt], reason="attempt_lost")

    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(attempt) == [attempt + "out.bin"]
    assert cleanup_row(seed, pid)["result_manifest"]["kept"] == [attempt]


@pytest.mark.parametrize("stage", RUN_STAGES)
def test_a_succeeded_run_keeps_only_the_attempts_layout_saves_retexture_from(
    db, seed, store, sup, stage
):
    # Layout saves read the poster, the dataset and the frame transform. Packaging copied
    # what the scene needs into its own prefix, and the rest only fed the next stage.
    pid = seed.project()
    rev = make_revision(seed, pid, state="valid", run_status="succeeded")
    job_id = make_run_job(seed, pid, rev["run"], "succeeded", stage=stage)
    attempt = attempt_of(pid, rev["run"], job_id, stage=stage)
    store.put_bytes(attempt + "out.bin", b"x")
    enqueue(db, pid, [attempt], reason="run_succeeded")

    assert run_due_cleanup(db, sup) == "succeeded"
    kept = stage in RETEXTURE_STAGES
    assert store.list_keys(attempt) == ([attempt + "out.bin"] if kept else [])
    assert cleanup_row(seed, pid)["result_manifest"]["kept" if kept else "deleted"] == [attempt]


@pytest.mark.parametrize(("run_status", "kept"), [("running", True), ("succeeded", False)])
def test_the_preparing_frames_are_kept_only_while_the_run_is_processing(
    db, seed, store, sup, run_status, kept
):
    # camera_reconstruction reads them, then packs undistorted copies into its dataset;
    # after the run, layout saves read only the poster beside them.
    pid = seed.project()
    state = "valid" if run_status == "succeeded" else "pending"
    rev = make_revision(seed, pid, state=state, run_status=run_status)
    job_id = make_run_job(seed, pid, rev["run"], "succeeded", stage="preparing")
    attempt = attempt_of(pid, rev["run"], job_id, stage="preparing")
    frames = attempt + FRAMES_DIRECTORY
    store.put_bytes(frames + "frame_0001.jpg", b"x")
    store.put_bytes(attempt + "poster.webp", b"x")
    enqueue(db, pid, [frames], reason="run_succeeded")

    assert run_due_cleanup(db, sup) == "succeeded"
    left = [frames + "frame_0001.jpg"] if kept else []
    assert sorted(store.list_keys(attempt)) == sorted([attempt + "poster.webp", *left])
    assert cleanup_row(seed, pid)["result_manifest"]["kept" if kept else "deleted"] == [frames]


@pytest.mark.parametrize("run_status", ["failed", "cancelled"])
def test_a_succeeded_attempt_of_a_run_that_ended_unfinished_is_deleted(
    db, seed, store, sup, run_status
):
    # Such a run never had a valid scene to re-texture, and a retry is a new run.
    pid = seed.project()
    rev = make_revision(seed, pid, run_status=run_status)
    job_id = make_run_job(seed, pid, rev["run"], "succeeded")
    attempt = attempt_of(pid, rev["run"], job_id)
    store.put_bytes(attempt + "frames.tar", b"x")
    enqueue(db, pid, [attempt], reason="run_ended")

    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(attempt) == []
    assert cleanup_row(seed, pid)["result_manifest"]["deleted"] == [attempt]


def test_a_run_ending_after_an_upgrade_cleans_up_its_legacy_training_outputs(db, seed, store, sup):
    # Migration 0006 keeps historical training rows. When such a run fails, its training
    # attempt is queued with the other outputs and must not make the whole cleanup refuse.
    pid = seed.project()
    rev = make_revision(seed, pid, run_status="failed")
    packaging = make_run_job(seed, pid, rev["run"], "succeeded")
    training = seed.job(pid, "training", {}, run_id=rev["run"])
    seed.exec("UPDATE jobs SET state = 'succeeded', fencing_token = 1 WHERE id = %s", training)
    legacy = f"tenants/{pid}/runs/{rev['run']}/attempts/training/{training}/1/"
    current = attempt_of(pid, rev["run"], packaging)
    store.put_bytes(legacy + "splat.ply", b"x")
    store.put_bytes(current + "out.bin", b"x")
    with db.transaction() as conn:
        queue.queue_run_outputs(conn, rev["run"])

    assert run_all_due_cleanups(db, sup) == ["succeeded"]
    assert store.list_keys(legacy) == [] and store.list_keys(current) == []
    assert sorted(cleanup_row(seed, pid)["result_manifest"]["deleted"]) == sorted([legacy, current])


def test_only_the_fence_a_succeeded_job_committed_under_is_kept(db, seed, store, sup):
    # A worker that could not sweep fence 1 before succeeding at fence 2 queues fence 1.
    pid = seed.project()
    rev = make_revision(seed, pid, run_status="running")
    job_id = make_run_job(seed, pid, rev["run"], "succeeded")
    seed.exec("UPDATE jobs SET fencing_token = 2 WHERE id = %s", job_id)
    stale, committed = (attempt_of(pid, rev["run"], job_id, fence) for fence in (1, 2))
    for prefix in (stale, committed):
        store.put_bytes(prefix + "out.bin", b"x")
    enqueue(db, pid, [stale, committed], reason="delete_failed")

    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(stale) == []
    assert store.list_keys(committed) == [committed + "out.bin"]
    manifest = cleanup_row(seed, pid)["result_manifest"]
    assert manifest["deleted"] == [stale] and manifest["kept"] == [committed]


@pytest.mark.parametrize(
    "job_state",
    [
        # The enqueue paths only name attempts of jobs that failed or were cancelled, so
        # the unfinished states never reach this stage in practice.
        pytest.param("queued", id="queued"),
        pytest.param("running", id="running"),
        pytest.param("retry_wait", id="retry_wait"),
        pytest.param("failed", id="failed"),
        pytest.param("cancelled", id="cancelled"),
        pytest.param(None, id="job-gone"),
    ],
)
def test_an_attempt_is_deleted_unless_its_job_succeeded(db, seed, store, sup, job_state):
    pid = seed.project()
    rev = make_revision(seed, pid)
    if job_state is None:
        job_id = str(uuid.uuid4())  # the job row no longer exists
    else:
        job_id = make_run_job(seed, pid, rev["run"], job_state)
    attempt = attempt_of(pid, rev["run"], job_id)
    store.put_bytes(attempt + "partial.bin", b"x")
    enqueue(db, pid, [attempt], reason="attempt_lost")

    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(attempt) == []
    assert cleanup_row(seed, pid)["result_manifest"]["deleted"] == [attempt]


@pytest.mark.parametrize(
    "kind",
    [
        "other_project_scene",
        "other_project_attempt",
        "malformed_scene",
        "malformed_attempt",
        "project_root",
        "traversal",
        "unknown_stage",
        "frames_of_another_stage",
        "manifest_project",
    ],
)
def test_the_stage_refuses_foreign_and_malformed_prefixes_before_deleting_anything(
    db, seed, store, sup, kind
):
    pid = seed.project()
    other = seed.project()
    rev = make_revision(seed, pid)
    own = scene_of(pid, rev)
    store.put_bytes(own + "scene.ply", b"x")
    # Another project's invalid scene and failed attempt, which a refused request names
    # on purpose: an accepted request would delete them, so they must survive.
    foreign = make_revision(seed, other)
    foreign_scene = scene_of(other, foreign)
    foreign_job = make_run_job(seed, other, foreign["run"], "failed")
    foreign_attempt = attempt_of(other, foreign["run"], foreign_job)
    store.put_bytes(foreign_scene + "scene.ply", b"x")
    store.put_bytes(foreign_attempt + "partial.bin", b"x")
    bad = {
        "other_project_scene": foreign_scene,
        "other_project_attempt": foreign_attempt,
        "malformed_scene": f"tenants/{pid}/scenes/not-a-uuid/",
        "malformed_attempt": attempt_of(pid, rev["run"], str(uuid.uuid4()), fence=0),
        "project_root": f"tenants/{pid}/",
        "traversal": f"tenants/{pid}/scenes/{rev['revision']}/../../",
        "unknown_stage": f"tenants/{pid}/runs/{rev['run']}/attempts/deleting/{uuid.uuid4()}/1/",
        # Only preparing stores frames; any other sub-prefix of an attempt is refused.
        "frames_of_another_stage": attempt_of(pid, rev["run"], str(uuid.uuid4()))
        + FRAMES_DIRECTORY,
        "manifest_project": own,  # valid itself, but the request names another project
    }[kind]
    claimed_project = other if kind == "manifest_project" else pid
    seed.job(
        pid, "artifact_cleanup",
        {"projectId": claimed_project, "prefixes": [own, bad], "reason": "revision_invalid"},
    )  # fmt: skip

    assert run_due_cleanup(db, sup) == "failed"
    row = cleanup_row(seed, pid)  # the only cleanup job: a refusal is never requeued
    assert row["state"] == "failed"
    assert row["error"]["code"] == "INTERNAL_ERROR" and row["error"]["retryable"] is False
    assert store.list_keys(own) == [own + "scene.ply"]  # the valid entry was not deleted
    assert sorted(store.list_keys(f"tenants/{other}/")) == sorted(
        [foreign_scene + "scene.ply", foreign_attempt + "partial.bin"]
    )


def test_a_rejected_scene_leaves_nothing_behind_once_its_cleanup_runs(db, seed, store, settings):
    def rejecting(ctx):
        def invalidate(conn, _job):  # scene_validation's failure hook
            conn.execute(
                "UPDATE scene_revisions SET validation_state = 'invalid', "
                "validation_findings = %s WHERE id = %s AND validation_state = 'pending'",
                (
                    Jsonb([{"code": "SPAWN_BLOCKED", "message": "spawn blocked"}]),
                    str(ctx.output("sceneRevisionId")),
                ),  # fmt: skip
            )

        raise StageError("SCENE_VALIDATION_FAILED", on_fail=invalidate)

    sup = Supervisor(db, store, settings, stages={**STAGES, "scene_validation": rejecting},
                     delay_fn=lambda attempt: 0.0)  # fmt: skip
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    assert drain(db, sup)[-1] == ("scene_validation", "failed")
    rev = seed.one("SELECT id, validation_state FROM scene_revisions")
    assert rev["validation_state"] == "invalid"
    prefix = f"tenants/{pid}/scenes/{rev['id']}/"
    assert store.list_keys(prefix)  # packaging wrote the viewer files before validation failed

    assert store.list_keys(f"tenants/{pid}/runs/{run_id}/")  # the finished stages' outputs

    # One cleanup for the scene, one for what the failed run's finished stages wrote.
    assert run_all_due_cleanups(db, sup) == ["succeeded", "succeeded"]
    assert store.list_keys(prefix) == []
    assert store.list_keys(f"tenants/{pid}/runs/{run_id}/") == []


def save_layout(seed, project_id: str, run_id: str, layout: dict) -> str:
    """A user layout save on the run's scene, as the API makes it: revision 2 and its job."""
    base = seed.one("SELECT id, scene_id FROM scene_revisions WHERE run_id = %s", run_id)
    revision_id = str(uuid.uuid4())
    seed.exec(
        "INSERT INTO scene_revisions (id, scene_id, project_id, run_id, base_revision_id, revision, "
        "layout_revision, layout, validation_state, approval_state, created_by) "
        "VALUES (%s, %s, %s, %s, %s, 2, 2, %s, 'pending', 'draft', 'user-1')",
        revision_id, base["scene_id"], project_id, run_id, base["id"], Jsonb(layout),
    )  # fmt: skip
    seed.job(project_id, "geometry_proposal", {"sceneRevisionId": revision_id,
             "layoutSource": "user"}, run_id=run_id, scene_revision_id=revision_id)  # fmt: skip
    return revision_id


def layout_attempts(seed, project_id: str, revision_id: str) -> list[str]:
    rows = seed.all(
        "SELECT id, run_id, stage, fencing_token FROM jobs "
        "WHERE scene_revision_id = %s AND state = 'succeeded'",
        revision_id,
    )
    return [
        attempt_prefix(project_id, str(r["run_id"]), r["stage"], str(r["id"]), r["fencing_token"])
        for r in rows
    ]


def test_a_valid_layout_save_leaves_only_its_scene_once_its_cleanup_runs(db, seed, store, sup):
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    drain(db, sup)
    assert run_all_due_cleanups(db, sup) == ["succeeded"]  # the run's disposable outputs
    run_outputs = sorted(store.list_keys(f"tenants/{pid}/runs/{run_id}/"))
    revision_id = save_layout(
        seed, pid, run_id, seed.one("SELECT layout FROM scene_revisions")["layout"]
    )
    assert [d[1] for d in drain(db, sup)] == ["succeeded"] * 4
    attempts = layout_attempts(seed, pid, revision_id)
    assert len(attempts) == 4
    assert any(key.endswith("/model.glb") for p in attempts for key in store.list_keys(p))

    assert run_all_due_cleanups(db, sup) == ["succeeded"]
    assert all(store.list_keys(p) == [] for p in attempts)
    # The revision's own files stay, and so does what the run wrote: later saves re-texture
    # from its reconstruction.
    assert store.list_keys(f"tenants/{pid}/scenes/{revision_id}/")
    assert sorted(store.list_keys(f"tenants/{pid}/runs/{run_id}/")) == run_outputs


def test_a_layout_save_that_fails_validation_leaves_nothing_behind(db, seed, store, sup):
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    drain(db, sup)
    assert run_all_due_cleanups(db, sup) == ["succeeded"]  # the run's disposable outputs
    run_outputs = sorted(store.list_keys(f"tenants/{pid}/runs/{run_id}/"))
    layout = copy.deepcopy(seed.one("SELECT layout FROM scene_revisions")["layout"])
    layout["spawn"]["feet"] = [0.0, 0.0, 1.5]  # inside the divider wall
    revision_id = save_layout(seed, pid, run_id, layout)
    assert [d[1] for d in drain(db, sup)] == ["succeeded"] * 3 + ["failed"]
    attempts = layout_attempts(seed, pid, revision_id)
    assert len(attempts) == 3

    # One cleanup for the invalid revision's scene, one for what its stages wrote.
    assert run_all_due_cleanups(db, sup) == ["succeeded", "succeeded"]
    assert all(store.list_keys(p) == [] for p in attempts)
    assert store.list_keys(f"tenants/{pid}/scenes/{revision_id}/") == []
    assert sorted(store.list_keys(f"tenants/{pid}/runs/{run_id}/")) == run_outputs


def test_an_attempt_of_a_layout_save_is_kept_only_while_its_revision_is_pending(
    db, seed, store, sup
):
    pid = seed.project()
    rev = make_revision(seed, pid, state="pending", run_status="succeeded")
    job_id = seed.job(pid, "packaging", {}, run_id=rev["run"], scene_revision_id=rev["revision"])
    seed.exec("UPDATE jobs SET state = 'succeeded', fencing_token = 1 WHERE id = %s", job_id)
    attempt = attempt_of(pid, rev["run"], job_id)
    store.put_bytes(attempt + "model.glb", b"x")
    enqueue(db, pid, [attempt], reason="layout_settled")
    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(attempt) == [attempt + "model.glb"]  # a later stage may read it

    seed.exec(
        "UPDATE scene_revisions SET validation_state = 'valid' WHERE id = %s", rev["revision"]
    )
    enqueue(db, pid, [attempt], reason="layout_settled")
    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(attempt) == []


def test_a_succeeded_run_leaves_only_what_layout_saves_retexture_from(db, seed, store, sup):
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    drain(db, sup)
    run_prefix = f"tenants/{pid}/runs/{run_id}/"
    revision = seed.one("SELECT id, layout FROM scene_revisions")
    scene = sorted(store.list_keys(f"tenants/{pid}/scenes/{revision['id']}/"))
    outputs = sorted(store.list_keys(run_prefix))

    def stage_of(key: str) -> str:
        return key.removeprefix(run_prefix).split("/")[1]

    assert "texturing" in {stage_of(key) for key in outputs}  # the model packaging copied

    assert run_all_due_cleanups(db, sup) == ["succeeded"]
    assert [c["reason"] for c in queued_cleanups(seed, pid)] == ["run_succeeded"]
    left = sorted(store.list_keys(run_prefix))
    assert left == [key for key in outputs if stage_of(key) in RETEXTURE_STAGES]
    assert sorted(store.list_keys(f"tenants/{pid}/scenes/{revision['id']}/")) == scene
    # A layout save still re-textures from what the run kept.
    save_layout(seed, pid, run_id, revision["layout"])
    assert [d[1] for d in drain(db, sup)] == ["succeeded"] * 4


def test_a_worker_lost_on_its_last_attempt_leaves_nothing_behind(db, seed, store, sup):
    pid = seed.project()
    run_id, job_id = seed.run(pid, max_attempts=1)
    queue.claim(db, "lost-worker", "cpu")
    attempt = f"tenants/{pid}/runs/{run_id}/attempts/validating/{job_id}/1/"
    store.put_bytes(attempt + "partial.bin", b"partial")  # written before the crash
    seed.exec(
        "UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id = %s", job_id
    )
    assert [o.state for o in queue.reap_expired(db)] == ["failed"]

    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(attempt) == []


def test_a_requeued_attempt_cancelled_before_its_retry_leaves_nothing_behind(db, seed, store, sup):
    pid = seed.project()
    run_id, job_id = seed.run(pid, max_attempts=3)
    queue.claim(db, "lost-worker", "cpu")  # fence 1 writes, then the worker vanishes
    first = f"tenants/{pid}/runs/{run_id}/attempts/validating/{job_id}/1/"
    store.put_bytes(first + "partial.bin", b"partial")
    seed.exec(
        "UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id = %s", job_id
    )
    assert [o.state for o in queue.reap_expired(db)] == ["queued"]  # re-queued at fence 2
    seed.exec("UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", run_id)
    assert queue.sweep_cancelled(db) == 1  # no worker will claim it, so nothing sweeps fence 1

    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(f"tenants/{pid}/runs/{run_id}/") == []


class UndeletableStore(LocalObjectStore):
    """Writes work but deletes fail, as when the object store refuses them for a while."""

    def delete_prefix(self, prefix: str) -> int:
        raise OSError("delete refused")


def writes_then_retries(ctx):
    ctx.store.put_bytes(ctx.attempt_key("partial.bin"), b"partial")
    raise TransientError()


def undeletable_worker(db, store, settings, stage) -> Supervisor:
    return Supervisor(db, UndeletableStore(store.root), settings,
                      stages={**STAGES, "validating": stage}, delay_fn=lambda attempt: 0.0)  # fmt: skip


def queued_cleanups(seed, project_id: str) -> list[dict]:
    rows = seed.all(
        "SELECT input_manifest FROM jobs WHERE project_id = %s AND stage = 'artifact_cleanup'",
        project_id,
    )
    return [row["input_manifest"] for row in rows]


def test_attempts_a_worker_could_not_delete_are_queued_when_retries_run_out(
    db, seed, store, settings, sup
):
    worker = undeletable_worker(db, store, settings, writes_then_retries)
    pid = seed.project()
    run_id, job_id = seed.run(pid, max_attempts=3)
    assert [d[1] for d in drain(db, worker)] == ["retry_wait", "retry_wait", "failed"]
    attempts = f"tenants/{pid}/runs/{run_id}/attempts/validating/{job_id}/"
    assert len(store.list_keys(attempts)) == 3  # every delete and every sweep failed
    # A retry sweeps again, so only the failure that ended the job queued them.
    assert queued_cleanups(seed, pid) == [
        {
            "projectId": pid,
            "prefixes": [f"{attempts}{fence}/" for fence in (1, 2, 3)],
            "reason": "delete_failed",
        }
    ]

    assert run_due_cleanup(db, sup) == "succeeded"  # a worker whose deletes work
    assert store.list_keys(attempts) == []


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_an_attempt_a_worker_could_not_delete_is_queued_when_the_job_ends(
    db, seed, store, settings, outcome
):
    def writes_then_ends(ctx):
        ctx.store.put_bytes(ctx.attempt_key("partial.bin"), b"partial")
        if outcome == "failed":
            raise StageError("INVALID_MEDIA")
        # Cancellation lands after the stage's own checks, before completion.
        seed.exec(
            "UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", ctx.run_id
        )
        seed.exec("UPDATE jobs SET cancel_requested_at = now() WHERE id = %s", ctx.job_id)
        return finish(ctx, {})

    worker = undeletable_worker(db, store, settings, writes_then_ends)
    pid = seed.project()
    run_id, job_id = seed.run(pid)
    assert worker.execute(queue.claim(db, "worker-test", "cpu")) == outcome
    attempt = f"tenants/{pid}/runs/{run_id}/attempts/validating/{job_id}/1/"
    assert [m["prefixes"] for m in queued_cleanups(seed, pid)] == [[attempt]]


def test_a_job_that_succeeds_queues_the_earlier_attempt_it_could_not_sweep(
    db, seed, store, settings, sup
):
    def second_time_lucky(ctx):
        ctx.store.put_bytes(ctx.attempt_key("output.bin"), b"x")
        if ctx.attempt == 1:
            raise TransientError()
        return finish(ctx, {})

    worker = undeletable_worker(db, store, settings, second_time_lucky)
    pid = seed.project()
    run_id, job_id = seed.run(pid)
    assert worker.execute(queue.claim(db, "worker-test", "cpu")) == "retry_wait"
    assert queue.promote_retry_wait(db) == 1
    assert worker.execute(queue.claim(db, "worker-test", "cpu")) == "succeeded"
    attempts = f"tenants/{pid}/runs/{run_id}/attempts/validating/{job_id}/"
    assert [m["prefixes"] for m in queued_cleanups(seed, pid)] == [[f"{attempts}1/"]]

    # The next stage waits so that the cleanup is the job claimed next.
    seed.exec("UPDATE jobs SET available_at = now() + interval '1 hour' WHERE stage = 'preparing'")
    assert run_due_cleanup(db, sup) == "succeeded"
    assert store.list_keys(f"{attempts}1/") == []
    assert f"{attempts}2/output.bin" in store.list_keys(f"{attempts}2/")


def test_a_job_failed_before_its_stage_runs_queues_what_earlier_attempts_left(
    db, seed, store, settings
):
    worker = undeletable_worker(db, store, settings, writes_then_retries)
    pid = seed.project()
    run_id, job_id = seed.run(pid, max_attempts=3)
    assert worker.execute(queue.claim(db, "worker-test", "cpu")) == "retry_wait"
    # Earlier stages and attempts consumed the whole run budget before the retry.
    budget = seed.one("SELECT resource_budget FROM reconstruction_runs")["resource_budget"]
    seed.exec("UPDATE usage_ledger SET actual_cpu_seconds = %s", int(budget["maxSeconds"]))
    assert queue.promote_retry_wait(db) == 1
    assert worker.execute(queue.claim(db, "worker-test", "cpu")) == "failed"
    attempt = f"tenants/{pid}/runs/{run_id}/attempts/validating/{job_id}/1/"
    assert [m["prefixes"] for m in queued_cleanups(seed, pid)] == [[attempt]]


def queued_cleanup_rows(seed, project_id: str) -> list[dict]:
    return seed.all(
        "SELECT input_manifest, available_at - now() AS wait FROM jobs "
        "WHERE project_id = %s AND stage = 'artifact_cleanup' AND state = 'queued'",
        project_id,
    )


def exhaust_cleanup(db, worker) -> list[str]:
    """Run the queued cleanup now and through every retry, without waiting out a requeue."""
    with db.transaction() as conn:
        conn.execute(
            "UPDATE jobs SET available_at = now() "
            "WHERE stage = 'artifact_cleanup' AND state = 'queued'"
        )
    outcomes = []
    while True:
        queue.promote_retry_wait(db)  # retries are due at once (delay_fn returns 0)
        job = queue.claim(db, "worker-test", "cpu")
        if job is None:
            return outcomes
        outcomes.append(worker.execute(job))


def test_a_cleanup_that_runs_out_of_retries_is_queued_again_later(db, seed, store, settings, sup):
    # A storage outage longer than the retry window must not leave the objects behind.
    pid = seed.project()
    rev = make_revision(seed, pid)
    scene = scene_of(pid, rev)
    store.put_bytes(scene + "model.glb", b"x")
    enqueue(db, pid, [scene], reason="revision_invalid", scene_revision_id=rev["revision"])
    outage = Supervisor(db, UndeletableStore(store.root), settings, delay_fn=lambda attempt: 0.0)

    assert exhaust_cleanup(db, outage) == ["retry_wait", "retry_wait", "failed"]
    [again] = queued_cleanup_rows(seed, pid)
    assert again["input_manifest"] == {
        "projectId": pid, "prefixes": [scene], "reason": "revision_invalid",
        "sceneRevisionId": rev["revision"], "requeues": 1,
    }  # fmt: skip
    expected = queue.ARTIFACT_CLEANUP_REQUEUE_SECONDS
    assert expected - 60 < again["wait"].total_seconds() <= expected

    # Once storage is back, the requeued cleanup deletes the scene.
    assert run_all_due_cleanups(db, sup) == ["succeeded"]
    assert store.list_keys(scene) == []


def test_requeued_cleanups_wait_longer_each_time_and_never_give_up(db, seed, store, settings):
    pid = seed.project()
    scene = scene_of(pid, make_revision(seed, pid))
    outage = Supervisor(db, UndeletableStore(store.root), settings, delay_fn=lambda attempt: 0.0)
    enqueue(db, pid, [scene], reason="revision_invalid", requeues=3)
    assert exhaust_cleanup(db, outage)[-1] == "failed"
    [again] = queued_cleanup_rows(seed, pid)
    assert again["input_manifest"]["requeues"] == 4
    assert again["wait"].total_seconds() > 8 * queue.ARTIFACT_CLEANUP_REQUEUE_SECONDS - 60

    # Days into an outage the wait stays at a day, and the request is still queued: nothing
    # else would find these prefixes again.
    seed.exec("DELETE FROM jobs WHERE stage = 'artifact_cleanup'")
    enqueue(db, pid, [scene], reason="revision_invalid", requeues=40)
    assert exhaust_cleanup(db, outage)[-1] == "failed"
    [later] = queued_cleanup_rows(seed, pid)
    assert later["input_manifest"]["requeues"] == 41
    cap = queue.ARTIFACT_CLEANUP_REQUEUE_CAP_SECONDS
    assert cap - 60 < later["wait"].total_seconds() <= cap


def test_a_cleanup_whose_worker_is_lost_on_its_last_attempt_is_queued_again(db, seed, store):
    pid = seed.project()
    scene = scene_of(pid, make_revision(seed, pid))
    enqueue(db, pid, [scene], reason="revision_invalid")
    seed.exec(
        "UPDATE jobs SET available_at = now(), max_attempts = 1 WHERE stage = 'artifact_cleanup'"
    )
    job = queue.claim(db, "lost-worker", "cpu")
    assert job is not None and job.stage == "artifact_cleanup"
    seed.exec(
        "UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id = %s", job.id
    )

    assert [o.state for o in queue.reap_expired(db)] == ["failed"]
    [again] = queued_cleanup_rows(seed, pid)
    assert again["input_manifest"]["prefixes"] == [scene]
    assert again["input_manifest"]["requeues"] == 1
