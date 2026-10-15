"""CPU end-to-end tests through the real supervisor, queue and Postgres schema."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from helpers import drain
from pdf_fixtures import classic_pdf, encrypted_pdf, object_stream_pdf, page_tree
from psycopg.types.json import Jsonb

from andarilho_worker import queue
from andarilho_worker.geometry.glb import inspect_textured_glb
from andarilho_worker.runner import ProcessFailed, RunResult
from andarilho_worker.stages import STAGES
from andarilho_worker.stages import inspect_upload as inspect_mod
from andarilho_worker.supervisor import Supervisor
from andarilho_worker.validation import manifest_sha256


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
        ("scene_analysis", "succeeded"),
        ("geometry_proposal", "succeeded"),
        ("texturing", "succeeded"),
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
    assert manifest["schemaVersion"] == "2.0" and manifest["renderMode"] == "mesh"
    assert "worldFromDecodedRender" not in manifest
    assert manifest["sceneRevisionId"] == str(rev["id"])
    assert manifest["sceneId"] == str(rev["scene_id"]) and manifest["runId"] == run_id
    assert manifest["navigation"]["mode"] == "approved_collision"
    assert manifest["provenance"]["geometrySource"] == "synthetic_fixture"
    assert rev["manifest_sha256"] == manifest_sha256(manifest)
    prefix = f"tenants/{pid}/scenes/{rev['id']}/"
    assert set(store.list_keys(prefix)) == {
        prefix + "model.glb",
        prefix + "collision.glb",
        prefix + "manifest.json",
    }
    render = manifest["assets"]["render"]
    assert render["key"] == prefix + "model.glb" and render["format"] == "glb"
    model = inspect_textured_glb(store.get_bytes(render["key"]))
    assert model["unlit"] and model["triangles"] > 0
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
    assert {"captureSetId", "modelKey", "collisionKey", "manifestKey"} <= set(
        last["input_manifest"]["outputs"]
    )
    texturing = seed.one("SELECT result_manifest FROM jobs WHERE stage = 'texturing'")
    assert texturing["result_manifest"]["outputs"]["textureCoverage"]["all"] == 1.0
    assert any(
        "/attempts/packaging/" in k and k.endswith("/1/result.json")
        for k in store.list_keys(f"tenants/{pid}/runs/")
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
    assert [d[1] for d in done] == ["succeeded"] * 4
    second = seed.one("SELECT * FROM scene_revisions WHERE id = %s", rev2)
    assert second["validation_state"] == "valid"
    m2 = second["manifest"]
    assert m2["navigation"]["mode"] == "approved_collision"
    assert m2["provenance"]["geometrySource"] == "user_confirmed_trace"
    assert m2["spawn"]["feet"] == [-3, 0, 2]
    # A layout save re-textures: its own model, here identical to the first (same surfaces).
    assert m2["assets"]["render"]["key"] == f"tenants/{pid}/scenes/{rev2}/model.glb"
    assert m2["assets"]["render"]["sha256"] == manifest["assets"]["render"]["sha256"]
    assert store.get_bytes(m2["assets"]["render"]["key"])[:4] == b"glTF"
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
    assert done == [("geometry_proposal", "succeeded"), ("texturing", "succeeded"),
                    ("packaging", "succeeded"), ("scene_validation", "failed")]  # fmt: skip
    third = seed.one(
        "SELECT validation_state, validation_findings FROM scene_revisions WHERE id = %s", rev3
    )
    assert third["validation_state"] == "invalid"
    assert {f["code"] for f in third["validation_findings"]} & {"SPAWN_BLOCKED"}
    assert seed.one("SELECT status FROM reconstruction_runs")["status"] == "succeeded"


WEBP = b"RIFF" + (4).to_bytes(4, "little") + b"WEBP"


def test_packaging_publishes_the_poster_and_carries_it_to_layout_revisions(
    db, seed, store, settings
):
    def preparing(ctx):  # the synthetic recipe has no capture, so stand in for its poster
        result = STAGES["preparing"](ctx)
        key = ctx.attempt_key("poster.webp")
        ctx.store.put_bytes(key, WEBP, "image/webp")
        # The job's result manifest is what later stages read back: the poster must be in it.
        outputs = {**result.outputs, "posterKey": key}
        return replace(result, manifest={**result.manifest, "outputs": outputs}, outputs=outputs)

    sup = Supervisor(db, store, settings, delay_fn=lambda attempt: 0.0,
                     stages={**STAGES, "preparing": preparing})  # fmt: skip
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    assert all(status == "succeeded" for _, status in drain(db, sup))
    rev = seed.one("SELECT * FROM scene_revisions")
    assert rev["validation_state"] == "valid"
    poster = rev["manifest"]["assets"]["poster"]
    assert poster["key"] == f"tenants/{pid}/scenes/{rev['id']}/poster.webp"
    assert poster["format"] == "webp" and poster["bytes"] == len(WEBP)
    assert store.get_bytes(poster["key"]) == WEBP
    arts = seed.all("SELECT kind FROM artifacts ORDER BY kind")
    assert [a["kind"] for a in arts] == [
        "scene_collision", "scene_manifest", "scene_poster", "scene_render"
    ]  # fmt: skip

    rev2 = _save_layout(seed, rev, pid, run_id, 2, rev["id"], copy.deepcopy(rev["layout"]))
    assert [status for _, status in drain(db, sup)] == ["succeeded"] * 4
    second = seed.one("SELECT validation_state, manifest FROM scene_revisions WHERE id = %s", rev2)
    assert second["validation_state"] == "valid"
    carried = second["manifest"]["assets"]["poster"]
    assert carried["key"] == f"tenants/{pid}/scenes/{rev2}/poster.webp"
    assert carried["sha256"] == poster["sha256"]


@pytest.mark.parametrize("analysis", [None, ["Two chairs were drawn from one photo."]])
def test_packaging_carries_the_capture_quality_poster_and_unseen_area_warning(
    db, seed, store, sup, analysis
):
    """A captured revision's packaging reads the run's outputs, as a layout save does.

    The scene analysis warnings are optional: a run without them packages as before, and when
    present they follow the frame warnings and come before the unseen-area warning.
    """
    from andarilho_worker.geometry.glb import write_textured_glb
    from andarilho_worker.geometry.layout import collision_glb, rectangle_layout
    from andarilho_worker.geometry.texture import TextureParams, procedural_texture_layout

    pid = seed.project()
    run_id, _ = seed.run(pid, quality="preview")
    seed.exec("UPDATE jobs SET state = 'succeeded' WHERE state = 'queued'")
    layout = rectangle_layout((-1.5, -1.25), (1.5, 1.25), elevation=0.0, height=2.6)
    scene_id, rid = str(uuid.uuid4()), str(uuid.uuid4())
    seed.exec("INSERT INTO scenes (id, project_id, run_id, version) VALUES (%s, %s, %s, 1)",
              scene_id, pid, run_id)  # fmt: skip
    seed.exec(
        "INSERT INTO scene_revisions (id, scene_id, project_id, run_id, revision, layout_revision, "
        "layout, validation_state, approval_state, created_by) "
        "VALUES (%s, %s, %s, %s, 1, 1, %s, 'pending', 'draft', 'user-1')",
        rid, scene_id, pid, run_id, Jsonb(layout),
    )  # fmt: skip
    model_key = f"tenants/{pid}/runs/{run_id}/attempts/texturing/x/1/model.glb"
    store.put_bytes(model_key, write_textured_glb(procedural_texture_layout(
        layout, TextureParams(texel_size=0.05, page_size=256))))  # fmt: skip
    collision_key = f"tenants/{pid}/runs/{run_id}/attempts/geometry_proposal/x/1/collision.glb"
    store.put_bytes(collision_key, collision_glb(layout))
    poster_key = f"tenants/{pid}/runs/{run_id}/attempts/preparing/x/1/poster.webp"
    store.put_bytes(poster_key, WEBP)
    seed.job(
        pid, "packaging",
        {"outputs": {
            "sceneRevisionId": rid, "layoutSource": "user", "modelKey": model_key,
            "collisionKey": collision_key, "posterKey": poster_key, "scaleStatus": "estimated",
            "frameWarnings": ["Only a short walk was recorded."],
            **({"analysisWarnings": analysis} if analysis is not None else {}),
            "textureCoverage": {"all": 0.5, "wall": 0.4, "floor": 0.9, "ceiling": 0.9},
            "coverage": {"registeredViews": 24, "selectedViews": 24, "disconnectedComponents": 0,
                         "warnings": ["An earlier warning."], "registeredRatio": 1.0},
        }},
        run_id=run_id, scene_revision_id=rid,
    )  # fmt: skip
    assert sup.execute(queue.claim(db, "worker-test", "cpu")) == "succeeded"
    outputs = seed.one("SELECT result_manifest FROM jobs WHERE stage = 'packaging'")[
        "result_manifest"
    ]["outputs"]
    manifest = json.loads(store.get_bytes(outputs["manifestKey"]))
    assert manifest["scaleStatus"] == "estimated"
    assert manifest["quality"]["registeredViews"] == 24
    assert manifest["quality"]["warnings"] == [
        "An earlier warning.",
        "Only a short walk was recorded.",
        *(analysis or []),
        "About 50% of the walls, floor and ceiling were not seen in the video and are shown "
        "in plain grey.",
    ]
    assert manifest["assets"]["render"]["key"] == f"tenants/{pid}/scenes/{rid}/model.glb"
    assert manifest["assets"]["poster"]["key"] == f"tenants/{pid}/scenes/{rid}/poster.webp"
    assert manifest["provenance"]["geometrySource"] == "user_confirmed_trace"
    assert manifest["navigation"]["mode"] == "approved_collision"
    assert store.get_bytes(manifest["assets"]["render"]["key"]) == store.get_bytes(model_key)


@pytest.mark.parametrize(
    "seen,expected",
    [
        (1.0, None),
        (0.99, None),  # the threshold itself is not a warning
        (
            0.9899,
            "About 1% of the walls, floor and ceiling were not seen in the video "
            "and are shown in plain grey.",
        ),
        (
            0.5,
            "About 50% of the walls, floor and ceiling were not seen in the video "
            "and are shown in plain grey.",
        ),
        (
            0.0,
            "About 100% of the walls, floor and ceiling were not seen in the video "
            "and are shown in plain grey.",
        ),
    ],
)
def test_unseen_area_warning_starts_below_ninety_nine_percent(seen, expected):
    from andarilho_worker.stages.package_scene import unseen_warning

    assert unseen_warning({"all": seen, "wall": seen}) == expected


def _scaled_layout(layout, k):
    out = copy.deepcopy(layout)
    s2 = lambda p: [p[0] * k, p[1] * k]  # noqa: E731
    for floor in out["floors"]:
        floor["elevation"] *= k
        floor["polygons"] = [[s2(p) for p in ring] for ring in floor["polygons"]]
        floor["holes"] = [[s2(p) for p in ring] for ring in floor.get("holes", [])]
    for wall in out["walls"]:
        wall.update(start=s2(wall["start"]), end=s2(wall["end"]),
                    thickness=wall["thickness"] * k, height=wall["height"] * k)  # fmt: skip
    for o in out["openings"]:
        o.update(offset=o["offset"] * k, width=o["width"] * k,
                 bottom=o.get("bottom", 0) * k, top=o["top"] * k)  # fmt: skip
    for o in out["obstacles"]:
        o.update(footprint=[s2(p) for p in o["footprint"]], height=o["height"] * k)
    out["spawn"]["feet"] = [v * k for v in out["spawn"]["feet"]]
    out["calibration"] = {"method": "known_distance", "renderScale": k}
    return out


def _save_layout(seed, rev, pid, run_id, number, base_id, layout):
    rid = str(uuid.uuid4())
    seed.exec(
        "INSERT INTO scene_revisions (id, scene_id, project_id, run_id, base_revision_id, revision, "
        "layout_revision, layout, validation_state, approval_state, created_by) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', 'draft', 'user-1')",
        rid, rev["scene_id"], pid, run_id, base_id, number, number, Jsonb(layout),
    )  # fmt: skip
    seed.job(pid, "geometry_proposal", {"sceneRevisionId": rid, "layoutSource": "user"},
             run_id=run_id, scene_revision_id=rid)  # fmt: skip
    return rid


def test_calibrated_layout_save_retextures_and_does_not_compound(db, seed, store, sup):
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    drain(db, sup)
    rev = seed.one("SELECT * FROM scene_revisions")
    original = rev["manifest"]["assets"]["render"]["sha256"]
    calibrated = _scaled_layout(rev["layout"], 2.0)
    rev2 = _save_layout(seed, rev, pid, run_id, 2, rev["id"], calibrated)
    assert [d[1] for d in drain(db, sup)] == ["succeeded"] * 4
    m2 = seed.one("SELECT manifest FROM scene_revisions WHERE id = %s", rev2)["manifest"]
    assert m2["scaleStatus"] == "user_measured"
    # Doubled walls and floor need a different atlas: the revision gets its own model.
    assert m2["assets"]["render"]["key"] == f"tenants/{pid}/scenes/{rev2}/model.glb"
    assert m2["assets"]["render"]["sha256"] != original
    assert m2["bounds"]["max"][0] > rev["manifest"]["bounds"]["max"][0]
    # Re-saving the same calibrated layout re-textures to the same model: the layout is
    # the absolute, already-scaled geometry, so nothing is scaled twice.
    rev3 = _save_layout(seed, rev, pid, run_id, 3, rev2, calibrated)
    assert [d[1] for d in drain(db, sup)] == ["succeeded"] * 4
    m3 = seed.one("SELECT manifest FROM scene_revisions WHERE id = %s", rev3)["manifest"]
    assert m3["assets"]["render"]["sha256"] == m2["assets"]["render"]["sha256"]
    assert m3["bounds"] == m2["bounds"]


def test_older_scene_revision_never_takes_the_active_draft_back(db, seed, sup):
    pid = seed.project()
    old_run, _ = seed.run(pid, quality="synthetic")
    drain(db, sup)
    old = seed.one("SELECT * FROM scene_revisions WHERE run_id = %s", old_run)
    new_run, _ = seed.run(pid, quality="synthetic")
    drain(db, sup)
    new = seed.one("SELECT id FROM scene_revisions WHERE run_id = %s", new_run)
    active = "SELECT active_draft_revision_id AS a FROM projects WHERE id = %s"
    assert seed.one(active, pid)["a"] == new["id"]
    # A layout revision of the older scene finishes after the newer scene exists.
    rev2 = str(uuid.uuid4())
    seed.exec(
        "INSERT INTO scene_revisions (id, scene_id, project_id, run_id, base_revision_id, revision, "
        "layout_revision, layout, validation_state, approval_state, created_by) "
        "VALUES (%s, %s, %s, %s, %s, 2, 2, %s, 'pending', 'draft', 'user-1')",
        rev2, old["scene_id"], pid, old_run, old["id"], Jsonb(old["layout"]),
    )  # fmt: skip
    seed.job(
        pid, "geometry_proposal",
        {"sceneRevisionId": rev2, "sceneId": str(old["scene_id"]), "baseRevisionId": str(old["id"]),
         "layoutRevision": 2, "layoutSource": "user"},
        run_id=old_run, scene_revision_id=rev2,
    )  # fmt: skip
    assert [d[1] for d in drain(db, sup)] == ["succeeded"] * 4
    assert (
        seed.one("SELECT validation_state AS v FROM scene_revisions WHERE id = %s", rev2)["v"]
        == "valid"
    )
    assert seed.one(active, pid)["a"] == new["id"]


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


def _writing_transient_stage(ctx):
    ctx.store.put_bytes(ctx.attempt_key("partial.bin"), b"partial")
    _transient_stage(ctx)


def _writing_sleeping_stage(ctx):
    ctx.store.put_bytes(ctx.attempt_key("partial.bin"), b"partial")
    _sleeping_stage(ctx)


def test_failed_attempts_delete_their_objects(db, seed, store, settings):
    sup = Supervisor(db, store, settings, stages={**STAGES, "validating": _writing_transient_stage},
                     delay_fn=lambda attempt: 0.0)  # fmt: skip
    pid = seed.project()
    run_id, _ = seed.run(pid, max_attempts=3)
    assert [d[1] for d in drain(db, sup)] == ["retry_wait", "retry_wait", "failed"]
    # Each retried attempt and the final failed one removed what it wrote.
    assert store.list_keys(f"tenants/{pid}/runs/{run_id}/attempts/") == []


def test_completion_that_loses_to_cancellation_deletes_its_outputs(db, seed, store, settings):
    from andarilho_worker.stages.base import finish

    def finishes_as_cancellation_arrives(ctx):
        ctx.store.put_bytes(ctx.attempt_key("output.bin"), b"done")
        # Cancellation lands after the stage's own checks, before completion.
        seed.exec(
            "UPDATE reconstruction_runs SET cancel_requested_at = now() WHERE id = %s", ctx.run_id
        )
        seed.exec("UPDATE jobs SET cancel_requested_at = now() WHERE id = %s", ctx.job_id)
        return finish(ctx, {})

    sup = Supervisor(
        db, store, settings, stages={**STAGES, "validating": finishes_as_cancellation_arrives}
    )
    pid = seed.project()
    run_id, _ = seed.run(pid)
    assert sup.execute(queue.claim(db, "worker-test", "cpu")) == "cancelled"
    assert store.list_keys(f"tenants/{pid}/runs/{run_id}/attempts/") == []


def test_a_new_attempt_sweeps_what_a_replaced_one_left(db, seed, store, settings):
    sup = Supervisor(db, store, settings, stages={**STAGES, "validating": _writing_sleeping_stage})
    pid = seed.project()
    run_id, _ = seed.run(pid)
    job = queue.claim(db, "worker-test", "cpu")

    def bump():
        time.sleep(1.0)
        seed.exec("UPDATE jobs SET fencing_token = fencing_token + 1, state = 'queued', "
                  "lease_owner = NULL, lease_expires_at = NULL WHERE id = %s", job.id)  # fmt: skip

    threading.Thread(target=bump, daemon=True).start()
    assert sup.execute(job) == "lost"
    attempts = f"tenants/{pid}/runs/{run_id}/attempts/"
    # A worker that lost its lease stops writing, deletes included.
    assert store.list_keys(attempts) == [f"{attempts}validating/{job.id}/1/partial.bin"]
    retry = queue.claim(db, "worker-test", "cpu")
    assert retry is not None and retry.fencing_token == 3
    replacement = Supervisor(
        db, store, settings, stages={**STAGES, "validating": _transient_stage},
        delay_fn=lambda attempt: 0.0,
    )  # fmt: skip
    assert replacement.execute(retry) == "retry_wait"
    assert store.list_keys(attempts) == []


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


def test_run_budget_is_shared_across_stages(db, seed, store, sup):
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    budget = seed.one("SELECT resource_budget FROM reconstruction_runs")["resource_budget"]
    # Earlier stages already consumed the whole run ceiling.
    seed.exec("UPDATE usage_ledger SET actual_cpu_seconds = %s", int(budget["maxSeconds"]))
    done = drain(db, sup)
    assert done == [("validating", "failed")]
    run = seed.one("SELECT status, error FROM reconstruction_runs WHERE id = %s", run_id)
    assert run["status"] == "failed" and run["error"]["code"] == "BUDGET_EXHAUSTED"


# ---------------------------------------------------------------------------
# Upload inspection
# ---------------------------------------------------------------------------

# Real, decodable images (FFmpeg decodes them during inspection where it is installed).
FIXTURES = Path(__file__).parent / "fixtures"
JPEG = (FIXTURES / "photo-32x16.jpg").read_bytes()
PNG = (FIXTURES / "photo-64x48.png").read_bytes()
PDF = classic_pdf(page_tree(1))
ENCRYPTED_PDF = encrypted_pdf()
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
    stored = store.list_keys(f"tenants/{pid}/originals/")
    keyed = seed.one("SELECT immutable_key FROM uploads WHERE id = %s", uid)["immutable_key"]
    if status == "rejected":
        # Rejected bytes are deleted and the key cleared, which releases the quota.
        assert stored == [] and keyed is None
    else:
        assert stored == [keyed]


def test_inspection_rejects_pdf_with_too_many_pages(db, seed, store, sup):
    pid = seed.project()
    uid = make_upload(seed, store, pid, classic_pdf(page_tree(50)), "floor_plan")
    drain(db, sup)
    row = seed.one("SELECT status, media_metadata FROM uploads WHERE id = %s", uid)
    assert row["status"] == "rejected"
    assert row["media_metadata"]["rejectionReason"] == "PDF has too many pages"


def test_inspection_reads_pages_from_compressed_object_streams(db, seed, store, sup):
    # Optimized PDFs keep their page objects in compressed object streams.
    pid = seed.project()
    uid = make_upload(seed, store, pid, object_stream_pdf(page_tree(2)), "floor_plan")
    assert drain(db, sup) == [("upload_inspection", "succeeded")]
    row = seed.one("SELECT status, media_metadata FROM uploads WHERE id = %s", uid)
    assert row["status"] == "verified" and row["media_metadata"]["pages"] == 2


class DecodeRunner:
    """Stands in for FFmpeg's image decode: succeeds, or fails as for a corrupt file."""

    def __init__(self, ok: bool) -> None:
        self.ok = ok
        self.calls: list[list[str]] = []

    def run(self, argv, *, timeout, check_cancelled=None, **kw):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        result = RunResult(tuple(argv), 0 if self.ok else 1, "", "", 0.01)
        if not self.ok:
            raise ProcessFailed("ffmpeg exited with status 1", result)
        return result


@pytest.mark.parametrize("ok", [True, False])
def test_inspection_decodes_images_before_verifying_them(
    db, seed, store, settings, monkeypatch, ok
):
    monkeypatch.setattr(inspect_mod, "tool_available", lambda name: True)
    runner = DecodeRunner(ok)
    sup = Supervisor(db, store, settings, runner=runner)
    pid = seed.project()
    uid = make_upload(seed, store, pid, JPEG, "photo")
    assert drain(db, sup) == [("upload_inspection", "succeeded")]
    (call,) = runner.calls
    assert call[0] == "ffmpeg" and call[-3:] == ["-f", "null", "-"] and "-xerror" in call
    row = seed.one("SELECT status, rejection_code, media_metadata FROM uploads WHERE id = %s", uid)
    if ok:
        assert row["status"] == "verified" and row["media_metadata"]["decoded"] is True
    else:
        # A valid header over truncated pixels is rejected before any run can pick it.
        assert row["status"] == "rejected" and row["rejection_code"] == "INVALID_MEDIA"
        assert row["media_metadata"]["rejectionReason"] == "image could not be decoded"


def test_inspection_finalizes_when_rejected_object_is_already_gone(db, seed, store, sup):
    # A previous attempt deleted the rejected bytes but lost its commit: the retry
    # must still finalize the upload instead of failing on the missing object.
    pid = seed.project()
    uid = make_upload(seed, store, pid, SVG, "photo")
    key = seed.one("SELECT immutable_key FROM uploads WHERE id = %s", uid)["immutable_key"]
    store.delete_prefix(key)
    assert drain(db, sup) == [("upload_inspection", "succeeded")]
    row = seed.one("SELECT status, rejection_code, immutable_key FROM uploads WHERE id = %s", uid)
    assert row["status"] == "rejected" and row["immutable_key"] is None


def test_upload_cleanup_deletes_removed_bytes_and_releases_the_key(db, seed, store, sup):
    pid = seed.project()
    uid = make_upload(seed, store, pid, JPEG, "photo")
    assert drain(db, sup) == [("upload_inspection", "succeeded")]
    key = seed.one("SELECT immutable_key FROM uploads WHERE id = %s", uid)["immutable_key"]
    seed.exec("UPDATE uploads SET status = 'aborted' WHERE id = %s", uid)  # as the API does
    for _ in range(2):  # a duplicate or retried job is harmless
        seed.job(pid, "upload_cleanup", {"uploadId": uid, "projectId": pid, "key": key},
                 upload_id=uid)  # fmt: skip
    assert drain(db, sup) == [("upload_cleanup", "succeeded")] * 2
    assert store.list_keys(f"tenants/{pid}/originals/") == []
    row = seed.one("SELECT status, immutable_key FROM uploads WHERE id = %s", uid)
    assert row["status"] == "aborted" and row["immutable_key"] is None


def test_upload_cleanup_deletes_a_staging_copy_the_api_could_not(db, seed, store, sup):
    # The API froze (or rejected) these uploads but failed to delete their staging
    # objects, so it queued cleanup jobs.
    pid = seed.project()
    kept = make_upload(seed, store, pid, JPEG, "photo")
    assert drain(db, sup) == [("upload_inspection", "succeeded")]
    rejected = make_upload(seed, store, pid, JPEG, "photo")
    seed.exec("DELETE FROM jobs WHERE upload_id = %s", rejected)
    seed.exec(
        "UPDATE uploads SET status = 'rejected', immutable_key = NULL WHERE id = %s", rejected
    )
    for uid in (kept, rejected):
        store.put_bytes(f"tenants/{pid}/staging/{uid}", JPEG)
        seed.job(pid, "upload_cleanup", {"uploadId": uid, "projectId": pid}, upload_id=uid)
    assert drain(db, sup) == [("upload_cleanup", "succeeded")] * 2
    assert store.list_keys(f"tenants/{pid}/staging/") == []
    # The verified upload keeps its frozen bytes and its status.
    row = seed.one("SELECT status, immutable_key FROM uploads WHERE id = %s", kept)
    assert row["status"] == "verified" and store.list_keys(row["immutable_key"]) != []


def test_inspection_out_of_retries_rejects_and_cleans_up(db, seed, store, settings):
    stages = {**STAGES, "upload_inspection": _transient_stage}
    sup = Supervisor(db, store, settings, stages=stages, delay_fn=lambda attempt: 0.0)
    pid = seed.project()
    uid = make_upload(seed, store, pid, JPEG, "photo")
    seed.exec("UPDATE jobs SET max_attempts = 2 WHERE upload_id = %s", uid)
    assert drain(db, sup) == [
        ("upload_inspection", "retry_wait"),
        ("upload_inspection", "failed"),
        ("upload_cleanup", "succeeded"),
    ]
    row = seed.one("SELECT status, rejection_code, immutable_key FROM uploads WHERE id = %s", uid)
    assert row["status"] == "rejected" and row["rejection_code"] == "INSPECTION_FAILED"
    assert row["immutable_key"] is None  # bytes deleted, quota released
    assert store.list_keys(f"tenants/{pid}/originals/") == []


def test_reaped_inspection_out_of_attempts_rejects_the_upload(db, seed, store):
    pid = seed.project()
    uid = make_upload(seed, store, pid, JPEG, "photo")
    seed.exec("UPDATE jobs SET max_attempts = 1 WHERE upload_id = %s", uid)
    job = queue.claim(db, "lost-worker", "cpu")
    assert job is not None and job.stage == "upload_inspection"
    seed.exec("UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id = %s",
              job.id)  # fmt: skip
    assert [o.state for o in queue.reap_expired(db)] == ["failed"]
    row = seed.one("SELECT status, rejection_code FROM uploads WHERE id = %s", uid)
    assert row["status"] == "rejected" and row["rejection_code"] == "INSPECTION_FAILED"
    cleanup = seed.one("SELECT state FROM jobs WHERE upload_id = %s AND stage = 'upload_cleanup'",
                       uid)  # fmt: skip
    assert cleanup["state"] == "queued"  # its bytes (and quota) are released next


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


class VideoRunner:
    """Stands in for FFprobe and the inspection decode of one H.264 video."""

    def __init__(self, *, duration: float = 10.0, declared: int = 300, decoded: int = 300,
                 corrupt: bool = False) -> None:  # fmt: skip
        self.duration, self.declared, self.decoded, self.corrupt = (
            duration, declared, decoded, corrupt,
        )  # fmt: skip
        self.calls: list[list[str]] = []

    def run(self, argv, *, timeout, check_cancelled=None, **kw):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        if argv[0] == "ffprobe":
            stream = {"codec_type": "video", "codec_name": "h264", "width": 1920,
                      "height": 1080, "duration": str(self.duration),
                      "avg_frame_rate": "30/1", "nb_frames": str(self.declared)}  # fmt: skip
            out = json.dumps({"streams": [stream], "format": {"duration": str(self.duration)}})
            return RunResult(tuple(argv), 0, out, "", 0.01)
        if self.corrupt:
            failed = RunResult(tuple(argv), 1, "", "", 0.01)
            raise ProcessFailed("ffmpeg exited with status 1", failed)
        progress = Path(argv[argv.index("-progress") + 1])
        lines = ["frame=1", "progress=continue", f"frame={self.decoded}", "progress=end"]
        progress.write_text("\n".join(lines) + "\n")
        return RunResult(tuple(argv), 0, "", "", 0.01)


@pytest.mark.parametrize(
    "runner,status,reason",
    [
        (VideoRunner(), "verified", None),
        # Cut at a packet boundary: decodes cleanly but stops halfway.
        (VideoRunner(decoded=149), "rejected", "video is truncated"),
        (VideoRunner(corrupt=True), "rejected", "video could not be decoded"),
        (VideoRunner(duration=0.0, declared=0), "rejected", "video is empty or too long"),
        (VideoRunner(duration=31 * 60.0), "rejected", "video is empty or too long"),
    ],
)
def test_inspection_decodes_videos_within_the_supported_duration(
    db, seed, store, settings, monkeypatch, runner, status, reason
):
    monkeypatch.setattr(inspect_mod, "tool_available", lambda name: True)
    sup = Supervisor(db, store, settings, runner=runner)
    pid = seed.project()
    uid = make_upload(seed, store, pid, MP4, "video")
    assert drain(db, sup) == [("upload_inspection", "succeeded")]
    row = seed.one("SELECT status, rejection_code, media_metadata FROM uploads WHERE id = %s",
                   uid)  # fmt: skip
    assert row["status"] == status
    decodes = [c for c in runner.calls if c[0] == "ffmpeg"]
    if status == "verified":
        assert row["media_metadata"]["decoded"] is True
        assert row["media_metadata"]["decodedFrames"] == 300
    else:
        assert row["rejection_code"] == "INVALID_MEDIA"
        assert row["media_metadata"]["rejectionReason"] == reason
    if reason == "video is empty or too long":
        assert decodes == []  # never decoded
    else:
        (decode,) = decodes
        assert "-xerror" in decode and "-progress" in decode and decode[-1] == "-"


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
    seed.exec(
        "INSERT INTO tour_imports (id, project_id, created_by, kind, provider, model_id, "
        "canonical_url, listing_url) VALUES (%s, %s, 'user-1', 'matterport_embed', "
        "'matterport', 'GzAxTe2zAQs', 'https://my.matterport.com/show/?m=GzAxTe2zAQs', "
        "'https://www.homeq.se/lagenhet/1')",
        str(uuid.uuid4()), pid,
    )  # fmt: skip
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
    assert seed.one("SELECT count(*) AS n FROM tour_imports WHERE project_id = %s", pid)["n"] == 0
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


def test_layout_revisions_are_not_limited_by_spent_run_budget(db, seed, sup):
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    drain(db, sup)
    rev = seed.one("SELECT * FROM scene_revisions")
    seed.exec("UPDATE usage_ledger SET actual_cpu_seconds = 999999")
    before = seed.one("SELECT actual_cpu_seconds, state FROM usage_ledger")
    rev2 = _save_layout(seed, rev, pid, run_id, 2, rev["id"], copy.deepcopy(rev["layout"]))
    assert [d[1] for d in drain(db, sup)] == ["succeeded"] * 4
    assert (
        seed.one("SELECT validation_state FROM scene_revisions WHERE id = %s", rev2)[
            "validation_state"
        ]
        == "valid"
    )
    # The settled reconstruction ledger is not charged for layout processing.
    assert seed.one("SELECT actual_cpu_seconds, state FROM usage_ledger") == before


def test_run_with_unknown_recipe_digest_is_rejected(db, seed, sup):
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    seed.exec("UPDATE reconstruction_runs SET recipe_digest = 'sha256:stale' WHERE id = %s", run_id)
    assert drain(db, sup) == [("validating", "failed")]
    run = seed.one("SELECT status, error FROM reconstruction_runs WHERE id = %s", run_id)
    assert run["status"] == "failed" and run["error"]["code"] == "RECIPE_MISMATCH"


def test_layout_revision_of_a_scene_built_by_an_older_recipe(db, seed, sup):
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    drain(db, sup)
    rev = seed.one("SELECT * FROM scene_revisions")
    # A deployment changed the recipe after this scene was reconstructed.
    seed.exec("UPDATE reconstruction_runs SET recipe_digest = 'sha256:older' WHERE id = %s", run_id)
    rev2 = _save_layout(seed, rev, pid, run_id, 2, rev["id"], rev["layout"])
    assert [d[1] for d in drain(db, sup)] == ["succeeded"] * 4
    second = seed.one("SELECT validation_state, manifest FROM scene_revisions WHERE id = %s", rev2)
    assert second["validation_state"] == "valid"
    # The scene keeps the reconstruction's recipe as provenance.
    assert second["manifest"]["provenance"]["recipeDigest"] == "sha256:older"


def test_layout_job_out_of_retries_marks_the_revision_invalid(db, seed, store, settings, sup):
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    drain(db, sup)
    sup = Supervisor(db, store, settings, stages={**STAGES, "geometry_proposal": _transient_stage},
                     delay_fn=lambda attempt: 0.0)  # fmt: skip
    rev = seed.one("SELECT * FROM scene_revisions")
    rev2 = _save_layout(seed, rev, pid, run_id, 2, rev["id"], rev["layout"])
    seed.exec("UPDATE jobs SET max_attempts = 2 WHERE scene_revision_id = %s", rev2)
    assert [d[1] for d in drain(db, sup)] == ["retry_wait", "failed"]
    row = seed.one("SELECT validation_state, validation_findings FROM scene_revisions "
                   "WHERE id = %s", rev2)  # fmt: skip
    assert row["validation_state"] == "invalid"
    assert row["validation_findings"][0]["code"] == "LAYOUT_PROCESSING_FAILED"
    assert seed.one("SELECT status FROM reconstruction_runs")["status"] == "succeeded"


def test_concurrent_layout_revisions_do_not_share_intermediates(db, seed, sup):
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    drain(db, sup)
    rev = seed.one("SELECT * FROM scene_revisions")
    thin, thick = copy.deepcopy(rev["layout"]), copy.deepcopy(rev["layout"])
    for wall in thick["walls"]:
        wall["thickness"] = 0.2
    # Both revisions are queued before either runs, so both jobs start at fence 1.
    rev2 = _save_layout(seed, rev, pid, run_id, 2, rev["id"], thin)
    rev3 = _save_layout(seed, rev, pid, run_id, 3, rev["id"], thick)
    assert [d[1] for d in drain(db, sup)] == ["succeeded"] * 8
    m2, m3 = (
        seed.one("SELECT manifest FROM scene_revisions WHERE id = %s", r)["manifest"]
        for r in (rev2, rev3)
    )
    assert m2["assets"]["collision"]["sha256"] != m3["assets"]["collision"]["sha256"]
