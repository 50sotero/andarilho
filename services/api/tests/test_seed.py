"""scripts/seed-fixture.py seeds a reviewable, deliverable fixture scene."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import uuid
from pathlib import Path

import pytest
from conftest import BUCKET
from sqlalchemy import select

from andarilho_api.models import Project, ReconstructionRun, SceneRevision
from andarilho_api.schemas.scene import SceneManifest

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "packages" / "scene-contract" / "fixtures"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "seed_fixture", REPO / "scripts" / "seed-fixture.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fixture_dir(tmp_path: Path) -> Path:
    """Use the committed fixture when complete; otherwise synthesize a stand-in."""
    if (FIXTURES / "synthetic-two-room.manifest.json").is_file() and (
        FIXTURES / "synthetic-two-room.splat.ply"
    ).is_file():
        return FIXTURES
    target = tmp_path / "fixtures"
    target.mkdir()
    shutil.copy(FIXTURES / "synthetic-two-room.layout.json", target)
    from factories import manifest

    blobs = {"render": b"ply-bytes", "collision": b"glb-bytes"}
    doc = manifest(
        "fixture-project",
        "fixture-scene",
        "fixture-revision",
        "fixture-run",
        "fixture-capture",
        assets={"scene": blobs["render"], "collision": blobs["collision"]},
    )
    del doc["assets"]["poster"]
    (target / "synthetic-two-room.manifest.json").write_text(json.dumps(doc))
    (target / "synthetic-two-room.splat.ply").write_bytes(blobs["render"])
    (target / "synthetic-two-room.collision.glb").write_bytes(blobs["collision"])
    return target


def test_seed_fixture_end_to_end(app, client, auth, db, s3, tmp_path):
    module = _load_script()
    fixtures = _fixture_dir(tmp_path)
    result = module.seed(
        settings=app.state.settings,
        subject="user-a",
        fixtures_dir=fixtures,
        session_factory=app.state.session_factory,
        storage=app.state.storage,
    )
    pid, rid = result["projectId"], result["sceneRevisionId"]
    revision = db.get(SceneRevision, uuid.UUID(rid))
    assert revision.validation_state == "valid"
    manifest = SceneManifest.model_validate(revision.manifest)
    assert manifest.scene_revision_id == rid and manifest.navigation.mode == "approved_collision"
    original = json.loads((fixtures / "synthetic-two-room.manifest.json").read_text())
    for slot, asset in revision.manifest["assets"].items():
        assert asset["key"].startswith(f"tenants/{pid}/scenes/{rid}/")
        assert asset["sha256"] == original["assets"][slot]["sha256"]
        body = s3.get_object(Bucket=BUCKET, Key=asset["key"])["Body"].read()
        assert hashlib.sha256(body).hexdigest() == asset["sha256"]
    run = db.execute(select(ReconstructionRun)).scalar_one()
    assert run.quality == "synthetic" and run.status == "succeeded"
    project = db.get(Project, uuid.UUID(pid))
    assert str(project.active_draft_revision_id) == rid

    # The seeded scene is deliverable, approvable and shareable through the API.
    scene_id = result["sceneId"]
    assert client.get(f"/v1/projects/{pid}", headers=auth()).json()["scene"]["id"] == scene_id
    delivery = client.post(
        f"/v1/scenes/{scene_id}/delivery", json={"sceneRevisionId": rid}, headers=auth()
    )
    assert delivery.status_code == 200, delivery.text
    assert {"render", "collision"} <= set(delivery.json()["assetUrls"])
    approve = client.post(
        f"/v1/scenes/{scene_id}/approve",
        json={"sceneRevisionId": rid},
        headers=auth(**{"If-Match": '"1"'}),
    )
    assert approve.status_code == 200
    # Another user cannot see the seeded project.
    assert client.get(f"/v1/projects/{pid}", headers=auth("user-b")).status_code == 404


def test_seed_rejects_tampered_assets(app, tmp_path):
    module = _load_script()
    source = _fixture_dir(tmp_path)
    target = tmp_path / "tampered"
    shutil.copytree(source, target)
    (target / "synthetic-two-room.collision.glb").write_bytes(b"tampered")
    with pytest.raises(module.SeedError):
        module.seed(
            settings=app.state.settings,
            subject="user-a",
            fixtures_dir=target,
            session_factory=app.state.session_factory,
            storage=app.state.storage,
        )
