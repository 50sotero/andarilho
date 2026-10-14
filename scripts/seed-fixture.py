"""Seed the synthetic two-room fixture as a reviewable project for one user.

Usage (inside the API container, README section 14):
  python /app/scripts/seed-fixture.py --subject <supabase-user-id>
Locally:
  uv run --project services/api python scripts/seed-fixture.py --subject <id>

Reads DATABASE_URL and S3_* from the API settings. It uploads the committed
fixture assets under ``tenants/{project_id}/scenes/{scene_revision_id}/``,
rewrites the manifest's IDs and keys to the new UUIDs (bytes and hashes are
unchanged and verified against the files), and inserts a project, owner
membership, capture set (no inputs), a succeeded synthetic run, a scene and a
valid draft revision with ``approved_collision`` navigation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_LOCAL_SRC = ROOT / "services" / "api" / "src"
if _LOCAL_SRC.is_dir() and str(_LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(_LOCAL_SRC))

from andarilho_api.db import make_engine, make_session_factory  # noqa: E402
from andarilho_api.domain.canonical import sha256_hex  # noqa: E402
from andarilho_api.domain.recipes import get_recipe  # noqa: E402
from andarilho_api.models import (  # noqa: E402
    CaptureSet,
    Project,
    ProjectMember,
    ReconstructionRun,
    Scene,
    SceneRevision,
    UsageLedger,
)
from andarilho_api.schemas.scene import Layout, SceneManifest  # noqa: E402
from andarilho_api.settings import Settings, get_settings  # noqa: E402
from andarilho_api.storage import Storage, scene_prefix  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

DEFAULT_FIXTURES = ROOT / "packages" / "scene-contract" / "fixtures"
FIXTURE_NAME = "synthetic-two-room"
# Local file suffix for each manifest asset slot.
ASSET_FILES = {
    "render": ".splat.ply",
    "collision": ".collision.glb",
    "poster": ".poster.webp",
}
CONTENT_TYPES = {
    "ply": "application/octet-stream",
    "sog": "application/octet-stream",
    "spz": "application/octet-stream",
    "glb": "model/gltf-binary",
    "webp": "image/webp",
    "png": "image/png",
    "jpeg": "image/jpeg",
}


class SeedError(Exception):
    pass


def load_fixture(fixtures_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
    manifest_path = fixtures_dir / f"{FIXTURE_NAME}.manifest.json"
    layout_path = fixtures_dir / f"{FIXTURE_NAME}.layout.json"
    for path in (manifest_path, layout_path):
        if not path.is_file():
            raise SeedError(f"missing fixture file: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    Layout.model_validate(layout)
    blobs: dict[str, bytes] = {}
    for slot, asset in manifest.get("assets", {}).items():
        if asset is None:
            continue
        suffix = ASSET_FILES.get(slot)
        path = fixtures_dir / f"{FIXTURE_NAME}{suffix}" if suffix else None
        if path is None or not path.is_file():
            raise SeedError(f"missing fixture asset for {slot}")
        data = path.read_bytes()
        if len(data) != asset["bytes"] or hashlib.sha256(data).hexdigest() != asset["sha256"]:
            raise SeedError(f"fixture asset {slot} does not match the manifest bytes/sha256")
        blobs[slot] = data
    return manifest, layout, blobs


def rewrite_manifest(
    manifest: dict[str, Any],
    *,
    project_id: uuid.UUID,
    scene_id: uuid.UUID,
    revision_id: uuid.UUID,
    run_id: uuid.UUID,
    capture_set_id: uuid.UUID,
) -> dict[str, Any]:
    """Replace fixture IDs/keys with real ones; everything else is kept verbatim."""
    doc = json.loads(json.dumps(manifest))
    doc["sceneId"] = str(scene_id)
    doc["sceneRevisionId"] = str(revision_id)
    doc["runId"] = str(run_id)
    doc["provenance"]["captureSetId"] = str(capture_set_id)
    prefix = scene_prefix(project_id, revision_id)
    for asset in doc["assets"].values():
        if asset is not None:
            asset["key"] = prefix + asset["key"].rsplit("/", 1)[-1]
    validated = SceneManifest.model_validate(doc)
    if validated.navigation.mode != "approved_collision":
        raise SeedError("fixture manifest must use approved_collision navigation")
    return doc


def seed(
    *,
    settings: Settings,
    subject: str,
    fixtures_dir: Path = DEFAULT_FIXTURES,
    title: str = "Synthetic two-room fixture",
    session_factory: sessionmaker[Session] | None = None,
    storage: Storage | None = None,
) -> dict[str, str]:
    if not subject or len(subject) > 255:
        raise SeedError("--subject must be a non-empty provider subject (max 255 chars)")
    manifest, layout, blobs = load_fixture(fixtures_dir)
    project_id, scene_id, revision_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    run_id, capture_set_id = uuid.uuid4(), uuid.uuid4()
    doc = rewrite_manifest(
        manifest,
        project_id=project_id,
        scene_id=scene_id,
        revision_id=revision_id,
        run_id=run_id,
        capture_set_id=capture_set_id,
    )

    storage = storage or Storage(settings)
    for slot, data in blobs.items():
        asset = doc["assets"][slot]
        storage.put_object(
            asset["key"], data, CONTENT_TYPES.get(asset["format"], "application/octet-stream")
        )

    engine = None
    if session_factory is None:
        engine = make_engine(settings.database_url)
        session_factory = make_session_factory(engine)
    recipe = get_recipe("synthetic")
    now = datetime.now(UTC)
    try:
        with session_factory() as db:
            db.add(Project(id=project_id, title=title, owner_subject=subject, lifecycle="active"))
            db.flush()
            db.add(ProjectMember(project_id=project_id, subject=subject, role="owner"))
            db.add(
                CaptureSet(
                    id=capture_set_id,
                    project_id=project_id,
                    created_by=subject,
                    inputs=[],
                    capture_metadata={"fixture": FIXTURE_NAME},
                    content_sha256=sha256_hex([]),
                )
            )
            db.flush()
            db.add(
                ReconstructionRun(
                    id=run_id,
                    project_id=project_id,
                    capture_set_id=capture_set_id,
                    created_by=subject,
                    recipe_name=recipe.name,
                    recipe_digest=recipe.digest,
                    quality="synthetic",
                    include_supplementary_photos=False,
                    resource_budget=recipe.budget(settings.run_max_seconds),
                    status="succeeded",
                    finished_at=now,
                )
            )
            db.flush()
            db.add(
                UsageLedger(
                    project_id=project_id,
                    run_id=run_id,
                    reserved_gpu_seconds=0,
                    reserved_cpu_seconds=0,
                    state="settled",
                    settled_at=now,
                )
            )
            db.add(Scene(id=scene_id, project_id=project_id, run_id=run_id))
            db.flush()
            db.add(
                SceneRevision(
                    id=revision_id,
                    scene_id=scene_id,
                    project_id=project_id,
                    run_id=run_id,
                    revision=int(doc["revision"]),
                    layout_revision=int(doc["navigation"]["layoutRevision"]),
                    layout=layout,
                    manifest=doc,
                    manifest_sha256=sha256_hex(doc),
                    validation_state="valid",
                    approval_state="draft",
                    created_by=subject,
                )
            )
            db.flush()
            scene = db.get(Scene, scene_id)
            project = db.get(Project, project_id)
            assert scene is not None and project is not None
            scene.current_draft_revision_id = revision_id
            project.active_draft_revision_id = revision_id
            db.commit()
    finally:
        if engine is not None:
            engine.dispose()
    return {
        "projectId": str(project_id),
        "sceneId": str(scene_id),
        "sceneRevisionId": str(revision_id),
        "runId": str(run_id),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_subject = os.environ.get("SEED_SUBJECT") or None
    parser.add_argument(
        "--subject",
        required=default_subject is None,
        default=default_subject,
        help="Supabase user id (JWT sub claim); defaults to $SEED_SUBJECT",
    )
    parser.add_argument("--title", default="Synthetic two-room fixture")
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES)
    args = parser.parse_args(argv)
    try:
        result = seed(
            settings=get_settings(),
            subject=args.subject,
            fixtures_dir=args.fixtures_dir,
            title=args.title,
        )
    except SeedError as exc:
        print(f"seed-fixture: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
