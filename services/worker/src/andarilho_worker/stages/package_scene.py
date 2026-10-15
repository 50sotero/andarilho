"""``packaging``: write the viewer derivatives and the persistent manifest.

Everything lands under ``tenants/{project}/scenes/{scene_revision_id}/`` and is
written with :func:`put_immutable` (an existing object with different bytes is a
conflict, never overwritten). The render asset is the textured model from
``texturing``, copied to this revision's prefix, so every revision keeps its own model.
Navigation is only ``approved_collision`` when the layout came from a user save
(``user_confirmed_trace``) or the synthetic fixture; automatic proposals are
``inspection_only``.

The poster, quality and scale status come from this run's outputs in both pipelines,
so a layout save reads the same run facts as the initial run did.
"""

from __future__ import annotations

from typing import Any

from andarilho_worker.artifacts import (
    MAX_POSTER_BYTES,
    MAX_RENDER_BYTES,
    put_immutable,
    scene_prefix,
)
from andarilho_worker.errors import StageError
from andarilho_worker.manifest import asset_entry, build_manifest
from andarilho_worker.stages.base import StageContext, StageResult, finish
from andarilho_worker.validation import canonical_json

GEOMETRY_SOURCE = {
    "user": "user_confirmed_trace",
    "synthetic": "synthetic_fixture",
    "proposal": "proposal",
}
MODEL_FORMAT = "glb"
MODEL_TYPE = "model/gltf-binary"
POSTER_FORMAT = "webp"
POSTER_TYPE = "image/webp"
# Share of walls, floors and ceilings that some frame saw, below which the model warns.
SEEN_THRESHOLD = 0.99
SYNTHETIC_QUALITY = {
    "registeredViews": 0,
    "selectedViews": 0,
    "disconnectedComponents": 0,
    "warnings": ["Synthetic scene: no capture was reconstructed."],
}


def _load(conn: Any, ctx: StageContext, revision_id: str) -> dict[str, Any]:
    revision = conn.execute(
        "SELECT id, scene_id, revision, layout_revision, layout FROM scene_revisions "
        "WHERE id = %s AND project_id = %s",
        (revision_id, ctx.project_id),
    ).fetchone()
    run = conn.execute(
        "SELECT capture_set_id FROM reconstruction_runs WHERE id = %s AND project_id = %s",
        (ctx.run_id, ctx.project_id),
    ).fetchone()
    return {"revision": revision, "run": run}


def _poster_bytes(ctx: StageContext, key: str) -> bytes:
    """The poster made by ``preparing``, as the texturing stage passes it on."""
    info = ctx.store.head(key) if key.startswith(f"tenants/{ctx.project_id}/") else None
    if info is None:
        raise StageError("MISSING_INPUT", details={"missing": "poster"})
    if info.size > MAX_POSTER_BYTES:
        raise StageError("BUDGET_EXHAUSTED", "The scene poster is too large to package.")
    return ctx.store.get_bytes(key)


def unseen_warning(texture_coverage: dict[str, float]) -> str | None:
    """A warning when much of the model was never seen in the video, else None."""
    seen = float(texture_coverage["all"])
    if seen >= SEEN_THRESHOLD:
        return None
    percent = max(1, round((1 - seen) * 100))
    return (
        f"About {percent}% of the walls, floor and ceiling were not seen in the video "
        "and are shown in plain grey."
    )


def run(ctx: StageContext) -> StageResult:
    out = ctx.outputs()
    revision_id = str(ctx.output("sceneRevisionId"))
    data = ctx.query(lambda conn: _load(conn, ctx, revision_id))
    revision, run_row = data["revision"], data["run"]
    if revision is None or run_row is None:
        raise StageError("MISSING_INPUT")
    prefix = scene_prefix(ctx.project_id, revision_id)
    model_source = str(ctx.output("modelKey"))
    if not model_source.startswith(f"tenants/{ctx.project_id}/"):
        raise StageError("MISSING_INPUT")

    quality = out.get("coverage") or SYNTHETIC_QUALITY
    warnings = [*quality.get("warnings", []), *out.get("frameWarnings", [])]
    unseen = unseen_warning(ctx.output("textureCoverage"))
    if unseen is not None:
        warnings.append(unseen)
    quality = {**quality, "warnings": warnings}
    calibrated = (revision["layout"] or {}).get("calibration")
    scale_status = "user_measured" if calibrated else str(ctx.output("scaleStatus"))

    ctx.progress("Packaging scene files.")
    source = ctx.store.head(model_source)
    if source is None:
        raise StageError("MISSING_INPUT", details={"missing": "render asset"})
    if source.size > MAX_RENDER_BYTES:
        raise StageError("BUDGET_EXHAUSTED", "The scene is too large to package.")
    # Streamed through scratch and hashed from disk, never held in memory.
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    model_path = ctx.scratch / f"scene.{MODEL_FORMAT}"
    model_sha = ctx.store.download(model_source, model_path)
    model_key = prefix + f"model.{MODEL_FORMAT}"
    model_info = put_immutable(ctx.store, model_key, model_path, MODEL_TYPE, sha256=model_sha)
    render = asset_entry(model_key, model_path, MODEL_FORMAT, sha256=model_sha)
    glb = ctx.store.get_bytes(str(ctx.output("collisionKey")))
    collision_key = prefix + "collision.glb"
    collision_info = put_immutable(ctx.store, collision_key, glb, "model/gltf-binary")
    poster, poster_info = None, None
    poster_source = out.get("posterKey")
    if poster_source:
        poster_key = prefix + f"poster.{POSTER_FORMAT}"
        image = _poster_bytes(ctx, str(poster_source))
        poster_info = put_immutable(ctx.store, poster_key, image, POSTER_TYPE)
        poster = asset_entry(poster_key, image, POSTER_FORMAT)

    layout_source = str(
        out.get("layoutSource", "user" if ctx.job.scene_revision_id else "proposal")
    )
    manifest = build_manifest(
        scene_id=str(revision["scene_id"]),
        scene_revision_id=revision_id,
        revision=int(revision["revision"]),
        run_id=str(ctx.run_id),
        layout=revision["layout"],
        layout_revision=int(revision["layout_revision"]),
        render=render,
        collision=asset_entry(collision_key, glb, "glb"),
        poster=poster,
        scale_status=scale_status,
        geometry_source=GEOMETRY_SOURCE[layout_source],
        recipe_digest=ctx.recipe_digest,
        capture_set_id=str(run_row["capture_set_id"]),
        quality={k: v for k, v in quality.items() if k != "registeredRatio"},
    )
    manifest_blob = canonical_json(manifest)
    manifest_key = prefix + "manifest.json"
    manifest_info = put_immutable(ctx.store, manifest_key, manifest_blob, "application/json")
    files = {
        "render": {**render, "version": model_info.version},
        "collision": {**asset_entry(collision_key, glb, "glb"), "version": collision_info.version},
        "manifest": {**asset_entry(manifest_key, manifest_blob, "json"),
                     "version": manifest_info.version},
    }  # fmt: skip
    if poster is not None and poster_info is not None:
        files["poster"] = {**poster, "version": poster_info.version}
    return finish(
        ctx,
        {"files": files},
        outputs={"manifestKey": manifest_key, "packagedFiles": files},
    )
