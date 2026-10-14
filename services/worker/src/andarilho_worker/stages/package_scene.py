"""``packaging``: write the viewer derivatives and the persistent manifest.

Everything lands under ``tenants/{project}/scenes/{scene_revision_id}/`` and is
written with :func:`put_immutable` (an existing object with different bytes is a
conflict, never overwritten). Navigation is only ``approved_collision`` when the
layout came from a user save (``user_confirmed_trace``) or the synthetic fixture;
automatic proposals are ``inspection_only``.
"""

from __future__ import annotations

from typing import Any

from andarilho_worker.artifacts import put_immutable, scene_prefix
from andarilho_worker.errors import StageError
from andarilho_worker.manifest import asset_entry, build_manifest
from andarilho_worker.stages.base import StageContext, StageResult, finish
from andarilho_worker.validation import canonical_json

GEOMETRY_SOURCE = {
    "user": "user_confirmed_trace",
    "synthetic": "synthetic_fixture",
    "proposal": "proposal",
}
CONTENT_TYPES = {"ply": "application/octet-stream", "sog": "application/octet-stream",
                 "spz": "application/octet-stream"}  # fmt: skip


def _load(conn: Any, ctx: StageContext, revision_id: str) -> dict[str, Any]:
    revision = conn.execute(
        "SELECT id, scene_id, revision, layout_revision, layout, base_revision_id "
        "FROM scene_revisions WHERE id = %s AND project_id = %s",
        (revision_id, ctx.project_id),
    ).fetchone()
    run = conn.execute(
        "SELECT capture_set_id FROM reconstruction_runs WHERE id = %s AND project_id = %s",
        (ctx.run_id, ctx.project_id),
    ).fetchone()
    previous = None
    if revision is not None and ctx.job.scene_revision_id:
        previous = conn.execute(
            """
            SELECT manifest FROM scene_revisions
            WHERE project_id = %s AND scene_id = %s AND manifest IS NOT NULL
              AND (id = %s OR revision < %s)
            ORDER BY (id = %s) DESC, revision DESC LIMIT 1
            """,
            (
                ctx.project_id,
                revision["scene_id"],
                revision["base_revision_id"],
                revision["revision"],
                revision["base_revision_id"],
            ),
        ).fetchone()
    original = None
    if revision is not None and ctx.job.scene_revision_id:
        # The first packaged revision carries the run's original render frame.
        original = conn.execute(
            """
            SELECT manifest FROM scene_revisions
            WHERE project_id = %s AND scene_id = %s AND manifest IS NOT NULL
            ORDER BY revision ASC LIMIT 1
            """,
            (ctx.project_id, revision["scene_id"]),
        ).fetchone()
    return {"revision": revision, "run": run, "previous": previous, "original": original}


def scaled_render_transform(matrix: list[float], scale: float) -> list[float]:
    """Return ``S(scale) * matrix`` for a column-major 4x4 matrix (uniform, about the origin)."""
    out = [float(v) for v in matrix]
    for column in range(4):
        for row in range(3):
            out[column * 4 + row] *= scale
    return out


def run(ctx: StageContext) -> StageResult:
    out = ctx.outputs()
    revision_id = str(ctx.output("sceneRevisionId"))
    data = ctx.query(lambda conn: _load(conn, ctx, revision_id))
    revision, run_row = data["revision"], data["run"]
    if revision is None or run_row is None:
        raise StageError("MISSING_INPUT")
    prefix = scene_prefix(ctx.project_id, revision_id)

    if ctx.job.scene_revision_id:
        previous = data["previous"]
        if previous is None:
            raise StageError("MISSING_INPUT", details={"missing": "previous revision manifest"})
        prev = previous["manifest"]
        source_key = prev["assets"]["render"]["key"]
        if not source_key.startswith(f"tenants/{ctx.project_id}/"):
            raise StageError("MISSING_INPUT")
        render_format = prev["assets"]["render"]["format"]
        matrix = prev["worldFromDecodedRender"]
        scale_status = prev["scaleStatus"]
        quality = prev["quality"]
        calibration = (revision["layout"] or {}).get("calibration")
        if calibration:
            # The layout was rescaled to a known distance: scale the splats with it so
            # collision and appearance stay aligned, and record the measured scale.
            original = (data["original"] or previous)["manifest"]
            matrix = scaled_render_transform(
                original["worldFromDecodedRender"], float(calibration["renderScale"])
            )
            scale_status = "user_measured"
    else:
        source_key = str(ctx.output("renderKey"))
        render_format = str(ctx.output("renderFormat"))
        matrix = ctx.output("worldFromDecodedRender")
        scale_status = str(ctx.output("scaleStatus"))
        quality = out.get("coverage") or {
            "registeredViews": 0,
            "selectedViews": 0,
            "disconnectedComponents": 0,
            "warnings": ["Synthetic scene: no capture was reconstructed."],
        }
    ctx.progress("Packaging scene files.")
    render_bytes = ctx.store.get_bytes(source_key)
    render_key = prefix + f"scene.{render_format}"
    render_info = put_immutable(ctx.store, render_key, render_bytes, CONTENT_TYPES[render_format])
    glb = ctx.store.get_bytes(str(ctx.output("collisionKey")))
    collision_key = prefix + "collision.glb"
    collision_info = put_immutable(ctx.store, collision_key, glb, "model/gltf-binary")

    source = str(out.get("layoutSource", "user" if ctx.job.scene_revision_id else "proposal"))
    manifest = build_manifest(
        scene_id=str(revision["scene_id"]),
        scene_revision_id=revision_id,
        revision=int(revision["revision"]),
        run_id=str(ctx.run_id),
        layout=revision["layout"],
        layout_revision=int(revision["layout_revision"]),
        render=asset_entry(render_key, render_bytes, render_format),
        collision=asset_entry(collision_key, glb, "glb"),
        poster=None,
        world_from_decoded_render=matrix,
        scale_status=scale_status,
        geometry_source=GEOMETRY_SOURCE[source],
        recipe_digest=ctx.recipe_digest,
        capture_set_id=str(run_row["capture_set_id"]),
        quality={k: v for k, v in quality.items() if k != "registeredRatio"},
    )
    manifest_blob = canonical_json(manifest)
    manifest_key = prefix + "manifest.json"
    manifest_info = put_immutable(ctx.store, manifest_key, manifest_blob, "application/json")
    files = {
        "render": {**asset_entry(render_key, render_bytes, render_format),
                   "version": render_info.version},
        "collision": {**asset_entry(collision_key, glb, "glb"), "version": collision_info.version},
        "manifest": {**asset_entry(manifest_key, manifest_blob, "json"),
                     "version": manifest_info.version},
    }  # fmt: skip
    return finish(
        ctx,
        {"files": files},
        outputs={"manifestKey": manifest_key, "packagedFiles": files},
    )
