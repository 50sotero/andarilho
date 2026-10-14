"""``geometry_proposal``: layout -> collision GLB in canonical world coordinates.

Layout source, in priority order:

1. a user layout from ``scene_revisions.layout`` when the job carries a
   ``scene_revision_id`` (layout-save pipeline);
2. the synthetic recipe's bundled two-room layout;
3. a single-room rectangle proposal from the normalized splat bounds (robust
   percentiles in the world frame).

For an initial run the stage idempotently creates the ``scenes`` row (unique
``run_id``) and revision 1 through a fenced write, so every later artifact can be
keyed by the immutable revision ID.
"""

from __future__ import annotations

import uuid
from typing import Any

from psycopg.types.json import Jsonb

from andarilho_worker.errors import StageError
from andarilho_worker.geometry.layout import collision_glb, rectangle_layout, validate_layout
from andarilho_worker.stages.base import StageContext, StageResult, finish
from andarilho_worker.stages.normalize import synthetic_layout


def _load_revision(conn: Any, ctx: StageContext) -> dict[str, Any] | None:
    return conn.execute(
        "SELECT id, scene_id, revision, layout_revision, layout, base_revision_id "
        "FROM scene_revisions WHERE id = %s AND project_id = %s",
        (ctx.job.scene_revision_id, ctx.project_id),
    ).fetchone()


def ensure_initial_revision(conn: Any, ctx: StageContext, layout: dict[str, Any]) -> dict[str, Any]:
    run = conn.execute(
        "SELECT created_by FROM reconstruction_runs WHERE id = %s AND project_id = %s",
        (ctx.run_id, ctx.project_id),
    ).fetchone()
    conn.execute(
        "INSERT INTO scenes (id, project_id, run_id, version) VALUES (%s, %s, %s, 1) "
        "ON CONFLICT (run_id) DO NOTHING",
        (str(uuid.uuid4()), ctx.project_id, ctx.run_id),
    )
    scene = conn.execute(
        "SELECT id FROM scenes WHERE run_id = %s AND project_id = %s", (ctx.run_id, ctx.project_id)
    ).fetchone()
    conn.execute(
        """
        INSERT INTO scene_revisions (id, scene_id, project_id, run_id, revision, layout_revision,
                                     layout, validation_state, approval_state, created_by)
        VALUES (%s, %s, %s, %s, 1, 1, %s, 'pending', 'draft', %s)
        ON CONFLICT (scene_id, revision) DO NOTHING
        """,
        (
            str(uuid.uuid4()),
            scene["id"],
            ctx.project_id,
            ctx.run_id,
            Jsonb(layout),
            run["created_by"],
        ),
    )
    return conn.execute(
        "SELECT id, scene_id, revision, layout_revision, layout FROM scene_revisions "
        "WHERE scene_id = %s AND revision = 1",
        (scene["id"],),
    ).fetchone()


def _mark_invalid(revision_id: str, findings: list[dict[str, str]]):
    def hook(conn: Any, job: Any) -> None:
        conn.execute(
            "UPDATE scene_revisions SET validation_state = 'invalid', validation_findings = %s "
            "WHERE id = %s AND validation_state = 'pending'",
            (Jsonb(findings), revision_id),
        )

    return hook


def run(ctx: StageContext) -> StageResult:
    revision: dict[str, Any] | None = None
    if ctx.job.scene_revision_id:
        revision = ctx.query(lambda conn: _load_revision(conn, ctx))
        if revision is None:
            raise StageError("MISSING_INPUT")
        layout, source = dict(revision["layout"]), "user"
    elif ctx.outputs().get("quality") == "synthetic":
        layout, source = synthetic_layout(), "synthetic"
    else:
        bounds = ctx.output("proposalBounds")
        layout = rectangle_layout(
            bounds["minXZ"], bounds["maxXZ"], elevation=bounds["floorY"], height=bounds["height"]
        )
        source = "proposal"
    findings = [f.as_dict() for f in validate_layout(layout)]
    if findings:
        on_fail = _mark_invalid(str(revision["id"]), findings) if revision else None
        raise StageError("INVALID_LAYOUT", details={"findings": findings[:20]}, on_fail=on_fail)
    if revision is None:
        revision = ctx.fenced(lambda conn: ensure_initial_revision(conn, ctx, layout))
        layout = dict(revision["layout"])  # a previous attempt may have created it
    ctx.progress("Building collision geometry.")
    glb = collision_glb(layout)
    key = ctx.attempt_key("collision.glb")
    ctx.store.put_bytes(key, glb, "model/gltf-binary")
    return finish(
        ctx,
        {"layoutSource": source, "collisionBytes": len(glb)},
        outputs={
            "sceneId": str(revision["scene_id"]),
            "sceneRevisionId": str(revision["id"]),
            "revision": int(revision["revision"]),
            "layoutRevision": int(revision["layout_revision"]),
            "layoutSource": source,
            "collisionKey": key,
        },
    )
