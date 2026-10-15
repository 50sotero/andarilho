"""``geometry_proposal``: layout -> collision GLB in canonical world coordinates.

Layout source, in priority order:

1. a user layout from ``scene_revisions.layout`` when the job carries a
   ``scene_revision_id`` (layout-save pipeline);
2. the synthetic recipe's bundled two-room layout;
3. the automatic proposal. Its rectangle is the capture's camera path with a margin, and
   each side snaps to the strongest wall-height density of the sparse points near it
   (``snap_rectangle``), or else reaches out to the wall beyond it
   (``extend_unsnapped_sides``). When the run's scene analysis completed, the rooms, doors
   and furniture it recognised are placed on that layout: rooms are segmented from the walk,
   doors and furniture are grounded in the posed keyframes. Any failure in that analysis
   keeps the single room, and the reason is recorded in the diagnostics.

For an initial run the stage idempotently creates the ``scenes`` row (unique
``run_id``) and revision 1 through a fenced write, so every later artifact can be
keyed by the immutable revision ID.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import numpy as np
from psycopg.types.json import Jsonb

from andarilho_worker.errors import Cancelled, LeaseLost, StageError, TransientError
from andarilho_worker.geometry.grounding import (
    floor_objects,
    ground_doors,
    ground_objects,
    views_from_analysis,
)
from andarilho_worker.geometry.layout import collision_glb, rectangle_layout, validate_layout
from andarilho_worker.geometry.rooms import (
    attach_doors,
    extend_unsnapped_sides,
    place_furniture,
    rooms_layout,
    segment_rooms,
    snap_rectangle,
)
from andarilho_worker.stages.base import StageContext, StageResult, finish
from andarilho_worker.stages.normalize import EVIDENCE_SCHEMA_VERSION, synthetic_layout

ANALYSIS_SCHEMA_VERSION = 1

# ``snap_rectangle`` lives with the other rectangle geometry; it stays importable from here.
__all__ = ["load_evidence", "proposed_layout", "run", "snap_rectangle"]


def load_evidence(ctx: StageContext) -> tuple[np.ndarray, np.ndarray] | None:
    """The run's sparse points and camera centres, or None when they are not usable.

    A key outside this run's prefix is never read. Evidence that is missing or malformed
    is ignored too: the proposal can always fall back to the camera-path rectangle, so it
    only costs the snap. Storage errors that are not missing objects still propagate.
    """
    key = ctx.outputs().get("evidenceKey")
    if not key or not str(key).startswith(f"tenants/{ctx.project_id}/runs/{ctx.run_id}/"):
        return None
    try:
        evidence = json.loads(ctx.store.get_bytes(str(key)))
        if evidence.get("schemaVersion") != EVIDENCE_SCHEMA_VERSION:
            return None
        points = np.asarray(evidence["points"], dtype=np.float64).reshape(-1, 3)
        cameras = np.asarray(evidence["cameras"], dtype=np.float64).reshape(-1, 3)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return points, cameras


def _load_analysis(ctx: StageContext) -> dict[str, Any] | None:
    """The run's completed scene analysis, or None when it is absent or unreadable.

    Like ``load_evidence``, a key outside this run's prefix is never read, and an artifact
    that is missing, not JSON or of another schema is ignored.
    """
    key = ctx.outputs().get("analysisKey")
    if not key or not str(key).startswith(f"tenants/{ctx.project_id}/runs/{ctx.run_id}/"):
        return None
    try:
        analysis = json.loads(ctx.store.get_bytes(str(key)))
    except (OSError, ValueError):
        return None
    if not isinstance(analysis, dict) or analysis.get("schemaVersion") != ANALYSIS_SCHEMA_VERSION:
        return None
    if analysis.get("status") != "complete":
        return None
    return analysis


def _analysed_layout(
    analysis: dict[str, Any],
    evidence: tuple[np.ndarray, np.ndarray] | None,
    bounds: dict[str, Any],
    spawn: dict[str, Any] | None,
    plain: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The plain room with the analysis's rooms, doors and furniture, and what was placed.

    Without evidence the keyframes' poses stand in for the camera path, and there are no
    points to snap to. The result must validate, or the analysis is not used at all.
    """
    reply = analysis["reply"]
    if not isinstance(reply, dict):
        raise ValueError("the analysis has no reply")
    keyframes = analysis["keyframes"]
    views = views_from_analysis(analysis)
    if evidence is not None:
        points, cameras = evidence
    else:
        points = np.empty((0, 3), dtype=np.float64)
        cameras = np.array([views[int(kf["frame"])].centre() for kf in keyframes])
        cameras = cameras.reshape(-1, 3)
    rooms = segment_rooms(keyframes, reply["frames"], cameras[:, [0, 2]])
    layout = plain
    room_count = 1
    if len(rooms) >= 2:
        layout = rooms_layout(
            rooms,
            points=points,
            cameras=cameras,
            height=bounds["height"],
            floor_y=bounds["floorY"],
            spawn=spawn,
        )
        room_count = len(rooms)
    layout = attach_doors(layout, ground_doors(analysis, views), [room.id for room in rooms])
    objects = ground_objects(analysis, views)
    ungrounded = len(floor_objects(analysis)) - len(objects)
    layout, dropped = place_furniture(layout, objects, cameras)
    if validate_layout(layout):
        raise ValueError("the analysed layout does not validate")
    counts = {
        "rooms": room_count,
        "doors": len(layout["openings"]),
        "furniture": len(layout["furniture"]),
        "droppedFurniture": ungrounded + dropped,
    }
    return layout, counts


def proposed_layout(ctx: StageContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """The automatic layout from the run's ``normalizing`` and ``scene_analysis`` outputs.

    Returns the layout and its diagnostics: the sides that snapped and that extended, the
    rooms, doors and furniture placed, the analysis status, and ``analysisError`` (the
    exception's class name) when the analysis could not be used. Shared with the smoke
    runner, which has no database.
    """
    bounds = ctx.output("proposalBounds")
    min_xz, max_xz = bounds["minXZ"], bounds["maxXZ"]
    spawn = ctx.outputs().get("proposalSpawn")
    snapped: dict[str, float] = {}
    extended: dict[str, float] = {}
    evidence = load_evidence(ctx)
    if evidence is not None:
        min_xz, max_xz, snapped = snap_rectangle(
            min_xz, max_xz, points=evidence[0], cameras=evidence[1], height=bounds["height"]
        )
        min_xz, max_xz, extended = extend_unsnapped_sides(
            min_xz,
            max_xz,
            points=evidence[0],
            cameras=evidence[1],
            height=bounds["height"],
            snapped=set(snapped),
        )
    layout = rectangle_layout(
        min_xz, max_xz, elevation=bounds["floorY"], height=bounds["height"], spawn=spawn
    )
    status = str(ctx.outputs().get("analysisStatus") or "absent")
    diagnostics: dict[str, Any] = {
        "snappedSides": snapped,
        "extendedSides": extended,
        "rooms": 1,
        "doors": 0,
        "furniture": 0,
        "droppedFurniture": 0,
        "analysisStatus": status,
    }
    if status != "complete":
        return layout, diagnostics
    try:
        analysis = _load_analysis(ctx)
        if analysis is None:
            return layout, diagnostics
        analysed, counts = _analysed_layout(analysis, evidence, bounds, spawn, layout)
    except (TransientError, Cancelled, LeaseLost):
        raise  # a storage outage retries the stage; only an unusable analysis falls back
    except Exception as exc:
        # The room is still a valid proposal: an analysis that cannot be used never fails it.
        diagnostics["analysisError"] = type(exc).__name__
        return layout, diagnostics
    diagnostics.update(counts)
    return analysed, diagnostics


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
    diagnostics: dict[str, Any] = {}
    if ctx.job.scene_revision_id:
        revision = ctx.query(lambda conn: _load_revision(conn, ctx))
        if revision is None:
            raise StageError("MISSING_INPUT")
        layout, source = dict(revision["layout"]), "user"
    elif ctx.outputs().get("quality") == "synthetic":
        layout, source = synthetic_layout(), "synthetic"
    else:
        layout, diagnostics = proposed_layout(ctx)
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
        {"layoutSource": source, "collisionBytes": len(glb), "diagnostics": diagnostics},
        outputs={
            "sceneId": str(revision["scene_id"]),
            "sceneRevisionId": str(revision["id"]),
            "revision": int(revision["revision"]),
            "layoutRevision": int(revision["layout_revision"]),
            "layoutSource": source,
            "collisionKey": key,
        },
    )
