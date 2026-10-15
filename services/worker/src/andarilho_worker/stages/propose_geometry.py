"""``geometry_proposal``: layout -> collision GLB in canonical world coordinates.

Layout source, in priority order:

1. a user layout from ``scene_revisions.layout`` when the job carries a
   ``scene_revision_id`` (layout-save pipeline);
2. the synthetic recipe's bundled two-room layout;
3. a single-room rectangle around the capture's camera path, with the spawn where the
   walk started. When the run's capture evidence is readable, each side of that
   rectangle snaps to the strongest wall-height density of the sparse points near it
   (``snap_rectangle``). Doors and multi-room outlines are left to review.

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

from andarilho_worker.errors import StageError
from andarilho_worker.geometry.layout import collision_glb, rectangle_layout, validate_layout
from andarilho_worker.stages.base import StageContext, StageResult, finish
from andarilho_worker.stages.normalize import EVIDENCE_SCHEMA_VERSION, synthetic_layout

# How far a side may move, and what a wall must look like to move it. The points are
# histogrammed in bins across the window, so a wall shows as a peak above the clutter.
SNAP_RANGE = 0.75  # metres either side of a side, where its wall is looked for
SNAP_BIN = 0.05  # metres per histogram bin
SNAP_MIN_POINTS = 8  # the peak bin holds at least this many points
SNAP_PEAK_RATIO = 2.0  # ... and at least this many times the median non-empty bin
SNAP_MIN_SPAN = 1.5  # the rectangle never shrinks below this on either axis, metres
SNAP_CLEARANCE = 0.3  # every camera keeps at least this distance from a moved side
# Wall-height evidence: above the floor clutter and below the ceiling.
WALL_MIN_HEIGHT = 0.3  # metres above the floor
WALL_MAX_SHARE = 0.85  # of the room height

# (side, coordinate of its normal in (x, y, z): 0 for X or 2 for Z, then 0 for the minimum
# or 1 for the maximum of that axis).
SIDES: tuple[tuple[str, int, int], ...] = (
    ("west", 0, 0),
    ("east", 0, 1),
    ("south", 2, 0),
    ("north", 2, 1),
)


def _density_peak(values: np.ndarray, side: float) -> float | None:
    """The median of the densest bin of ``values`` near ``side``, when it reads as a wall.

    The bins span SNAP_RANGE either side of ``side``. The peak must hold SNAP_MIN_POINTS
    points and SNAP_PEAK_RATIO times the median non-empty bin, so a flat scatter of
    clutter never moves a side.
    """
    bins = round(2 * SNAP_RANGE / SNAP_BIN)
    index = np.floor((values - (side - SNAP_RANGE)) / SNAP_BIN)
    index = np.clip(index, 0, bins - 1).astype(np.int64)
    counts = np.bincount(index, minlength=bins)
    peak = int(np.argmax(counts))
    if counts[peak] < SNAP_MIN_POINTS:
        return None
    if counts[peak] < SNAP_PEAK_RATIO * float(np.median(counts[counts > 0])):
        return None
    return float(np.median(values[index == peak]))


def snap_rectangle(
    min_xz: Any,
    max_xz: Any,
    *,
    points: np.ndarray,
    cameras: np.ndarray,
    height: float,
) -> tuple[list[float], list[float], dict[str, float]]:
    """Move each side of a rectangle onto the wall-height density peak of the points near it.

    ``points`` (x, y, z) are the evidence's sparse points and ``cameras`` its camera
    centres, both in the canonical world. A side moves only when its peak passes
    ``_density_peak``, the rectangle stays at least SNAP_MIN_SPAN wide and every camera
    keeps SNAP_CLEARANCE from the new side. Sides are tried in ``SIDES`` order, each
    against the rectangle as already snapped. Returns the new corners and the shift in
    metres of each side that moved.
    """
    lo = np.asarray(min_xz, dtype=np.float64).copy()
    hi = np.asarray(max_xz, dtype=np.float64).copy()
    moved: dict[str, float] = {}
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cameras = np.asarray(cameras, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0 or len(cameras) == 0:
        return lo.tolist(), hi.tolist(), moved
    band = (points[:, 1] >= WALL_MIN_HEIGHT) & (points[:, 1] <= WALL_MAX_SHARE * height)
    walls = points[band]
    for name, axis, which in SIDES:
        # The rectangle is indexed (x, z) and the points (x, y, z): ``k`` is this side's
        # rectangle index, ``other`` the other horizontal one, ``along`` the points' one.
        k, other, along = axis // 2, 1 - axis // 2, 2 - axis
        side = float((lo if which == 0 else hi)[k])
        in_extent = (walls[:, along] >= lo[other]) & (walls[:, along] <= hi[other])
        near = walls[in_extent & (np.abs(walls[:, axis] - side) <= SNAP_RANGE)]
        new = _density_peak(near[:, axis], side)
        if new is None:
            continue
        candidate_lo, candidate_hi = lo.copy(), hi.copy()
        if which == 0:
            candidate_lo[k] = new
            gap = cameras[:, axis] - new
        else:
            candidate_hi[k] = new
            gap = new - cameras[:, axis]
        if candidate_hi[k] - candidate_lo[k] < SNAP_MIN_SPAN:
            continue
        if gap.min() < SNAP_CLEARANCE:
            continue
        lo, hi = candidate_lo, candidate_hi
        moved[name] = new - side
    return lo.tolist(), hi.tolist(), moved


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


def proposed_layout(ctx: StageContext) -> tuple[dict[str, Any], dict[str, float]]:
    """The automatic single-room layout from the run's ``normalizing`` outputs.

    Returns the layout and the shift of each side that snapped to the evidence. Shared
    with the smoke runner, which has no database.
    """
    bounds = ctx.output("proposalBounds")
    min_xz, max_xz = bounds["minXZ"], bounds["maxXZ"]
    snapped: dict[str, float] = {}
    evidence = load_evidence(ctx)
    if evidence is not None:
        min_xz, max_xz, snapped = snap_rectangle(
            min_xz, max_xz, points=evidence[0], cameras=evidence[1], height=bounds["height"]
        )
    layout = rectangle_layout(
        min_xz,
        max_xz,
        elevation=bounds["floorY"],
        height=bounds["height"],
        spawn=ctx.outputs().get("proposalSpawn"),
    )
    return layout, snapped


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
    snapped: dict[str, float] = {}
    if ctx.job.scene_revision_id:
        revision = ctx.query(lambda conn: _load_revision(conn, ctx))
        if revision is None:
            raise StageError("MISSING_INPUT")
        layout, source = dict(revision["layout"]), "user"
    elif ctx.outputs().get("quality") == "synthetic":
        layout, source = synthetic_layout(), "synthetic"
    else:
        layout, snapped = proposed_layout(ctx)
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
        {"layoutSource": source, "collisionBytes": len(glb), "snappedSides": snapped},
        outputs={
            "sceneId": str(revision["scene_id"]),
            "sceneRevisionId": str(revision["id"]),
            "revision": int(revision["revision"]),
            "layoutRevision": int(revision["layout_revision"]),
            "layoutSource": source,
            "collisionKey": key,
        },
    )
