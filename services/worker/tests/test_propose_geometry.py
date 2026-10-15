"""The initial proposal snapped to the capture's wall evidence (stages.propose_geometry)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from stage_fixtures import stage_context

from andarilho_worker import queue
from andarilho_worker.stages.propose_geometry import load_evidence, snap_rectangle
from andarilho_worker.supervisor import Supervisor

HEIGHT = 2.5  # the room height of every synthetic scene here
RECTANGLE = ([-0.3, -0.2], [4.4, 3.3])  # a proposal about 0.3 m outside a 4 x 3 m room
OWN_EVIDENCE = "tenants/p/runs/run/attempts/normalizing/job/1/evidence.json"


def _wall_points(rng, x_range, z_range, *, per_wall: int = 400, noise: float = 0.01):
    """Wall-height points on the four walls of an axis-aligned room, with a little noise."""
    (x0, x1), (z0, z1) = x_range, z_range
    heights = rng.uniform(0.5, 2.0, (4, per_wall))
    along_x = rng.uniform(x0, x1, (2, per_wall))
    along_z = rng.uniform(z0, z1, (2, per_wall))
    walls = np.vstack([
        np.column_stack([along_x[0], heights[0], np.full(per_wall, z0)]),
        np.column_stack([along_x[1], heights[1], np.full(per_wall, z1)]),
        np.column_stack([np.full(per_wall, x0), heights[2], along_z[0]]),
        np.column_stack([np.full(per_wall, x1), heights[3], along_z[1]]),
    ])  # fmt: skip
    return walls + rng.normal(0, noise, walls.shape)


def _flat_clutter(n: int = 2000) -> np.ndarray:
    """Points spread evenly in x and in z, so every histogram bin holds about the same count."""
    return np.column_stack([np.linspace(0.0, 4.0, n), np.full(n, 1.0), np.linspace(0.0, 3.0, n)])


def _walk(x_range=(1.0, 3.0), n: int = 20) -> np.ndarray:
    """Camera centres walking along the middle of the room, 1.5 m up."""
    return np.column_stack([np.linspace(*x_range, n), np.full(n, 1.5), np.full(n, 1.5)])


def _evidence_blob(points: np.ndarray, cameras: np.ndarray) -> bytes:
    return json.dumps({
        "schemaVersion": 1,
        "units": "m",
        "points": np.round(points, 3).tolist(),
        "cameras": np.round(cameras, 3).tolist(),
    }).encode()  # fmt: skip


def test_each_side_snaps_to_the_wall_behind_it_despite_noise():
    rng = np.random.default_rng(1)
    points = np.vstack([_wall_points(rng, (0.0, 4.0), (0.0, 3.0)), _flat_clutter()])

    min_xz, max_xz, moved = snap_rectangle(
        *RECTANGLE, points=points, cameras=_walk(), height=HEIGHT
    )

    assert min_xz == pytest.approx([0.0, 0.0], abs=0.02)
    assert max_xz == pytest.approx([4.0, 3.0], abs=0.02)
    assert moved == pytest.approx(
        {"west": 0.3, "east": -0.4, "south": 0.2, "north": -0.3}, abs=0.02
    )


@pytest.mark.parametrize(
    "points",
    [
        np.empty((0, 3)),
        np.tile([0.0, 1.0, 1.0], (5, 1)),  # five points at one spot: fewer than eight
        np.tile([0.0, 1.0, 1.0], (7, 1)),  # seven points at one spot, still too few
    ],
)
def test_sparse_or_empty_evidence_keeps_the_rectangle(points):
    assert snap_rectangle(*RECTANGLE, points=points, cameras=_walk(), height=HEIGHT) == (
        [-0.3, -0.2],
        [4.4, 3.3],
        {},
    )


def test_evidence_without_cameras_keeps_the_rectangle():
    rng = np.random.default_rng(2)
    points = _wall_points(rng, (0.0, 4.0), (0.0, 3.0))
    no_cameras = np.empty((0, 3))
    assert snap_rectangle(*RECTANGLE, points=points, cameras=no_cameras, height=HEIGHT) == (
        [-0.3, -0.2],
        [4.4, 3.3],
        {},
    )


def test_flat_clutter_never_moves_a_side():
    assert snap_rectangle(*RECTANGLE, points=_flat_clutter(), cameras=_walk(), height=HEIGHT) == (
        [-0.3, -0.2],
        [4.4, 3.3],
        {},
    )


def test_only_points_between_the_floor_clutter_and_the_ceiling_count():
    rng = np.random.default_rng(3)
    # A dense plane at x = 0 that is either floor-level clutter or above the wall band.
    low = np.column_stack([np.zeros(300), rng.uniform(0.0, 0.2, 300), rng.uniform(0.5, 2.5, 300)])
    high = np.column_stack([np.zeros(300), np.full(300, 2.4), rng.uniform(0.5, 2.5, 300)])
    points = np.vstack([low, high, _flat_clutter()])
    assert snap_rectangle(*RECTANGLE, points=points, cameras=_walk(), height=HEIGHT) == (
        [-0.3, -0.2],
        [4.4, 3.3],
        {},
    )


def test_a_wall_that_would_cut_off_the_camera_path_is_ignored():
    rng = np.random.default_rng(4)
    # A wall at x = 0.4 with the walk starting at x = 0.5: 0.1 m from the wall to a camera.
    wall = np.column_stack([np.full(400, 0.4), rng.uniform(0.5, 2.0, 400), rng.uniform(0, 3, 400)])
    points = np.vstack([wall, _flat_clutter()])
    rectangle = ([-0.25, 0.0], [4.25, 3.0])

    cut_off = snap_rectangle(*rectangle, points=points, cameras=_walk((0.5, 3.5)), height=HEIGHT)
    assert cut_off[0][0] == -0.25
    # The same wall snaps once the walk keeps clear of it.
    clear = snap_rectangle(*rectangle, points=points, cameras=_walk((1.0, 3.5)), height=HEIGHT)
    assert clear[0][0] == pytest.approx(0.4, abs=0.02)


def test_a_snap_that_leaves_less_than_one_and_a_half_metres_is_ignored():
    rng = np.random.default_rng(6)
    wall = np.column_stack([np.full(400, 0.8), rng.uniform(0.5, 2.0, 400), rng.uniform(0, 3, 400)])
    points = np.vstack([wall, _flat_clutter()])
    cameras = _walk((1.2, 1.9))

    narrow = snap_rectangle([0.1, 0.0], [2.1, 3.0], points=points, cameras=cameras, height=HEIGHT)
    assert narrow[0][0] == 0.1  # 2.1 - 0.8 = 1.3 m would be left
    wide = snap_rectangle([0.1, 0.0], [2.6, 3.0], points=points, cameras=cameras, height=HEIGHT)
    assert wide[0][0] == pytest.approx(0.8, abs=0.02)  # 1.8 m is left


def test_load_evidence_never_reads_another_runs_key(tmp_path, store, settings, monkeypatch):
    other = "tenants/p/runs/other/attempts/normalizing/job/1/evidence.json"
    store.put_bytes(other, _evidence_blob(_flat_clutter(), _walk()))

    def refuse(key, *args, **kwargs):
        raise AssertionError(f"read {key}")

    monkeypatch.setattr(store, "get_bytes", refuse)
    ctx = stage_context(
        tmp_path, store, settings, {"evidenceKey": other}, stage="geometry_proposal"
    )
    assert load_evidence(ctx) is None


def test_load_evidence_reads_this_runs_points_and_cameras(tmp_path, store, settings):
    points, cameras = _flat_clutter(50), _walk()
    store.put_bytes(OWN_EVIDENCE, _evidence_blob(points, cameras))
    ctx = stage_context(
        tmp_path, store, settings, {"evidenceKey": OWN_EVIDENCE}, stage="geometry_proposal"
    )
    got_points, got_cameras = load_evidence(ctx)
    # The evidence holds millimetres: what comes back is what was written, rounded.
    assert got_points == pytest.approx(np.round(points, 3))
    assert got_cameras == pytest.approx(np.round(cameras, 3))


@pytest.mark.parametrize(
    "blob",
    [
        None,  # missing
        b"not json",
        b'{"schemaVersion": 2, "points": [], "cameras": []}',
        b'{"schemaVersion": 1}',
        b"[1, 2]",
    ],
)
def test_unreadable_evidence_falls_back_to_the_rectangle(tmp_path, store, settings, blob):
    if blob is not None:
        store.put_bytes(OWN_EVIDENCE, blob)
    ctx = stage_context(
        tmp_path, store, settings, {"evidenceKey": OWN_EVIDENCE}, stage="geometry_proposal"
    )
    assert load_evidence(ctx) is None


def test_no_evidence_key_is_no_evidence(tmp_path, store, settings):
    ctx = stage_context(tmp_path, store, settings, {}, stage="geometry_proposal")
    assert load_evidence(ctx) is None


@pytest.mark.parametrize("with_evidence", [True, False])
def test_the_initial_proposal_takes_the_snapped_rectangle_only_with_evidence(
    db, seed, store, settings, with_evidence
):
    from andarilho_worker.stages import STAGES

    rng = np.random.default_rng(7)
    points = np.vstack([_wall_points(rng, (0.0, 4.0), (0.0, 3.0)), _flat_clutter()])
    pid = seed.project()
    run_id, job_id = seed.run(pid, quality="preview", first_stage="geometry_proposal")
    outputs = {
        "proposalBounds": {
            "minXZ": RECTANGLE[0], "maxXZ": RECTANGLE[1], "floorY": 0.0, "height": HEIGHT,
        },
        "proposalSpawn": {"feet": [1.0, 0.0, 1.5], "yawRadians": 0.0},
        "quality": "preview",
    }  # fmt: skip
    if with_evidence:
        key = f"tenants/{pid}/runs/{run_id}/attempts/normalizing/n1/1/evidence.json"
        store.put_bytes(key, _evidence_blob(points, _walk()))
        outputs["evidenceKey"] = key
    seed.exec(
        "UPDATE jobs SET input_manifest = input_manifest || %s::jsonb WHERE id = %s",
        json.dumps({"outputs": outputs}), job_id,
    )  # fmt: skip
    sup = Supervisor(db, store, settings, stages=STAGES)
    assert sup.execute(queue.claim(db, "cpu-worker", "cpu")) == "succeeded"

    layout = seed.one("SELECT layout FROM scene_revisions WHERE run_id = %s", run_id)["layout"]
    corners = np.array(layout["floors"][0]["polygons"][0])
    expected = (
        [[0.0, 0.0], [4.0, 3.0]] if with_evidence else [list(RECTANGLE[0]), list(RECTANGLE[1])]
    )
    assert corners.min(axis=0) == pytest.approx(expected[0], abs=0.02)
    assert corners.max(axis=0) == pytest.approx(expected[1], abs=0.02)
    result = seed.one("SELECT result_manifest FROM jobs WHERE stage = 'geometry_proposal'")
    assert bool(result["result_manifest"]["snappedSides"]) == with_evidence
