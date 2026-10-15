"""``texturing`` stage: real runs paint from the run's frames, synthetic runs from patterns.

The capture is a small box room rendered into JPEG frames by a test-only ray caster and
written as a COLMAP dataset in a similarity frame that is not the canonical one (scale 2,
yawed, shifted), as ``normalizing`` reports it. The run's rows (``preparing``,
``camera_reconstruction``, ``normalizing``) are seeded in the database, as the pipeline
leaves them.
"""

from __future__ import annotations

import copy
import io
import tarfile
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from PIL import Image
from psycopg.types.json import Jsonb

from andarilho_worker import queue
from andarilho_worker.db import Database
from andarilho_worker.errors import Cancelled, StageError
from andarilho_worker.geometry.colmap import (
    ColmapCamera,
    ColmapImage,
    SparsePoints,
    quaternion_to_rotation,
    write_cameras_bin,
    write_images_bin,
    write_points3d_bin,
)
from andarilho_worker.geometry.glb import _split_glb, inspect_textured_glb
from andarilho_worker.geometry.layout import layout_bounds, rectangle_layout
from andarilho_worker.recipes import recipe_for
from andarilho_worker.stages import STAGES
from andarilho_worker.stages import texture as texture_stage
from andarilho_worker.stages.base import CancellationToken, JobInfo, StageContext, StageServices
from andarilho_worker.supervisor import Supervisor

WIDTH, HEIGHT = 160, 120
K = np.array([[110.0, 0.0, 80.0], [0.0, 110.0, 60.0], [0.0, 0.0, 1.0]])
# The room in the estimated (uncalibrated) frame, in metres: x, y (up), z.
ROOM_LO = np.array([-1.5, 0.0, -1.25])
ROOM_HI = np.array([1.5, 2.6, 1.25])
# One colour per face: x low (west), x high (east), floor, ceiling, z low, z high.
FACE_COLOURS = np.array(
    [(200, 70, 60), (60, 160, 90), (70, 110, 210), (210, 160, 60), (150, 70, 190), (60, 180, 190)],
    dtype=np.float64,
)
SCALE = 2.0  # the similarity's scale: COLMAP units to metres
YAW = 0.3
TRANSLATION = np.array([0.5, -0.2, 1.0])
# Vertical view is +-28.6 degrees (the frame is 160 x 120 at focal length 110 px). Steep
# pitches reach the floor and ceiling close to the camera; shallow ones reach the walls' ends.
PITCHES = (-0.9, -0.35, 0.35, 0.9)
CAMERAS = 24
TOTAL_PASSES = 2 * CAMERAS  # texture_layout reports every frame once per pass


def _world_from_colmap() -> np.ndarray:
    """The 4x4 similarity from COLMAP to the canonical world."""
    c, s = np.cos(YAW), np.sin(YAW)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    matrix = np.eye(4)
    matrix[:3, :3] = SCALE * rotation
    matrix[:3, 3] = TRANSLATION
    return matrix


def _quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """(w, x, y, z) of a rotation matrix (Shepperd's method, the largest term first)."""
    trace = np.trace(R)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)


def _canonical_cameras() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(R, t, eye) of each camera in the canonical frame: inside the room, sweeping it."""
    poses = []
    up = np.array([0.0, 1.0, 0.0])
    for k in range(CAMERAS):
        yaw = 2.0 * np.pi * k / CAMERAS
        pitch = PITCHES[k % len(PITCHES)]
        eye = np.array([0.15 * np.cos(yaw), 1.4, 0.15 * np.sin(yaw)])
        forward = np.array(
            [np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)]
        )
        x_axis = np.cross(forward, up)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(forward, x_axis)  # down in the image, as COLMAP and OpenCV have it
        R = np.stack([x_axis, y_axis, forward])
        poses.append((R, -R @ eye, eye))
    return poses


def _render_room(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """RGB of the box room seen from a canonical world-to-camera pose, by ray casting."""
    u, v = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
    rays = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3) @ np.linalg.inv(K).T
    dirs = rays @ R  # camera rays in the world, as row vectors
    centre = -R.T @ t
    best = np.full(len(dirs), np.inf)
    face = np.zeros(len(dirs), dtype=np.int64)
    for axis in range(3):
        for side, plane in enumerate((ROOM_LO[axis], ROOM_HI[axis])):
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (plane - centre[axis]) / dirs[:, axis]
            hit = centre + s[:, None] * dirs
            ok = np.isfinite(s) & (s > 1e-9)
            for other in range(3):
                if other != axis:
                    ok &= (hit[:, other] >= ROOM_LO[other] - 1e-9) & (
                        hit[:, other] <= ROOM_HI[other] + 1e-9
                    )
            closer = ok & (s < best)
            best[closer] = s[closer]
            face[closer] = axis * 2 + side
    colours = FACE_COLOURS[np.where(np.isfinite(best), face, 0)]
    return np.rint(colours).reshape(HEIGHT, WIDTH, 3).astype(np.uint8)


def _recipe(quality: str = "preview", **texture: Any) -> dict[str, Any]:
    """A recipe's dict (as the supervisor passes it), with some texture settings replaced."""
    recipe = recipe_for(quality).as_dict()
    settings = {**recipe["params"]["texture"], **texture}
    return {**recipe, "params": {**recipe["params"], "texture": settings}}


def _succeeded(seed, pid: str, run_id: str, stage: str, outputs: dict[str, Any]) -> None:
    job_id = seed.job(pid, stage, {}, run_id=run_id)
    seed.exec(
        "UPDATE jobs SET state = 'succeeded', result_manifest = %s WHERE id = %s",
        Jsonb({"stage": stage, "outputs": outputs}),
        job_id,
    )


def _revision(
    seed,
    pid: str,
    run_id: str,
    layout: dict[str, Any],
    *,
    number: int = 1,
    scene_id: str | None = None,
    base: str | None = None,
) -> tuple[str, str]:
    """A scene revision of ``run_id`` (the scene is created for the first revision)."""
    revision_id = str(uuid.uuid4())
    if scene_id is None:
        scene_id = str(uuid.uuid4())
        seed.exec(
            "INSERT INTO scenes (id, project_id, run_id, version) VALUES (%s, %s, %s, 1)",
            scene_id,
            pid,
            run_id,
        )
    seed.exec(
        "INSERT INTO scene_revisions (id, scene_id, project_id, run_id, base_revision_id, "
        "revision, layout_revision, layout, validation_state, approval_state, created_by) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', 'draft', 'user-1')",
        revision_id,
        scene_id,
        pid,
        run_id,
        base,
        number,
        number,
        Jsonb(layout),
    )
    return revision_id, scene_id


def _set_dataset_key(seed, value: str | None) -> None:
    seed.exec(
        "UPDATE jobs SET result_manifest = jsonb_set(result_manifest, '{outputs,datasetKey}', %s) "
        "WHERE stage = 'camera_reconstruction'",
        Jsonb(value),
    )


@pytest.fixture
def capture(db, seed, store, tmp_path) -> dict[str, Any]:
    """A run with its dataset of the room, the rows its stages leave, and its first scene."""
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="preview")
    work = tmp_path / "dataset"
    (work / "images" / "00").mkdir(parents=True)
    (work / "sparse" / "0").mkdir(parents=True)
    world = _world_from_colmap()
    # Canonical X = s R_w X_c + t_w, so a canonical pose (R, t) is the COLMAP pose
    # (R R_w, (R t_w + t) / s): the same pixels, seen from the COLMAP frame.
    R_w, t_w = world[:3, :3] / SCALE, world[:3, 3]
    images = []
    for k, (R, t_canonical, _eye) in enumerate(_canonical_cameras()):
        name = f"00/frame_{k + 1:05d}.jpg"
        Image.fromarray(_render_room(R, t_canonical)).save(
            work / "images" / name, "JPEG", quality=95
        )
        R_colmap, t_colmap = R @ R_w, (R @ t_w + t_canonical) / SCALE
        images.append(ColmapImage(k + 1, 1, name, _quaternion(R_colmap), tuple(t_colmap)))
    write_cameras_bin(
        work / "sparse" / "0" / "cameras.bin",
        [ColmapCamera(1, "PINHOLE", WIDTH, HEIGHT, (110.0, 110.0, 80.0, 60.0))],
    )
    write_images_bin(work / "sparse" / "0" / "images.bin", images)
    points = SparsePoints(
        xyz=np.array([[0.0, 1.0, 0.0], [1.0, 0.5, 0.5]]),
        rgb=np.zeros((2, 3), dtype=np.uint8),
        error=np.zeros(2),
        track_length=np.full(2, 3, dtype=np.int64),
    )
    write_points3d_bin(work / "sparse" / "0" / "points3D.bin", points)
    tar_path = tmp_path / "dataset.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(work / "images", arcname="images")
        tar.add(work / "sparse", arcname="sparse")
    dataset_key = f"tenants/{pid}/runs/{run_id}/attempts/camera_reconstruction/x/1/dataset.tar"
    store.put_file(dataset_key, tar_path, "application/x-tar")
    coverage = {
        "registeredViews": CAMERAS,
        "selectedViews": CAMERAS,
        "disconnectedComponents": 0,
        "warnings": [],
        "registeredRatio": 1.0,
    }
    _succeeded(
        seed,
        pid,
        run_id,
        "camera_reconstruction",
        {"datasetKey": dataset_key, "camerasKey": "unused", "coverage": coverage},
    )
    _succeeded(
        seed,
        pid,
        run_id,
        "normalizing",
        {
            "worldFromColmap": world.reshape(-1, order="F").tolist(),
            "scaleStatus": "estimated",
            "proposalBounds": None,
            "frameWarnings": ["Only a short walk was recorded."],
            "evidenceKey": "unused",
        },
    )
    layout = rectangle_layout(ROOM_LO[[0, 2]], ROOM_HI[[0, 2]], elevation=0.0, height=2.6)
    revision, scene_id = _revision(seed, pid, run_id, layout)
    return {
        "project": pid,
        "run": run_id,
        "revision": revision,
        "scene": scene_id,
        "layout": layout,
    }


def _read(db: Database, fn):
    with db.transaction() as conn:
        return fn(conn)


def _context(
    db, store, settings, tmp_path, capture: dict[str, Any], *, recipe: dict[str, Any],
    progress=None, token: CancellationToken | None = None,
) -> StageContext:  # fmt: skip
    job_id = str(uuid.uuid4())
    return StageContext(
        job_id=job_id,
        run_id=capture["run"],
        stage="texturing",
        attempt=1,
        fencing_token=1,
        scratch=tmp_path / "scratch" / job_id,
        recipe_digest="sha256:test",
        recipe=recipe,
        input_manifest={"outputs": {"sceneRevisionId": capture["revision"]}},
        deadline=datetime.now(UTC) + timedelta(hours=1),
        max_scratch_bytes=settings.max_scratch_bytes,
        cancellation=token or CancellationToken(),
        job=JobInfo(capture["project"], capture["run"], None, None, "cpu"),
        services=StageServices(
            store=store,
            runner=None,
            settings=settings,
            db=db,
            progress=progress or (lambda message, data: True),
            fenced=None,
            read=lambda fn: _read(db, fn),
        ),
    )


def _run(db, store, settings, tmp_path, capture, recipe=None):
    ctx = _context(db, store, settings, tmp_path, capture, recipe=recipe or _recipe(texelSize=0.05))
    return texture_stage.run(ctx), ctx


def _atlas_pages(glb: bytes) -> list[np.ndarray]:
    doc, bin_blob = _split_glb(glb)
    pages = []
    for image in doc["images"]:
        view = doc["bufferViews"][image["bufferView"]]
        start = view["byteOffset"]
        raw = bin_blob[start : start + view["byteLength"]]
        pages.append(np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")))
    return pages


def _shows_colour(pages: list[np.ndarray], colour: np.ndarray) -> bool:
    texels = np.concatenate([p.reshape(-1, 3) for p in pages]).astype(np.float64)
    return int((np.linalg.norm(texels - colour, axis=1) < 40).sum()) >= 50


def test_quaternion_helper_round_trips_rotations():
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = quaternion_to_rotation(q)
        np.testing.assert_allclose(quaternion_to_rotation(_quaternion(R)), R, atol=1e-9)


def test_real_run_paints_the_layout_from_the_video_frames(db, store, settings, tmp_path, capture):
    result, ctx = _run(db, store, settings, tmp_path, capture)
    outputs = result.outputs
    assert outputs["modelKey"] == ctx.attempt_key("model.glb")
    model = store.get_bytes(outputs["modelKey"])
    summary = inspect_textured_glb(model)
    assert summary["unlit"] and summary["triangles"] > 0
    pages = _atlas_pages(model)
    assert len(pages) == len(summary["images"]) == result.manifest["pages"]
    for face in FACE_COLOURS:
        assert _shows_colour(pages, face), f"no texel shows the face colour {face.tolist()}"
    coverage = outputs["textureCoverage"]
    assert coverage["floor"] > 0.9 and coverage["ceiling"] > 0.9, coverage
    # Each wall is two faces and only the room side is ever in view from inside, so the
    # walls' share is about half of their area (see the open issue on the coverage metric).
    assert 0.4 < coverage["wall"] < 0.6, coverage
    assert all(0.0 <= value <= 1.0 for value in coverage.values())
    # The reconstruction's quality and the run's normalizing facts pass through unchanged.
    assert outputs["coverage"]["registeredViews"] == CAMERAS
    assert outputs["scaleStatus"] == "estimated"
    assert outputs["frameWarnings"] == ["Only a short walk was recorded."]
    assert "posterKey" not in outputs
    assert result.manifest["frames"] == CAMERAS
    assert result.manifest["modelBytes"] == len(model)
    assert result.manifest["texelSize"] == pytest.approx(0.05)


def test_frames_are_posed_in_the_canonical_frame_scaled_by_the_calibration(
    db, seed, store, settings, tmp_path, capture, monkeypatch
):
    seen: dict[str, Any] = {}
    real_texture = texture_stage.texture_layout

    def spy(layout, frames, params, *, progress=None):
        seen["frames"] = list(frames)
        return real_texture(layout, frames, params, progress=progress)

    monkeypatch.setattr(texture_stage, "texture_layout", spy)
    # The user measured the room: the calibrated layout is twice the estimated room.
    calibrated = rectangle_layout(
        2 * ROOM_LO[[0, 2]], 2 * ROOM_HI[[0, 2]], elevation=0.0, height=2 * 2.6
    )
    calibrated["calibration"] = {"method": "known_distance", "renderScale": SCALE}
    revision, _ = _revision(
        seed, capture["project"], capture["run"], calibrated,
        number=2, scene_id=capture["scene"], base=capture["revision"],
    )  # fmt: skip
    result, _ = _run(db, store, settings, tmp_path, {**capture, "revision": revision})
    assert result.outputs["scaleStatus"] == "estimated"  # the run's fact, unchanged
    frames = sorted(seen["frames"], key=lambda frame: frame.name)
    assert len(frames) == CAMERAS
    bounds = layout_bounds(calibrated)
    for frame, (R_canonical, t_canonical, eye) in zip(frames, _canonical_cameras(), strict=True):
        centre = -frame.R.T @ frame.t
        np.testing.assert_allclose(centre, SCALE * eye, atol=1e-6)
        np.testing.assert_allclose(frame.R, R_canonical, atol=1e-6)
        np.testing.assert_allclose(frame.t, SCALE * t_canonical, atol=1e-6)
        assert np.all(centre >= np.array(bounds["min"]))
        assert np.all(centre <= np.array(bounds["max"]))


def test_layout_save_retextures_from_the_runs_rows_through_the_supervisor(
    db, seed, store, settings, capture
):
    """A layout job carries the run ID but none of the run's outputs: it reads the rows."""
    seed.exec("UPDATE jobs SET state = 'succeeded' WHERE state = 'queued'")  # the run's validating
    layout = copy.deepcopy(capture["layout"])
    layout["spawn"]["feet"] = [-0.5, 0.0, 0.25]
    revision, _ = _revision(
        seed, capture["project"], capture["run"], layout,
        number=2, scene_id=capture["scene"], base=capture["revision"],
    )  # fmt: skip
    seed.job(
        capture["project"],
        "texturing",
        {"outputs": {"sceneRevisionId": revision, "layoutSource": "user"}},
        run_id=capture["run"],
        scene_revision_id=revision,
    )
    sup = Supervisor(db, store, settings, delay_fn=lambda attempt: 0.0)
    assert sup.execute(queue.claim(db, "worker-test", "cpu")) == "succeeded"
    job = seed.one("SELECT state, result_manifest FROM jobs WHERE stage = 'texturing'")
    assert job["state"] == "succeeded"
    outputs = job["result_manifest"]["outputs"]
    assert outputs["modelKey"].startswith(f"tenants/{capture['project']}/runs/{capture['run']}/")
    assert outputs["scaleStatus"] == "estimated"
    assert outputs["coverage"]["registeredViews"] == CAMERAS
    assert inspect_textured_glb(store.get_bytes(outputs["modelKey"]))["unlit"]
    assert seed.one("SELECT stage FROM jobs WHERE stage = 'packaging'")["stage"] == "packaging"


def test_a_synthetic_run_paints_patterns_and_passes_the_poster_through(
    db, seed, store, settings, tmp_path
):
    pid = seed.project()
    run_id, _ = seed.run(pid, quality="synthetic")
    layout = rectangle_layout((-1.5, -1.25), (1.5, 1.25), elevation=0.0, height=2.6)
    revision, _ = _revision(seed, pid, run_id, layout)
    poster = f"tenants/{pid}/runs/{run_id}/attempts/preparing/x/1/poster.webp"
    store.put_bytes(poster, b"RIFF\x04\x00\x00\x00WEBP")
    _succeeded(seed, pid, run_id, "preparing", {"frameCount": 0, "posterKey": poster})
    _succeeded(
        seed, pid, run_id, "normalizing", {"scaleStatus": "user_measured", "proposalBounds": None}
    )
    capture = {"project": pid, "run": run_id, "revision": revision, "layout": layout}
    result, _ = _run(
        db, store, settings, tmp_path, capture, recipe=_recipe("synthetic", texelSize=0.05)
    )
    outputs = result.outputs
    assert set(outputs["textureCoverage"].values()) == {1.0}
    assert outputs["scaleStatus"] == "user_measured"
    assert outputs["frameWarnings"] == []
    assert outputs["posterKey"] == poster
    assert "coverage" not in outputs
    assert result.manifest["frames"] == 0
    assert inspect_textured_glb(store.get_bytes(outputs["modelKey"]))["unlit"]


def test_texture_settings_come_from_the_recipe(db, store, settings, tmp_path, capture):
    recipe = _recipe(texelSize=0.05, pageSize=256, maxPages=1, jpegQuality=90)
    result, _ = _run(db, store, settings, tmp_path, capture, recipe=recipe)
    summary = inspect_textured_glb(store.get_bytes(result.outputs["modelKey"]))
    assert summary["images"] == [[256, 256]]
    assert result.manifest["pages"] == 1


def _no_camera_row(seed, capture) -> None:
    seed.exec("DELETE FROM jobs WHERE stage = 'camera_reconstruction'")


def _no_normalizing_row(seed, capture) -> None:
    seed.exec("DELETE FROM jobs WHERE stage = 'normalizing'")


def _dataset_of_another_project(seed, capture) -> None:
    _set_dataset_key(seed, f"tenants/other-project/runs/{capture['run']}/dataset.tar")


def _dataset_of_another_run(seed, capture) -> None:
    _set_dataset_key(seed, f"tenants/{capture['project']}/runs/other-run/dataset.tar")


def _dataset_not_stored(seed, capture) -> None:
    _set_dataset_key(seed, f"tenants/{capture['project']}/runs/{capture['run']}/missing.tar")


def _no_dataset_key(seed, capture) -> None:
    _set_dataset_key(seed, None)


@pytest.mark.parametrize(
    "break_run",
    [
        _no_camera_row,
        _no_normalizing_row,
        _dataset_of_another_project,
        _dataset_of_another_run,
        _dataset_not_stored,
        _no_dataset_key,
    ],
    ids=lambda f: f.__name__.lstrip("_"),
)
def test_a_real_run_missing_its_inputs_fails_and_never_falls_back(
    db, seed, store, settings, tmp_path, capture, break_run
):
    break_run(seed, capture)
    with pytest.raises(StageError) as raised:
        _run(db, store, settings, tmp_path, capture)
    assert raised.value.code == "MISSING_INPUT"


def test_a_revision_of_another_project_is_missing_input(
    db, seed, store, settings, tmp_path, capture
):
    other = seed.project()
    other_run, _ = seed.run(other, quality="preview")
    stranger, _ = _revision(seed, other, other_run, capture["layout"])
    with pytest.raises(StageError) as raised:
        _run(db, store, settings, tmp_path, {**capture, "revision": stranger})
    assert raised.value.code == "MISSING_INPUT"


def test_a_scene_from_the_splat_pipeline_is_outdated(db, seed, store, settings, tmp_path, capture):
    seed.exec(
        "UPDATE jobs SET result_manifest = %s WHERE stage = 'normalizing'",
        Jsonb(
            {
                "outputs": {
                    "renderKey": "tenants/x/y.ply",
                    "scaleStatus": "estimated",
                    "worldFromDecodedRender": [1.0] * 16,
                }
            }
        ),
    )
    with pytest.raises(StageError) as raised:
        _run(db, store, settings, tmp_path, capture)
    assert raised.value.code == "SCENE_OUTDATED"
    assert raised.value.body()["suggestedAction"]


def test_cancellation_stops_the_frame_loop(db, store, settings, tmp_path, capture, monkeypatch):
    loaded: list[str] = []
    real_loader = texture_stage._loader

    def counting_loader(path):
        load = real_loader(path)

        def counted():
            loaded.append(path.name)
            return load()

        return counted

    monkeypatch.setattr(texture_stage, "_loader", counting_loader)
    token = CancellationToken()

    def cancel_as_texturing_starts(message, data):
        if message.startswith("Texturing"):
            token.cancel()
        return True

    ctx = _context(
        db, store, settings, tmp_path, capture, recipe=_recipe(texelSize=0.05),
        progress=cancel_as_texturing_starts, token=token,
    )  # fmt: skip
    with pytest.raises(Cancelled):
        texture_stage.run(ctx)
    assert len(loaded) == 1  # the second frame's check raised before that frame was read


def test_progress_is_reported_about_every_tenth_of_the_passes(
    db, store, settings, tmp_path, capture
):
    reports: list[dict[str, Any]] = []

    def record(message, data):
        if message.startswith("Texturing"):
            reports.append(data)
        return True

    ctx = _context(
        db, store, settings, tmp_path, capture, recipe=_recipe(texelSize=0.05), progress=record
    )
    texture_stage.run(ctx)
    step = TOTAL_PASSES // 10
    assert [r["current"] for r in reports] == list(range(0, TOTAL_PASSES, step))
    assert all(r["total"] == TOTAL_PASSES and r["unit"] == "steps" for r in reports)


def test_the_stage_is_registered_as_texturing():
    assert STAGES["texturing"] is texture_stage.run
