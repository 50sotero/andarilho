"""The smoke runner chains the real capture stages into a textured bundle and fails loudly.

The capture stages are stand-ins (no ffmpeg or COLMAP here) that leave the outputs the
real ones do, including a real dataset of a rendered box room, so the proposal,
texturing and packaging steps run for real on it.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from room_dataset import (
    CAMERAS,
    HEIGHT,
    ROOM_HI,
    ROOM_LO,
    WIDTH,
    K,
    canonical_cameras,
    write_room_dataset,
)
from scene_analysis_fixtures import box_corners

from andarilho_worker import smoke
from andarilho_worker.errors import StageError
from andarilho_worker.geometry.glb import inspect_textured_glb
from andarilho_worker.geometry.layout import rectangle_layout
from andarilho_worker.stages import STAGES
from andarilho_worker.stages.base import StageResult

MP4 = b"\x00\x00\x00\x18ftypisom"
WEBP = b"RIFF" + (4).to_bytes(4, "little") + b"WEBP"
COVERAGE = {"registeredViews": CAMERAS, "selectedViews": CAMERAS, "disconnectedComponents": 1,
            "registeredRatio": 1.0, "warnings": []}  # fmt: skip
ROOM_BOUNDS = {
    "minXZ": ROOM_LO[[0, 2]].tolist(),
    "maxXZ": ROOM_HI[[0, 2]].tolist(),
    "floorY": 0.0,
    "height": float(ROOM_HI[1]),
}


@pytest.fixture
def dataset(tmp_path):
    """The room's dataset.tar and its COLMAP-to-world matrix, column-major as normalizing has it."""
    world = write_room_dataset(tmp_path / "dataset", tmp_path / "dataset.tar")
    return tmp_path / "dataset.tar", world.reshape(-1, order="F").tolist()


def _normalizing(world, bounds):
    def stage(ctx):
        assert ctx.output("datasetKey").startswith(f"tenants/{ctx.project_id}/runs/{ctx.run_id}/")
        outputs = {
            "worldFromColmap": world,
            "scaleStatus": "estimated",
            "proposalBounds": bounds,
            "proposalSpawn": None,
            "frameWarnings": ["Only a short walk was recorded."],
        }
        return StageResult("", "0" * 64, {}, outputs)

    return stage


def _capture_stages(dataset, bounds=ROOM_BOUNDS):
    tar_path, world = dataset

    def preparing(ctx):
        assert ctx.outputs()["inputs"][0]["mediaKind"] == "video"
        key = ctx.attempt_key("poster.webp")
        ctx.store.put_bytes(key, WEBP)
        return StageResult("", "0" * 64, {}, {"frameCount": 120, "posterKey": key})

    def camera_reconstruction(ctx):
        key = ctx.attempt_key("dataset.tar")
        ctx.store.put_file(key, tar_path, "application/x-tar")
        return StageResult("", "0" * 64, {}, {"datasetKey": key, "coverage": COVERAGE})

    return {
        "preparing": preparing,
        "camera_reconstruction": camera_reconstruction,
        "normalizing": _normalizing(world, bounds),
        "scene_analysis": STAGES["scene_analysis"],  # the real stage, on the room's dataset
    }


def _video(tmp_path):
    video = tmp_path / "capture.mp4"
    video.write_bytes(MP4)
    return video


def test_smoke_chains_capture_stages_into_a_textured_bundle(tmp_path, monkeypatch, dataset):
    monkeypatch.setattr(smoke, "STAGES", _capture_stages(dataset))
    out = tmp_path / "out"
    assert smoke.main(["--video", str(_video(tmp_path)), "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert [s["stage"] for s in report["stages"]] == [*smoke.CAPTURE_STAGES, "texturing"]
    assert report["frameCount"] == 120 and report["layoutSource"] == "proposal"
    # Without --analysis nothing is sent, and the proposal is the plain room.
    assert report["analysis"]["status"] == "disabled"
    assert report["proposal"]["analysisStatus"] == "disabled"
    assert report["proposal"]["rooms"] == 1 and report["proposal"]["furniture"] == 0
    assert not (out / "analysis").exists()
    assert report["bundle"]["findings"] == []
    assert report["bundle"]["renderFile"] == "bundle/model.glb"
    assert report["texturing"]["frames"] == CAMERAS
    # The rendered room fills the frames, so the walk sees nearly all of its walls.
    assert report["textureCoverage"]["wall"] > 0.9, report["textureCoverage"]
    bundle = out / "bundle"
    summary = inspect_textured_glb((bundle / "model.glb").read_bytes())
    assert summary["unlit"] and summary["triangles"] > 0
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["renderMode"] == "mesh"
    assert manifest["assets"]["render"]["format"] == "glb"
    assert manifest["assets"]["render"]["key"].endswith("/model.glb")
    assert manifest["navigation"]["mode"] == "approved_collision"  # a proposal is walkable at once
    assert manifest["quality"]["registeredViews"] == CAMERAS
    assert "Only a short walk was recorded." in manifest["quality"]["warnings"]
    assert manifest["scaleStatus"] == "estimated"
    assert (bundle / "collision.glb").stat().st_size > 0
    assert (bundle / "poster.webp").read_bytes() == WEBP
    assert manifest["assets"]["poster"]["format"] == "webp"
    assert json.loads((bundle / "layout.json").read_text())["spawn"]["floorId"] == "floor-0"


def test_smoke_textures_a_stored_dataset_with_a_reviewed_layout(tmp_path, monkeypatch, dataset):
    # Only normalizing runs on a stored dataset; the layout file stands in for review.
    tar_path, world = dataset
    stages = {
        "normalizing": _normalizing(world, ROOM_BOUNDS),
        "scene_analysis": STAGES["scene_analysis"],
    }
    monkeypatch.setattr(smoke, "STAGES", stages)
    layout_path = tmp_path / "layout.json"
    layout = rectangle_layout(ROOM_BOUNDS["minXZ"], ROOM_BOUNDS["maxXZ"], elevation=0.0, height=2.6)
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    out = tmp_path / "out"
    argv = ["--dataset", str(tar_path), "--layout", str(layout_path), "--out", str(out)]
    assert smoke.main(argv) == 0
    report = json.loads((out / "report.json").read_text())
    assert [s["stage"] for s in report["stages"]] == ["normalizing", "scene_analysis", "texturing"]
    assert report["layoutSource"] == "user" and report["proposal"] == {}
    manifest = json.loads((out / "bundle" / "manifest.json").read_text())
    assert manifest["navigation"]["mode"] == "approved_collision"
    assert manifest["quality"]["registeredViews"] == CAMERAS  # the frames in the archive
    assert smoke.DATASET_WARNING in manifest["quality"]["warnings"]
    assert "poster" not in manifest["assets"]
    assert json.loads((out / "bundle" / "layout.json").read_text()) == layout


# A white cabinet with its back to the room's south wall (+z), facing into the room.
CABINET_CENTRE, CABINET_YAW, CABINET_SIZE = (0.2, 0.95), math.pi, (0.8, 0.9, 0.5)


def _cabinet_reply() -> tuple[dict, list[int]]:
    """A model reply that saw the cabinet, and the frames that show its bottom edge.

    Each box is the cabinet's corners projected through that camera and clipped to the
    image, so the reply is what a perfect model would answer about the rendered room.
    """
    corners = box_corners(CABINET_CENTRE, CABINET_YAW, CABINET_SIZE)
    bottom = corners[:, 1] < 1e-9
    frames, seen = [], []
    for k, (R, t, _) in enumerate(canonical_cameras()):
        cam = corners @ R.T + t
        if cam[:, 2].min() <= 0.05:
            continue
        x = (K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]) / WIDTH
        y = (K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]) / HEIGHT
        if min(x[bottom].min(), y[bottom].min()) < 0.0 or x[bottom].max() > 1.0:
            continue
        if y[bottom].max() > 0.98:
            continue
        x, y = np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0)
        box = [float(x.min()), float(y.min()), float(x.max()), float(y.max())]
        detection = {"id": "o1", "box": box, "bottomVisible": True}
        frames.append({"frame": k, "room": "r1", "doors": [], "objects": [detection]})
        seen.append(k)
    cabinet = {
        "id": "o1", "category": "cabinet", "label": "white cabinet", "mount": "floor",
        "solid": True, "size": list(CABINET_SIZE),
        "parts": [{"shape": "box", "min": [0, 0, 0], "max": [1, 1, 1], "color": "#f0f0f0"}],
    }  # fmt: skip
    return {
        "rooms": [{"id": "r1", "label": "bedroom"}],
        "frames": frames,
        "objects": [cabinet],
    }, seen


def test_smoke_replays_a_recorded_analysis_into_furniture(tmp_path, monkeypatch, dataset):
    reply, seen = _cabinet_reply()
    assert seen, "some camera must see the bottom of the cabinet"
    reply_path = tmp_path / "reply.json"
    reply_path.write_text(json.dumps(reply), encoding="utf-8")
    monkeypatch.setattr(smoke, "STAGES", _capture_stages(dataset))
    out = tmp_path / "out"
    argv = [
        "--video",
        str(_video(tmp_path)),
        "--out",
        str(out),
        "--analysis-reply",
        str(reply_path),
    ]
    assert smoke.main(argv) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["analysis"]["status"] == "complete" and report["analysis"]["warnings"] == []
    assert report["proposal"]["analysisStatus"] == "complete"
    assert report["proposal"]["furniture"] == 1 and report["proposal"]["droppedFurniture"] == 0
    # The recorder kept what the model was shown, labelled by frame, and the reply.
    shown = sorted(int(f.stem.split("-")[1]) for f in (out / "analysis").glob("frame-*.jpg"))
    assert len(shown) == report["analysis"]["keyframes"] and set(shown) & set(seen)
    assert json.loads((out / "analysis" / "reply.json").read_text()) == reply
    assert "Frame " in (out / "analysis" / "request.txt").read_text(encoding="utf-8")
    layout = json.loads((out / "bundle" / "layout.json").read_text())
    (cabinet,) = layout["furniture"]
    assert cabinet["category"] == "cabinet" and cabinet["solid"] is True
    # Grounded from the frames near where it stands, facing into the room.
    assert np.hypot(*np.subtract(cabinet["position"], CABINET_CENTRE)) < 0.35, cabinet
    assert math.isclose(math.cos(cabinet["yawRadians"]), -1.0, abs_tol=1e-6)
    assert report["bundle"]["findings"] == []
    assert report["textureCoverage"]["furniture"] > 0.0


def test_the_recorder_needs_exactly_one_of_a_client_and_a_reply(tmp_path):
    with pytest.raises(ValueError):
        smoke.AnalysisRecorder(tmp_path)
    with pytest.raises(ValueError):
        smoke.AnalysisRecorder(tmp_path, client=object(), reply={})


def test_smoke_takes_at_most_one_analysis_source(tmp_path):
    with pytest.raises(SystemExit):
        smoke.main(["--video", "v.mp4", "--out", str(tmp_path), "--analysis",
                    "--analysis-reply", "r.json"])  # fmt: skip


def test_smoke_fails_when_the_layout_does_not_validate(tmp_path, monkeypatch, dataset):
    flat = {"minXZ": [0.0, 0.0], "maxXZ": [0.0, 0.0], "floorY": 0.0, "height": 2.5}  # no floor
    monkeypatch.setattr(smoke, "STAGES", _capture_stages(dataset, flat))
    out = tmp_path / "out"
    assert smoke.main(["--video", str(_video(tmp_path)), "--out", str(out)]) == 1
    report = json.loads((out / "report.json").read_text())
    assert report["bundle"]["findings"]
    assert "texturing" not in [s["stage"] for s in report["stages"]]


def test_smoke_exits_nonzero_on_stage_failure(tmp_path, monkeypatch):
    def broken(ctx):
        raise StageError("CAMERA_REGISTRATION_FAILED")

    monkeypatch.setattr(smoke, "STAGES", {"preparing": broken})
    assert smoke.main(["--video", str(_video(tmp_path)), "--out", str(tmp_path / "out")]) == 1


def test_smoke_rejects_synthetic_quality(tmp_path):
    with pytest.raises(SystemExit):
        smoke.main(["--video", "v.mp4", "--out", str(tmp_path), "--quality", "synthetic"])


def test_smoke_takes_exactly_one_capture_source(tmp_path):
    with pytest.raises(SystemExit):
        smoke.main(["--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        smoke.main(["--video", "v.mp4", "--dataset", "d.tar", "--out", str(tmp_path)])
