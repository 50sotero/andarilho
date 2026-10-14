"""The GPU smoke runner chains the real capture stages and fails loudly."""

from __future__ import annotations

import json

import pytest

from andarilho_worker import smoke
from andarilho_worker.errors import StageError
from andarilho_worker.geometry.ply import write_splat_ply
from andarilho_worker.geometry.synthetic import synthetic_splats
from andarilho_worker.manifest import IDENTITY_MATRIX
from andarilho_worker.stages.base import StageResult
from andarilho_worker.stages.normalize import synthetic_layout

PLY = write_splat_ply(synthetic_splats(synthetic_layout()))
RENDER_KEY = "tenants/x/runs/y/scene.ply"
MP4 = b"\x00\x00\x00\x18ftypisom"


def _fake(outputs):
    def stage(ctx):
        assert ctx.outputs()["inputs"][0]["mediaKind"] == "video"
        if ctx.stage == "normalizing":
            ctx.store.put_bytes(RENDER_KEY, PLY)
        return StageResult("", "0" * 64, {}, outputs)

    return stage


def _fakes(bounds):
    coverage = {"registeredViews": 110, "selectedViews": 120, "disconnectedComponents": 1,
                "registeredRatio": 0.92}  # fmt: skip
    normalized = {"renderKey": RENDER_KEY, "renderFormat": "ply", "scaleStatus": "estimated",
                  "worldFromDecodedRender": IDENTITY_MATRIX, "proposalBounds": bounds}  # fmt: skip
    return {
        "preparing": _fake({"frameCount": 120}),
        "camera_reconstruction": _fake({"coverage": coverage}),
        "training": _fake({"splatCount": 5000}),
        "normalizing": _fake(normalized),
    }


def test_smoke_chains_capture_stages_into_a_validated_bundle(tmp_path, monkeypatch):
    video = tmp_path / "capture.mp4"
    video.write_bytes(MP4)
    bounds = {"minXZ": [-2.0, -1.5], "maxXZ": [2.5, 3.0], "floorY": 0.1, "height": 2.5}
    monkeypatch.setattr(smoke, "STAGES", _fakes(bounds))
    assert smoke.main(["--video", str(video), "--out", str(tmp_path / "out")]) == 0
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert [s["stage"] for s in report["stages"]] == list(smoke.CAPTURE_STAGES)
    assert report["frameCount"] == 120 and report["splatCount"] == 5000
    bundle = tmp_path / "out" / "bundle"
    assert report["bundle"]["findings"] == []
    assert report["bundle"]["renderFile"] == "bundle/scene.ply"
    assert (bundle / "scene.ply").read_bytes() == PLY
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["navigation"]["mode"] == "inspection_only"  # a proposal awaits review
    assert manifest["quality"]["registeredViews"] == 110
    assert manifest["floors"][0]["elevation"] == 0.1  # the proposal sits on the splats' floor
    assert (bundle / "collision.glb").stat().st_size > 0
    assert json.loads((bundle / "layout.json").read_text())["spawn"]["floorId"] == "floor-0"


def test_smoke_fails_when_the_bundle_does_not_validate(tmp_path, monkeypatch):
    video = tmp_path / "capture.mp4"
    video.write_bytes(MP4)
    flat = {"minXZ": [0.0, 0.0], "maxXZ": [0.0, 0.0], "floorY": 0.0, "height": 2.5}  # no floor
    monkeypatch.setattr(smoke, "STAGES", _fakes(flat))
    assert smoke.main(["--video", str(video), "--out", str(tmp_path / "out")]) == 1


def test_smoke_exits_nonzero_on_stage_failure(tmp_path, monkeypatch):
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"x")

    def broken(ctx):
        raise StageError("CAMERA_REGISTRATION_FAILED")

    monkeypatch.setattr(smoke, "STAGES", {"preparing": broken})
    assert smoke.main(["--video", str(video), "--out", str(tmp_path / "out")]) == 1


def test_smoke_rejects_synthetic_quality(tmp_path):
    with pytest.raises(SystemExit):
        smoke.main(["--video", "v.mp4", "--out", str(tmp_path), "--quality", "synthetic"])
