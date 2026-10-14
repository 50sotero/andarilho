"""The GPU smoke runner chains the real capture stages and fails loudly."""

from __future__ import annotations

import json

import pytest

from andarilho_worker import smoke
from andarilho_worker.errors import StageError
from andarilho_worker.stages.base import StageResult


def _fake(outputs):
    def stage(ctx):
        assert ctx.outputs()["inputs"][0]["mediaKind"] == "video"
        if ctx.stage == "normalizing":
            ctx.store.put_bytes("tenants/x/runs/y/scene.ply", b"ply-bytes")
        return StageResult("", "0" * 64, {}, outputs)

    return stage


def test_smoke_chains_capture_stages(tmp_path, monkeypatch):
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypisom")
    fakes = {
        "preparing": _fake({"frameCount": 120}),
        "camera_reconstruction": _fake({"coverage": {"registeredViews": 110}}),
        "training": _fake({"splatCount": 5000}),
        "normalizing": _fake({"renderKey": "tenants/x/runs/y/scene.ply", "renderFormat": "ply"}),
    }
    monkeypatch.setattr(smoke, "STAGES", fakes)
    assert smoke.main(["--video", str(video), "--out", str(tmp_path / "out")]) == 0
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert [s["stage"] for s in report["stages"]] == list(smoke.CAPTURE_STAGES)
    assert report["frameCount"] == 120 and report["splatCount"] == 5000
    assert (tmp_path / "out" / "scene.ply").read_bytes() == b"ply-bytes"


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
