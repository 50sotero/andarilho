"""Recipe, coverage-gate and media-stage tests with fake external tools (no FFmpeg/COLMAP/GPU)."""

from __future__ import annotations

import json
import struct
import tarfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from colmap_fixtures import (
    FLOOR,
    apply_affine,
    cameras_for,
    colmap_transform,
    join_points,
    l_walk,
    make_room,
    similarity_matrix_problem,
    to_sparse,
)
from exif_fixtures import exif_tiff, jpeg_with_exif
from stage_fixtures import stage_context

from andarilho_worker import queue
from andarilho_worker.errors import StageError
from andarilho_worker.geometry.colmap import (
    ColmapCamera,
    ColmapImage,
    SparsePoints,
    write_cameras_bin,
    write_images_bin,
    write_points3d_bin,
)
from andarilho_worker.geometry.frame import frame_from_sparse, reliable_points
from andarilho_worker.recipes import (
    ARGV,
    GPU_STAGES,
    LAYOUT_STAGES,
    RECIPES,
    RUN_STAGES,
    TIMEOUTS,
    TOOL_VERSIONS,
    recipe_for,
    render_argv,
)
from andarilho_worker.runner import ProcessFailed, RunResult
from andarilho_worker.settings import Settings
from andarilho_worker.stages import normalize
from andarilho_worker.stages import prepare as prepare_mod
from andarilho_worker.stages.reconstruct import SparseModel, coverage_gate
from andarilho_worker.supervisor import Supervisor


def test_recipes_stage_order_and_capabilities():
    synthetic = recipe_for("synthetic")
    assert synthetic.stage_names() == [
        "validating", "preparing", "normalizing", "scene_analysis", "geometry_proposal",
        "texturing", "packaging", "scene_validation",
    ]  # fmt: skip
    assert {s.capability for s in synthetic.stages} == {"cpu"}
    preview = recipe_for("preview")
    assert preview.stage_names() == list(RUN_STAGES)
    caps = {s.stage: s.capability for s in preview.stages}
    assert caps["camera_reconstruction"] == "gpu"
    assert all(caps[s] == "cpu" for s in RUN_STAGES if s != "camera_reconstruction")
    assert "training" not in caps and "training" not in TIMEOUTS
    assert TIMEOUTS["texturing"] == 1800 and TIMEOUTS["scene_analysis"] == 600
    assert preview.next_stage("camera_reconstruction").stage == "normalizing"
    assert preview.next_stage("normalizing").stage == "scene_analysis"
    assert preview.next_stage("scene_analysis").stage == "geometry_proposal"
    assert preview.next_stage("geometry_proposal").stage == "texturing"
    assert preview.next_stage("texturing").stage == "packaging"
    assert preview.next_stage("scene_validation") is None
    # A layout save runs the layout stages only, texturing included.
    assert preview.next_stage("geometry_proposal", layout_pipeline=True).stage == "texturing"
    assert preview.next_stage("texturing", layout_pipeline=True).stage == "packaging"
    assert preview.next_stage("packaging", layout_pipeline=True).stage == "scene_validation"
    assert LAYOUT_STAGES == ("geometry_proposal", "texturing", "packaging", "scene_validation")
    assert frozenset({"camera_reconstruction"}) == GPU_STAGES
    digests = {r.digest for r in RECIPES.values()}
    assert len(digests) == 3 and all(d.startswith("sha256:") for d in digests)
    assert recipe_for("preview").digest == recipe_for("preview").digest
    with pytest.raises(KeyError):
        recipe_for("ultra")


def test_recipe_names_and_parameters_after_the_textured_model():
    assert {r.name for r in RECIPES.values()} == {
        "preview-v3", "standard-v3", "synthetic-v3"
    }  # fmt: skip
    texture = {"texelSize": 0.01, "pageSize": 2048, "maxPages": 8, "jpegQuality": 85}
    for quality in ("preview", "standard"):
        params = recipe_for(quality).params
        assert params["texture"] == texture
        assert params["analysis"]["keyframes"] == 12 and params["analysis"]["imageLongEdge"] == 1024
        assert not {"iterations", "saveSteps", "renderShDegree"} & set(params)
    synthetic = recipe_for("synthetic").params
    assert synthetic == {"fixture": "synthetic-two-room", "texture": texture}


def test_digest_changes_with_the_texture_parameters():
    preview = recipe_for("preview")
    texture = {**preview.params["texture"], "texelSize": 0.02}
    coarser = replace(preview, params={**preview.params, "texture": texture})
    assert coarser.digest != preview.digest


def test_argv_templates_are_pinned():
    argv = render_argv("ffprobe", input=Path("/scratch/x.mp4"))
    assert argv[0] == "ffprobe" and argv[-1].endswith("x.mp4")
    with pytest.raises(KeyError):
        render_argv("ffprobe", input="a", extra="--evil")
    # The splat trainer and compressor are gone: their templates and tool pins with them.
    assert not {"ns_train", "ns_export", "splat_transform"} & set(ARGV)
    assert not {"nerfstudio", "gsplat", "torch", "splat-transform"} & set(TOOL_VERSIONS)
    assert set(TOOL_VERSIONS) == {"colmap", "ffmpeg"}
    assert all(isinstance(t, tuple) for t in ARGV.values())


def test_coverage_gate_codes():
    def models(*counts):
        return [SparseModel(Path(f"m{i}"), c, 100) for i, c in enumerate(counts)]

    with pytest.raises(StageError) as e:
        coverage_gate([], 50, min_views=30, min_ratio=0.8)
    assert e.value.code == "CAMERA_REGISTRATION_FAILED"
    with pytest.raises(StageError) as e:
        coverage_gate(models(25, 25), 50, min_views=30, min_ratio=0.8)
    assert e.value.code == "DISCONNECTED_CAPTURE"
    with pytest.raises(StageError) as e:
        coverage_gate(models(20), 22, min_views=30, min_ratio=0.8)
    assert e.value.code == "INSUFFICIENT_OVERLAP"
    with pytest.raises(StageError) as e:
        coverage_gate(models(60, 2), 100, min_views=30, min_ratio=0.8)
    assert e.value.code == "INSUFFICIENT_OVERLAP"
    quality = coverage_gate(models(95, 2), 100, min_views=30, min_ratio=0.8)
    assert quality["registeredViews"] == 95 and quality["disconnectedComponents"] == 1


def test_settings_from_env(tmp_path):
    s = Settings.from_env({"DATABASE_URL": "postgresql://x", "WORKER_CAPABILITY": "gpu",
                           "WORKER_MAX_CONCURRENCY": "2", "WORKER_SCRATCH_DIR": str(tmp_path)})  # fmt: skip
    assert s.worker_capability == "gpu" and s.worker_max_concurrency == 2
    assert s.worker_id
    with pytest.raises(ValueError):
        Settings.from_env({"DATABASE_URL": "x", "WORKER_CAPABILITY": "tpu"})
    with pytest.raises(ValueError):
        Settings.from_env({"DATABASE_URL": "postgresql://a:local-password@h/d", "APP_ENV": "production",
                           "S3_ACCESS_KEY_ID": "local-development-only"})  # fmt: skip


class FakeMediaRunner:
    """Stands in for ffprobe/ffmpeg: records argv and writes plausible outputs."""

    def __init__(
        self,
        codec: str = "h264",
        duration: float = 10.0,
        photo_sizes=None,
        frame_rate="30/1",
        poster_fails=False,
    ) -> None:
        self.calls: list[list[str]] = []
        self.codec, self.duration, self.frame_rate = codec, duration, frame_rate
        self.photo_sizes = list(photo_sizes or [(1600, 1200)])
        self.photos = 0
        self.poster_fails = poster_fails

    def run(self, argv, *, timeout, check_cancelled=None, **kw):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        if check_cancelled:
            check_cancelled()
        stdout = ""
        if argv[0] == "ffprobe":
            stdout = json.dumps({
                "streams": [{"codec_type": "video", "codec_name": self.codec, "width": 1920,
                             "height": 1080, "avg_frame_rate": self.frame_rate, "duration": str(self.duration)},
                            {"codec_type": "audio", "codec_name": "aac"}],
                "format": {"duration": str(self.duration), "start_time": "0"},
            })  # fmt: skip
        elif argv[0] == "ffmpeg":
            out = Path(argv[-1])
            count = int(argv[argv.index("-frames:v") + 1]) if "-frames:v" in argv else 1
            if "rawvideo" in argv:
                # Sharpness-scoring decode: only every third candidate frame has texture.
                size = int(argv[argv.index("-vf") + 1].split("scale=")[1].split(":")[0])
                texture = np.random.default_rng(3).integers(0, 256, (size, size), dtype=np.uint8)
                flat = np.full((size, size), 128, dtype=np.uint8)
                np.stack([texture if i % 3 == 1 else flat for i in range(count)]).tofile(out)
            elif out.suffix == ".webp":
                if self.poster_fails:
                    raise ProcessFailed("ffmpeg exited with status 1")
                out.write_bytes(b"RIFF" + (4).to_bytes(4, "little") + b"WEBP")
            elif "%05d" in out.name:
                for i in range(1, count + 1):
                    frame = b"\xff\xd8\xff" + i.to_bytes(4, "big")  # remembers its candidate index
                    (out.parent / out.name.replace("%05d", f"{i:05d}")).write_bytes(frame)
            else:
                w, h = self.photo_sizes[self.photos % len(self.photo_sizes)]
                self.photos += 1
                out.write_bytes(jpeg_with_size(w, h))
        return RunResult(tuple(argv), 0, stdout, "", 0.01)


def jpeg_with_size(width: int, height: int) -> bytes:
    sof = b"\xff\xc0\x00\x11\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big")
    return b"\xff\xd8" + sof + b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01" + b"\xff\xd9"


def _stored_upload(seed, store, pid, kind: str, data: bytes, status: str = "verified") -> dict:
    """An upload row with its bytes stored, and its capture-set input entry."""
    import hashlib
    import uuid

    uid = str(uuid.uuid4())
    key = f"tenants/{pid}/originals/{uid}"
    store.put_bytes(key, data)
    digest = hashlib.sha256(data).hexdigest()
    seed.exec(
        "INSERT INTO uploads (id, project_id, created_by, media_kind, display_filename, "
        "declared_content_type, expected_bytes, actual_bytes, sha256, staging_key, "
        "immutable_key, part_size, part_count, status, expires_at) VALUES (%s, %s, 'user-1', "
        "%s, 'file', 'application/octet-stream', %s, %s, %s, %s, %s, 16777216, 1, %s, "
        "now() + interval '1 day')",
        uid, pid, kind, len(data), len(data), digest, f"tenants/{pid}/staging/{uid}", key,
        status,
    )  # fmt: skip
    return {"uploadId": uid, "mediaKind": kind, "key": key, "storageVersion": None,
            "sha256": digest, "bytes": len(data)}  # fmt: skip


@pytest.mark.parametrize("include_photos", [False, True])
def test_validation_skips_photos_a_video_run_leaves_out(db, seed, store, settings,
                                                        include_photos):  # fmt: skip
    pid = seed.project()
    video = _stored_upload(seed, store, pid, "video", b"\x00\x00\x00\x18ftypisom" + b"v" * 64)
    # Removed while the run was queued, as the API allows for an excluded photo.
    photo = _stored_upload(seed, store, pid, "photo", b"\xff\xd8\xff" + b"p" * 64, "aborted")
    capture = seed.capture_set(pid, [video, photo])
    run_id, _ = seed.run(pid, quality="preview", capture_set_id=capture)
    seed.exec(
        "UPDATE reconstruction_runs SET include_supplementary_photos = %s WHERE id = %s",
        include_photos, run_id,
    )  # fmt: skip
    job = queue.claim(db, "w", "cpu")
    state = Supervisor(db, store, settings).execute(job)
    if include_photos:
        # The run reads that photo, so a removed one is a missing input.
        assert state == "failed"
        error = seed.one("SELECT error FROM jobs WHERE id = %s", job.id)["error"]
        assert error["code"] == "INPUT_NOT_VERIFIED"
    else:
        assert state == "succeeded"
        nxt = seed.one("SELECT input_manifest FROM jobs WHERE stage = 'preparing'")
        inputs = nxt["input_manifest"]["outputs"]["inputs"]
        assert [i["uploadId"] for i in inputs] == [video["uploadId"]]


def _preparing_job(seed, store, pid, *, codec="h264", photos=1, photo_bytes=None):
    run_id, job_id = seed.run(pid, quality="preview", first_stage="preparing")
    inputs = []
    sources = [("video", b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)]
    if photo_bytes is None:
        photo_bytes = [b"\xff\xd8\xff" + bytes([i]) * 64 for i in range(photos)]
    sources += [("photo", data) for data in photo_bytes]
    for n, (kind, data) in enumerate(sources):
        uid = f"{kind}-{n:04d}" if kind == "photo" else f"{kind}-0000"
        key = f"tenants/{pid}/originals/{uid}"
        store.put_bytes(key, data)
        import hashlib

        inputs.append({"uploadId": uid, "mediaKind": kind, "key": key, "storageVersion": None,
                       "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})  # fmt: skip
    seed.exec(
        "UPDATE jobs SET input_manifest = input_manifest || %s::jsonb WHERE id = %s",
        json.dumps({"outputs": {"inputs": inputs, "quality": "preview"}}), job_id,
    )  # fmt: skip
    return run_id, job_id


def test_prepare_extracts_bounded_frames_and_strips_metadata(
    db, seed, store, settings, monkeypatch
):
    monkeypatch.setattr(prepare_mod, "tool_available", lambda name: True)
    fake = FakeMediaRunner()
    sup = Supervisor(db, store, settings, runner=fake)
    pid = seed.project()
    run_id, _ = _preparing_job(seed, store, pid)
    job = queue.claim(db, "w", "cpu")
    assert sup.execute(job) == "succeeded"
    ffmpeg = [c for c in fake.calls if c[0] == "ffmpeg"]
    candidates, scoring = ffmpeg[0], ffmpeg[1]
    for call in (candidates, scoring):
        assert "-an" in call and call[call.index("-map_metadata") + 1] == "-1"
        assert int(call[call.index("-frames:v") + 1]) == 120  # 10 s at 4 fps, 3 candidates each
        assert "fps=12.0000" in call[call.index("-vf") + 1]  # the same frames in both decodes
    assert any("min(1600,iw)" in a for a in candidates)
    assert "rawvideo" in scoring and "format=gray" in scoring[scoring.index("-vf") + 1]
    nxt = seed.one(
        "SELECT stage, capability, input_manifest FROM jobs WHERE stage = 'camera_reconstruction'"
    )
    assert nxt["capability"] == "gpu"
    outputs = nxt["input_manifest"]["outputs"]
    assert outputs["frameCount"] == 41
    # The sharpest candidate of each window of three is kept: candidate 3k+1 (0-based),
    # the second of every window, at (3k + 1) / 12 seconds.
    source_map = json.loads(store.get_bytes(outputs["sourceMapKey"]))
    assert source_map["00/frame_00003.jpg"] == {"uploadId": "video-0000", "timestampSeconds": 0.583}
    third = store.get_bytes(outputs["framesPrefix"] + "00/frame_00003.jpg")
    assert int.from_bytes(third[3:], "big") == 8  # ffmpeg's 1-based name for candidate 7
    assert source_map["photos_1600x1200/photo_00000.jpg"]["photo"] is True
    assert len(store.list_keys(outputs["framesPrefix"])) == 41
    result = seed.one("SELECT result_manifest FROM jobs WHERE stage = 'preparing'")
    stats = result["result_manifest"]["keyframes"][0]
    assert stats["candidates"] == 120 and stats["kept"] == 40
    assert stats["keptMedianSharpness"] > stats["medianSharpness"] == 0
    # The poster is the sharpest kept frame of the middle half (all kept frames tie
    # here, so the first of frames 10..29), encoded as WebP from the prepared frame.
    (poster_call,) = [c for c in ffmpeg if c[-1].endswith(".webp")]
    assert poster_call[poster_call.index("-i") + 1].endswith("frame_00011.jpg")
    assert poster_call[poster_call.index("-c:v") + 1] == "libwebp"
    assert "min(1280,iw)" in poster_call[poster_call.index("-vf") + 1]
    assert result["result_manifest"]["poster"] == "00/frame_00011.jpg"
    assert store.get_bytes(outputs["posterKey"])[8:12] == b"WEBP"


def test_prepare_never_samples_above_the_native_frame_rate(db, seed, store, settings, monkeypatch):
    monkeypatch.setattr(prepare_mod, "tool_available", lambda name: True)
    fake = FakeMediaRunner(frame_rate="1/1")  # a 10 s, 1 fps time-lapse
    sup = Supervisor(db, store, settings, runner=fake)
    pid = seed.project()
    _preparing_job(seed, store, pid, photos=0)
    assert sup.execute(queue.claim(db, "w", "cpu")) == "succeeded"
    video_calls = [c for c in fake.calls if c[0] == "ffmpeg" and not c[-1].endswith(".webp")]
    (video_call,) = video_calls  # no candidates to choose from
    assert "fps=1.0000" in video_call[video_call.index("-vf") + 1]
    assert int(video_call[video_call.index("-frames:v") + 1]) == 10  # one per source frame
    outputs = seed.one("SELECT input_manifest FROM jobs WHERE stage = 'camera_reconstruction'")[
        "input_manifest"
    ]["outputs"]
    assert outputs["frameCount"] == 10
    poster = seed.one("SELECT result_manifest FROM jobs WHERE stage = 'preparing'")
    assert poster["result_manifest"]["poster"] == "00/frame_00006.jpg"  # unscored: the middle


def test_prepare_continues_without_a_poster_when_encoding_fails(
    db, seed, store, settings, monkeypatch
):
    monkeypatch.setattr(prepare_mod, "tool_available", lambda name: True)
    sup = Supervisor(db, store, settings, runner=FakeMediaRunner(poster_fails=True))
    pid = seed.project()
    _preparing_job(seed, store, pid)
    assert sup.execute(queue.claim(db, "w", "cpu")) == "succeeded"
    nxt = seed.one("SELECT input_manifest FROM jobs WHERE stage = 'camera_reconstruction'")
    assert nxt["input_manifest"]["outputs"]["posterKey"] is None


def test_prepare_groups_photos_by_dimensions(db, seed, store, settings, monkeypatch):
    monkeypatch.setattr(prepare_mod, "tool_available", lambda name: True)
    fake = FakeMediaRunner(photo_sizes=[(1600, 1200), (1200, 1600)])
    sup = Supervisor(db, store, settings, runner=fake)
    pid = seed.project()
    _preparing_job(seed, store, pid, photos=3)
    assert sup.execute(queue.claim(db, "w", "cpu")) == "succeeded"
    outputs = seed.one("SELECT input_manifest FROM jobs WHERE stage = 'camera_reconstruction'")[
        "input_manifest"
    ]["outputs"]
    folders = {name.split("/")[0] for name in outputs["frameNames"] if name.startswith("photos")}
    # Portrait and landscape photos get separate COLMAP camera folders.
    assert folders == {"photos_1600x1200", "photos_1200x1600"}


def test_prepare_gives_photos_from_different_cameras_their_own_folders(
    db, seed, store, settings, monkeypatch
):
    monkeypatch.setattr(prepare_mod, "tool_available", lambda name: True)
    sup = Supervisor(db, store, settings, runner=FakeMediaRunner())
    pid = seed.project()
    iphone = exif_tiff(make="Apple", model="iPhone 15", focal_mm=6.1)
    pixel = exif_tiff(make="Google", model="Pixel 8", focal_mm=6.9)
    no_exif = b"\xff\xd8\xff" + b"\x00" * 64
    photos = [jpeg_with_exif(iphone), jpeg_with_exif(iphone), jpeg_with_exif(pixel), no_exif]
    _preparing_job(seed, store, pid, photo_bytes=photos)
    assert sup.execute(queue.claim(db, "w", "cpu")) == "succeeded"
    outputs = seed.one("SELECT input_manifest FROM jobs WHERE stage = 'camera_reconstruction'")[
        "input_manifest"
    ]["outputs"]
    folders = Counter(n.split("/")[0] for n in outputs["frameNames"] if n.startswith("photos"))
    # All four normalize to 1600x1200: the two iPhone photos share a camera folder,
    # the Pixel photo gets its own, and the photo without EXIF keeps the size folder.
    assert sorted(folders.values()) == [1, 1, 2] and folders["photos_1600x1200"] == 1
    assert all(f.startswith("photos_1600x1200") for f in folders)


def test_prepare_rejects_unsupported_codec(db, seed, store, settings, monkeypatch):
    monkeypatch.setattr(prepare_mod, "tool_available", lambda name: True)
    sup = Supervisor(db, store, settings, runner=FakeMediaRunner(codec="vp9"))
    pid = seed.project()
    run_id, _ = _preparing_job(seed, store, pid)
    job = queue.claim(db, "w", "cpu")
    assert sup.execute(job) == "failed"
    run = seed.one("SELECT status, error FROM reconstruction_runs WHERE id = %s", run_id)
    assert run["status"] == "failed" and run["error"]["code"] == "UNSUPPORTED_MEDIA"
    assert seed.one("SELECT state FROM usage_ledger")["state"] == "released"


def test_plan_frames_caps_total():
    fps, budget = prepare_mod.plan_frames([600.0], 100, fps=2, cap=400)
    assert budget == 300 and fps == pytest.approx(0.5)
    assert prepare_mod.plan_frames([60.0], 0, fps=2, cap=400) == (2, 400)


def _images_bin(path: Path, count: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "images.bin").write_bytes(struct.pack("<Q", count))
    (path / "points3D.bin").write_bytes(struct.pack("<Q", 0))


class FakeColmap:
    def __init__(self, components: list[int]) -> None:
        self.components = components

    def run(self, argv, *, timeout, check_cancelled=None, **kw):
        argv = [str(a) for a in argv]
        if argv[:2] == ["colmap", "mapper"]:
            sparse = Path(argv[argv.index("--output_path") + 1])
            for i, count in enumerate(self.components):
                _images_bin(sparse / str(i), count)
        return RunResult(tuple(argv), 0, "", "", 0.01)


def test_reconstruct_reports_disconnected_capture(db, seed, store, settings):
    from andarilho_worker.stages import STAGES

    settings_gpu = Settings(**{**settings.__dict__, "worker_capability": "gpu"})
    sup = Supervisor(db, store, settings_gpu, runner=FakeColmap([25, 25]), stages=STAGES)
    pid = seed.project()
    run_id, job_id = seed.run(pid, quality="preview", first_stage="camera_reconstruction",
                              capability="gpu")  # fmt: skip
    names = [f"00/frame_{i:05d}.jpg" for i in range(50)]
    prefix = f"tenants/{pid}/runs/{run_id}/attempts/preparing/1/images/"
    for name in names:
        store.put_bytes(prefix + name, b"\xff\xd8\xff")
    seed.exec(
        "UPDATE jobs SET input_manifest = input_manifest || %s::jsonb WHERE id = %s",
        json.dumps({"outputs": {"framesPrefix": prefix, "frameNames": names, "quality": "preview"}}),
        job_id,
    )  # fmt: skip
    job = queue.claim(db, "gpu-worker", "gpu")
    assert sup.execute(job) == "failed"
    error = seed.one("SELECT error FROM reconstruction_runs WHERE id = %s", run_id)["error"]
    assert error["code"] == "DISCONNECTED_CAPTURE"
    assert error["suggestedAction"] and "details" in error


class FakeColmapRegistering(FakeColmap):
    """Registers every frame in one model, then undistorts it (images + sparse files)."""

    def run(self, argv, *, timeout, check_cancelled=None, **kw):
        argv = [str(a) for a in argv]
        if argv[:2] == ["colmap", "mapper"]:
            images = sorted(Path(argv[argv.index("--image_path") + 1]).rglob("*.jpg"))
            model = Path(argv[argv.index("--output_path") + 1]) / "0"
            model.mkdir(parents=True)
            blob = struct.pack("<Q", len(images))
            for i, image in enumerate(images):
                name = image.relative_to(image.parent.parent).as_posix()
                # identity rotation, camera i at x = i (t = -centre)
                blob += struct.pack("<I7dI", i + 1, 1.0, 0, 0, 0, -float(i), 0, 0, 1)
                blob += name.encode() + bytes(1) + struct.pack("<Q", 0)
            (model / "images.bin").write_bytes(blob)
            (model / "points3D.bin").write_bytes(struct.pack("<Q", 0))
        elif argv[:2] == ["colmap", "image_undistorter"]:
            out = Path(argv[argv.index("--output_path") + 1])
            model = Path(argv[argv.index("--input_path") + 1])
            (out / "images").mkdir(parents=True)
            (out / "sparse").mkdir()
            for item in model.iterdir():
                (out / "sparse" / item.name).write_bytes(item.read_bytes())
        return RunResult(tuple(argv), 0, "", "", 0.01)


def test_reconstruct_records_the_camera_path_in_walk_order(db, seed, store, settings):
    from andarilho_worker.stages import STAGES

    settings_gpu = Settings(**{**settings.__dict__, "worker_capability": "gpu"})
    sup = Supervisor(db, store, settings_gpu, runner=FakeColmapRegistering([]), stages=STAGES)
    pid = seed.project()
    run_id, job_id = seed.run(pid, quality="preview", first_stage="camera_reconstruction",
                              capability="gpu")  # fmt: skip
    names = [f"00/frame_{i:05d}.jpg" for i in range(40, 0, -1)]  # not in walk order
    prefix = f"tenants/{pid}/runs/{run_id}/attempts/preparing/1/images/"
    for name in names:
        store.put_bytes(prefix + name, b"\xff\xd8\xff")
    seed.exec(
        "UPDATE jobs SET input_manifest = input_manifest || %s::jsonb WHERE id = %s",
        json.dumps({"outputs": {"framesPrefix": prefix, "frameNames": names, "quality": "preview"}}),
        job_id,
    )  # fmt: skip
    assert sup.execute(queue.claim(db, "gpu-worker", "gpu")) == "succeeded"
    outputs = seed.one("SELECT input_manifest FROM jobs WHERE stage = 'normalizing'")[
        "input_manifest"
    ]["outputs"]
    record = json.loads(store.get_bytes(outputs["camerasKey"]))
    assert record["frame"] == "colmap"
    cameras = record["cameras"]
    assert [c["name"] for c in cameras] == sorted(names)
    assert cameras[0]["center"] == pytest.approx([0, 0, 0])
    assert cameras[-1]["center"] == pytest.approx([39, 0, 0])
    assert cameras[0]["forward"] == pytest.approx([0, 0, 1])


class _StubS3:
    """Just enough of a versioned S3 client for delete_prefix."""

    def __init__(self, keys: list[str], failing: set[str]) -> None:
        self.versions = {k: f"v-{k}" for k in keys}
        self.failing = failing

    def get_paginator(self, name):
        stub = self

        class Paginator:
            def paginate(self, Bucket, Prefix):  # noqa: N803 - boto3 names
                keys = [k for k in stub.versions if k.startswith(Prefix)]
                if name == "list_object_versions":
                    yield {"Versions": [{"Key": k, "VersionId": stub.versions[k]} for k in keys]}
                else:
                    yield {"Contents": []}

        return Paginator()

    def delete_objects(self, Bucket, Delete):  # noqa: N803 - boto3 names
        errors = []
        for item in Delete["Objects"]:
            if item["Key"] in self.failing:
                errors.append({"Key": item["Key"], "Code": "AccessDenied"})
            else:
                self.versions.pop(item["Key"], None)
        return {"Errors": errors} if errors else {}


def test_delete_prefix_fails_when_s3_reports_objects_it_did_not_delete():
    from andarilho_worker.artifacts import S3ObjectStore
    from andarilho_worker.errors import TransientError

    keys = [f"tenants/p1/uploads/u1/part-{i}" for i in range(3)]
    flaky = S3ObjectStore("b", _StubS3(keys, failing={keys[1]}))
    with pytest.raises(TransientError):
        flaky.delete_prefix("tenants/p1/uploads/u1")
    healthy = S3ObjectStore("b", _StubS3(keys, failing=set()))
    assert healthy.delete_prefix("tenants/p1/uploads/u1") == 3
    assert healthy.client.versions == {}


def test_put_immutable_never_replaces_committed_bytes(tmp_path):
    from andarilho_worker.artifacts import LocalObjectStore, put_immutable
    from andarilho_worker.errors import StageError

    store = LocalObjectStore(tmp_path)
    key = "tenants/p1/scenes/r1/collision.glb"
    assert store.create_new(key, b"winner") is not None
    assert store.create_new(key, b"stale") is None  # atomic create refuses to overwrite
    assert put_immutable(store, key, b"winner", "model/gltf-binary") is not None
    with pytest.raises(StageError):
        put_immutable(store, key, b"stale", "model/gltf-binary")
    assert store.get_bytes(key) == b"winner"


class _RecordingCtx:
    def __init__(self):
        self.calls = []

    def run(self, argv, **kw):
        self.calls.append([str(a) for a in argv])


@pytest.mark.parametrize(
    "names,expected",
    [
        (["00/frame_00001.jpg", "00/frame_00002.jpg"], [["sequential_matcher"]]),
        (["photos_1600x1200/photo_00000.jpg", "photos_1600x1200/photo_00001.jpg"],
         [["exhaustive_matcher"]]),
        (["00/frame_00001.jpg", "photos_1600x1200/photo_00000.jpg"],
         [["sequential_matcher"], ["matches_importer"]]),
    ],
)  # fmt: skip
def test_matcher_respects_photo_ordering(tmp_path, names, expected):
    from andarilho_worker.stages.reconstruct import match_features, photo_pairs

    ctx = _RecordingCtx()
    match_features(ctx, tmp_path / "db", names, tmp_path, overlap=10)
    assert [c[1:2] for c in ctx.calls] == expected
    if expected[-1] == ["matches_importer"]:
        written = (tmp_path / "photo-pairs.txt").read_text().split("\n")
        assert written[0] == "00/frame_00001.jpg photos_1600x1200/photo_00000.jpg"
    assert photo_pairs(["a/1.jpg", "photos_1x1/p.jpg", "photos_1x1/q.jpg"]) == [
        ("a/1.jpg", "photos_1x1/p.jpg"),
        ("a/1.jpg", "photos_1x1/q.jpg"),
        ("photos_1x1/p.jpg", "photos_1x1/q.jpg"),
    ]


def _dataset_archive(tmp_path: Path, images: list[ColmapImage], points: SparsePoints) -> Path:
    """A dataset.tar as camera_reconstruction writes it: frames under images/, the model in sparse/0."""
    root = tmp_path / "dataset"
    model = root / "sparse" / "0"
    model.mkdir(parents=True)
    write_cameras_bin(
        model / "cameras.bin", [ColmapCamera(1, "PINHOLE", 640, 480, (500.0, 500.0, 320.0, 240.0))]
    )
    write_images_bin(model / "images.bin", images)
    write_points3d_bin(model / "points3D.bin", points)
    frames = root / "images"
    frames.mkdir(parents=True)
    for image in images:
        (frames / image.name).write_bytes(b"\xff\xd8\xff" + bytes(64 * 1024))
    archive = tmp_path / "dataset.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(frames, arcname="images")
        tar.add(root / "sparse", arcname="sparse")
    return archive


def _walk_scene(seed: int = 3, *, extra_points: int = 0):
    """A walk through a 5 x 3.5 m room, as COLMAP stores it: (kinds, images, points)."""
    rng = np.random.default_rng(seed)
    transform = colmap_transform()
    room, kind = make_room(rng)
    centres, forwards = l_walk(rng)
    images = cameras_for(centres, forwards, transform)
    points = to_sparse(room, transform, rng)
    if extra_points:
        clutter = rng.uniform([0, 0, 0], [5, 2.6, 3.5], (extra_points, 3))
        points = join_points(points, to_sparse(clutter, transform, rng))
    return kind, images, points


DATASET_KEY = "tenants/p/runs/run/attempts/camera_reconstruction/job/1/dataset.tar"


def test_normalize_frames_the_sparse_model_and_writes_its_evidence(tmp_path, store, settings):
    kind, images, points = _walk_scene()
    store.put_file(DATASET_KEY, _dataset_archive(tmp_path, images, points), "application/x-tar")
    ctx = stage_context(
        tmp_path, store, settings, {"datasetKey": DATASET_KEY, "quality": "preview"}
    )

    result = normalize.run(ctx)

    outputs = result.outputs
    assert set(outputs) == {
        "worldFromColmap", "scaleStatus", "proposalBounds", "proposalSpawn", "frameWarnings",
        "evidenceKey",
    }  # fmt: skip
    assert similarity_matrix_problem(outputs["worldFromColmap"]) is None
    assert outputs["scaleStatus"] == "estimated"
    floor = apply_affine(outputs["worldFromColmap"], points.xyz[kind == FLOOR])
    assert np.percentile(np.abs(floor[:, 1]), 95) < 0.03
    cameras = apply_affine(outputs["worldFromColmap"], np.array([i.center() for i in images]))
    bounds = outputs["proposalBounds"]
    assert bounds["floorY"] == 0.0
    assert np.all(np.array(bounds["minXZ"]) <= cameras[:, [0, 2]].min(axis=0))
    assert np.all(np.array(bounds["maxXZ"]) >= cameras[:, [0, 2]].max(axis=0))
    assert outputs["proposalSpawn"]["feet"][0] == pytest.approx(cameras[0, 0], abs=0.01)
    assert outputs["proposalSpawn"]["feet"][1] == 0.0
    assert result.manifest["sparsePoints"] == len(points.xyz)
    assert result.manifest["cameras"] == len(images)
    assert result.manifest["frame"]["method"] == "floor_plane"


def test_capture_evidence_has_the_editor_shape_and_three_decimals(tmp_path, store, settings):
    _kind, images, points = _walk_scene()
    store.put_file(DATASET_KEY, _dataset_archive(tmp_path, images, points), "application/x-tar")
    ctx = stage_context(
        tmp_path, store, settings, {"datasetKey": DATASET_KEY, "quality": "preview"}
    )
    result = normalize.run(ctx)

    blob = store.get_bytes(result.outputs["evidenceKey"])
    evidence = json.loads(blob)
    assert list(evidence) == ["schemaVersion", "units", "points", "cameras"]
    assert evidence["schemaVersion"] == 1 and evidence["units"] == "m"
    assert b" " not in blob  # compact separators
    assert result.manifest["evidencePoints"] == len(evidence["points"]) == len(points.xyz)
    assert all(v == round(v, 3) for row in evidence["points"] for v in row)
    assert all(v == round(v, 3) for row in evidence["cameras"] for v in row)
    # The cameras are in walk order (sorted by name) and in the canonical world.
    canonical = apply_affine(
        result.outputs["worldFromColmap"], np.array([i.center() for i in images])
    )
    assert np.array(evidence["cameras"]) == pytest.approx(canonical, abs=1e-3)
    assert len(evidence["cameras"]) == len(images)


def test_capture_evidence_keeps_at_most_five_thousand_points_in_their_order():
    _kind, images, points = _walk_scene(extra_points=4000)
    frame = frame_from_sparse(points, images, seed=0)
    blob, kept = normalize.capture_evidence(frame, images, points, seed=0)
    evidence = json.loads(blob)
    assert kept == len(evidence["points"]) == normalize.EVIDENCE_POINTS
    assert normalize.capture_evidence(frame, images, points, seed=0)[0] == blob
    assert normalize.capture_evidence(frame, images, points, seed=1)[0] != blob
    # Every kept point is one of the reliable points, in the order they were stored.
    reliable = np.round(apply_affine(frame.matrix, reliable_points(points)), 3).tolist()
    position: dict[tuple, int] = {}
    for index, row in enumerate(reliable):
        position.setdefault(tuple(row), index)
    order = [position[tuple(row)] for row in evidence["points"]]
    assert order == sorted(order)


def test_image_frames_are_never_extracted(tmp_path, store, settings):
    _kind, images, points = _walk_scene()
    store.put_file(DATASET_KEY, _dataset_archive(tmp_path, images, points), "application/x-tar")
    ctx = stage_context(
        tmp_path, store, settings, {"datasetKey": DATASET_KEY, "quality": "preview"}
    )

    normalize.run(ctx)

    names = {p.relative_to(ctx.scratch).as_posix() for p in ctx.scratch.rglob("*")}
    assert "normalize/sparse/0/images.bin" in names
    assert not [n for n in names if "images" in n.split("/")]
    assert not (ctx.scratch / "dataset.tar").exists()  # the archive is removed once read


@pytest.mark.parametrize("empty", ["cameras", "points", "archive"])
def test_a_model_without_cameras_or_points_is_a_registration_failure(
    tmp_path, store, settings, empty
):
    _kind, images, points = _walk_scene()
    if empty == "archive":
        store.put_bytes(DATASET_KEY, b"not a tar archive")
    else:
        if empty == "cameras":
            images = []
        else:
            points = SparsePoints(
                np.empty((0, 3)), np.empty((0, 3), np.uint8), np.empty(0), np.empty(0, np.int64)
            )
        store.put_file(DATASET_KEY, _dataset_archive(tmp_path, images, points), "application/x-tar")
    ctx = stage_context(
        tmp_path, store, settings, {"datasetKey": DATASET_KEY, "quality": "preview"}
    )

    with pytest.raises(StageError) as error:
        normalize.run(ctx)
    assert error.value.code == "CAMERA_REGISTRATION_FAILED"


def test_a_missing_dataset_is_a_missing_input(tmp_path, store, settings):
    ctx = stage_context(tmp_path, store, settings, {"datasetKey": DATASET_KEY})
    with pytest.raises(StageError) as error:
        normalize.run(ctx)
    assert error.value.code == "MISSING_INPUT"


def test_the_synthetic_recipe_writes_no_files_and_only_the_bundled_scale(tmp_path, store, settings):
    ctx = stage_context(tmp_path, store, settings, {"quality": "synthetic"}, quality="synthetic")

    result = normalize.run(ctx)

    assert result.outputs == {"scaleStatus": "user_measured", "proposalBounds": None}
    assert store.list_keys(ctx.attempt_key("")) == [ctx.attempt_key("result.json")]


def test_normalizing_hands_the_frame_and_the_evidence_to_the_proposal(
    db, seed, store, settings, tmp_path
):
    from andarilho_worker.stages import STAGES

    _kind, images, points = _walk_scene()
    pid = seed.project()
    run_id, job_id = seed.run(pid, quality="preview", first_stage="normalizing")
    key = f"tenants/{pid}/runs/{run_id}/attempts/camera_reconstruction/c1/1/dataset.tar"
    store.put_file(key, _dataset_archive(tmp_path, images, points), "application/x-tar")
    seed.exec(
        "UPDATE jobs SET input_manifest = input_manifest || %s::jsonb WHERE id = %s",
        json.dumps({"outputs": {"datasetKey": key, "quality": "preview"}}), job_id,
    )  # fmt: skip
    sup = Supervisor(db, store, settings, stages=STAGES)
    assert sup.execute(queue.claim(db, "cpu-worker", "cpu")) == "succeeded"
    # scene_analysis comes next and passes every earlier output on to the proposal.
    row = seed.one("SELECT input_manifest FROM jobs WHERE stage = 'scene_analysis'")
    outputs = row["input_manifest"]["outputs"]
    assert outputs["scaleStatus"] == "estimated"
    assert similarity_matrix_problem(outputs["worldFromColmap"]) is None
    assert outputs["evidenceKey"].startswith(f"tenants/{pid}/runs/{run_id}/attempts/normalizing/")
    assert json.loads(store.get_bytes(outputs["evidenceKey"]))["schemaVersion"] == 1
