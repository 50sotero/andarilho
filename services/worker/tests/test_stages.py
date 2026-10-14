"""Recipe, coverage-gate and media-stage tests with fake external tools (no FFmpeg/COLMAP/GPU)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from andarilho_worker import queue
from andarilho_worker.errors import StageError
from andarilho_worker.recipes import ARGV, RECIPES, recipe_for, render_argv
from andarilho_worker.runner import ProcessFailed, RunResult
from andarilho_worker.settings import Settings
from andarilho_worker.stages import prepare as prepare_mod
from andarilho_worker.stages.reconstruct import SparseModel, coverage_gate
from andarilho_worker.supervisor import Supervisor


def test_recipes_stage_order_and_capabilities():
    synthetic = recipe_for("synthetic")
    assert synthetic.stage_names() == [
        "validating", "preparing", "normalizing", "geometry_proposal", "packaging",
        "scene_validation",
    ]  # fmt: skip
    assert {s.capability for s in synthetic.stages} == {"cpu"}
    preview = recipe_for("preview")
    caps = {s.stage: s.capability for s in preview.stages}
    assert caps["camera_reconstruction"] == caps["training"] == "gpu"
    assert caps["preparing"] == "cpu"
    assert preview.next_stage("training").stage == "normalizing"
    assert preview.next_stage("scene_validation") is None
    assert preview.next_stage("geometry_proposal", layout_pipeline=True).stage == "packaging"
    digests = {r.digest for r in RECIPES.values()}
    assert len(digests) == 3 and all(d.startswith("sha256:") for d in digests)
    assert recipe_for("preview").digest == recipe_for("preview").digest
    with pytest.raises(KeyError):
        recipe_for("ultra")


def test_argv_templates_are_pinned():
    argv = render_argv("ffprobe", input=Path("/scratch/x.mp4"))
    assert argv[0] == "ffprobe" and argv[-1].endswith("x.mp4")
    with pytest.raises(KeyError):
        render_argv("ffprobe", input="a", extra="--evil")
    with pytest.raises(KeyError):
        render_argv("ns_train", data="d")
    train = render_argv("ns_train", data="d", output_dir="o", timestamp="t", iterations=10,
                        save_steps=5, seed=1)  # fmt: skip
    assert "--max-num-iterations" in train and "--viewer.quit-on-train-completion" in train
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


def _preparing_job(seed, store, pid, *, codec="h264", photos=1):
    run_id, job_id = seed.run(pid, quality="preview", first_stage="preparing")
    inputs = []
    sources = [("video", b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)]
    sources += [("photo", b"\xff\xd8\xff" + bytes([i]) * 64) for i in range(photos)]
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


def test_slow_sog_compression_falls_back_to_the_ply(db, seed, store, settings, monkeypatch):
    from andarilho_worker.geometry.ply import write_splat_ply
    from andarilho_worker.geometry.synthetic import synthetic_splats
    from andarilho_worker.runner import ProcessTimeout
    from andarilho_worker.stages import STAGES
    from andarilho_worker.stages import normalize as normalize_mod

    monkeypatch.setattr(normalize_mod, "tool_available", lambda name: True)
    timeouts: list[float] = []

    class SlowCompressor:
        def run(self, argv, *, timeout, **kw):
            timeouts.append(timeout)
            raise ProcessTimeout(f"{argv[0]} exceeded {timeout:.0f}s")

    pid = seed.project()
    run_id, job_id = seed.run(pid, quality="preview", first_stage="normalizing")
    ply_key = f"tenants/{pid}/runs/{run_id}/attempts/training/1/scene.ply"
    store.put_bytes(ply_key, write_splat_ply(synthetic_splats(normalize_mod.synthetic_layout())))
    seed.exec(
        "UPDATE jobs SET input_manifest = input_manifest || %s::jsonb WHERE id = %s",
        json.dumps({"outputs": {"plyKey": ply_key, "quality": "preview"}}), job_id,
    )  # fmt: skip
    sup = Supervisor(db, store, settings, runner=SlowCompressor(), stages=STAGES)
    assert sup.execute(queue.claim(db, "cpu-worker", "cpu")) == "succeeded"
    assert len(timeouts) == 1 and timeouts[0] <= normalize_mod.SOG_MAX_SECONDS
    outputs = seed.one("SELECT input_manifest FROM jobs WHERE stage = 'geometry_proposal'")
    assert outputs["input_manifest"]["outputs"]["renderFormat"] == "ply"
    result = seed.one("SELECT result_manifest FROM jobs WHERE stage = 'normalizing'")
    assert result["result_manifest"]["compression"]["reason"] == "timeout"


def test_sog_compression_drops_the_bands_above_the_recipe_sh_degree(
    db, seed, store, settings, monkeypatch
):
    from andarilho_worker.geometry.ply import write_splat_ply
    from andarilho_worker.geometry.synthetic import synthetic_splats
    from andarilho_worker.stages import STAGES
    from andarilho_worker.stages import normalize as normalize_mod

    monkeypatch.setattr(normalize_mod, "tool_available", lambda name: True)
    calls: list[list[str]] = []

    class Compressor:
        def run(self, argv, *, timeout, **kw):
            argv = [str(a) for a in argv]
            calls.append(argv)
            Path(argv[-1]).write_bytes(b"PK\x03\x04 sog")
            return RunResult(tuple(argv), 0, "", "", 0.01)

    pid = seed.project()
    run_id, job_id = seed.run(pid, quality="preview", first_stage="normalizing")
    ply_key = f"tenants/{pid}/runs/{run_id}/attempts/training/1/scene.ply"
    store.put_bytes(ply_key, write_splat_ply(synthetic_splats(normalize_mod.synthetic_layout())))
    seed.exec(
        "UPDATE jobs SET input_manifest = input_manifest || %s::jsonb WHERE id = %s",
        json.dumps({"outputs": {"plyKey": ply_key, "quality": "preview"}}), job_id,
    )  # fmt: skip
    sup = Supervisor(db, store, settings, runner=Compressor(), stages=STAGES)
    assert sup.execute(queue.claim(db, "cpu-worker", "cpu")) == "succeeded"
    (argv,) = calls
    assert argv[0] == "splat-transform"
    # An action after the input applies to it: drop the bands, then write the SOG.
    flag = argv.index("--filter-harmonics")
    assert argv[flag - 1].endswith("normalized.ply")
    assert argv[flag + 1] == str(RECIPES["preview"].params["renderShDegree"]) == "1"
    assert argv[-1].endswith("scene.sog")
    outputs = seed.one("SELECT input_manifest FROM jobs WHERE stage = 'geometry_proposal'")
    assert outputs["input_manifest"]["outputs"]["renderFormat"] == "sog"
    result = seed.one("SELECT result_manifest FROM jobs WHERE stage = 'normalizing'")
    compression = result["result_manifest"]["compression"]
    assert compression["format"] == "sog" and compression["shDegree"] == 1
    assert compression["bytes"] == len(b"PK\x03\x04 sog")


def test_normalize_streams_the_export_instead_of_reading_it_inline(
    db, seed, store, settings, monkeypatch
):
    from andarilho_worker.geometry.ply import write_splat_ply
    from andarilho_worker.geometry.synthetic import synthetic_splats
    from andarilho_worker.stages import STAGES
    from andarilho_worker.stages import normalize as normalize_mod

    layout = normalize_mod.synthetic_layout()

    monkeypatch.setattr(normalize_mod, "tool_available", lambda name: False)
    pid = seed.project()
    run_id, job_id = seed.run(pid, quality="preview", first_stage="normalizing")
    ply_key = f"tenants/{pid}/runs/{run_id}/attempts/training/1/scene.ply"
    store.put_bytes(ply_key, write_splat_ply(synthetic_splats(layout)))
    seed.exec(
        "UPDATE jobs SET input_manifest = input_manifest || %s::jsonb WHERE id = %s",
        json.dumps({"outputs": {"plyKey": ply_key, "quality": "preview"}}), job_id,
    )  # fmt: skip
    inline = store.get_bytes

    def no_inline_export(key, *args, **kwargs):
        assert key != ply_key, "large exports must not be read inline"
        return inline(key, *args, **kwargs)

    monkeypatch.setattr(store, "get_bytes", no_inline_export)
    sup = Supervisor(db, store, settings, stages=STAGES)
    job = queue.claim(db, "cpu-worker", "cpu")
    assert sup.execute(job) == "succeeded"
    outputs = seed.one("SELECT input_manifest FROM jobs WHERE stage = 'geometry_proposal'")
    assert outputs["input_manifest"]["outputs"]["renderFormat"] == "ply"
