"""``scene_analysis`` stage: the model is shown posed keyframes, photos and the floor plan.

The capture is the box room of ``room_dataset`` written as a real dataset archive under a
run's prefix. The model is a fake gateway (``httpx.MockTransport``), so nothing touches the
network. A provider failure is recorded in the artifact and never fails the stage.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import shutil
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from pdf_fixtures import classic_pdf, page_tree
from PIL import Image
from room_dataset import write_room_dataset
from stage_fixtures import stage_context

from andarilho_worker.ai import VisionClient
from andarilho_worker.errors import Cancelled, StageError
from andarilho_worker.geometry.colmap import read_images_bin, write_images_bin
from andarilho_worker.geometry.texture import camera_in_world
from andarilho_worker.settings import Settings
from andarilho_worker.stages import analyze_scene
from andarilho_worker.stages.analyze_scene import INVALID_WARNING, UNAVAILABLE_WARNING

FIXTURES = Path(__file__).parent / "fixtures" / "analysis"
PID, RUN = "p", "run"
KEY = "sk-test-0123456789abcdef"
BASE = "https://gateway.test/v1"
MODEL = "deepseek-v4.1-flash"
ROOM_FRAMES = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]  # the keyframes of the box room


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Capture:
    key: str  # the dataset's object key under the run
    world: list[float]  # worldFromColmap, column major, as normalizing records it
    tar: Path  # the local archive that was stored


def _store_capture(store, tar: Path, world: np.ndarray) -> Capture:
    key = f"tenants/{PID}/runs/{RUN}/attempts/camera_reconstruction/c1/1/dataset.tar"
    store.put_file(key, tar, "application/x-tar")
    return Capture(key, world.reshape(-1, order="F").tolist(), tar)


@pytest.fixture
def capture(store, tmp_path) -> Capture:
    tar = tmp_path / "dataset.tar"
    world = write_room_dataset(tmp_path / "dataset", tar)
    return _store_capture(store, tar, world)


def _photo_capture(store, tmp_path, photos: set[int]) -> Capture:
    """The box room with the frames at ``photos`` registered as supplementary photos."""
    world = write_room_dataset(tmp_path / "room", tmp_path / "room.tar")
    work = tmp_path / "photos"
    with tarfile.open(tmp_path / "room.tar") as source:
        source.extractall(work, filter="data")
    images = read_images_bin(work / "sparse" / "0" / "images.bin")
    (work / "images" / "photos_160x120").mkdir()
    renamed = []
    for image in images:
        index = int(image.name.rsplit("_", 1)[1].split(".")[0]) - 1
        name = image.name
        if index in photos:
            name = f"photos_160x120/photo_{index:05d}.jpg"
            shutil.move(work / "images" / image.name, work / "images" / name)
        renamed.append(dataclasses.replace(image, name=name))
    write_images_bin(work / "sparse" / "0" / "images.bin", renamed)
    tar = tmp_path / "photo-dataset.tar"
    with tarfile.open(tar, "w") as out:
        out.add(work / "images", arcname="images")
        out.add(work / "sparse", arcname="sparse")
    return _store_capture(store, tar, world)


@pytest.fixture
def ai_key(tmp_path) -> Path:
    path = tmp_path / "ai-key"
    path.write_text(KEY + "\n", encoding="utf-8")
    return path


@pytest.fixture
def ai_settings(tmp_path, ai_key) -> Settings:
    return Settings(
        database_url="postgresql+psycopg://unused@localhost:1/unused",
        ai_scene_analysis=True,
        ai_base_url=BASE,
        ai_model=MODEL,
        opencode_api_key_file=str(ai_key),
        ai_timeout_seconds=30.0,
        worker_scratch_dir=tmp_path / "scratch",
    )


def _outputs(capture: Capture, **extra: Any) -> dict[str, Any]:
    return {"datasetKey": capture.key, "worldFromColmap": capture.world, **extra}


def _context(tmp_path, store, settings, outputs, *, quality="preview", params=None):
    ctx = stage_context(tmp_path, store, settings, outputs, stage="scene_analysis", quality=quality)
    if params:
        analysis = {**ctx.recipe["params"]["analysis"], **params}
        recipe = {**ctx.recipe, "params": {**ctx.recipe["params"], "analysis": analysis}}
        ctx = dataclasses.replace(ctx, recipe=recipe)
    return ctx


class Gateway:
    """A fake model: records each request and answers it with ``respond``."""

    def __init__(self, respond: Callable[[httpx.Request], httpx.Response]) -> None:
        self.respond = respond
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    def parts(self) -> list[dict[str, Any]]:
        (body,) = self.bodies()
        return body["messages"][1]["content"]


@pytest.fixture
def gateway(monkeypatch) -> Callable[[Callable[[httpx.Request], httpx.Response]], Gateway]:
    def install(respond) -> Gateway:
        fake = Gateway(respond)

        def from_settings(cls, settings):
            return VisionClient(
                base_url=settings.ai_base_url,
                model=settings.ai_model,
                key_file=settings.opencode_api_key_file,
                timeout_seconds=settings.ai_timeout_seconds,
                transport=httpx.MockTransport(fake.handler),
            )

        monkeypatch.setattr(VisionClient, "from_settings", classmethod(from_settings))
        return fake

    return install


@pytest.fixture
def no_call(monkeypatch) -> None:
    def from_settings(cls, settings):
        raise AssertionError("the model must not be called")

    monkeypatch.setattr(VisionClient, "from_settings", classmethod(from_settings))


def answer(obj: Any, usage: dict[str, int] | None = None):
    """A responder for a chat completion whose message is ``obj`` as JSON."""

    def respond(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = {
            "choices": [{"message": {"role": "assistant", "content": json.dumps(obj)}}]
        }
        if usage is not None:
            body["usage"] = usage
        return httpx.Response(200, json=body)

    return respond


def _good() -> dict[str, Any]:
    return json.loads((FIXTURES / "good_reply.json").read_text(encoding="utf-8"))


def _decode(url: str) -> Image.Image:
    prefix, encoded = url.split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    image.load()
    return image


def _image_urls(parts: list[dict[str, Any]]) -> list[str]:
    return [part["image_url"]["url"] for part in parts if part["type"] == "image_url"]


def _texts(parts: list[dict[str, Any]]) -> list[str]:
    return [part["text"] for part in parts if part["type"] == "text"]


def _artifact(store, result) -> dict[str, Any]:
    return json.loads(store.get_bytes(result.outputs["analysisKey"]))


# --------------------------------------------------------------------------
# A complete analysis
# --------------------------------------------------------------------------


def test_a_complete_analysis_is_written_for_the_proposal(
    tmp_path, store, ai_settings, capture, gateway
):
    usage = {"prompt_tokens": 900, "completion_tokens": 120, "total_tokens": 1020}
    fake = gateway(answer(_good(), usage))
    ctx = _context(tmp_path, store, ai_settings, _outputs(capture))
    result = analyze_scene.run(ctx)

    assert result.outputs["analysisStatus"] == "complete"
    assert result.outputs["analysisWarnings"] == []
    assert result.outputs["analysisKey"] == ctx.attempt_key("analysis.json")
    artifact = _artifact(store, result)
    assert artifact["schemaVersion"] == 1 and artifact["status"] == "complete"
    assert artifact["reason"] is None and artifact["floorY"] == 0.0
    assert artifact["model"] == MODEL
    assert artifact["reply"] == _good()  # the good reply is already in the reply's shape

    assert [record["frame"] for record in artifact["keyframes"]] == ROOM_FRAMES
    extracted = tmp_path / "check"
    with tarfile.open(capture.tar) as tar:
        tar.extractall(extracted, filter="data")
    images = {i.name: i for i in read_images_bin(extracted / "sparse" / "0" / "images.bin")}
    for record in artifact["keyframes"]:
        image = images[record["name"]]
        rotation, translation = camera_in_world(
            image.rotation(), np.asarray(image.tvec), capture.world, 1.0
        )
        assert np.allclose(record["rotation"], rotation.reshape(-1), atol=1e-9)
        assert np.allclose(record["translation"], translation, atol=1e-9)
        assert record["kind"] == "video"
        assert (record["width"], record["height"]) == (160, 120)
        assert record["intrinsics"] == [110.0, 110.0, 80.0, 60.0]

    (request,) = fake.requests
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["user-agent"] == "andarilho-worker"
    assert KEY not in json.dumps(artifact) and KEY not in json.dumps(result.manifest)

    manifest = result.manifest
    assert manifest["status"] == "complete" and manifest["reason"] is None
    assert manifest["keyframes"] == 12 and manifest["photos"] == 0
    assert manifest["floorPlanSent"] is False and manifest["imageBytes"] > 0
    assert manifest["usage"] == {"promptTokens": 900, "completionTokens": 120, "totalTokens": 1020}
    assert manifest["model"] == MODEL and manifest["seconds"] >= 0
    assert "data:image" not in json.dumps(manifest)


def test_the_model_is_sent_only_downscaled_jpegs_of_the_selected_frames(
    tmp_path, store, ai_settings, capture, gateway
):
    fake = gateway(answer(_good()))
    ctx = _context(tmp_path, store, ai_settings, _outputs(capture), params={"imageLongEdge": 80})
    analyze_scene.run(ctx)

    urls = _image_urls(fake.parts())
    assert len(urls) == 12
    for url in urls:
        image = _decode(url)
        assert image.size == (80, 60)
    # Only the selected frames were extracted from the archive; nothing else left scratch.
    extracted = sorted(p.name for p in (ctx.scratch / "analysis" / "images").rglob("*.jpg"))
    assert len(extracted) == 12
    assert not (ctx.scratch / "analysis" / "dataset.tar").exists()
    labels = _texts(fake.parts())
    assert labels.count("Frame 0:") == 1 and "Frame 1:" not in labels


def test_frame_labels_come_before_their_images(tmp_path, store, ai_settings, capture, gateway):
    fake = gateway(answer({"rooms": [], "frames": [], "objects": []}))
    analyze_scene.run(_context(tmp_path, store, ai_settings, _outputs(capture)))
    parts = fake.parts()
    labelled = [p for p in parts if p["type"] == "text" and p["text"].startswith("Frame ")]
    assert len(labelled) == 12
    for label_part in labelled:
        index = parts.index(label_part)
        assert parts[index + 1]["type"] == "image_url"
    assert parts[-1]["text"].startswith("Reply with one JSON object")


# --------------------------------------------------------------------------
# Photos and the floor plan
# --------------------------------------------------------------------------


def test_registered_photos_come_first_and_are_marked_as_photos(
    tmp_path, store, ai_settings, gateway
):
    capture = _photo_capture(store, tmp_path, photos={2, 10, 20})
    fake = gateway(answer({"rooms": [], "frames": [], "objects": []}))
    ctx = _context(
        tmp_path, store, ai_settings, _outputs(capture), params={"keyframes": 5, "maxPhotos": 2}
    )
    result = analyze_scene.run(ctx)
    artifact = _artifact(store, result)
    kinds = [record["kind"] for record in artifact["keyframes"]]
    assert kinds.count("photo") == 2 and kinds.count("video") == 5
    photo_frames = [r["frame"] for r in artifact["keyframes"] if r["kind"] == "photo"]
    labels = _texts(fake.parts())
    for frame in photo_frames:
        assert f"Frame {frame} (photo):" in labels
    assert result.manifest["photos"] == 2


def test_a_png_floor_plan_is_sent_at_its_own_size_limit(
    tmp_path, store, ai_settings, capture, gateway
):
    plan = Image.new("RGB", (400, 300), (255, 255, 255))
    buffer = io.BytesIO()
    plan.save(buffer, format="PNG")
    key = f"tenants/{PID}/uploads/u1/original"
    store.put_bytes(key, buffer.getvalue(), "image/png")
    floor = {
        "uploadId": "u1",
        "key": key,
        "sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
        "contentType": "image/png",
    }
    fake = gateway(answer({"rooms": [], "frames": [], "objects": []}))
    ctx = _context(
        tmp_path,
        store,
        ai_settings,
        _outputs(capture, floorPlan=floor),
        params={"floorPlanLongEdge": 100},
    )
    result = analyze_scene.run(ctx)
    parts = fake.parts()
    index = parts.index({"type": "text", "text": "Floor plan:"})
    assert _decode(parts[index + 1]["image_url"]["url"]).size == (100, 75)
    assert result.manifest["floorPlanSent"] is True


def test_a_pdf_floor_plan_without_an_image_is_skipped(
    tmp_path, store, ai_settings, capture, gateway
):
    pdf = classic_pdf(page_tree(1))
    key = f"tenants/{PID}/uploads/u2/original"
    store.put_bytes(key, pdf, "application/pdf")
    floor = {
        "uploadId": "u2",
        "key": key,
        "sha256": hashlib.sha256(pdf).hexdigest(),
        "contentType": "application/pdf",
    }
    fake = gateway(answer({"rooms": [], "frames": [], "objects": []}))
    ctx = _context(tmp_path, store, ai_settings, _outputs(capture, floorPlan=floor))
    result = analyze_scene.run(ctx)
    assert result.outputs["analysisStatus"] == "complete"
    assert "Floor plan:" not in _texts(fake.parts())
    assert result.manifest["floorPlanSent"] is False


@pytest.mark.parametrize("problem", ["outside", "changed", "too_large"])
def test_a_floor_plan_that_is_not_ours_or_not_as_verified_is_skipped(
    problem, tmp_path, store, ai_settings, capture, gateway, monkeypatch
):
    plan = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 200, 200)).save(plan, format="PNG")
    data = plan.getvalue()
    key = "tenants/other/uploads/u3/original" if problem == "outside" else f"tenants/{PID}/u3"
    store.put_bytes(key, data, "image/png")
    sha = "0" * 64 if problem == "changed" else hashlib.sha256(data).hexdigest()
    if problem == "too_large":
        monkeypatch.setattr(analyze_scene, "MAX_FLOOR_PLAN_BYTES", 16)
    floor = {"uploadId": "u3", "key": key, "sha256": sha, "contentType": "image/png"}
    fake = gateway(answer({"rooms": [], "frames": [], "objects": []}))
    result = analyze_scene.run(
        _context(tmp_path, store, ai_settings, _outputs(capture, floorPlan=floor))
    )
    assert result.manifest["floorPlanSent"] is False
    assert "Floor plan:" not in _texts(fake.parts())


# --------------------------------------------------------------------------
# Failures the run survives
# --------------------------------------------------------------------------


def _timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("the model is slow", request=request)


@pytest.mark.parametrize(
    ("respond", "reason"),
    [
        pytest.param(lambda request: httpx.Response(402), "no_credit", id="402"),
        pytest.param(lambda request: httpx.Response(401), "auth", id="401"),
        pytest.param(lambda request: httpx.Response(429), "rate_limited", id="429"),
        pytest.param(lambda request: httpx.Response(503), "server", id="503"),
        pytest.param(_timeout, "timeout", id="timeout"),
    ],
)
def test_an_unavailable_model_is_recorded_and_the_stage_succeeds(
    respond, reason, tmp_path, store, ai_settings, capture, gateway
):
    gateway(respond)
    ctx = _context(tmp_path, store, ai_settings, _outputs(capture))
    result = analyze_scene.run(ctx)
    assert result.outputs["analysisStatus"] == "unavailable"
    assert result.outputs["analysisWarnings"] == [UNAVAILABLE_WARNING]
    artifact = _artifact(store, result)
    assert artifact["status"] == "unavailable" and artifact["reason"] == reason
    assert artifact["reply"] is None
    assert len(artifact["keyframes"]) == 12
    assert result.manifest["status"] == "unavailable" and result.manifest["reason"] == reason
    assert result.manifest["usage"] == {}


def test_a_reply_that_is_not_usable_is_invalid_and_the_stage_succeeds(
    tmp_path, store, ai_settings, capture, gateway
):
    gateway(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "nope"}}]})
    )
    result = analyze_scene.run(_context(tmp_path, store, ai_settings, _outputs(capture)))
    assert result.outputs["analysisStatus"] == "invalid"
    assert result.outputs["analysisWarnings"] == [INVALID_WARNING]
    artifact = _artifact(store, result)
    assert artifact["status"] == "invalid" and artifact["reason"] == "bad_reply"
    assert artifact["reply"] is None


def test_a_json_reply_without_rooms_frames_or_objects_is_invalid(
    tmp_path, store, ai_settings, capture, gateway
):
    gateway(answer({"unexpected": True}))
    result = analyze_scene.run(_context(tmp_path, store, ai_settings, _outputs(capture)))
    assert result.outputs["analysisStatus"] == "invalid"
    assert _artifact(store, result)["reply"] is None


def test_without_a_key_the_model_is_unavailable_and_never_called(
    tmp_path, store, ai_settings, capture, gateway
):
    fake = gateway(answer(_good()))
    settings = dataclasses.replace(ai_settings, opencode_api_key_file=None)
    result = analyze_scene.run(_context(tmp_path, store, settings, _outputs(capture)))
    assert fake.requests == []
    assert result.outputs["analysisStatus"] == "unavailable"
    assert _artifact(store, result)["reason"] == "no_key"


# --------------------------------------------------------------------------
# Runs that make no call
# --------------------------------------------------------------------------


def test_with_the_switch_off_nothing_is_read_or_sent(tmp_path, store, ai_settings, no_call):
    settings = dataclasses.replace(ai_settings, ai_scene_analysis=False)
    ctx = _context(tmp_path, store, settings, {})  # no dataset: none is needed
    result = analyze_scene.run(ctx)
    assert result.outputs == {"analysisStatus": "disabled", "analysisWarnings": []}
    assert ctx.attempt_key("analysis.json") not in store.list_keys(ctx.attempt_key(""))
    assert not (ctx.scratch / "analysis").exists()


@pytest.mark.parametrize("where", ["recipe", "outputs"])
def test_the_synthetic_recipe_is_skipped(where, tmp_path, store, ai_settings, no_call):
    if where == "recipe":
        ctx = _context(tmp_path, store, ai_settings, {}, quality="synthetic")
    else:
        ctx = _context(tmp_path, store, ai_settings, {"quality": "synthetic"})
    result = analyze_scene.run(ctx)
    assert result.outputs == {"analysisStatus": "skipped", "analysisWarnings": []}
    assert ctx.attempt_key("analysis.json") not in store.list_keys(ctx.attempt_key(""))


# --------------------------------------------------------------------------
# Refusals and cancellation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        f"tenants/{PID}/runs/other-run/attempts/camera_reconstruction/c/1/dataset.tar",
        f"tenants/other/runs/{RUN}/attempts/camera_reconstruction/c/1/dataset.tar",
    ],
    ids=["another run", "another project"],
)
def test_a_dataset_outside_this_run_is_missing_input(key, tmp_path, store, ai_settings, no_call):
    ctx = _context(tmp_path, store, ai_settings, {"datasetKey": key, "worldFromColmap": [1.0]})
    with pytest.raises(StageError) as caught:
        analyze_scene.run(ctx)
    assert caught.value.code == "MISSING_INPUT"


def test_without_a_dataset_the_stage_is_missing_input(tmp_path, store, ai_settings, no_call):
    ctx = _context(tmp_path, store, ai_settings, {"worldFromColmap": [1.0]})
    with pytest.raises(StageError) as caught:
        analyze_scene.run(ctx)
    assert caught.value.code == "MISSING_INPUT"


def test_cancellation_before_the_call_sends_nothing(tmp_path, store, ai_settings, capture, gateway):
    fake = gateway(answer(_good()))
    ctx = _context(tmp_path, store, ai_settings, _outputs(capture))

    def cancel_before_asking(message, progress):
        if message.startswith("Asking"):
            ctx.cancellation.cancel()
        return True

    ctx.services.progress = cancel_before_asking
    with pytest.raises(Cancelled):
        analyze_scene.run(ctx)
    assert fake.requests == []
    assert ctx.attempt_key("analysis.json") not in store.list_keys(ctx.attempt_key(""))


def test_cancellation_after_the_call_writes_no_artifact(
    tmp_path, store, ai_settings, capture, gateway
):
    ctx = _context(tmp_path, store, ai_settings, _outputs(capture))

    def answer_then_cancel(request: httpx.Request) -> httpx.Response:
        ctx.cancellation.cancel()
        return answer(_good())(request)

    gateway(answer_then_cancel)
    with pytest.raises(Cancelled):
        analyze_scene.run(ctx)
    assert ctx.attempt_key("analysis.json") not in store.list_keys(ctx.attempt_key(""))
