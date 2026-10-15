"""``preparing``: probe media and extract bounded, metadata-free frames.

* FFprobe records codec, size, duration, frame rate and rotation;
* FFmpeg extracts 2-4 fps (recipe) capped at 400 frames overall, at most 1600 px
  on the long edge, drops audio/subtitle/data streams and strips EXIF/GPS
  (``-map_metadata -1``); FFmpeg's default autorotation normalizes orientation;
* video frames are chosen, not just sampled: candidates are decoded at
  ``keyframeOversample`` times that rate and the sharpest of each consecutive
  window is kept (:mod:`andarilho_worker.keyframes`), so fast turns contribute
  their least blurred frames while coverage stays even in time;
* every retained image keeps a source map entry (upload ID + timestamp);
* one retained frame becomes the scene's poster (WebP, at most 1280 px wide): the
  sharpest of the middle half of the first video, else the middle photo.

Frames are grouped per source (``images/<nn>/``) so COLMAP can assign one camera
per source instead of forcing a shared calibration onto unrelated cameras.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from andarilho_worker.artifacts import MAX_POSTER_BYTES
from andarilho_worker.errors import StageError
from andarilho_worker.keyframes import (
    SCORE_SIZE,
    pick_poster,
    pick_sharpest,
    sharpness_from_raw,
)
from andarilho_worker.media import (
    MAX_VIDEO_SECONDS,
    VIDEO_CODECS,
    camera_identity,
    jpeg_dimensions,
    parse_ffprobe,
)
from andarilho_worker.recipes import render_argv
from andarilho_worker.runner import ProcessFailed, ProcessTimeout, tool_available
from andarilho_worker.stages.base import StageContext, StageResult, finish

log = logging.getLogger(__name__)

PROBE_TIMEOUT = 120
MAX_CANDIDATES_PER_FRAME = 4  # bounds decode work and scratch for keyframe candidates
POSTER_WIDTH = 1280


def probe_video(ctx: StageContext, path: Path) -> dict[str, Any]:
    try:
        result = ctx.run(render_argv("ffprobe", input=path), timeout=PROBE_TIMEOUT)
        meta = parse_ffprobe(result.stdout)
    except ProcessFailed as exc:
        raise StageError("INVALID_MEDIA") from exc
    except (ValueError, KeyError) as exc:
        raise StageError("INVALID_MEDIA") from exc
    if meta["codec"] not in VIDEO_CODECS:
        raise StageError("UNSUPPORTED_MEDIA", details={"codec": str(meta["codec"])})
    if meta["durationSeconds"] <= 0 or meta["durationSeconds"] > MAX_VIDEO_SECONDS:
        raise StageError("INVALID_MEDIA", "A video is empty or too long.")
    return meta


def plan_frames(durations: list[float], photos: int, *, fps: float, cap: int) -> tuple[float, int]:
    """Choose an effective fps so videos + photos fit the frame cap. Returns (fps, video budget)."""
    budget = max(0, cap - photos)
    total = sum(durations)
    if total <= 0 or budget == 0:
        return fps, budget
    return min(fps, budget / total), budget


def extract_frames(
    ctx: StageContext,
    video: Path,
    folder: Path,
    rate: float,
    count: int,
    long_edge: int,
    timeout: float,
) -> list[float]:
    """Evenly sampled frames as ``frame_00001.jpg``...; returns their offsets in seconds."""
    ctx.run(
        render_argv(
            "ffmpeg_frames",
            input=video,
            fps=f"{rate:.4f}",
            long_edge=long_edge,
            max_frames=count,
            output_pattern=folder / "frame_%05d.jpg",
        ),
        timeout=timeout,
    )
    return [k / rate for k in range(len(list(folder.glob("frame_*.jpg"))))]


def extract_keyframes(
    ctx: StageContext,
    video: Path,
    folder: Path,
    staging: Path,
    rate: float,
    candidates: int,
    keep: int,
    long_edge: int,
    timeout: float,
) -> tuple[list[float], dict[str, Any], np.ndarray]:
    """The sharpest candidate of each window, renamed in order.

    Returns the kept frames' offsets, selection stats and the kept frames' sharpness.

    The candidates and their tiny grayscale copies come from the same ``fps``
    filter, so candidate ``i`` of both decodes is the same source frame.
    """
    staging.mkdir(parents=True, exist_ok=True)
    try:
        ctx.run(
            render_argv(
                "ffmpeg_frames",
                input=video,
                fps=f"{rate:.4f}",
                long_edge=long_edge,
                max_frames=candidates,
                output_pattern=staging / "cand_%05d.jpg",
            ),
            timeout=timeout,
        )
        raw = staging / "score.gray"
        ctx.run(
            render_argv(
                "ffmpeg_score_frames",
                input=video,
                fps=f"{rate:.4f}",
                size=SCORE_SIZE,
                max_frames=candidates,
                output=raw,
            ),
            timeout=timeout,
        )
        scores = sharpness_from_raw(raw)
        produced = sorted(staging.glob("cand_*.jpg"))
        n = min(len(produced), len(scores))
        picked = pick_sharpest(scores[:n], keep)
        for k, i in enumerate(picked):
            produced[i].replace(folder / f"frame_{k + 1:05d}.jpg")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    stats = {
        "candidates": n,
        "kept": len(picked),
        "medianSharpness": round(float(np.median(scores[:n])), 1) if n else None,
        "keptMedianSharpness": round(float(np.median(scores[picked])), 1) if picked else None,
    }
    return [i / rate for i in picked], stats, scores[picked]


EXIF_HEAD_BYTES = 256 * 1024  # EXIF sits in one segment (at most 64 KiB) near the start


def _head(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(EXIF_HEAD_BYTES)


def is_webp(head: bytes) -> bool:
    return len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP"


def make_poster(ctx: StageContext, frame: Path, work: Path) -> str | None:
    """Encode ``frame`` as the poster and store it with the attempt; its key or None.

    The poster is only the viewer's fallback image, so a failed encode is logged
    and the run continues without one instead of failing the reconstruction.
    """
    out = work / "poster.webp"
    try:
        ctx.run(
            render_argv("ffmpeg_poster", input=frame, width=POSTER_WIDTH, output=out),
            timeout=120,
        )
    except (ProcessFailed, ProcessTimeout) as exc:
        log.warning("poster encoding failed: %s", type(exc).__name__)
        return None
    size = out.stat().st_size if out.is_file() else 0
    if not 0 < size <= MAX_POSTER_BYTES or not is_webp(out.read_bytes()[:12]):
        log.warning("poster encoding produced no usable WebP (%d bytes)", size)
        return None
    key = ctx.attempt_key("poster.webp")
    ctx.store.put_file(key, out, "image/webp")
    return key


def run(ctx: StageContext) -> StageResult:
    inputs = list(ctx.output("inputs"))
    quality = ctx.outputs().get("quality")
    if quality == "synthetic":
        return finish(
            ctx,
            {"frames": 0, "note": "synthetic recipe: no capture to prepare"},
            outputs={"frameCount": 0, "framesPrefix": None, "sourceMapKey": None},
        )
    for tool in ("ffprobe", "ffmpeg"):
        if not tool_available(tool):
            raise StageError("INTERNAL_ERROR", "A media tool is missing from the worker image.")
    params = ctx.params
    long_edge = int(params.get("longEdge", 1600))
    cap = int(params.get("maxFrames", 400))
    work = ctx.scratch / "prepare"
    images = work / "images"
    images.mkdir(parents=True, exist_ok=True)

    videos = [i for i in inputs if i["mediaKind"] == "video"]
    photos = [i for i in inputs if i["mediaKind"] == "photo"][:cap]
    local: dict[str, Path] = {}
    for item in videos + photos:
        ctx.cancellation.raise_if_cancelled()
        dest = work / "src" / item["uploadId"]
        digest = ctx.store.download(item["key"], dest, item.get("storageVersion"))
        if digest != item["sha256"]:
            raise StageError("INPUT_NOT_VERIFIED", details={"uploadId": item["uploadId"]})
        local[item["uploadId"]] = dest

    metas = {v["uploadId"]: probe_video(ctx, local[v["uploadId"]]) for v in videos}
    fps, budget = plan_frames(
        [metas[v["uploadId"]]["durationSeconds"] for v in videos],
        len(photos),
        fps=float(params.get("fps", 2)),
        cap=cap,
    )
    source_map: dict[str, dict[str, Any]] = {}
    sharpness: list[dict[str, Any]] = []
    oversample = max(1, int(params.get("keyframeOversample", 1)))
    poster_frames: list[Path] = []  # the first video's kept frames, in order
    poster_scores: np.ndarray | None = None
    remaining = budget
    for index, video in enumerate(videos):
        if remaining <= 0:
            break
        meta = metas[video["uploadId"]]
        folder = images / f"{index:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        duration = meta["durationSeconds"]
        # Never sample above the video's own frame rate: FFmpeg's fps filter would
        # duplicate frames (a 1 fps time-lapse would yield copies of each frame).
        native = float(meta.get("frameRate") or 0)
        rate = min(fps, native) if native > 0 else fps
        want = min(remaining, max(1, int(duration * rate)))
        candidates = min(int(duration * rate * oversample), want * MAX_CANDIDATES_PER_FRAME)
        if native > 0:
            candidates = min(candidates, int(duration * native))
        ctx.progress("Extracting frames.", source=index, frames=want)
        timeout = max(300.0, duration * 4)
        kept_scores = None
        if candidates <= want:
            take = extract_frames(
                ctx, local[video["uploadId"]], folder, rate, want, long_edge, timeout
            )
        else:
            take, stats, kept_scores = extract_keyframes(
                ctx,
                local[video["uploadId"]],
                folder,
                work / "cand",
                candidates / duration,
                candidates,
                want,
                long_edge,
                timeout,
            )
            sharpness.append({"source": index, **stats})
        if not poster_frames:
            poster_frames = sorted(folder.glob("frame_*.jpg"))
            poster_scores = kept_scores
        for k, seconds in enumerate(take):
            source_map[f"{index:02d}/frame_{k + 1:05d}.jpg"] = {
                "uploadId": video["uploadId"],
                "timestampSeconds": round(meta["startTime"] + seconds, 3),
            }
        remaining -= len(take)
    if photos:
        # COLMAP gives each folder one camera and rejects a folder whose images differ
        # in size, so photos are grouped by their output dimensions (e.g. portrait
        # and landscape shots land in separate camera folders) and, where EXIF names
        # it, by the camera and lens: unrelated phones must not share intrinsics.
        # Photos without EXIF cannot be told apart and share their size's folder.
        staging = images.parent / "photo-staging"
        staging.mkdir(parents=True, exist_ok=True)
        for k, photo in enumerate(photos):
            out = staging / f"photo_{k:05d}.jpg"
            try:
                ctx.run(
                    render_argv(
                        "ffmpeg_photo",
                        input=local[photo["uploadId"]],
                        long_edge=long_edge,
                        output=out,
                    ),
                    timeout=120,
                )
            except ProcessFailed as exc:
                raise StageError("INVALID_MEDIA", details={"uploadId": photo["uploadId"]}) from exc
            size = jpeg_dimensions(out.read_bytes()) if out.is_file() else None
            if size is None:
                raise StageError("INVALID_MEDIA", details={"uploadId": photo["uploadId"]})
            group = f"photos_{size[0]}x{size[1]}"
            camera = camera_identity(_head(local[photo["uploadId"]]))
            if camera is not None:
                group += "_" + hashlib.sha256(camera.encode()).hexdigest()[:8]
            (images / group).mkdir(parents=True, exist_ok=True)
            out.replace(images / group / out.name)
            source_map[f"{group}/{out.name}"] = {"uploadId": photo["uploadId"], "photo": True}

    frames = sorted(p for p in images.rglob("*.jpg"))
    if not frames:
        raise StageError("INVALID_MEDIA", "No frames could be extracted.")
    prefix = ctx.attempt_key("images/")
    for frame in frames:
        ctx.cancellation.raise_if_cancelled()
        rel = frame.relative_to(images).as_posix()
        ctx.store.put_file(prefix + rel, frame, "image/jpeg")
    source_map_key = ctx.attempt_key("source-map.json")
    ctx.store.put_bytes(
        source_map_key, json.dumps(source_map, sort_keys=True).encode(), "application/json"
    )
    if not poster_frames:  # photos only
        poster_frames = frames
    poster = poster_frames[pick_poster(poster_scores, len(poster_frames))]
    poster_key = make_poster(ctx, poster, work)
    return finish(
        ctx,
        {
            "frames": len(frames),
            "effectiveFps": round(fps, 4),
            "videos": metas,
            "keyframes": sharpness,
            "poster": poster.relative_to(images).as_posix() if poster_key else None,
        },
        outputs={
            "frameCount": len(frames),
            "framesPrefix": prefix,
            "frameNames": sorted(source_map),
            "sourceMapKey": source_map_key,
            "posterKey": poster_key,
        },
    )
