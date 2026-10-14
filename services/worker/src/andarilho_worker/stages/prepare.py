"""``preparing``: probe media and extract bounded, metadata-free frames.

* FFprobe records codec, size, duration, frame rate and rotation;
* FFmpeg extracts 2-4 fps (recipe) capped at 400 frames overall, at most 1600 px
  on the long edge, drops audio/subtitle/data streams and strips EXIF/GPS
  (``-map_metadata -1``); FFmpeg's default autorotation normalizes orientation;
* every retained image keeps a source map entry (upload ID + timestamp).

Frames are grouped per source (``images/<nn>/``) so COLMAP can assign one camera
per source instead of forcing a shared calibration onto unrelated cameras.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from andarilho_worker.errors import StageError
from andarilho_worker.media import MAX_VIDEO_SECONDS, VIDEO_CODECS, parse_ffprobe
from andarilho_worker.recipes import render_argv
from andarilho_worker.runner import ProcessFailed, tool_available
from andarilho_worker.stages.base import StageContext, StageResult, finish

PROBE_TIMEOUT = 120


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
    remaining = budget
    for index, video in enumerate(videos):
        if remaining <= 0:
            break
        meta = metas[video["uploadId"]]
        folder = images / f"{index:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        want = min(remaining, max(1, int(meta["durationSeconds"] * fps)))
        ctx.progress("Extracting frames.", source=index, frames=want)
        ctx.run(
            render_argv(
                "ffmpeg_frames",
                input=local[video["uploadId"]],
                fps=f"{fps:.4f}",
                long_edge=long_edge,
                max_frames=want,
                output_pattern=folder / "frame_%05d.jpg",
            ),
            timeout=max(300.0, meta["durationSeconds"] * 4),
        )
        produced = sorted(folder.glob("frame_*.jpg"))
        for k, frame in enumerate(produced):
            source_map[f"{index:02d}/{frame.name}"] = {
                "uploadId": video["uploadId"],
                "timestampSeconds": round(meta["startTime"] + k / fps, 3),
            }
        remaining -= len(produced)
    if photos:
        folder = images / "photos"
        folder.mkdir(parents=True, exist_ok=True)
        for k, photo in enumerate(photos):
            out = folder / f"photo_{k:05d}.jpg"
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
            source_map[f"photos/{out.name}"] = {"uploadId": photo["uploadId"], "photo": True}

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
    return finish(
        ctx,
        {"frames": len(frames), "effectiveFps": round(fps, 4), "videos": metas},
        outputs={
            "frameCount": len(frames),
            "framesPrefix": prefix,
            "frameNames": sorted(source_map),
            "sourceMapKey": source_map_key,
        },
    )
