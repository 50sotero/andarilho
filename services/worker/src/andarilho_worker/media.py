"""File-signature sniffing, image dimension headers and FFprobe output parsing.

Content is checked by bounded subprocesses: FFmpeg decodes images and
:mod:`andarilho_worker.pdf_inspect` walks PDF page trees.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

VIDEO_BRANDS = {
    b"isom", b"iso2", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"avc1", b"qt  ",
    b"M4V ", b"M4VP", b"MSNV", b"3gp4", b"3gp5", b"3gp6", b"hvc1", b"dash",
}  # fmt: skip
VIDEO_CODECS = {"h264", "hevc"}
MAX_IMAGE_PIXELS = 100_000_000
MAX_VIDEO_SECONDS = 30 * 60


@dataclass(frozen=True)
class Sniff:
    kind: str  # mp4 | mov | jpeg | png | pdf | unknown | forbidden
    detail: str = ""


def sniff(head: bytes) -> Sniff:
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand == b"qt  ":
            return Sniff("mov", brand.decode("latin-1"))
        if brand in VIDEO_BRANDS:
            return Sniff("mp4", brand.decode("latin-1"))
        if brand in (b"heic", b"heix", b"mif1", b"avif"):
            return Sniff("unknown", "heif image")
        return Sniff("unknown", "unsupported ftyp brand")
    if head.startswith(b"\xff\xd8\xff"):
        return Sniff("jpeg")
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return Sniff("png")
    if head.startswith(b"%PDF-"):
        return Sniff("pdf")
    lowered = head[:1024].lstrip().lower()
    if (
        lowered.startswith((b"<svg", b"<?xml", b"<!doctype", b"<html", b"<script"))
        or b"<svg" in lowered
    ):
        return Sniff("forbidden", "markup")
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06", b"\x1f\x8b", b"Rar!", b"7z\xbc\xaf", b"BZh")):
        return Sniff("forbidden", "archive")
    if head.startswith((b"MZ", b"\x7fELF", b"#!")):
        return Sniff("forbidden", "executable")
    return Sniff("unknown")


ALLOWED = {
    "video": {"mp4", "mov"},
    "photo": {"jpeg", "png"},
    "floor_plan": {"jpeg", "png", "pdf"},
}


def png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    i = 2
    n = len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        length = struct.unpack(">H", data[i + 2 : i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return width, height
        if marker == 0xDA:
            return None
        i += 2 + length
    return None


def parse_ffprobe(stdout: str) -> dict[str, Any]:
    info = json.loads(stdout or "{}")
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("no video stream")
    fmt = info.get("format", {})
    duration = float(video.get("duration") or fmt.get("duration") or 0)
    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = int(float(side["rotation"]))
    rotation = int(float(video.get("tags", {}).get("rotate", rotation)))
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    num, _, den = rate.partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {
        "codec": video.get("codec_name"),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "durationSeconds": round(duration, 3),
        # Frames the container declares (0 when it does not say).
        "frameCount": int(video.get("nb_frames") or 0),
        "frameRate": round(fps, 3),
        "rotation": rotation,
        "hasAudio": any(s.get("codec_type") == "audio" for s in streams),
        "startTime": float(fmt.get("start_time") or 0),
    }


def progress_frames(text: str) -> int:
    """The last ``frame=`` count in FFmpeg ``-progress`` output (0 when there is none)."""
    frames = 0
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "frame" and value.strip().isdigit():
            frames = int(value.strip())
    return frames
