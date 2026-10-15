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


# EXIF tags that tell cameras apart: the make and model in IFD0, and in the Exif
# sub-IFD the lens and focal length (a phone's wide and ultra-wide lenses differ).
_EXIF_MAKE, _EXIF_MODEL, _EXIF_SUB_IFD = 0x010F, 0x0110, 0x8769
_EXIF_FOCAL_LENGTH, _EXIF_LENS_MODEL = 0x920A, 0xA434
_EXIF_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}
_EXIF_MAX_ENTRIES = 512
_EXIF_MAX_VALUE_BYTES = 256


def camera_identity(data: bytes) -> str | None:
    """Describe the camera and lens that took a JPEG, from its EXIF.

    Returns None when the file carries none (stripped by a messaging app, or a
    PNG). Only the EXIF segment before the image data is read; malformed EXIF
    counts as none.
    """
    tiff = _exif_payload(data)
    if tiff is None or tiff[:2] not in (b"II", b"MM"):
        return None
    order = "<" if tiff[:2] == b"II" else ">"
    try:
        if struct.unpack_from(order + "H", tiff, 2)[0] != 42:
            return None
        ifd0 = _exif_entries(tiff, order, struct.unpack_from(order + "I", tiff, 4)[0])
        sub: dict[int, tuple[int, bytes]] = {}
        if ifd0.get(_EXIF_SUB_IFD, (0, b""))[0] == 4:
            offset = struct.unpack(order + "I", ifd0[_EXIF_SUB_IFD][1][:4])[0]
            sub = _exif_entries(tiff, order, offset)
    except struct.error:
        return None
    parts = [
        _exif_text(ifd0.get(_EXIF_MAKE)),
        _exif_text(ifd0.get(_EXIF_MODEL)),
        _exif_text(sub.get(_EXIF_LENS_MODEL)),
        _exif_focal(sub.get(_EXIF_FOCAL_LENGTH), order),
    ]
    return "|".join(parts) if any(parts) else None


def _exif_payload(data: bytes) -> bytes | None:
    i = 2
    n = len(data)
    if data[:2] != b"\xff\xd8":
        return None
    while i + 4 <= n:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker in (0xD9, 0xDA):
            return None
        length = struct.unpack(">H", data[i + 2 : i + 4])[0]
        segment = data[i + 4 : i + 2 + length]
        if marker == 0xE1 and segment[:6] == b"Exif\x00\x00":
            return segment[6:]
        i += 2 + length
    return None


def _exif_entries(tiff: bytes, order: str, offset: int) -> dict[int, tuple[int, bytes]]:
    """One IFD's entries as ``{tag: (type, raw value bytes)}``, skipping bad ones."""
    if offset < 8 or offset + 2 > len(tiff):
        return {}
    count = struct.unpack_from(order + "H", tiff, offset)[0]
    entries: dict[int, tuple[int, bytes]] = {}
    for k in range(min(count, _EXIF_MAX_ENTRIES)):
        pos = offset + 2 + 12 * k
        if pos + 12 > len(tiff):
            break
        tag, kind, n = struct.unpack_from(order + "HHI", tiff, pos)
        size = _EXIF_TYPE_SIZES.get(kind, 0) * n
        if size == 0 or size > _EXIF_MAX_VALUE_BYTES:
            continue
        if size <= 4:
            raw = tiff[pos + 8 : pos + 8 + size]
        else:
            at = struct.unpack_from(order + "I", tiff, pos + 8)[0]
            raw = tiff[at : at + size]
        if len(raw) == size:
            entries[tag] = (kind, raw)
    return entries


def _exif_text(entry: tuple[int, bytes] | None) -> str:
    if entry is None or entry[0] != 2:
        return ""
    text = entry[1].split(b"\x00", 1)[0].decode("ascii", "replace")
    return "".join(c for c in text if c.isprintable()).strip()[:64]


def _exif_focal(entry: tuple[int, bytes] | None, order: str) -> str:
    if entry is None or entry[0] != 5:
        return ""
    num, den = struct.unpack(order + "II", entry[1][:8])
    return f"{num / den:.1f}mm" if den else ""


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
