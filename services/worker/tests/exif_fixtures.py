"""Tiny JPEGs carrying EXIF camera tags, built in memory."""

from __future__ import annotations

import struct

MAKE, MODEL, SUB_IFD, FOCAL_LENGTH, LENS_MODEL = 0x010F, 0x0110, 0x8769, 0x920A, 0xA434
ASCII, LONG, RATIONAL = 2, 4, 5


def _ascii(text: str) -> tuple[int, int, bytes]:
    raw = text.encode() + b"\x00"
    return ASCII, len(raw), raw


def exif_tiff(
    *,
    make: str | None = None,
    model: str | None = None,
    lens: str | None = None,
    focal_mm: float | None = None,
    big_endian: bool = False,
) -> bytes:
    """A TIFF structure with IFD0 (make, model) and an Exif sub-IFD (lens, focal)."""
    order = ">" if big_endian else "<"
    ifd0 = [(t, *_ascii(v)) for t, v in ((MAKE, make), (MODEL, model)) if v is not None]
    sub = [(LENS_MODEL, *_ascii(lens))] if lens is not None else []
    if focal_mm is not None:
        sub.append(
            (FOCAL_LENGTH, RATIONAL, 1, struct.pack(order + "II", round(focal_mm * 100), 100))
        )
    ifd0_at = 8
    n0 = len(ifd0) + (1 if sub else 0)
    sub_at = ifd0_at + 2 + 12 * n0 + 4
    data_at = sub_at + (2 + 12 * len(sub) + 4 if sub else 0)
    data = bytearray()

    def encode(items: list[tuple[int, int, int, bytes]]) -> bytes:
        out = bytearray(struct.pack(order + "H", len(items)))
        for tag, kind, count, raw in items:
            if len(raw) <= 4:
                out += struct.pack(order + "HHI", tag, kind, count) + raw.ljust(4, b"\x00")
            else:
                out += struct.pack(order + "HHII", tag, kind, count, data_at + len(data))
                data.extend(raw)
        return bytes(out + struct.pack(order + "I", 0))

    pointer = [(SUB_IFD, LONG, 1, struct.pack(order + "I", sub_at))] if sub else []
    first = encode(ifd0 + pointer)
    second = encode(sub) if sub else b""
    header = (b"MM" if big_endian else b"II") + struct.pack(order + "HI", 42, ifd0_at)
    return header + first + second + bytes(data)


def jpeg_with_exif(tiff: bytes, width: int = 64, height: int = 48) -> bytes:
    payload = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    sof = b"\xff\xc0\x00\x11\x08" + struct.pack(">HH", height, width)
    sof += b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    return b"\xff\xd8" + app1 + sof + b"\xff\xd9"
