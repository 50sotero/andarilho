"""EXIF camera identity, used to keep unrelated cameras in separate COLMAP folders."""

from __future__ import annotations

import pytest
from exif_fixtures import exif_tiff, jpeg_with_exif

from andarilho_worker.media import camera_identity


@pytest.mark.parametrize("big_endian", [False, True])
def test_camera_identity_reads_make_model_lens_and_focal_length(big_endian):
    tiff = exif_tiff(
        make="Apple",
        model="iPhone 15",
        lens="iPhone 15 back camera 6.1mm f/1.6",
        focal_mm=6.1,
        big_endian=big_endian,
    )
    assert camera_identity(jpeg_with_exif(tiff)) == (
        "Apple|iPhone 15|iPhone 15 back camera 6.1mm f/1.6|6.1mm"
    )


def test_camera_identity_tells_lenses_and_phones_apart():
    main = camera_identity(jpeg_with_exif(exif_tiff(make="Apple", model="iPhone 15", focal_mm=6.1)))
    wide = camera_identity(jpeg_with_exif(exif_tiff(make="Apple", model="iPhone 15", focal_mm=2.2)))
    other = camera_identity(jpeg_with_exif(exif_tiff(make="Google", model="Pixel 8")))
    assert len({main, wide, other}) == 3


def test_camera_identity_is_none_without_exif():
    plain = jpeg_with_exif(exif_tiff())[:2] + b"\xff\xc0\x00\x11\x08\x00\x30\x00\x40"
    assert camera_identity(plain) is None
    assert camera_identity(jpeg_with_exif(exif_tiff())) is None  # EXIF with no camera tags
    assert camera_identity(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32) is None
    assert camera_identity(b"") is None


def test_camera_identity_survives_truncated_or_corrupt_exif():
    data = jpeg_with_exif(exif_tiff(make="Apple", model="iPhone 15", lens="x" * 40, focal_mm=6.1))
    for cut in range(len(data)):
        camera_identity(data[:cut])  # never raises
    corrupt = bytearray(data)
    corrupt[30:60] = b"\xff" * 30  # offsets and counts pointing far outside the segment
    camera_identity(bytes(corrupt))
