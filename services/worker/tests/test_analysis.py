"""Scene analysis without I/O: keyframes, image preparation, the request and the reply."""

from __future__ import annotations

import io
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
from pdf_fixtures import classic_pdf, encrypted_pdf, page_tree
from PIL import Image
from room_dataset import canonical_cameras

from andarilho_worker.ai import AIInvalidReply
from andarilho_worker.analysis import (
    INTRODUCTION,
    MAX_DOORS_PER_FRAME,
    MAX_OBJECTS,
    MAX_OBJECTS_PER_FRAME,
    MAX_PARTS,
    MAX_ROOMS,
    REPLY_FORMAT,
    build_content,
    data_url,
    encode_image,
    floor_plan_image,
    frame_kind,
    normalise_reply,
    select_keyframes,
)

FIXTURES = Path(__file__).parent / "fixtures" / "analysis"
ROOM_FRAMES = {0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23}


def _jpeg(size: tuple[int, int], colour=(120, 40, 200), exif: bytes | None = None) -> bytes:
    buffer = io.BytesIO()
    kwargs = {"exif": exif} if exif is not None else {}
    Image.new("RGB", size, colour).save(buffer, format="JPEG", quality=95, **kwargs)
    return buffer.getvalue()


def _png(size: tuple[int, int], mode: str = "RGB", colour=(10, 20, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _walk(centres: list[tuple[float, float]], headings: list[float], kinds=None):
    """Views for ``select_keyframes``, named in walk order."""
    kinds = kinds or ["video"] * len(centres)
    return [
        (f"00/frame_{k + 1:05d}.jpg" if kind == "video" else f"photos_1x1/p{k:03d}.jpg",
         kind, centre, heading)
        for k, (kind, centre, heading) in enumerate(zip(kinds, centres, headings, strict=True))
    ]  # fmt: skip


def _straight(count: int, step: float = 0.1):
    centres = [(k * step, 0.0) for k in range(count)]
    return _walk(centres, [0.0] * count)


def _spin(count: int):
    """A camera turning on the spot: one frame every 360/count degrees."""
    headings = [2 * math.pi * k / count for k in range(count)]
    return _walk([(0.0, 0.0)] * count, headings)


# --------------------------------------------------------------------------
# Keyframe selection
# --------------------------------------------------------------------------


def test_frame_kind_takes_supplementary_photos_by_their_folder():
    assert frame_kind("photos_1600x1200/photo_00001.jpg") == "photo"
    assert frame_kind("photos/photo_00001.jpg") == "photo"
    assert frame_kind("00/frame_00001.jpg") == "video"


def test_selection_is_deterministic_and_in_walk_order():
    rng = random.Random(7)
    centres = [(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(80)]
    headings = [rng.uniform(-math.pi, math.pi) for _ in range(80)]
    views = _walk(centres, headings)
    first = select_keyframes(views, 12, 6)
    assert select_keyframes(views, 12, 6) == first
    assert first == sorted(first)


def test_a_video_is_chosen_for_every_step_of_the_walk():
    views = _straight(100)
    chosen = select_keyframes(views, 5, 0)
    assert len(chosen) == 5 and chosen[0] == 0 and chosen[-1] == 99
    expected = [0, 25, 50, 75, 99]
    assert all(abs(got - want) <= 2 for got, want in zip(chosen, expected, strict=True))


def test_a_spin_in_place_is_shown_around_the_turn():
    views = _spin(36)
    chosen = select_keyframes(views, 4, 0)
    assert chosen[0] == 0 and chosen[-1] == 35
    expected = [0, 12, 24, 35]
    assert all(abs(got - want) <= 2 for got, want in zip(chosen, expected, strict=True))


def test_a_stationary_camera_still_gives_distinct_frames():
    views = _walk([(0.0, 0.0)] * 30, [0.0] * 30)
    chosen = select_keyframes(views, 6, 0)
    assert len(chosen) == 6 and len(set(chosen)) == 6
    assert chosen[0] == 0 and chosen[-1] == 29


def test_the_first_and_last_video_frames_are_kept_for_two_or_more():
    views = _straight(50)
    assert select_keyframes(views, 2, 0) == [0, 49]
    assert select_keyframes(views, 1, 0) == [25]  # a single frame is the middle of the walk


def test_photos_come_first_capped_and_spread_over_the_photos():
    views = _walk(
        [(0.0, 0.0)] * 20,
        [0.0] * 20,
        kinds=["photo"] * 10 + ["video"] * 10,
    )
    chosen = select_keyframes(views, 0, 3)
    photos = [i for i in chosen if views[i][1] == "photo"]
    assert len(photos) == 3
    assert photos[0] == 0 and photos[-1] == 9  # spread to both ends of the photo list
    assert len(select_keyframes(views, 0, 50)) == 10  # every photo when there are few


def test_no_photo_budget_means_no_photos():
    views = _walk([(0.0, 0.0)] * 6, [0.0] * 6, kinds=["photo", "video"] * 3)
    assert all(views[i][1] == "video" for i in select_keyframes(views, 3, 0))


def test_the_result_has_no_duplicates_and_respects_the_budgets():
    rng = random.Random(11)
    kinds = ["photo" if k % 5 == 0 else "video" for k in range(120)]
    centres = [(rng.uniform(0, 4), rng.uniform(0, 4)) for _ in range(120)]
    headings = [rng.uniform(-math.pi, math.pi) for _ in range(120)]
    views = _walk(centres, headings, kinds)
    chosen = select_keyframes(views, 7, 4)
    assert len(chosen) == len(set(chosen)) <= 7 + 4
    assert sum(views[i][1] == "photo" for i in chosen) == 4
    assert sum(views[i][1] == "video" for i in chosen) == 7


def test_a_short_walk_gives_every_video_frame():
    assert select_keyframes(_straight(4), 12, 0) == [0, 1, 2, 3]
    assert select_keyframes([], 12, 6) == []


def test_the_box_room_capture_gets_frames_spread_over_its_turns():
    # The room's 24 cameras sweep a full turn from the middle of the box room.
    views = [
        (f"00/frame_{k + 1:05d}.jpg", "video", (eye[0], eye[2]), math.atan2(R[2][0], R[2][2]))
        for k, (R, _t, eye) in enumerate(canonical_cameras())
    ]
    assert select_keyframes(views, 12, 6) == sorted(ROOM_FRAMES)


# --------------------------------------------------------------------------
# Image preparation
# --------------------------------------------------------------------------


def test_an_image_is_shrunk_to_its_long_edge_and_keeps_its_ratio():
    jpeg, width, height = encode_image(Image.new("RGB", (400, 200)), 100, 80)
    assert (width, height) == (100, 50)
    decoded = Image.open(io.BytesIO(jpeg))
    assert decoded.format == "JPEG" and decoded.size == (100, 50)


def test_an_image_already_small_enough_is_never_upscaled():
    jpeg, width, height = encode_image(Image.new("RGB", (50, 40)), 1024, 80)
    assert (width, height) == (50, 40)
    assert Image.open(io.BytesIO(jpeg)).size == (50, 40)


def test_a_path_and_an_array_encode_like_an_image(tmp_path):
    path = tmp_path / "frame.jpg"
    path.write_bytes(_jpeg((320, 240)))
    from_path = encode_image(path, 160, 80)
    from_array = encode_image(np.zeros((240, 320, 3), dtype=np.uint8), 160, 80)
    assert from_path[1:] == from_array[1:] == (160, 120)


def test_the_metadata_of_the_source_is_not_carried_into_the_jpeg():
    camera = Image.Exif()
    camera[0x010F] = "TestMake"  # the camera's Make tag
    source = _jpeg((64, 48), exif=camera.tobytes())
    assert b"TestMake" in source
    assert Image.open(io.BytesIO(source)).getexif()  # the input really carries EXIF
    jpeg, _, _ = encode_image(Image.open(io.BytesIO(source)), 1024, 80)
    assert b"Exif" not in jpeg and b"TestMake" not in jpeg
    assert not Image.open(io.BytesIO(jpeg)).getexif()


def test_a_transparent_picture_is_flattened_onto_white():
    picture = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    jpeg, _, _ = encode_image(picture, 1024, 95)
    pixel = Image.open(io.BytesIO(jpeg)).convert("RGB").getpixel((4, 4))
    assert all(channel > 240 for channel in pixel)


def test_the_jpeg_quality_setting_changes_the_size():
    rng = np.random.default_rng(3)
    noisy = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    high, _, _ = encode_image(noisy, 1024, 95)
    low, _, _ = encode_image(noisy, 1024, 20)
    assert len(low) < len(high)


def test_data_url_wraps_jpeg_bytes_in_base64():
    assert data_url(b"\xff\xd8") == "data:image/jpeg;base64,/9g="


# --------------------------------------------------------------------------
# Floor plans
# --------------------------------------------------------------------------


def test_a_png_or_jpeg_floor_plan_decodes_directly():
    png = floor_plan_image(_png((120, 80)), "image/png")
    jpg = floor_plan_image(_jpeg((90, 60)), "image/jpeg")
    assert png is not None and png.size == (120, 80)
    assert jpg is not None and jpg.size == (90, 60)


def test_a_pdf_floor_plan_gives_its_largest_embedded_image():
    buffer = io.BytesIO()
    Image.new("RGB", (200, 150), (250, 250, 250)).save(buffer, format="PDF", resolution=100.0)
    image = floor_plan_image(buffer.getvalue(), "application/pdf")
    assert image is not None and image.size == (200, 150)


def test_a_pdf_without_an_image_gives_none():
    pdf = classic_pdf(page_tree(1))
    assert floor_plan_image(pdf, "application/pdf") is None


def test_an_encrypted_or_damaged_pdf_gives_none():
    assert floor_plan_image(encrypted_pdf(), "application/pdf") is None
    assert floor_plan_image(b"%PDF-1.4 truncated", "application/pdf") is None


def test_a_corrupt_image_or_another_type_gives_none():
    assert floor_plan_image(b"not a png at all", "image/png") is None
    assert floor_plan_image(_png((10, 10)), "text/plain") is None


# --------------------------------------------------------------------------
# The request
# --------------------------------------------------------------------------


def test_the_content_puts_each_frame_label_before_its_image():
    content = build_content(
        [(3, "data:image/jpeg;base64,AAA"), (7, "data:image/jpeg;base64,BBB")],
        "data:image/jpeg;base64,FFF",
        photos={7},
    )
    assert content[0] == {"type": "text", "text": INTRODUCTION}
    assert content[1] == {"type": "text", "text": "Frame 3:"}
    assert content[2] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}}
    assert content[3] == {"type": "text", "text": "Frame 7 (photo):"}
    assert content[4]["image_url"]["url"] == "data:image/jpeg;base64,BBB"
    assert content[5] == {"type": "text", "text": "Floor plan:"}
    assert content[6]["image_url"]["url"] == "data:image/jpeg;base64,FFF"
    assert content[-1] == {"type": "text", "text": REPLY_FORMAT}
    assert len(content) == 8


def test_the_content_without_a_floor_plan_still_ends_with_the_format():
    content = build_content([(0, "data:image/jpeg;base64,AAA")], None)
    assert [part["type"] for part in content] == ["text", "text", "image_url", "text"]
    assert content[-1]["text"] == REPLY_FORMAT


def test_the_instructions_ask_for_the_things_the_scene_needs():
    for phrase in (
        "walking order",
        "about 1.5 metres",
        "tight box",
        "bottomVisible",
        "unit cube",
        "#rrggbb",
        "JSON",
    ):
        assert phrase in INTRODUCTION + REPLY_FORMAT


# --------------------------------------------------------------------------
# The reply
# --------------------------------------------------------------------------


def _good() -> dict:
    return json.loads((FIXTURES / "good_reply.json").read_text(encoding="utf-8"))


def _messy() -> dict:
    return json.loads((FIXTURES / "messy_reply.json").read_text(encoding="utf-8"))


def test_a_good_reply_passes_through_unchanged():
    good = _good()
    assert normalise_reply(good, {0, 3, 5, 9, 17}) == good


def test_a_messy_reply_is_checked_item_by_item():
    reply = normalise_reply(_messy(), {0, 3, 5, 9})
    assert reply["rooms"] == [
        {"id": "r1", "label": "Living room"},
        {"id": "r2", "label": "Kitchen"},
        {"id": "r3", "label": "r3"},
    ]
    assert [obj["id"] for obj in reply["objects"]] == ["o1", "o2", "o4", "2"]
    sofa, rug, plant, shelf = reply["objects"]
    assert sofa["category"] == "sofa" and sofa["mount"] == "floor" and sofa["solid"] is True
    assert sofa["parts"] == [
        {"shape": "box", "min": [0.0, 0.0, 0.0], "max": [1.0, 0.4, 1.0], "color": "#7a7f87"},
        {"shape": "box", "min": [0.0, 0.0, 0.0], "max": [0.5, 1.0, 1.0], "color": "#7a7f87"},
        {"shape": "box", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0], "color": "#ffffff"},
    ]
    assert rug["category"] == "rug" and rug["label"] == "rug"  # an empty label takes the category
    assert rug["mount"] == "floor" and rug["solid"] is False
    assert rug["size"] == [0.05, 4.0, 0.5]
    assert rug["parts"][0]["color"] == "#aabbcc"
    assert plant["category"] == "potted plant" and plant["label"] == "tall green plant"
    assert shelf["mount"] == "wall" and shelf["parts"][0]["color"] == "#8b5a2b"
    frames = {frame["frame"]: frame for frame in reply["frames"]}
    assert sorted(frames) == [0, 3, 5, 9]
    zero = frames[0]
    assert zero["room"] == "r1"
    assert zero["doors"] == [
        {"box": [0.1, 0.1, 0.3, 0.9], "state": "open", "leadsTo": "r1"},
        {"box": [0.0, 0.2, 0.4, 0.9], "state": "closed", "leadsTo": None},
    ]
    assert zero["objects"] == [
        {"id": "o1", "box": [0.2, 0.5, 0.8, 1.0], "bottomVisible": True},
        {"id": "o2", "box": [0.0, 0.0, 1.0, 1.0], "bottomVisible": False},
    ]
    assert frames[3]["room"] is None
    assert frames[5]["objects"] == [
        {"id": "2", "box": [0.3, 0.3, 0.6, 0.6], "bottomVisible": False}
    ]
    assert frames[9]["room"] is None


def test_reply_items_are_bounded_as_well_as_checked():
    reply = {
        "rooms": [{"id": f"room{k}", "label": "x" * 200} for k in range(MAX_ROOMS + 3)],
        "frames": [
            {
                "frame": 0,
                "doors": [
                    {"box": [0.0, 0.0, 0.5, 0.5], "state": "closed"}
                    for _ in range(MAX_DOORS_PER_FRAME + 4)
                ],
                "objects": [
                    {"id": f"o{k}", "box": [0.0, 0.0, 1.0, 1.0], "bottomVisible": True}
                    for k in range(MAX_OBJECTS_PER_FRAME + 5)
                ],
            }
        ],
        "objects": [
            {
                "id": f"o{k}",
                "category": "Chair",
                "size": [1.0, 1.0, 1.0],
                "parts": [{"shape": "box", "min": [0, 0, 0], "max": [1, 1, 1], "color": "#000000"}]
                * (MAX_PARTS + 3),
            }
            for k in range(MAX_OBJECTS + 4)
        ],
    }
    frame_numbers = {0}
    out = normalise_reply(reply, frame_numbers)
    assert len(out["rooms"]) == MAX_ROOMS
    assert all(len(room["label"]) == 120 for room in out["rooms"])
    assert len(out["objects"]) == MAX_OBJECTS
    assert all(len(obj["parts"]) == MAX_PARTS for obj in out["objects"])
    assert all(obj["category"] == "chair" for obj in out["objects"])
    frame = out["frames"][0]
    assert len(frame["doors"]) == MAX_DOORS_PER_FRAME
    # Only detections of kept objects survive, and at most the per-frame cap.
    assert len(frame["objects"]) == MAX_OBJECTS_PER_FRAME
    assert all(det["id"] in {obj["id"] for obj in out["objects"]} for det in frame["objects"])


def test_size_clamps_and_long_ids_are_trimmed():
    reply = {
        "objects": [
            {
                "id": "x" * 60,
                "category": "c" * 60,
                "size": [0.0, 99.0, -4.0],
                "parts": [{"shape": "box", "min": [0, 0, 0], "max": [1, 1, 1], "color": "#000000"}],
            }
        ]
    }
    (obj,) = normalise_reply(reply, set())["objects"]
    assert obj["id"] == "x" * 40 and obj["category"] == "c" * 40
    assert obj["size"] == [0.05, 4.0, 0.05]


def test_a_reply_that_is_not_an_object_is_invalid():
    for raw in ([], "text", None, 3):
        with pytest.raises(AIInvalidReply):
            normalise_reply(raw, set())


def test_a_reply_without_rooms_frames_or_objects_lists_is_invalid():
    with pytest.raises(AIInvalidReply):
        normalise_reply({"rooms": "none", "frames": None, "objects": {}}, set())


def test_a_missing_list_is_empty_when_another_list_is_present():
    assert normalise_reply({"rooms": []}, set()) == {"rooms": [], "frames": [], "objects": []}
    with pytest.raises(AIInvalidReply):
        normalise_reply({}, set())
