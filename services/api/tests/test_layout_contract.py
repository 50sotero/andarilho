"""The layout contract's furniture: primitive parts in a unit box, placed on a floor."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from andarilho_api.schemas.scene import Layout


def _layout() -> dict[str, Any]:
    square = [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]
    return {
        "schemaVersion": "1.0",
        "floors": [{"id": "floor-0", "elevation": 0.0, "polygons": [square]}],
        "spawn": {"feet": [1.0, 0.0, 1.0], "yawRadians": 0.0, "floorId": "floor-0"},
        "furniture": [
            {
                "id": "sofa-1",
                "floorId": "floor-0",
                "category": "sofa",
                "label": "grey three-seat sofa",
                "position": [2.0, 2.5],
                "yawRadians": 3.14159,
                "size": [2.0, 0.85, 0.9],
                "parts": [
                    {"shape": "box", "min": [0, 0, 0], "max": [1, 0.45, 1], "color": "#7a7d80"},
                    {"shape": "box", "min": [0, 0, 0], "max": [1, 1, 0.25], "color": "#7a7d80"},
                    {
                        "shape": "cylinder",
                        "min": [0, 0, 0.8],
                        "max": [0.05, 0.1, 0.9],
                        "color": "#2b2118",
                    },
                ],
                "confidence": 0.6,
                "provenance": "proposal",
            }
        ],
    }


def test_furniture_round_trips_with_defaults():
    layout = Layout.model_validate(_layout())
    sofa = layout.furniture[0]
    assert sofa.solid is True and sofa.parts[2].shape == "cylinder"
    assert layout.model_dump(by_alias=True)["furniture"][0]["yawRadians"] == pytest.approx(3.14159)


def test_layouts_without_furniture_stay_valid():
    layout = _layout()
    del layout["furniture"]
    assert Layout.model_validate(layout).furniture == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda f: f["parts"][0].update(min=[0, 0.5, 0], max=[1, 0.5, 1]),  # empty along y
        lambda f: f["parts"][0].update(max=[1, 1.2, 1]),  # outside the unit box
        lambda f: f["parts"][0].update(color="grey"),
        lambda f: f["parts"].clear(),
        lambda f: f.update(size=[7.0, 0.85, 0.9]),
        lambda f: f.update(size=[2.0, 4.5, 0.9]),
        lambda f: f.update(size=[2.0, 0.0, 0.9]),
        lambda f: f.update(floorId="floor-9"),
    ],
)
def test_furniture_rejects_impossible_shapes_and_references(mutate):
    layout = _layout()
    mutate(layout["furniture"][0])
    with pytest.raises(ValidationError):
        Layout.model_validate(layout)


def test_furniture_ids_are_unique():
    layout = _layout()
    layout["furniture"].append(copy.deepcopy(layout["furniture"][0]))
    with pytest.raises(ValidationError, match="unique"):
        Layout.model_validate(layout)
