"""Build the synthetic two-room viewer fixture from its layout using worker code.

Writes into packages/scene-contract/fixtures/:
  synthetic-two-room.collision.glb  collision geometry (world coordinates, identity nodes)
  synthetic-two-room.splat.ply      tiny deterministic 3DGS splat PLY (world frame)
  synthetic-two-room.manifest.json  persistent scene manifest referencing both

Run from the repository root:
  uv run --project services/worker python scripts/build-fixture.py [--check]

Output is deterministic. ``--check`` exits non-zero when a file is out of date.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from andarilho_worker.geometry.layout import collision_glb
from andarilho_worker.geometry.ply import write_splat_ply
from andarilho_worker.geometry.synthetic import synthetic_splats
from andarilho_worker.manifest import IDENTITY_MATRIX, asset_entry, build_manifest, scene_prefix
from andarilho_worker.validation import loads_strict, validate_scene

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "packages" / "scene-contract" / "fixtures"
WORKER_SRC = ROOT / "services" / "worker" / "src" / "andarilho_worker"
BUNDLED_LAYOUT = WORKER_SRC / "data" / "synthetic-two-room.layout.json"
STEM = "synthetic-two-room"
PROJECT, REVISION = "fixture-project", "fixture-revision"


def format_json(value: Any, indent: int = 0, lead: int = 0, width: int = 100) -> str:
    """JSON formatted the way Prettier formats it (objects expanded, short arrays inline).

    ``lead`` is the number of characters already on the current line.
    """
    pad = "  " * indent
    inner = "  " * (indent + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = []
        for k, v in value.items():
            head = f"{inner}{json.dumps(k)}: "
            items.append(head + format_json(v, indent + 1, len(head), width))
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        flat = "[" + ", ".join(json.dumps(v) for v in value) + "]"
        scalars = all(not isinstance(v, dict | list) for v in value)
        if scalars and lead + len(flat) + 1 <= width:
            return flat
        items = [inner + format_json(v, indent + 1, len(inner), width) for v in value]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(value, allow_nan=False)


def build() -> dict[str, bytes]:
    layout_bytes = (FIXTURES / f"{STEM}.layout.json").read_bytes()
    layout = loads_strict(layout_bytes)
    glb = collision_glb(layout)
    ply = write_splat_ply(
        synthetic_splats(layout), comments=["andarilho synthetic two-room fixture (world frame)"]
    )
    prefix = scene_prefix(PROJECT, REVISION)
    manifest = build_manifest(
        scene_id="fixture-scene",
        scene_revision_id=REVISION,
        revision=1,
        run_id="fixture-run",
        layout=layout,
        layout_revision=1,
        render=asset_entry(prefix + "scene.ply", ply, "ply"),
        collision=asset_entry(prefix + "collision.glb", glb, "glb"),
        poster=None,
        world_from_decoded_render=IDENTITY_MATRIX,
        scale_status="user_measured",
        geometry_source="synthetic_fixture",
        recipe_digest="synthetic-fixture-v1",
        capture_set_id="fixture-capture",
        quality={
            "registeredViews": 0,
            "selectedViews": 0,
            "disconnectedComponents": 0,
            "warnings": ["Synthetic fixture: no capture was reconstructed."],
        },
    )
    blobs = {prefix + "scene.ply": ply, prefix + "collision.glb": glb}
    findings = validate_scene(manifest, lambda key: blobs[key], project_id=PROJECT, layout=layout)
    if findings:
        raise SystemExit("fixture failed validation: " + "; ".join(f.message for f in findings))
    return {
        f"{STEM}.collision.glb": glb,
        f"{STEM}.splat.ply": ply,
        f"{STEM}.manifest.json": (format_json(manifest) + "\n").encode(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if outputs are out of date")
    args = parser.parse_args()
    outputs = build()
    stale = []
    if BUNDLED_LAYOUT.read_bytes() != (FIXTURES / f"{STEM}.layout.json").read_bytes():
        stale.append(str(BUNDLED_LAYOUT.relative_to(ROOT)))
        if not args.check:
            BUNDLED_LAYOUT.write_bytes((FIXTURES / f"{STEM}.layout.json").read_bytes())
    for name, data in outputs.items():
        path = FIXTURES / name
        if path.exists() and path.read_bytes() == data:
            continue
        stale.append(str(path.relative_to(ROOT)))
        if not args.check:
            path.write_bytes(data)
    if args.check and stale:
        print("out of date: " + ", ".join(stale), file=sys.stderr)
        return 1
    for name, data in outputs.items():
        print(f"{name}: {len(data)} bytes sha256={hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
