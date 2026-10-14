"""Validate a scene manifest and its local asset files.

Checks the JSON Schema, manifest semantics (finite numbers, similarity matrix,
spawn floor/bounds, asset prefixes), asset bytes/sha256, render asset header, and
the collision GLB (identity node transforms, finite vertices within bounds, spawn
capsule clearance and floor support).

Usage (from the repository root):
  uv run --project services/worker python scripts/check-scene.py \
      packages/scene-contract/fixtures/synthetic-two-room.manifest.json

Asset files are resolved next to the manifest. For ``<stem>.manifest.json`` the
render asset is ``<stem>.splat.<format>``, the collision asset ``<stem>.collision.glb``
and an optional layout ``<stem>.layout.json``; otherwise each key's basename is used.
Override with --render/--collision/--poster/--layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from andarilho_worker.validation import loads_strict, validate_scene


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a scene manifest and local assets.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--collision", type=Path)
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--project-id", help="expected project prefix of every asset key")
    args = parser.parse_args()

    manifest = loads_strict(args.manifest.read_bytes())
    base = args.manifest.parent
    name = args.manifest.name
    stem = name[: -len(".manifest.json")] if name.endswith(".manifest.json") else None
    assets = manifest.get("assets", {}) if isinstance(manifest, dict) else {}

    def default_path(kind: str) -> Path | None:
        asset = assets.get(kind)
        if not isinstance(asset, dict):
            return None
        if stem is not None:
            suffix = {"render": f"splat.{asset.get('format')}", "collision": "collision.glb"}.get(
                kind, f"{kind}.{asset.get('format')}"
            )
            return base / f"{stem}.{suffix}"
        return base / str(asset.get("key", "")).rsplit("/", 1)[-1]

    paths: dict[str, Path | None] = {
        "render": args.render or default_path("render"),
        "collision": args.collision or default_path("collision"),
        "poster": args.poster or default_path("poster"),
    }
    key_to_path = {
        assets[kind]["key"]: path
        for kind, path in paths.items()
        if path is not None and isinstance(assets.get(kind), dict)
    }

    def fetch(key: str) -> bytes:
        path = key_to_path.get(key)
        if path is None or not path.exists():
            raise FileNotFoundError(key)
        return path.read_bytes()

    layout_path = args.layout or (base / f"{stem}.layout.json" if stem else None)
    layout = None
    if layout_path is not None and layout_path.exists():
        layout = loads_strict(layout_path.read_bytes())

    findings = validate_scene(manifest, fetch, project_id=args.project_id, layout=layout)
    if findings:
        for finding in findings:
            print(f"FAIL {finding.code}: {finding.message}", file=sys.stderr)
        return 1
    used = ", ".join(str(p) for p in key_to_path.values())
    print(f"OK {args.manifest} ({used}{', layout ' + str(layout_path) if layout else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
