"""scripts/check-scene.py finds the render model next to the manifest, as ``<stem>.model.glb``."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "packages" / "scene-contract" / "fixtures"
MANIFEST = FIXTURES / "synthetic-two-room.manifest.json"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "check_scene", REPO / "scripts" / "check-scene.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_committed_fixture_passes_and_resolves_the_model_glb(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check-scene.py", str(MANIFEST)])
    assert _load_script().main() == 0
    assert "synthetic-two-room.model.glb" in capsys.readouterr().out


def test_missing_render_model_is_reported(tmp_path, monkeypatch, capsys):
    fixtures = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, fixtures)
    (fixtures / "synthetic-two-room.model.glb").unlink()
    monkeypatch.setattr(sys, "argv", ["check-scene.py", str(fixtures / MANIFEST.name)])
    assert _load_script().main() == 1
    assert "ASSET_MISSING" in capsys.readouterr().err
