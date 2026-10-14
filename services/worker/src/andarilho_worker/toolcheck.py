"""Check the pinned recipe's command-line options against the installed tools.

Run inside the GPU image (``python -m andarilho_worker.toolcheck``); the image
build does this so a recipe written for another tool version fails the build
instead of failing a user's run. It needs no GPU: every check reads ``--help``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence

from andarilho_worker.recipes import ARGV

# Help commands whose combined output must list every ``--option`` of a template.
# ffmpeg/ffprobe use single-dash options and splat-transform takes none.
HELP: dict[str, tuple[tuple[str, ...], ...]] = {
    "colmap_feature_extractor": (("colmap", "feature_extractor", "-h"),),
    "colmap_sequential_matcher": (("colmap", "sequential_matcher", "-h"),),
    "colmap_exhaustive_matcher": (("colmap", "exhaustive_matcher", "-h"),),
    "colmap_matches_importer": (("colmap", "matches_importer", "-h"),),
    "colmap_mapper": (("colmap", "mapper", "-h"),),
    "colmap_image_undistorter": (("colmap", "image_undistorter", "-h"),),
    "ns_train": (
        ("ns-train", "splatfacto", "--help"),
        ("ns-train", "splatfacto", "colmap", "--help"),
    ),
    "ns_export": (("ns-export", "gaussian-splat", "--help"),),
}

Runner = Callable[[Sequence[str]], str]


def _help_text(argv: Sequence[str]) -> str:
    env = {**os.environ, "COLUMNS": "200"}  # keep long option names on one line
    proc = subprocess.run(  # noqa: S603 - fixed argument lists, no shell
        list(argv), capture_output=True, text=True, timeout=300, env=env, check=False
    )
    return proc.stdout + proc.stderr


def options(template: Sequence[str]) -> list[str]:
    return [token for token in template if token.startswith("--")]


def missing_options(run_help: Runner = _help_text) -> dict[str, list[str]]:
    """Template name -> options its tool's help does not list (empty when all match)."""
    missing: dict[str, list[str]] = {}
    for name, commands in HELP.items():
        text = "\n".join(run_help(argv) for argv in commands)
        absent = [
            opt
            for opt in options(ARGV[name])
            if not re.search(rf"(?<![\w.-]){re.escape(opt)}(?![\w.-])", text)
        ]
        if absent:
            missing[name] = absent
    return missing


def main() -> int:
    missing = missing_options()
    for name, absent in missing.items():
        print(f"{name}: not accepted by the installed tool: {', '.join(absent)}", file=sys.stderr)
    if not missing:
        print(f"recipe options match the installed tools ({len(HELP)} commands checked)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
