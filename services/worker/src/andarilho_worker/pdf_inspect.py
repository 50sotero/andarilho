"""Bounded PDF floor-plan inspection, run as ``python -I -m andarilho_worker.pdf_inspect FILE``.

pypdf reads the cross-reference tables or streams (object streams included, as
optimized PDFs use) and this module walks the actual page tree, counting leaf
pages with a node budget and cycle detection; nothing is rendered or executed.
pypdf caps every decompressed stream, and on Linux the process also limits its
own address space. The worker runs it through the stage runner, which bounds its
time and scratch space and kills it on cancellation.

Prints one JSON object: ``{"ok": true, "pages": n, "hasJavaScript": bool}`` or
``{"ok": false, "reason": "..."}`` for a PDF that must be rejected. Any other
outcome (a crash, a timeout) is a rejection too.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

MAX_PDF_PAGES = 20
# Pages plus intermediate /Pages nodes visited before giving up on a tree.
MAX_TREE_NODES = 1000
ADDRESS_SPACE_BYTES = 1024**3


class Unsupported(ValueError):
    """A structurally readable PDF that the intake does not accept."""


def _limit_memory() -> None:
    try:
        import resource
    except ImportError:  # not on Windows; the worker images are Linux
        return
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES))


def _get(node: Any, key: str) -> Any:
    """A dictionary entry with indirect references resolved (``get`` returns them raw)."""
    value = node.get(key)
    return None if value is None else value.get_object()


def count_pages(root: Any) -> int:
    """Leaf pages under the catalog's /Pages tree (stops past MAX_PDF_PAGES)."""
    from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject

    stack = [root.raw_get("/Pages")] if "/Pages" in root else []
    seen: set[tuple[int, int]] = set()
    nodes = pages = 0
    while stack:
        ref = stack.pop()
        if isinstance(ref, IndirectObject):
            key = (ref.idnum, ref.generation)
            if key in seen:
                raise Unsupported("PDF page tree is malformed")
            seen.add(key)
        node = ref.get_object()
        nodes += 1
        if nodes > MAX_TREE_NODES:
            raise Unsupported("PDF page tree is too large")
        if not isinstance(node, DictionaryObject):
            raise Unsupported("PDF page tree is malformed")
        kind = _get(node, "/Type")
        if kind == "/Pages" or (kind != "/Page" and "/Kids" in node):
            kids = _get(node, "/Kids")
            if not isinstance(kids, ArrayObject):
                raise Unsupported("PDF page tree is malformed")
            stack.extend(reversed(kids))
        else:
            pages += 1
            if pages > MAX_PDF_PAGES:
                raise Unsupported("PDF has too many pages")
    return pages


def has_javascript(root: Any) -> bool:
    """Document-level JavaScript (named scripts or an open action). Informational only."""
    from pypdf.generic import DictionaryObject

    names = _get(root, "/Names")
    if isinstance(names, DictionaryObject) and "/JavaScript" in names:
        return True
    action = _get(root, "/OpenAction")
    return isinstance(action, DictionaryObject) and _get(action, "/S") == "/JavaScript"


def inspect(path: Path) -> dict[str, Any]:
    from pypdf import PdfReader

    reader = PdfReader(path, strict=False)
    if reader.is_encrypted:
        raise Unsupported("encrypted PDF")
    root = reader.root_object
    pages = count_pages(root)
    if pages == 0:
        raise Unsupported("PDF has no pages")
    return {"pages": pages, "hasJavaScript": has_javascript(root)}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m andarilho_worker.pdf_inspect FILE", file=sys.stderr)
        return 2
    _limit_memory()
    logging.basicConfig(level=logging.ERROR)  # pypdf warns about every repaired quirk
    try:
        report: dict[str, Any] = {"ok": True, **inspect(Path(args[0]))}
    except Unsupported as exc:
        report = {"ok": False, "reason": str(exc)}
    except Exception:  # noqa: BLE001 - any parse failure means the file is not a usable PDF
        report = {"ok": False, "reason": "PDF could not be parsed"}
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
