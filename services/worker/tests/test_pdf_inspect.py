from __future__ import annotations

import json

import pytest
from pdf_fixtures import (
    CATALOG,
    classic_pdf,
    encrypted_pdf,
    object_stream_pdf,
    page_tree,
)

from andarilho_worker import pdf_inspect


def inspect(tmp_path, data: bytes) -> dict:
    path = tmp_path / "plan.pdf"
    path.write_bytes(data)
    return pdf_inspect.inspect(path)


def test_counts_pages_in_the_page_tree(tmp_path):
    assert inspect(tmp_path, classic_pdf(page_tree(3))) == {"pages": 3, "hasJavaScript": False}


def test_counts_pages_stored_in_compressed_object_streams(tmp_path):
    data = object_stream_pdf(page_tree(2))
    assert b"/Type /Page" not in data  # invisible to a scan of the raw bytes
    assert inspect(tmp_path, data)["pages"] == 2


def test_walks_nested_page_tree_nodes(tmp_path):
    objects = [
        CATALOG,
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 3 >>",
        b"<< /Type /Pages /Parent 2 0 R /Kids [5 0 R 6 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R >>",
        b"<< /Type /Page /Parent 3 0 R >>",
        b"<< /Type /Page /Parent 3 0 R >>",
    ]
    assert inspect(tmp_path, classic_pdf(objects))["pages"] == 3


@pytest.mark.parametrize(
    "data,reason",
    [
        (encrypted_pdf(), "encrypted PDF"),
        (classic_pdf(page_tree(pdf_inspect.MAX_PDF_PAGES + 1)), "PDF has too many pages"),
        (object_stream_pdf(page_tree(pdf_inspect.MAX_PDF_PAGES + 1)), "PDF has too many pages"),
        (classic_pdf(page_tree(0)), "PDF has no pages"),
        # A /Pages node that lists itself must not loop forever.
        (
            classic_pdf([CATALOG, b"<< /Type /Pages /Kids [2 0 R] /Count 1 >>"]),
            "PDF page tree is malformed",
        ),
    ],
)
def test_rejects_pdfs_the_intake_does_not_accept(tmp_path, data, reason):
    with pytest.raises(pdf_inspect.Unsupported, match=reason):
        inspect(tmp_path, data)


def test_reports_document_javascript(tmp_path):
    catalog = b"<< /Type /Catalog /Pages 2 0 R /OpenAction << /S /JavaScript /JS (1) >> >>"
    assert inspect(tmp_path, classic_pdf(page_tree(1, catalog)))["hasJavaScript"] is True


def test_main_prints_one_json_verdict(tmp_path, capsys, monkeypatch):
    # main() caps its own process's address space (on Linux). In-process, that cap would
    # outlive this test and starve every later one, so record the call instead.
    limited: list[bool] = []
    monkeypatch.setattr(pdf_inspect, "_limit_memory", lambda: limited.append(True))
    good, bad = tmp_path / "good.pdf", tmp_path / "bad.pdf"
    good.write_bytes(classic_pdf(page_tree(1)))
    bad.write_bytes(b"%PDF-1.7\nnot a document")
    assert pdf_inspect.main([str(good)]) == 0
    assert limited == [True]
    assert json.loads(capsys.readouterr().out) == {"ok": True, "pages": 1, "hasJavaScript": False}
    assert pdf_inspect.main([str(bad)]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": False, "reason": "PDF could not be parsed"}
