"""Small, structurally real PDFs for the inspection tests, built in memory."""

from __future__ import annotations

import io
import zlib

from pypdf import PdfWriter

CATALOG = b"<< /Type /Catalog /Pages 2 0 R >>"
PAGE = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] >>"


def page_tree(pages: int, catalog: bytes = CATALOG) -> list[bytes]:
    """Objects 1.. of a document: the catalog, one /Pages node and its pages."""
    kids = " ".join(f"{3 + i} 0 R" for i in range(pages))
    root = f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode()
    return [catalog, root] + [PAGE] * pages


def classic_pdf(objects: list[bytes]) -> bytes:
    """Objects numbered from 1, with a plain cross-reference table; object 1 is the root."""
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def object_stream_pdf(objects: list[bytes]) -> bytes:
    """Every object inside one Flate object stream, indexed by a compressed xref stream.

    As in optimized PDFs, no ``/Type /Page`` token appears in the file's bytes.
    """
    body, offsets = b"", []
    for obj in objects:
        offsets.append(len(body))
        body += obj + b"\n"
    index = " ".join(f"{n} {offset}" for n, offset in enumerate(offsets, start=1)).encode() + b"\n"
    packed = zlib.compress(index + body)
    stream_number, xref_number = len(objects) + 1, len(objects) + 2
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    stream_offset = len(out)
    out += (
        (
            f"{stream_number} 0 obj\n<< /Type /ObjStm /N {len(objects)} /First {len(index)} "
            f"/Filter /FlateDecode /Length {len(packed)} >>\nstream\n"
        ).encode()
        + packed
        + b"\nendstream\nendobj\n"
    )
    xref_offset = len(out)

    def row(kind: int, field2: int, field3: int) -> bytes:
        return bytes([kind]) + field2.to_bytes(4, "big") + field3.to_bytes(2, "big")

    rows = [row(0, 0, 65535)]
    rows += [row(2, stream_number, i) for i in range(len(objects))]
    rows += [row(1, stream_offset, 0), row(1, xref_offset, 0)]
    table = zlib.compress(b"".join(rows))
    out += (
        (
            f"{xref_number} 0 obj\n<< /Type /XRef /Size {xref_number + 1} /W [1 4 2] /Root 1 0 R "
            f"/Filter /FlateDecode /Length {len(table)} >>\nstream\n"
        ).encode()
        + table
        + b"\nendstream\nendobj\n"
    )
    out += f"startxref\n{xref_offset}\n%%EOF\n".encode()
    return bytes(out)


def encrypted_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=100)
    writer.encrypt("owner-only", algorithm="RC4-128")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
