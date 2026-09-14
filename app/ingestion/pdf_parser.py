"""Thin wrapper around Docling's PDF parser.

Docling's `DocumentConverter` turns a PDF into a `DoclingDocument` — a
structured representation that keeps page numbers and heading hierarchy
attached to every piece of content. That structure is what lets the
chunker (see `chunker.py`) report *which page* and *which section heading*
an answer came from.
"""

from pathlib import Path

from docling.document_converter import DocumentConverter

# A single converter instance is reused across calls — Docling initializes
# its layout/OCR models once and keeps them warm.
_converter = DocumentConverter()


def parse_pdf(path: str | Path):
    """Convert a PDF file into a Docling `DoclingDocument`.

    Raises whatever Docling raises on a corrupt/unreadable file — callers
    are expected to handle that (e.g. return a 400 from the /ingest route).
    """
    result = _converter.convert(str(path))
    return result.document
