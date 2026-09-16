"""PDF parsing and chunking, via Docling.

Docling's `DocumentConverter` turns a PDF into a structured
`DoclingDocument` that keeps page numbers and heading hierarchy attached
to every piece of content. `HybridChunker` then splits that into
token-aware chunks while preserving that same metadata. We pull the page
numbers and heading straight out and store them on a plain `ChunkRecord`,
so the rest of the app (vector store, prompts) never has to know anything
about Docling's object model.

Docling's page-number metadata can be inconsistent for chunks whose
source spans multiple pages (see docling-project/docling discussion
#1012), so every page a chunk touches is kept in `page_numbers`, with
`primary_page` (the minimum) exposed as the single citation page.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-checking only — kept out of the runtime import so modules that
    # import this file (e.g. app/retriever.py, for its non-Docling
    # functions) don't need `docling_core` installed unless they actually
    # call parse_pdf/chunk_document. `from __future__ import annotations`
    # above makes the DoclingDocument annotations below lazy strings, so
    # this works with no other code changes.
    from docling_core.types.doc.document import DoclingDocument


def _log(msg: str) -> None:
    print(f"[ingestion] {msg}", file=sys.stderr, flush=True)


@lru_cache(maxsize=1)
def _get_converter():
    """Build Docling's DocumentConverter lazily, on first use, and cache it.

    Constructing this eagerly at import time was blocking uvicorn from
    ever binding to its port — Docling's layout/OCR model initialization
    has to finish before `app.main:app` finishes importing, which on
    Lambda meant /health never responded within the cold-start timeout
    even though nothing was actually broken, just slow, with zero
    logging in the way to show that. Lazy construction decouples "the
    server is up" from "Docling is ready" and, logged like this, makes
    the actual cost visible instead of silent.
    """
    from docling.document_converter import DocumentConverter

    _log("constructing DocumentConverter...")
    start = time.monotonic()
    converter = DocumentConverter()
    _log(f"DocumentConverter ready in {time.monotonic() - start:.1f}s")
    return converter


@lru_cache(maxsize=1)
def _get_chunker():
    from docling.chunking import HybridChunker

    _log("constructing HybridChunker...")
    start = time.monotonic()
    # 512 tokens is well under OpenAI's embedding input limit; chosen for
    # retrieval granularity (a chunk small enough to cite precisely)
    # rather than to avoid truncation.
    chunker = HybridChunker(max_tokens=512)
    _log(f"HybridChunker ready in {time.monotonic() - start:.1f}s")
    return chunker


@dataclass
class ChunkRecord:
    chunk_index: int
    text: str  # heading-enriched text (from chunker.contextualize) — embed this
    page_numbers: list[int]
    heading: str | None

    @property
    def primary_page(self) -> int | None:
        return min(self.page_numbers) if self.page_numbers else None


def parse_pdf(path: str | Path) -> DoclingDocument:
    """Convert a PDF file into a Docling `DoclingDocument`.

    Raises whatever Docling raises on a corrupt/unreadable file — callers
    are expected to handle that (e.g. return a 400 from the /ingest route).
    """
    _log(f"parsing {path}...")
    start = time.monotonic()
    result = _get_converter().convert(str(path)).document
    _log(f"parsed in {time.monotonic() - start:.1f}s")
    return result


def chunk_document(doc: DoclingDocument) -> list[ChunkRecord]:
    """Chunk a DoclingDocument into a list of ChunkRecords."""
    chunker = _get_chunker()
    records: list[ChunkRecord] = []

    for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
        page_numbers = sorted(
            {prov.page_no for item in chunk.meta.doc_items for prov in item.prov}
        )
        heading = chunk.meta.headings[-1] if chunk.meta.headings else None

        records.append(
            ChunkRecord(
                chunk_index=i,
                text=chunker.contextualize(chunk),
                page_numbers=page_numbers,
                heading=heading,
            )
        )

    return records
