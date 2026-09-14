"""PDF parsing and chunking, via Docling.

Docling's `DocumentConverter` turns a PDF into a structured
`DoclingDocument` that keeps page numbers and heading hierarchy attached
to every piece of content. `HybridChunker` then splits that into
token-aware chunks while preserving that same metadata. We pull the page
numbers and heading straight out and store them on a plain `ChunkRecord`,
so the rest of the app (vector store, prompts) never has to know anything
about Docling's object model.

Known gotcha: Docling's page-number metadata can be inconsistent for
chunks whose source spans multiple pages (see docling-project/docling
discussion #1012). We don't hide that — we keep every page the chunk
touches in `page_numbers` and expose `primary_page` (the minimum) as the
single citation page, so nothing is silently wrong.
"""

from dataclasses import dataclass
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument

# A single converter instance is reused across calls — Docling initializes
# its layout/OCR models once and keeps them warm.
_converter = DocumentConverter()

# 512 tokens is well under OpenAI's embedding input limit; it's chosen for
# retrieval granularity (a chunk small enough to cite precisely) rather
# than to avoid truncation.
_chunker = HybridChunker(max_tokens=512)


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
    return _converter.convert(str(path)).document


def chunk_document(doc: DoclingDocument) -> list[ChunkRecord]:
    """Chunk a DoclingDocument into a list of ChunkRecords."""
    records: list[ChunkRecord] = []

    for i, chunk in enumerate(_chunker.chunk(dl_doc=doc)):
        page_numbers = sorted(
            {prov.page_no for item in chunk.meta.doc_items for prov in item.prov}
        )
        heading = chunk.meta.headings[-1] if chunk.meta.headings else None

        records.append(
            ChunkRecord(
                chunk_index=i,
                text=_chunker.contextualize(chunk),
                page_numbers=page_numbers,
                heading=heading,
            )
        )

    return records
