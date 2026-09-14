"""Splits a parsed Docling document into citable chunks.

Uses Docling's `HybridChunker`, which produces token-aware chunks while
keeping each chunk's page numbers and section heading as metadata. We pull
that metadata out immediately and store it on a plain `ChunkRecord` so the
rest of the app (embeddings, vector store, prompts) never has to know
anything about Docling's internal object model.

Known gotcha (see SPECS.md §7 Phase 1): Docling's page-number metadata can
be inconsistent for chunks whose source spans multiple pages (see
docling-project/docling discussion #1012). We don't hide that — we keep
every page the chunk touches in `page_numbers` and expose `primary_page`
(the minimum) as the single citation page, so nothing is silently wrong.
"""

from dataclasses import dataclass

from docling.chunking import HybridChunker
from docling_core.types.doc.document import DoclingDocument

# max_tokens is kept comfortably under common embedding-model limits
# (all-MiniLM-L6-v2 / bge-small-en-v1.5 both handle up to ~512 tokens) so
# chunks don't get silently truncated at embedding time.
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


def chunk_document(doc: DoclingDocument) -> list[ChunkRecord]:
    """Chunk a DoclingDocument into a list of ChunkRecords."""
    records: list[ChunkRecord] = []

    for i, chunk in enumerate(_chunker.chunk(dl_doc=doc)):
        page_numbers = sorted(
            {
                prov.page_no
                for item in chunk.meta.doc_items
                for prov in item.prov
            }
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
