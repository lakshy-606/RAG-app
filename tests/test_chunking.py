"""Validates PDF parsing + chunking against the real sample PDF.

This isn't just a smoke test — it's the spot-check Phase 1 of SPECS.md
calls for, since Docling's page-number metadata has documented edge cases.
If this ever starts failing after a docling upgrade, treat it as a real
signal, not a flaky test.
"""

from pathlib import Path

from app.ingestion import chunk_document, parse_pdf

SAMPLE_PDF = Path(__file__).parent.parent / "data" / "relevant_section_identification-sample.pdf"


def test_chunks_are_produced_with_page_numbers():
    doc = parse_pdf(SAMPLE_PDF)
    chunks = chunk_document(doc)

    assert len(chunks) > 0, "expected at least one chunk from the sample PDF"

    for chunk in chunks:
        assert chunk.text.strip(), f"chunk {chunk.chunk_index} has empty text"
        assert chunk.page_numbers, f"chunk {chunk.chunk_index} has no page numbers"
        assert chunk.primary_page == min(chunk.page_numbers)
        # sample PDF is 40 pages — every page number must fall in that range
        assert all(1 <= p <= 40 for p in chunk.page_numbers)


def test_some_chunks_have_headings():
    # Not every chunk necessarily has a heading (e.g. a cover page), but a
    # 40-page structured document should have *some* heading-tagged chunks.
    doc = parse_pdf(SAMPLE_PDF)
    chunks = chunk_document(doc)

    headed = [c for c in chunks if c.heading]
    assert len(headed) > 0, "expected at least some chunks to carry a section heading"


def test_manual_spot_check_sample(capsys):
    """Prints the first few chunks so a human can eyeball them against the
    actual PDF (open data/relevant_section_identification-sample.pdf and
    compare). Run with `pytest -s tests/test_chunking.py::test_manual_spot_check_sample`
    to see the output.
    """
    doc = parse_pdf(SAMPLE_PDF)
    chunks = chunk_document(doc)

    for chunk in chunks[:5]:
        print(
            f"\n--- chunk {chunk.chunk_index} | page(s) {chunk.page_numbers} "
            f"| heading: {chunk.heading!r} ---\n{chunk.text[:200]}..."
        )
