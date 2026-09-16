#!/usr/bin/env python3
"""Manual end-to-end ingest of the sample PDF — useful for testing the
ingestion pipeline before the FastAPI layer is running.

Usage (from the project root, with the venv active):
    python scripts/ingest_sample.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retriever import ingest_pdf

SAMPLE_PDF = Path(__file__).parent.parent / "data" / "relevant_section_identification-sample.pdf"


def main():
    print(f"Ingesting {SAMPLE_PDF.name} (parse -> chunk -> embed -> upsert to Pinecone) ...")
    result = ingest_pdf(path=str(SAMPLE_PDF), source_filename=SAMPLE_PDF.name)
    print(
        f"Done. doc_id={result['doc_id']} — {result['num_chunks']} chunks "
        f"across {result['num_pages']} pages. Use this doc_id to query it via Phase 3/4."
    )


if __name__ == "__main__":
    main()
