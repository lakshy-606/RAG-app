#!/usr/bin/env python3
"""Manual end-to-end ingest of the sample PDF — useful for testing Phases 1-2
before the FastAPI layer (Phase 4) exists.

Usage (from the project root, with the venv active):
    python scripts/ingest_sample.py
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.chunker import chunk_document
from app.ingestion.pdf_parser import parse_pdf
from app.vectorstore.pinecone_client import upsert_chunks

SAMPLE_PDF = Path(__file__).parent.parent / "data" / "relevant_section_identification-sample.pdf"


def main():
    print(f"Parsing {SAMPLE_PDF.name} ...")
    doc = parse_pdf(SAMPLE_PDF)

    print("Chunking ...")
    chunks = chunk_document(doc)
    print(f"  -> {len(chunks)} chunks")

    doc_id = str(uuid.uuid4())
    print(f"Embedding + upserting to Pinecone as doc_id={doc_id} ...")
    upsert_chunks(doc_id=doc_id, source_filename=SAMPLE_PDF.name, chunks=chunks)

    print(f"Done. doc_id={doc_id} — use this to query the sample doc in Phase 3.")


if __name__ == "__main__":
    main()
