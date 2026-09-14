"""Orchestration façade used by the FastAPI routers (Phase 4).

Keeps the routers thin and framework-agnostic: they call `ingest_pdf` /
`answer_query` here rather than reaching into ingestion/embeddings/rag
internals directly.
"""

import uuid

from app.ingestion.chunker import chunk_document
from app.ingestion.pdf_parser import parse_pdf
from app.rag.pipeline import answer_query as _answer_query
from app.vectorstore.pinecone_client import upsert_chunks


def ingest_pdf(path: str, source_filename: str) -> dict:
    """Parse, chunk, embed, and store one PDF. Returns ingest summary info
    matching the POST /ingest response contract (SPECS.md §6).
    """
    doc = parse_pdf(path)
    chunks = chunk_document(doc)

    doc_id = str(uuid.uuid4())
    upsert_chunks(doc_id=doc_id, source_filename=source_filename, chunks=chunks)

    return {
        "doc_id": doc_id,
        "source_filename": source_filename,
        "num_pages": len(doc.pages) if hasattr(doc, "pages") else None,
        "num_chunks": len(chunks),
        "status": "ingested",
    }


def answer_query(query: str, doc_id: str | None = None, top_k: int = 5) -> dict:
    """Answer a query, optionally scoped to one previously-ingested doc_id."""
    return _answer_query(query=query, doc_id=doc_id, top_k=top_k)
