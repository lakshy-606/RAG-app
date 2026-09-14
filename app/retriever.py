"""Orchestration façade used by the FastAPI routes in main.py.

Keeps main.py thin and framework-agnostic: it calls `ingest_pdf` /
`answer_query` here rather than reaching into ingestion/vectorstore/rag
internals directly.
"""

import uuid

from langchain_core.documents import Document

from app.ingestion import chunk_document, parse_pdf
from app.rag import answer_query as _answer_query
from app.vectorstore import get_vectorstore


def ingest_pdf(path: str, source_filename: str) -> dict:
    """Parse, chunk, embed, and store one PDF. Returns ingest summary info
    matching the POST /ingest response contract (SPECS.md §6).

    Each chunk becomes a LangChain `Document`; `vectorstore.add_documents`
    embeds and upserts it to Pinecone in one call (OpenAI embeddings
    happen automatically inside the vector store wrapper).
    """
    doc = parse_pdf(path)
    chunks = chunk_document(doc)
    doc_id = str(uuid.uuid4())

    documents = [
        Document(
            page_content=chunk.text,
            metadata={
                "doc_id": doc_id,
                "source_filename": source_filename,
                "page_number": chunk.primary_page,
                "page_numbers": chunk.page_numbers,
                "section_heading": chunk.heading or "",
                "chunk_index": chunk.chunk_index,
            },
        )
        for chunk in chunks
    ]
    ids = [f"{doc_id}-chunk-{chunk.chunk_index}" for chunk in chunks]

    get_vectorstore().add_documents(documents=documents, ids=ids)

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
