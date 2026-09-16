"""Orchestration façade used by the FastAPI routes in main.py.

Keeps main.py thin and framework-agnostic: it calls `ingest_pdf` /
`answer_query` / `reindex_with_model` here rather than reaching into
ingestion/vectorstore/rag internals directly.
"""

import json
import uuid
from pathlib import Path

from langchain_core.documents import Document

from app.ingestion import chunk_document, parse_pdf
from app.rag import DEFAULT_CHAT_MODEL
from app.rag import answer_query as _answer_query
from app.vectorstore import get_vectorstore, index_has_data

CACHED_CHUNKS_PATH = Path(__file__).parent.parent / "data" / "sample_chunks.json"


def ingest_pdf(path: str, source_filename: str) -> dict:
    """Parse, chunk, embed, and store one PDF. Returns an ingest summary
    (doc_id, page/chunk counts, status).

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
                # Pinecone's metadata schema only accepts lists of strings
                # (not numbers) for array-valued fields.
                "page_numbers": [str(p) for p in chunk.page_numbers],
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


def reindex_with_model(embedding_model: str) -> dict:
    """(Re-)embed the already-parsed sample chunks with a different
    embedding model, into that model's own Pinecone index.

    Deliberately doesn't touch Docling or the source PDF at all — it
    reuses data/sample_chunks.json (saved from the last real ingestion),
    since switching embedding models only requires new *vectors* for
    text that's already been extracted, not re-parsing. That's what makes
    this safe to expose on a public endpoint: no Docling cold start, and
    it's idempotent — if the target index already has vectors, this is a
    no-op, so the real embedding cost is paid at most once per model,
    ever, no matter how many times it's requested.
    """
    if index_has_data(embedding_model):
        return {"embedding_model": embedding_model, "status": "already_indexed"}

    cached = json.loads(CACHED_CHUNKS_PATH.read_text())
    doc_id = cached["doc_id"]

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "doc_id": doc_id,
                "source_filename": cached["source_filename"],
                "page_number": chunk["page_number"],
                "page_numbers": chunk["page_numbers"],
                "section_heading": chunk["section_heading"] or "",
                "chunk_index": chunk["chunk_index"],
            },
        )
        for chunk in cached["chunks"]
    ]
    ids = [f"{doc_id}-chunk-{chunk['chunk_index']}" for chunk in cached["chunks"]]

    get_vectorstore(embedding_model).add_documents(documents=documents, ids=ids)

    return {
        "embedding_model": embedding_model,
        "doc_id": doc_id,
        "num_chunks": len(documents),
        "status": "reindexed",
    }


def answer_query(
    query: str,
    doc_id: str | None = None,
    top_k: int = 5,
    embedding_model: str | None = None,
    chat_model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.0,
    use_reranking: bool = True,
) -> dict:
    """Answer a query, optionally scoped to one previously-ingested doc_id."""
    return _answer_query(
        query=query,
        doc_id=doc_id,
        top_k=top_k,
        embedding_model=embedding_model,
        chat_model=chat_model,
        temperature=temperature,
        use_reranking=use_reranking,
    )
