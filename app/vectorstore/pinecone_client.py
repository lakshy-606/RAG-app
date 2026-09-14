"""Pinecone vector store: index lifecycle + upsert/query.

Index schema (see SPECS.md §5): serverless, dense, 384 dimensions (matches
both candidate HF embedding models), cosine metric, pinned to aws/us-east-1
since not every region is available on Pinecone's free tier.
"""

from pinecone import Pinecone, ServerlessSpec

from app.config import settings

EMBEDDING_DIMENSION = 384

_pc = Pinecone(api_key=settings.pinecone_api_key)


def get_index():
    """Return a handle to the app's Pinecone index, creating it if needed.

    Safe to call repeatedly — `has_index` makes this idempotent so it can
    run on every app startup / ingest without erroring on the second call.
    """
    if not _pc.has_index(settings.pinecone_index_name):
        _pc.create_index(
            name=settings.pinecone_index_name,
            vector_type="dense",
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            deletion_protection="disabled",
        )
    return _pc.Index(settings.pinecone_index_name)


# Metadata payloads are capped around 40KB by Pinecone; chunk_text is kept
# comfortably under that (chunks are <=512 tokens) but we truncate
# defensively rather than risk an upsert failing on an outlier chunk.
_MAX_METADATA_TEXT_CHARS = 4000


def upsert_chunks(doc_id: str, source_filename: str, chunks) -> None:
    """Embed-and-store a list of ChunkRecord (see app/ingestion/chunker.py).

    Imported lazily to avoid a hard import-time dependency from vectorstore
    on embeddings — keeps each module independently testable/mockable.
    """
    from app.embeddings.hf_embedder import embed_texts

    vectors = embed_texts([c.text for c in chunks])
    index = get_index()

    payload = [
        {
            "id": f"{doc_id}-chunk-{chunk.chunk_index}",
            "values": vector,
            "metadata": {
                "doc_id": doc_id,
                "source_filename": source_filename,
                "page_number": chunk.primary_page,
                "page_numbers": chunk.page_numbers,
                "section_heading": chunk.heading or "",
                "chunk_text": chunk.text[:_MAX_METADATA_TEXT_CHARS],
                "chunk_index": chunk.chunk_index,
            },
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    # Batch upserts (Pinecone recommends <=100-500 vectors per call) rather
    # than one call per vector, which would be slow and wasteful for a
    # 40-page PDF's worth of chunks.
    batch_size = 100
    for i in range(0, len(payload), batch_size):
        index.upsert(vectors=payload[i : i + batch_size])


def query_chunks(query_vector: list[float], top_k: int = 5, doc_id: str | None = None):
    """Return the top_k most similar chunks, optionally scoped to one doc_id."""
    index = get_index()
    query_filter = {"doc_id": {"$eq": doc_id}} if doc_id else None

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=query_filter,
    )
    return result.matches
