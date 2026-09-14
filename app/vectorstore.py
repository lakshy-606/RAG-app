"""Pinecone vector store, accessed through LangChain's PineconeVectorStore.

We still talk to the raw `pinecone` SDK for one thing only — creating the
index if it doesn't exist yet, since that's index *administration*, not
something LangChain's wrapper does for you. Everything else (embedding
text, adding documents, similarity search) goes through
`langchain_pinecone.PineconeVectorStore`, which calls OpenAI's embeddings
API automatically whenever documents are added or searched — callers
never build or pass raw vectors themselves.

Index schema (see SPECS.md §5): serverless, dense, 1536 dimensions
(matches OpenAI's `text-embedding-3-small`), cosine metric, pinned to
aws/us-east-1 since not every region is available on Pinecone's free tier.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.config import settings

EMBEDDING_DIMENSION = 1536

_pc = Pinecone(api_key=settings.pinecone_api_key)


def _get_or_create_index():
    """Return a handle to the app's raw Pinecone index, creating it first
    if needed. Safe to call repeatedly — `has_index` makes this idempotent.
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


def get_vectorstore() -> PineconeVectorStore:
    """Return a LangChain-wrapped handle to the app's Pinecone index.

    Built fresh on each call (cheap — no network call happens until you
    actually add_documents/search) so it always reflects current settings;
    the underlying Pinecone SDK client (`_pc`) and index creation are the
    only parts kept process-wide.
    """
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
    return PineconeVectorStore(index=_get_or_create_index(), embedding=embeddings)
