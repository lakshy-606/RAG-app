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


def _get_or_create_index():
    """Return a handle to the app's raw Pinecone index, creating it first
    if needed. Safe to call repeatedly — `has_index` makes this idempotent.

    The Pinecone client is constructed here rather than at module import
    time: it raises immediately if `api_key` is empty, and app/config.py
    deliberately defaults secrets to "" so the app can still boot (and
    /health respond, and CI run tests that don't touch Pinecone) without
    real keys present. Constructing it lazily keeps that promise — this
    only runs, and only fails, when something actually needs the index.
    """
    pc = Pinecone(api_key=settings.pinecone_api_key)
    if not pc.has_index(settings.pinecone_index_name):
        pc.create_index(
            name=settings.pinecone_index_name,
            vector_type="dense",
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            deletion_protection="disabled",
        )
    return pc.Index(settings.pinecone_index_name)


def get_vectorstore() -> PineconeVectorStore:
    """Return a LangChain-wrapped handle to the app's Pinecone index.

    Built fresh on each call — cheap, since no network call happens until
    something actually calls add_documents/search on the result.
    """
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
    return PineconeVectorStore(index=_get_or_create_index(), embedding=embeddings)
