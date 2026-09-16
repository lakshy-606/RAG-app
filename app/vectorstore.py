"""Pinecone vector store, accessed through LangChain's PineconeVectorStore.

We still talk to the raw `pinecone` SDK for one thing only — creating the
index if it doesn't exist yet, since that's index *administration*, not
something LangChain's wrapper does for you. Everything else (embedding
text, adding documents, similarity search) goes through
`langchain_pinecone.PineconeVectorStore`, which calls OpenAI's embeddings
API automatically whenever documents are added or searched — callers
never build or pass raw vectors themselves.

Parameterized by embedding model (see EMBEDDING_MODELS below) rather than
hardcoded to one: different models produce different-sized vectors that
aren't compatible with each other, so each model gets its own Pinecone
index (name derived from the model), sized for that model's output. This
is what lets the UI's "embedding model" selector actually work — there's
no such thing as changing an index's dimension in place, so switching
models means switching which index you're reading/writing.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.config import settings

# Fixed, small allowlist — never accept an arbitrary model name from a
# request. Keeps the "reindex" cost/abuse surface bounded: at most one
# real embedding pass per model, ever (see get_vectorstore's idempotency
# note below), across everyone who ever uses the public demo.
EMBEDDING_MODELS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _index_name_for_model(embedding_model: str) -> str:
    # Pinecone index names must be lowercase alphanumeric + hyphens.
    slug = embedding_model.replace(".", "-").replace("_", "-").lower()
    return f"{settings.pinecone_index_name}-{slug}"


def _get_or_create_index(embedding_model: str):
    """Return a handle to the Pinecone index for one embedding model,
    creating it first if needed. Safe to call repeatedly — `has_index`
    makes this idempotent.

    The Pinecone client is constructed here rather than at module import
    time: it raises immediately if `api_key` is empty, and app/config.py
    deliberately defaults secrets to "" so the app can still boot (and
    /health respond, and CI run tests that don't touch Pinecone) without
    real keys present. Constructing it lazily keeps that promise — this
    only runs, and only fails, when something actually needs the index.
    """
    if embedding_model not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown embedding model: {embedding_model!r}")

    index_name = _index_name_for_model(embedding_model)
    pc = Pinecone(api_key=settings.pinecone_api_key)
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=EMBEDDING_MODELS[embedding_model],
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            deletion_protection="disabled",
        )
    return pc.Index(index_name)


def index_has_data(embedding_model: str) -> bool:
    """Whether this model's index already has vectors in it. Used to make
    reindexing idempotent — only actually re-embed content the first time
    a given model is selected, ever."""
    stats = _get_or_create_index(embedding_model).describe_index_stats()
    return stats.total_vector_count > 0


def get_vectorstore(embedding_model: str | None = None) -> PineconeVectorStore:
    """Return a LangChain-wrapped handle to the Pinecone index for one
    embedding model (defaults to the configured default).

    Built fresh on each call — cheap, since no network call happens until
    something actually calls add_documents/search on the result.
    """
    embedding_model = embedding_model or settings.openai_embedding_model
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=settings.openai_api_key)
    return PineconeVectorStore(index=_get_or_create_index(embedding_model), embedding=embeddings)
