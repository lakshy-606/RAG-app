"""Retrieval-augmented answer generation: query -> retrieve -> rerank -> generate -> cited answer.

Composed directly with LangChain's LCEL pipe (`prompt | llm`) rather than
a prebuilt chain helper — this is a single "stuff the context in and ask"
step, so the direct composition is just as readable and has no extra
moving parts.

`sources` in the returned dict comes straight from Pinecone's retrieval
metadata (page/heading/snippet/score) rather than being parsed out of the
LLM's free-text answer — that's more reliable. The LLM is still
instructed to cite inline in its answer text, for readability.

`chat_model`/`temperature`/`top_k`/`embedding_model`/`use_reranking` are
all per-request overrides (see the UI's parameter sidebar) — none of them
need a redeploy or touch Docling. `chat_model` is checked against a fixed
allowlist for the same reason `embedding_model` is in app/vectorstore.py:
this is a public, unauthenticated endpoint, so request-supplied values
never get passed straight through to a paid API untrusted.
"""

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pinecone import Pinecone

from app.config import settings
from app.vectorstore import get_vectorstore

CHAT_MODELS = ["gpt-4o-mini", "gpt-4o"]
DEFAULT_CHAT_MODEL = "gpt-4o-mini"

# Free Pinecone-hosted reranking model — no extra cost, unlike the paid
# Cohere options. Confirmed live: correctly separates a genuinely relevant
# chunk (score ~0.99) from an irrelevant one (score ~0.01) on a real query.
RERANK_MODEL = "bge-reranker-v2-m3"
# When reranking, retrieve more dense-search candidates than top_k before
# narrowing back down — reranking only helps if it has a wider pool to
# pick the truly best matches from. Capped so a large top_k doesn't turn
# into an unbounded number of rerank units.
_RERANK_CANDIDATE_MULTIPLIER = 4
_RERANK_CANDIDATE_MAX = 40

SYSTEM_PROMPT = """\
You are a document Q&A assistant. Answer the user's question using ONLY \
the context excerpts provided below — do not use outside knowledge, and \
do not guess. Each excerpt is labeled with its page number and (when \
known) its section heading, like [p.12, "Section 3.2 Limits"].

Rules:
- Cite the page/heading label for every claim you make, inline, e.g. \
  "...as stated in [p.12, \\"Section 3.2 Limits\\"]."
- If the context does not contain the answer, say plainly that the \
  document does not appear to contain this information — do not fabricate \
  an answer.
- Be concise and direct.
"""

_NOT_FOUND_ANSWER = "This document does not appear to contain information that answers this question."

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context excerpts:\n\n{context}\n\nQuestion: {question}"),
    ]
)


@lru_cache(maxsize=len(CHAT_MODELS) * 11)  # 11 = temperature 0.0..1.0 in 0.1 steps
def _get_chain(chat_model: str, temperature: float):
    """Build the prompt|llm chain lazily, on first use, and cache it per
    (model, temperature) pair — cheap to keep a handful of these around,
    and avoids reconstructing a ChatOpenAI client on every single request.

    `ChatOpenAI` validates its `api_key` eagerly in its constructor and
    raises if it's empty — and app/config.py deliberately defaults secrets
    to "" so the app can still boot (and /health respond, and CI run tests
    that don't touch OpenAI) without real keys present. Building this at
    import time would break that; building it lazily means construction —
    and any missing-key failure — only happens when a query is actually
    answered.
    """
    llm = ChatOpenAI(model=chat_model, api_key=settings.openai_api_key, temperature=temperature, max_retries=3)
    return _prompt | llm


@lru_cache(maxsize=1)
def _get_pinecone_client() -> Pinecone:
    # Lazy for the same reason as app/vectorstore.py's client — raises
    # immediately on an empty api_key, and app boot shouldn't depend on one.
    return Pinecone(api_key=settings.pinecone_api_key)


def _rerank(query: str, results: list[tuple], top_k: int) -> list[tuple]:
    """Rerank dense-search (Document, score) pairs with Pinecone's hosted
    reranker, returning the best `top_k` — same (Document, score) shape,
    with `score` replaced by the reranker's relevance score.
    """
    if not results:
        return results

    documents = [{"id": str(i), "chunk_text": doc.page_content} for i, (doc, _score) in enumerate(results)]
    reranked = _get_pinecone_client().inference.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=min(top_k, len(results)),
        rank_fields=["chunk_text"],
        return_documents=False,
    )
    return [(results[item.index][0], item.score) for item in reranked.data]


def _build_context_block(results) -> str:
    """Turn retrieved (Document, score) pairs into the labeled context
    block the LLM reads. LangChain Documents carry the chunk text as
    `.page_content` and our page/heading metadata as `.metadata`.
    """
    lines = []
    for doc, _score in results:
        heading = doc.metadata.get("section_heading") or "Untitled section"
        # Pinecone returns numeric metadata as float (e.g. 7.0) even though
        # we write plain ints at ingest time — display as a clean int so
        # the model's inline citations read "p.7", not "p.7.0".
        page = doc.metadata.get("page_number")
        page = int(page) if page is not None else "?"
        lines.append(f'[p.{page}, "{heading}"]\n{doc.page_content}')
    return "\n\n---\n\n".join(lines)


def _results_to_sources(results) -> list[dict]:
    return [
        {
            # Same float-vs-int Pinecone quirk as above — cast back so this
            # matches the `page: int | None` API contract.
            "page": (
                int(page) if (page := doc.metadata.get("page_number")) is not None else None
            ),
            "heading": doc.metadata.get("section_heading") or None,
            "snippet": doc.page_content,
            "score": score,
        }
        for doc, score in results
    ]


def answer_query(
    query: str,
    doc_id: str | None = None,
    top_k: int = 5,
    embedding_model: str | None = None,
    chat_model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.0,
    use_reranking: bool = True,
) -> dict:
    """Run the full retrieve-then-generate pipeline for one user query."""
    if chat_model not in CHAT_MODELS:
        raise ValueError(f"Unknown chat model: {chat_model!r}")

    vectorstore = get_vectorstore(embedding_model)
    metadata_filter = {"doc_id": doc_id} if doc_id else None

    # With reranking on, cast a wider net in the (cheap) dense search step
    # so the (also-free, but still worth not over-calling) reranker has a
    # real pool to pick the best top_k out of, rather than just reordering
    # the same top_k dense search would've returned anyway.
    fetch_k = (
        min(top_k * _RERANK_CANDIDATE_MULTIPLIER, _RERANK_CANDIDATE_MAX) if use_reranking else top_k
    )
    results = vectorstore.similarity_search_with_score(query, k=fetch_k, filter=metadata_filter)

    if not results:
        return {"answer": _NOT_FOUND_ANSWER, "sources": [], "doc_id": doc_id}

    if use_reranking:
        results = _rerank(query, results, top_k)

    context_block = _build_context_block(results)
    chain = _get_chain(chat_model, round(temperature, 1))
    response = chain.invoke({"context": context_block, "question": query})

    return {"answer": response.content, "sources": _results_to_sources(results), "doc_id": doc_id}
