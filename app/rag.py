"""Retrieval-augmented answer generation: query -> retrieve -> generate -> cited answer.

Composed directly with LangChain's LCEL pipe (`prompt | llm`) rather than
a prebuilt chain helper — this is a single "stuff the context in and ask"
step, so the direct composition is just as readable and has no extra
moving parts.

`sources` in the returned dict comes straight from Pinecone's retrieval
metadata (page/heading/snippet/score) rather than being parsed out of the
LLM's free-text answer — that's more reliable, and matches the /query
response contract in SPECS.md §6. The LLM is still instructed to cite
inline in its answer text, for readability.
"""

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings
from app.vectorstore import get_vectorstore

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


@lru_cache(maxsize=1)
def _get_chain():
    """Build the prompt|llm chain lazily, on first use, and cache it.

    `ChatOpenAI` validates its `api_key` eagerly in its constructor and
    raises if it's empty — and app/config.py deliberately defaults secrets
    to "" so the app can still boot (and /health respond, and CI run tests
    that don't touch OpenAI) without real keys present. Building this at
    import time would break that; building it lazily means construction —
    and any missing-key failure — only happens when a query is actually
    answered.
    """
    # max_retries handles transient rate-limit/network errors; temperature=0
    # keeps answers deterministic and grounded rather than creative, which
    # matters for a citation-constrained Q&A task.
    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_retries=3,
    )
    return _prompt | llm


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


def answer_query(query: str, doc_id: str | None = None, top_k: int = 5) -> dict:
    """Run the full retrieve-then-generate pipeline for one user query."""
    vectorstore = get_vectorstore()
    metadata_filter = {"doc_id": doc_id} if doc_id else None
    results = vectorstore.similarity_search_with_score(query, k=top_k, filter=metadata_filter)

    if not results:
        return {"answer": _NOT_FOUND_ANSWER, "sources": [], "doc_id": doc_id}

    context_block = _build_context_block(results)
    response = _get_chain().invoke({"context": context_block, "question": query})

    return {"answer": response.content, "sources": _results_to_sources(results), "doc_id": doc_id}
