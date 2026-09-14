"""The core RAG pipeline: query -> retrieve -> generate -> cited answer.

`sources` in the returned dict comes straight from Pinecone's retrieval
metadata (page/heading/snippet/score) rather than being parsed out of the
LLM's free-text answer — that's more reliable, and matches the /query
response contract in SPECS.md §6. The LLM is still instructed (see
prompts.py) to cite inline in its answer text, for readability.
"""

from groq import APIStatusError, Groq

from app.config import settings
from app.embeddings.hf_embedder import embed_text
from app.rag.prompts import SYSTEM_PROMPT, build_context_block, build_user_prompt
from app.vectorstore.pinecone_client import query_chunks

_client = Groq(api_key=settings.groq_api_key)

_NOT_FOUND_ANSWER = "This document does not appear to contain information that answers this question."


def _matches_to_sources(matches) -> list[dict]:
    return [
        {
            "page": m.metadata.get("page_number"),
            "heading": m.metadata.get("section_heading") or None,
            "snippet": m.metadata.get("chunk_text", ""),
            "score": m.score,
        }
        for m in matches
    ]


def _call_groq(messages: list[dict]) -> str:
    """Call Groq's primary model, falling back to a smaller model on a
    rate-limit response rather than failing the whole request — Groq's
    free tier has fairly low request limits (see SPECS.md §8 risk 4), and
    a demo/grading session can plausibly hit them.
    """
    try:
        resp = _client.chat.completions.create(model=settings.groq_model, messages=messages)
        return resp.choices[0].message.content
    except APIStatusError as e:
        if e.status_code == 429:
            resp = _client.chat.completions.create(
                model=settings.groq_fallback_model, messages=messages
            )
            return resp.choices[0].message.content
        raise


def answer_query(query: str, doc_id: str | None = None, top_k: int = 5) -> dict:
    """Run the full retrieve-then-generate pipeline for one user query."""
    query_vector = embed_text(query)
    matches = query_chunks(query_vector, top_k=top_k, doc_id=doc_id)

    if not matches:
        return {"answer": _NOT_FOUND_ANSWER, "sources": [], "doc_id": doc_id}

    context_block = build_context_block(matches)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, context_block)},
    ]
    answer = _call_groq(messages)

    return {"answer": answer, "sources": _matches_to_sources(matches), "doc_id": doc_id}
