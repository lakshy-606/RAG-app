"""The core RAG pipeline: query -> retrieve -> generate -> cited answer.

Built on LangChain's current (1.x) composition style: a `ChatPromptTemplate`
piped into a `ChatOpenAI` model with `|` (LCEL). We compose this directly
rather than via the older `create_retrieval_chain` helpers — those moved
to the separate `langchain_classic` package in LangChain 1.0, and a direct
pipe is just as readable for a single-step "stuff the context in and ask"
chain like this one.

`sources` in the returned dict comes straight from Pinecone's retrieval
metadata (page/heading/snippet/score) rather than being parsed out of the
LLM's free-text answer — that's more reliable, and matches the /query
response contract in SPECS.md §6. The LLM is still instructed (see
prompts.py) to cite inline in its answer text, for readability.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings
from app.rag.prompts import SYSTEM_PROMPT, build_context_block
from app.vectorstore.pinecone_client import get_vectorstore

_NOT_FOUND_ANSWER = "This document does not appear to contain information that answers this question."

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context excerpts:\n\n{context}\n\nQuestion: {question}"),
    ]
)

# max_retries handles transient rate-limit/network errors; temperature=0
# keeps answers deterministic and grounded rather than creative, which
# matters for a citation-constrained Q&A task.
_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    api_key=settings.openai_api_key,
    temperature=0,
    max_retries=3,
)

_chain = _prompt | _llm


def _results_to_sources(results) -> list[dict]:
    return [
        {
            # Pinecone stores all numeric metadata as float and returns it
            # that way on retrieval, even though we write plain ints at
            # ingest time — cast back explicitly so this matches the
            # `page: int | None` API contract instead of leaking a 7.0.
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

    context_block = build_context_block(results)
    response = _chain.invoke({"context": context_block, "question": query})

    return {"answer": response.content, "sources": _results_to_sources(results), "doc_id": doc_id}
