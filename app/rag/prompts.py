"""Prompt template for the answer-generation step.

The one rule that matters here: the model must answer only from the
retrieved context and must cite where each part of the answer came from,
in the [p.X, "Heading"] format the /query response's `sources` are built
from independently (see pipeline.py).
"""

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


def build_context_block(results) -> str:
    """Turn retrieved (Document, score) pairs into the labeled context
    block the LLM reads.

    `results` is what `vectorstore.similarity_search_with_score` returns
    (see app/vectorstore/pinecone_client.py) — LangChain Documents carry
    the chunk text as `.page_content` and our page/heading metadata as
    `.metadata`.
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
