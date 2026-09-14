"""Prompt templates for the answer-generation step.

The one rule that matters here: the model must answer only from the
retrieved context and must cite where each part of the answer came from,
in the [p.X, "Heading"] format the /query response parses sources from.
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


def build_context_block(matches) -> str:
    """Turn Pinecone matches into the labeled context block the LLM reads.

    `matches` are Pinecone query results (see pinecone_client.query_chunks),
    each with `.metadata` containing page_number/section_heading/chunk_text.
    """
    lines = []
    for match in matches:
        meta = match.metadata
        heading = meta.get("section_heading") or "Untitled section"
        page = meta.get("page_number")
        lines.append(f'[p.{page}, "{heading}"]\n{meta.get("chunk_text", "")}')
    return "\n\n---\n\n".join(lines)


def build_user_prompt(query: str, context_block: str) -> str:
    return f"Context excerpts:\n\n{context_block}\n\nQuestion: {query}"
