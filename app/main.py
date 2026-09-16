"""FastAPI app: health check, PDF ingest, cited Q&A, and the static UI.

Run locally with:
    uvicorn app.main:app --reload
"""

import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.rag import CHAT_MODELS, DEFAULT_CHAT_MODEL
from app.retriever import answer_query, ingest_pdf, reindex_with_model
from app.vectorstore import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS

app = FastAPI(title="RAG App", description="PDF Q&A with page/heading citations")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def config():
    """Available/default hyperparameter values, for the UI's settings
    sidebar to populate itself from — one source of truth instead of
    duplicating this list in JS."""
    return {
        "embedding_models": list(EMBEDDING_MODELS.keys()),
        "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chat_models": CHAT_MODELS,
        "default_chat_model": DEFAULT_CHAT_MODEL,
    }


@app.post("/ingest")
async def ingest(file: UploadFile):
    """Upload a PDF: parse, chunk, embed, and store it in Pinecone.

    Runs synchronously — blocking on Docling conversion plus one OpenAI
    embedding call per chunk. Fine for a demo-sized PDF (~40 pages); a
    larger corpus would need a background job/queue instead.

    Not exposed in the deployed UI (static/index.html): Docling's
    cold-start time on Lambda is too slow for a synchronous request.
    Still fully functional for local use or any environment without
    Lambda's cold-start constraint — see README.md "Vector Database &
    Reindexing" for how the deployed app ingests instead.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only application/pdf uploads are supported")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()

        try:
            result = ingest_pdf(path=tmp.name, source_filename=file.filename or Path(tmp.name).name)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to process PDF: {e}") from e

    return result


class ReindexRequest(BaseModel):
    embedding_model: str


@app.post("/reindex")
def reindex(request: ReindexRequest):
    """Re-embed the already-ingested sample chunks with a different
    embedding model, into that model's own Pinecone index.

    Safe to expose on this public, unauthenticated endpoint specifically
    because it's cheap and idempotent: no Docling/PDF involved (reuses
    data/sample_chunks.json), and a no-op if that model's index already
    has data — so the real embedding cost is paid at most once per model,
    ever, regardless of how many times/people call this.
    """
    if request.embedding_model not in EMBEDDING_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown embedding model. Choose one of: {list(EMBEDDING_MODELS)}",
        )
    return reindex_with_model(request.embedding_model)


class QueryRequest(BaseModel):
    query: str
    doc_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    embedding_model: str | None = None
    chat_model: str = DEFAULT_CHAT_MODEL
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    use_reranking: bool = True


class Source(BaseModel):
    page: int | None
    heading: str | None
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    doc_id: str | None


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Answer a question from previously-ingested PDF(s), citing page/heading."""
    if request.embedding_model is not None and request.embedding_model not in EMBEDDING_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown embedding model. Choose one of: {list(EMBEDDING_MODELS)}",
        )
    if request.chat_model not in CHAT_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown chat model. Choose one of: {CHAT_MODELS}")

    return answer_query(
        query=request.query,
        doc_id=request.doc_id,
        top_k=request.top_k,
        embedding_model=request.embedding_model,
        chat_model=request.chat_model,
        temperature=request.temperature,
        use_reranking=request.use_reranking,
    )


# Serves static/index.html at "/" and static/app.js, static/style.css
# alongside it (html=True makes "/" resolve to index.html automatically).
# Registered last so it only catches requests the routes above don't.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
