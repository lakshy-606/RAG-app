"""FastAPI app: health check, PDF ingest, cited Q&A, and the static UI.

Run locally with:
    uvicorn app.main:app --reload
"""

import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.retriever import answer_query, ingest_pdf

app = FastAPI(title="RAG App", description="PDF Q&A with page/heading citations")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile):
    """Upload a PDF: parse, chunk, embed, and store it in Pinecone.

    Runs synchronously — blocking on Docling conversion plus one OpenAI
    embedding call per chunk. Fine for a demo-sized PDF (~40 pages); a
    larger corpus would need a background job/queue instead.
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


class QueryRequest(BaseModel):
    query: str
    doc_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


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
    return answer_query(query=request.query, doc_id=request.doc_id, top_k=request.top_k)


# Serves static/index.html at "/" and static/app.js, static/style.css
# alongside it (html=True makes "/" resolve to index.html automatically).
# Registered last so it only catches requests the routes above don't.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
