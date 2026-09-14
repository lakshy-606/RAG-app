"""POST /query — answer a question from previously-ingested PDF(s), with
page/heading citations (see SPECS.md §6 for the exact request/response
shape)."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.retriever import answer_query

router = APIRouter()


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


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    return answer_query(query=request.query, doc_id=request.doc_id, top_k=request.top_k)
