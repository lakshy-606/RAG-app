"""POST /ingest — upload a PDF, chunk it, embed it, store it in Pinecone.

Note (SPECS.md §7 Phase 4 gotcha): this runs synchronously, blocking on
Docling conversion + one HF embedding call per chunk. Fine for a demo-sized
PDF (~40 pages); a larger corpus would need a background job/queue instead.
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.retriever import ingest_pdf

router = APIRouter()


@router.post("/ingest")
async def ingest(file: UploadFile):
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
