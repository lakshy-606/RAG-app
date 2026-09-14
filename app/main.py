"""FastAPI entrypoint: wires up the routers and serves the minimal static UI.

Run locally with:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import health, ingest, query

app = FastAPI(title="RAG App", description="PDF Q&A with page/heading citations")

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)

# Serves static/index.html at "/" and static/app.js, static/style.css
# alongside it — html=True makes "/" resolve to index.html automatically.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
