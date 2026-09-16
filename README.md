# RAG App — PDF Q&A with Page/Section Citations

Given a PDF and a question, this app retrieves the most relevant chunks
of the document and generates an answer that cites the exact page and
section heading it came from — e.g. *"The maximum input current limit is
3A, as stated in [p.21, "Input Current Limit"]."*

**Live app**: https://roazbaj4ksojmjfl2e4mxmohx40zvsqk.lambda-url.us-east-1.on.aws/

## Features

- **Cited answers** — every answer references the page number(s) and
  section heading(s) it was drawn from, not just a generic response.
- **Reranking** — retrieval results are reordered by a dedicated
  relevance model (Pinecone's hosted `bge-reranker-v2-m3`) before being
  passed to the LLM, improving precision over similarity search alone.
- **Live hyperparameter controls** — the UI exposes retrieval/generation
  parameters (embedding model, answer model, `top_k`, temperature,
  reranking on/off) that take effect immediately, no redeploy required.
- **Collapsible citations** — long supporting snippets collapse behind
  an expandable preview so answers stay readable.
- **Serverless, zero-cost hosting** — deployed on AWS Lambda behind a
  public Function URL, within AWS's always-free tier.

## Architecture

```
Ingestion:
  PDF → Docling (parse + chunk, preserving page/heading structure)
      → LangChain Document objects
      → PineconeVectorStore.add_documents()
        (OpenAI embeddings generated automatically)

Query:
  question → PineconeVectorStore.similarity_search_with_score()
           → Pinecone hosted reranking (optional, on by default)
           → ChatPromptTemplate | ChatOpenAI  (LangChain LCEL chain)
           → cited answer + ranked sources
```

- **Docling** parses PDFs and chunks them with `HybridChunker`, which
  preserves page numbers and section-heading hierarchy as metadata on
  every chunk — the structure that makes citation possible.
- **LangChain** provides the framework layer: `Document` objects carry
  chunk text and metadata end-to-end, `PineconeVectorStore` wraps the
  vector database, and an LCEL (`prompt | llm`) chain composes the
  citation-constrained answer step.
- **OpenAI** provides both embeddings (`text-embedding-3-small` by
  default) and answer generation (`gpt-4o-mini` by default).
- **Pinecone** (serverless) stores vectors and metadata; one index per
  embedding model, since different models produce incompatible vector
  spaces.
- **FastAPI** exposes the REST API and serves the static UI.
- **AWS Lambda** (container image, behind a Function URL) runs the app
  serverlessly, via the [AWS Lambda Web
  Adapter](https://github.com/awslabs/aws-lambda-web-adapter) — the same
  container image runs unmodified locally and on Lambda.

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check. |
| `/config` | GET | Available embedding/chat models and their defaults. |
| `/query` | POST | Ask a question; returns a cited answer + ranked sources. |
| `/reindex` | POST | Re-embed the document with a different embedding model. |
| `/ingest` | POST | Upload and ingest a new PDF. |

**`POST /query`**
```json
Request:
{
  "query": "What is the maximum input current limit?",
  "doc_id": "d3f1...",                          // optional — scope to one document
  "top_k": 5,                                    // optional, default 5
  "embedding_model": "text-embedding-3-small",  // optional
  "chat_model": "gpt-4o-mini",                   // optional
  "temperature": 0.0,                            // optional, 0.0-1.0
  "use_reranking": true                          // optional, default true
}

Response:
{
  "answer": "The maximum input current limit is 3A, as stated in [p.21, \"Input Current Limit\"].",
  "sources": [
    { "page": 21, "heading": "Input Current Limit", "snippet": "...", "score": 0.92 }
  ],
  "doc_id": "d3f1..."
}
```

**`POST /reindex`**
```json
Request:  { "embedding_model": "text-embedding-3-large" }
Response: { "embedding_model": "text-embedding-3-large", "num_chunks": 93, "status": "reindexed" }
```

## Project Structure

```
RAG_app/
├── .github/workflows/
│   ├── ci-cd.yml            # test → build → deploy, on every push to main
│   └── ingest-sample.yml    # ingest/reindex a PDF into Pinecone
├── app/
│   ├── main.py         # FastAPI app: routes + static UI mount
│   ├── config.py       # Settings (environment variables)
│   ├── ingestion.py    # Docling: PDF parsing + chunking
│   ├── vectorstore.py  # Pinecone + LangChain, one index per embedding model
│   ├── rag.py          # Retrieval, reranking, and answer generation
│   └── retriever.py    # Orchestration façade used by the API layer
├── static/              # Query UI: HTML/CSS/vanilla JS
├── data/
│   ├── relevant_section_identification-sample.pdf
│   └── sample_chunks.json   # Cached extracted chunks, for fast reindexing
├── tests/
├── scripts/ingest_sample.py
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Local Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # fill in PINECONE_API_KEY and OPENAI_API_KEY

uvicorn app.main:app --reload
# open http://localhost:8000
```

Requires Python 3.10+.

Run tests: `pytest`

## Environment Variables

| Variable | Purpose |
|---|---|
| `PINECONE_API_KEY` | Pinecone authentication. |
| `PINECONE_INDEX_NAME` | Base index name (suffixed per embedding model). |
| `PINECONE_CLOUD` / `PINECONE_REGION` | Pinecone serverless region. |
| `OPENAI_API_KEY` | OpenAI authentication. |
| `OPENAI_EMBEDDING_MODEL` / `OPENAI_CHAT_MODEL` | Default models. |
| `APP_ENV` | `dev` or `prod`. |
| `PORT` | Server port (default 8000). |

Secrets are never committed — `.env` is gitignored locally, and in
production the deployed Lambda function receives them as encrypted
environment variables set by the CI/CD pipeline from GitHub Actions
secrets.

## Vector Database & Reindexing

Pinecone stores one serverless index per embedding model (1536 or 3072
dimensions depending on the model), created automatically on first use.
To ingest a new PDF or reindex the existing one with a different
embedding model:

- **From the UI** — changing the embedding model in the sidebar triggers
  `POST /reindex` automatically, re-embedding the cached chunk text with
  the new model. This is idempotent: it's a no-op if that model's index
  already has data.
- **From GitHub Actions** — the `Ingest Sample PDF` workflow (Actions tab
  → Run workflow) runs the full parse → chunk → embed → upsert pipeline
  for a new document.

## CI/CD

GitHub Actions (`.github/workflows/ci-cd.yml`), triggered on every push
to `main`:

1. **Test** — lint (`ruff`) and run the test suite.
2. **Build & deploy** — builds the Docker image, pushes it to Amazon ECR,
   and updates the AWS Lambda function, all authenticated via a
   GitHub OIDC identity — no long-lived AWS credentials are stored
   anywhere in the repo or CI configuration.

## Assumptions & Limitations

- OpenAI's embedding and chat APIs are pay-per-use; there is no
  meaningful free tier beyond initial trial credit.
- `POST /ingest` runs synchronously and is intended for local/development
  use — the deployed UI performs ingestion via the GitHub Actions
  workflow instead, since a large PDF parse can exceed a single Lambda
  invocation's practical request window.
- Lambda cold starts add latency (dominated by model initialization) on
  the first request after an idle period; subsequent requests are fast.
  This doesn't affect `/query`, which only depends on Pinecone and
  OpenAI.
- Retrieval currently uses dense (embedding) search only; true hybrid
  dense+sparse retrieval is a natural next iteration for queries that
  hinge on exact terms or part numbers.
