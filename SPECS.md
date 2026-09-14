# RAG App — Build Specification

Status: planning document. No application code exists yet — this file is the step-by-step spec for building, containerizing, and deploying the app described below.

## 1. Overview

**Goal:** Given a PDF and a user query, identify the page(s) and, where possible, the section heading(s) that contain the information needed to answer the query — and return a synthesized answer citing those pages/headings.

**Fixed tech stack:**

| Concern | Choice | Why |
|---|---|---|
| PDF parsing + chunking | **Docling** (`DocumentConverter` + `HybridChunker`) | Preserves document structure — page numbers and heading hierarchy travel with each chunk as metadata, which is exactly what the citation requirement needs. |
| Embeddings | **HuggingFace Inference API** (free tier), `feature_extraction` task | Groq has no embeddings endpoint. HF's serverless free tier gives a hosted embedding model with no local GPU/CPU cost. |
| Vector DB | **Pinecone** (Serverless, free/Starter tier) | Already the intended DB per the existing `.env`; simple SDK, generous free tier for a small demo corpus. |
| Answer generation | **Groq** (free tier), `llama-3.3-70b-versatile` | Fast, free, OpenAI-compatible-ish SDK; good at instruction-following for citation-constrained answers. |
| Backend | **FastAPI** | REST API + easy static-file serving for a minimal UI, automatic OpenAPI docs. |
| UI | Minimal static HTML/JS page served by FastAPI | Enough to demo the app via a browser URL without a separate frontend deploy. |
| Container | **Docker** | Required for App Runner deployment. |
| Cloud | **AWS App Runner** | Simplest "push a container, get a URL" path on AWS — no ALB/ECS task-definition management, supports auto-deploy on new image. |
| CI/CD | **GitHub Actions** | Test → build → push to ECR → deploy to App Runner, triggered on push/merge to `main`. |

**Required deliverables (per the assignment):**
1. Source-code repository (new GitHub repo — see Phase 0 / Phase 7).
2. A deployed application URL (AWS App Runner — see Phase 6).
3. The CI/CD pipeline configuration (`.github/workflows/ci-cd.yml` — see Phase 6).
4. A brief README covering approach, cloud-service selection, and assumptions (see Phase 7).

## 2. Architecture

**Ingest path** (one-time or per-document, via `POST /ingest`):
```
PDF file
  → Docling DocumentConverter.convert()        (parse into structured DoclingDocument)
  → Docling HybridChunker.chunk()               (token-aware chunks + page/heading metadata)
  → HF Inference API feature_extraction()        (embed each chunk's contextualized text)
  → Pinecone upsert()                             (vector + metadata: doc_id, page_number(s), heading, chunk_text)
```

**Query path** (via `POST /query`):
```
user query
  → HF Inference API feature_extraction()        (embed the query)
  → Pinecone query()                              (top-k nearest chunks, optional doc_id filter)
  → build context block (chunk_text tagged with [p.X, "Heading"])
  → Groq chat.completions.create()                (LLM answers only from context, must cite page/heading)
  → parse into {answer, sources:[{page, heading, snippet, score}]}
```

## 3. Final Project Structure

```
RAG_app/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app: mounts routers + StaticFiles
│   ├── config.py                  # pydantic Settings, loads .env via python-dotenv
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── ingest.py              # POST /ingest
│   │   ├── query.py                # POST /query
│   │   └── health.py              # GET /health
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py          # DocumentConverter wrapper
│   │   └── chunker.py             # HybridChunker wrapper + page/heading extraction
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── hf_embedder.py         # InferenceClient(provider="hf-inference") wrapper, retry/backoff
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── pinecone_client.py     # index create/upsert/query
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── prompts.py             # system/user prompt templates with citation instructions
│   │   └── pipeline.py            # retrieve -> build context -> Groq call -> parse answer+sources
│   └── retriever.py                # orchestration façade used by routers (renamed from retreiver.py)
├── static/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── data/
│   └── relevant_section_identification-sample.pdf
├── tests/
│   ├── test_health.py
│   ├── test_chunking.py           # validates page/heading extraction against the sample PDF
│   └── test_pipeline_mocked.py    # mocks HF/Pinecone/Groq calls
├── scripts/
│   └── ingest_sample.py           # CLI: one-off ingest of the sample PDF for local/demo use
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── .env.example
├── .env                            # gitignored, never committed
└── README.md
```

> The existing `app/retreiver.py` is an empty stub with a filename typo. Phase 0 renames it to `app/retriever.py` and Phase 3 gives it real content (a thin façade over `app/rag/pipeline.py`).

## 4. Secrets / Environment Variables

**Application runtime** (`.env` locally; AWS App Runner environment variables / secrets in production):
```
PINECONE_API_KEY
PINECONE_INDEX_NAME
PINECONE_CLOUD          # aws
PINECONE_REGION         # us-east-1
HF_API_TOKEN
HF_EMBEDDING_MODEL      # sentence-transformers/all-MiniLM-L6-v2 (fallback: BAAI/bge-small-en-v1.5)
GROQ_API_KEY
GROQ_MODEL              # llama-3.3-70b-versatile
GROQ_FALLBACK_MODEL     # llama-3.1-8b-instant
APP_ENV                 # dev | prod
PORT                    # 8000
```

**CI/CD only** (GitHub Actions repo secrets/variables — never in `.env`, never committed):
```
AWS_ROLE_ARN                       # OIDC-assumed IAM role for CI — no long-lived keys
AWS_REGION
ECR_REPOSITORY
APP_RUNNER_SERVICE_NAME
APP_RUNNER_ECR_ACCESS_ROLE_ARN     # role App Runner itself uses to pull from ECR
# fallback only if OIDC bootstrap is skipped:
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

**⚠️ Immediate action required, independent of this spec:** the existing `.env` in this project already contains a real-looking Pinecone API key (`pinecone_api=...`) sitting in a folder that is not yet a git repo and has no `.gitignore`. Before doing anything else: (a) add `.env` to `.gitignore` in Phase 0 so it's never committed, and (b) rotate that Pinecone key in the Pinecone console, since it has been sitting exposed in plain text. Rename the variable itself to `PINECONE_API_KEY` for consistency.

## 5. Pinecone Index Schema

- **Name:** `rag-app-index` (configurable via `PINECONE_INDEX_NAME`)
- **Type:** Serverless, `vector_type="dense"`
- **Dimension:** `384` (matches both `all-MiniLM-L6-v2` and `bge-small-en-v1.5`)
- **Metric:** `cosine`
- **Cloud / Region:** `aws` / `us-east-1` (pinned — not every region is available on the free tier)
- **Vector ID scheme:** `{doc_id}-chunk-{chunk_index}`
- **Metadata per vector:**
  - `doc_id` (string) — stable id for the ingested PDF
  - `source_filename` (string)
  - `page_number` (number) — primary/min page for the chunk
  - `page_numbers` (list[number]) — full page span if the chunk crosses pages
  - `section_heading` (string, nullable)
  - `chunk_text` (string) — used to render citation snippets
  - `chunk_index` (number) — ordinal position in the document

## 6. API Contracts

**`POST /ingest`** — multipart file upload
```json
Request: multipart/form-data { file: <pdf bytes> }

Response 200:
{
  "doc_id": "d3f1...",
  "source_filename": "relevant_section_identification-sample.pdf",
  "num_pages": 40,
  "num_chunks": 87,
  "status": "ingested"
}
```

**`POST /query`**
```json
Request:
{
  "query": "What is the maximum allowed X?",
  "doc_id": "d3f1...",     // optional — restrict to one ingested doc; omit to search all
  "top_k": 5                // optional, default 5
}

Response 200:
{
  "answer": "The maximum allowed X is ... [p.12, \"Section 3.2 Limits\"]",
  "sources": [
    { "page": 12, "heading": "Section 3.2 Limits", "snippet": "...", "score": 0.87 },
    { "page": 13, "heading": "Section 3.2 Limits", "snippet": "...", "score": 0.81 }
  ],
  "doc_id": "d3f1..."
}
```

**`GET /health`** → `{ "status": "ok" }`

**`GET /`** → serves `static/index.html`: a query box + rendered answer with clickable citations, calling `POST /query` via `fetch`.

## 7. Phased Build Plan

### Phase 0 — Repo & environment bootstrap
- `git init`; `.gitignore` covering `.env`, `__pycache__/`, `.venv/`, `*.pyc`, `.DS_Store`, and any Docling model-cache directory.
- Rename `.env`'s `pinecone_api` → `PINECONE_API_KEY`; **rotate the key** (see §4 warning); create `.env.example` listing every variable name from §4 with no values.
- `requirements.txt` pinning: `docling`, `fastapi`, `uvicorn[standard]`, `python-dotenv`, `pinecone`, `huggingface_hub`, `groq`, `python-multipart`, `pydantic-settings`, `tenacity`, `pytest`, `httpx`.
- Scaffold the `app/` package; `app/config.py` defines a pydantic `Settings` class reading all env vars from §4.
- Rename `app/retreiver.py` → `app/retriever.py`.
- **Gotcha:** Docling pulls in `torch` and layout-model weights on first use — do a local sanity install/run before writing the Dockerfile, since it affects image size and build time later.

### Phase 1 — PDF ingestion & chunking (Docling)
- `app/ingestion/pdf_parser.py`:
  ```python
  from docling.document_converter import DocumentConverter
  converter = DocumentConverter()
  doc = converter.convert(path).document
  ```
- `app/ingestion/chunker.py`:
  ```python
  from docling.chunking import HybridChunker
  chunker = HybridChunker(max_tokens=512)
  for chunk in chunker.chunk(dl_doc=doc):
      text = chunker.contextualize(chunk)          # heading-enriched text — embed this, not raw chunk.text
      page_numbers = sorted({
          prov.page_no
          for item in chunk.meta.doc_items
          for prov in item.prov
      })
      heading = chunk.meta.headings[-1] if chunk.meta.headings else None
  ```
  Normalize each chunk into a plain `ChunkRecord` dataclass (`text, page_numbers, heading, chunk_index`) decoupled from Docling internals, so the rest of the pipeline doesn't depend on Docling's object model.
- `tests/test_chunking.py`: run against `data/relevant_section_identification-sample.pdf`; assert chunks are non-empty, every chunk has ≥1 page number, and print a handful for manual spot-check.
- **Gotcha (real, documented):** Docling provenance page numbers can be inconsistent for chunks whose source items span multiple pages or come from tables/captions (see docling-project/docling discussion #1012). Don't assume correctness — validate against the sample PDF before building on top. Store `page_numbers` as a list; use `min(page_numbers)` as the "primary" citation page but keep the full list in metadata.
- **Gotcha:** pick `max_tokens` to fit under the embedding model's max sequence length (MiniLM/bge-small are ~256–512 tokens) to avoid silent truncation at embedding time.

### Phase 2 — Embeddings & vector store
- `app/embeddings/hf_embedder.py`:
  ```python
  from huggingface_hub import InferenceClient
  client = InferenceClient(provider="hf-inference", api_key=HF_API_TOKEN, timeout=30)
  vec = client.feature_extraction(text, model=HF_EMBEDDING_MODEL)
  ```
  Wrap calls with `tenacity` retry/backoff. The `provider` must be explicit — `feature_extraction` is currently supported only by `hf-inference` and `Scaleway`; the default `"auto"` provider selection cannot be trusted for this task. Free-tier serverless calls can silently queue rather than return an explicit 429, so set an explicit timeout and treat timeouts as a distinct failure mode.
- Before locking in `HF_EMBEDDING_MODEL`, live smoke-test both `sentence-transformers/all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5` — free-tier serverless doesn't guarantee every model is hosted/"warm". Pick whichever responds; both are 384-dim so the Pinecone index dimension doesn't change either way. If neither is available at implementation time, fall back to running the same model locally via the `sentence-transformers` package inside the container (documented contingency, not the default plan).
- `app/vectorstore/pinecone_client.py`:
  ```python
  from pinecone import Pinecone, ServerlessSpec
  pc = Pinecone(api_key=PINECONE_API_KEY)
  if not pc.has_index(PINECONE_INDEX_NAME):
      pc.create_index(
          name=PINECONE_INDEX_NAME, vector_type="dense", dimension=384, metric="cosine",
          spec=ServerlessSpec(cloud="aws", region="us-east-1"), deletion_protection="disabled",
      )
  index = pc.Index(PINECONE_INDEX_NAME)
  index.upsert(vectors=[...])          # batch 100–500 at a time
  index.query(vector=[...], top_k=5, include_metadata=True, filter={"doc_id": {"$eq": doc_id}})
  ```
  Metadata payload per vector is capped around 40KB — `chunk_text` should stay well under this given 512-token chunks, but add a defensive truncation/log-warning.
- `scripts/ingest_sample.py`: CLI chaining Phase 1 + Phase 2 to ingest the sample PDF end-to-end, for manual testing before the API layer exists.

### Phase 3 — Retrieval & answer generation
- `app/rag/prompts.py`: system prompt instructing the model to answer **only** from the provided context and to cite every claim as `[p.X, "Heading"]`.
- `app/rag/pipeline.py`:
  ```python
  from groq import Groq
  client = Groq(api_key=GROQ_API_KEY)
  # embed query -> pinecone query (top_k, optional doc_id filter) -> assemble context
  # with each chunk tagged [p.X, "Heading"] -> call Groq, fall back to GROQ_FALLBACK_MODEL
  # on rate-limit errors -> parse into {answer, sources}
  resp = client.chat.completions.create(model=GROQ_MODEL, messages=[...])
  ```
  Handle the zero-retrieval-result case explicitly (return "not found in this document" rather than letting the LLM hallucinate an answer).
- `app/retriever.py` becomes the thin façade the routers call (`ingest_pdf(...)`, `answer_query(...)`).

### Phase 4 — FastAPI app & minimal UI
- `app/main.py`: FastAPI instance, `StaticFiles` mount, router includes, CORS only if the UI is ever split out.
- `app/routers/{ingest,query,health}.py` implementing the §6 contracts.
- `static/index.html` + `app.js`: one query box, `fetch('/query', {...})`, render `answer` with `sources` shown as clickable/highlighted citations.
- **Gotcha:** synchronous `/ingest` blocks on Docling conversion + per-chunk embedding calls for the whole PDF — fine for a ~40-page demo doc, but call this out in the README as a known v1 scaling limitation (future: background task/queue).

### Phase 5 — Containerization
- `Dockerfile`: slim Python base image, install Docling's system dependencies, `pip install -r requirements.txt`, copy the app, `EXPOSE 8000`, `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`.
- `.dockerignore`: exclude `.env`, `.git`, test caches, local venv.
- **Gotcha:** Docling downloads layout/OCR model weights on first conversion call unless pre-cached. Either run a throwaway conversion during `docker build` to bake the weights into the image, or accept a slower first request in production — document the trade-off made.

### Phase 6 — CI/CD & cloud deploy (AWS App Runner)
**One-time manual AWS bootstrap** (documented here, not scripted by the workflow):
1. Create an ECR repository for the app image.
2. Register GitHub's OIDC provider in AWS IAM (`https://token.actions.githubusercontent.com`, audience `sts.amazonaws.com`).
3. Create an IAM role trusted for `repo:<org>/<repo>:ref:refs/heads/main`, with a least-privilege policy: push to the specific ECR repo, `apprunner:UpdateService`/`DescribeService`/`StartDeployment`, and `iam:PassRole` for the App Runner access role. Store only the **role ARN** as a GitHub secret/variable — never a static key pair — unless OIDC setup is explicitly out of scope, in which case fall back to `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` secrets.
4. Create the App Runner service (or let the first CI deploy create it — see workflow below) pointing at that ECR repo.

**`.github/workflows/ci-cd.yml`**, triggered on `push`/`merge` to `main` (plus `pull_request` for the test job only):

```yaml
name: CI/CD
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: pytest

  build-and-deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
      - uses: aws-actions/amazon-ecr-login@v2
        id: login-ecr
      - run: |
          docker build \
            -t ${{ steps.login-ecr.outputs.registry }}/${{ vars.ECR_REPOSITORY }}:${{ github.sha }} \
            -t ${{ steps.login-ecr.outputs.registry }}/${{ vars.ECR_REPOSITORY }}:latest .
          docker push ${{ steps.login-ecr.outputs.registry }}/${{ vars.ECR_REPOSITORY }} --all-tags
      - uses: awslabs/amazon-app-runner-deploy@main
        with:
          service: ${{ vars.APP_RUNNER_SERVICE_NAME }}
          image: ${{ steps.login-ecr.outputs.registry }}/${{ vars.ECR_REPOSITORY }}:${{ github.sha }}
          access-role-arn: ${{ secrets.APP_RUNNER_ECR_ACCESS_ROLE_ARN }}
          region: ${{ vars.AWS_REGION }}
          cpu: 1
          memory: 2
          wait-for-service-stability-seconds: 1200
```

This action is idempotent — it creates the App Runner service on the first run and updates the image on every subsequent run — and outputs the live service URL.

- **Test job requirements:** lint (`ruff`/`flake8`) + `pytest` (health-endpoint smoke test, the Phase-1 chunking-metadata validation test, and a mocked-pipeline test for `/query` that stubs HF/Pinecone/Groq). This job must pass before `build-and-deploy` runs — that's the "basic build/test/validation step" requirement.
- **Secrets handling requirement:** application secrets (Pinecone/HF/Groq keys) are set as App Runner environment variables/secrets in production, never baked into the image; CI secrets are GitHub Actions repo secrets, and AWS auth uses OIDC role assumption rather than long-lived keys — satisfies "secure handling of credentials and secrets."

### Phase 7 — README & repo deliverables
- `README.md` covering: architecture overview (§2), why each cloud/AI service was chosen (§1 rationale column), local setup instructions, the assumptions/limitations list (§8 below), and the live App Runner URL once deployed.
- `gh repo create` (new repo), push to `main`, confirm the Actions run is green and the App Runner URL responds.

## 8. Assumptions & Risks

1. **HF free-tier model availability isn't guaranteed** — not every embedding model is hosted/warm on serverless inference; smoke-test the chosen model live and keep a same-dimension fallback (and a local-inference contingency).
2. **HF `feature_extraction` requires an explicit provider** (`"hf-inference"`) — the client's default `"auto"` provider selection can't be trusted for this task.
3. **Docling provenance accuracy** — page-number metadata can be inconsistent for chunks spanning multiple pages/tables (documented upstream issue); validate against the sample PDF rather than assuming correctness.
4. **Groq free-tier rate limits are low** (e.g. ~30 RPM / 1,000 RPD on the 70B model) — easy to hit during a demo/grading session; mitigated with retry + a smaller fallback model.
5. **Pinecone free-tier region must be pinned** to a supported combination (`aws`/`us-east-1`) rather than left to default.
6. **AWS App Runner is not free** — it bills for provisioned vCPU/memory continuously (no scale-to-zero), roughly $5–25+/month depending on sizing. Accepted as a cost trade-off for deployment simplicity; worth disclosing explicitly.
7. **Docling's model-weight download** on first conversion affects cold start / image build time — mitigated by pre-baking weights into the Docker image.
8. **Synchronous `/ingest`** won't scale past demo-sized PDFs (large documents risk request timeouts) — explicitly a v1 limitation, not solved here.
9. **Exposed secret in the current project** — the pre-existing `.env` contains a real Pinecone key that was sitting in plaintext in an unversioned, unprotected folder. It must be rotated regardless of anything else in this spec.
