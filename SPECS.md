# RAG App — Build Specification

Status: planning document. No application code exists yet — this file is the step-by-step spec for building, containerizing, and deploying the app described below.

## 1. Overview

**Goal:** Given a PDF and a user query, identify the page(s) and, where possible, the section heading(s) that contain the information needed to answer the query — and return a synthesized answer citing those pages/headings.

**Fixed tech stack:**

| Concern | Choice | Why |
|---|---|---|
| PDF parsing + chunking | **Docling** (`DocumentConverter` + `HybridChunker`) | Preserves document structure — page numbers and heading hierarchy travel with each chunk as metadata, which is exactly what the citation requirement needs. |
| Framework | **LangChain** (`Document`, `PineconeVectorStore`, `ChatPromptTemplate` \| `ChatOpenAI`) | Standardizes the chunk→vector-store→retrieval→prompt plumbing on well-known abstractions instead of hand-rolled glue code around each provider's raw SDK. |
| Embeddings | **OpenAI** `text-embedding-3-small` (1536-dim), via LangChain | One provider for both embeddings and generation; `langchain-pinecone`'s `PineconeVectorStore` calls it automatically on add/search, so the app never builds raw vectors by hand. |
| Vector DB | **Pinecone** (Serverless, free/Starter tier) | Already the intended DB per the existing `.env`; simple SDK, generous free tier for a small demo corpus. |
| Answer generation | **OpenAI** `gpt-4o-mini`, via LangChain `ChatOpenAI` | Strong instruction-following for citation-constrained answers; same provider as embeddings keeps the model-facing stack to one API key. |
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
  → Docling DocumentConverter.convert()          (parse into structured DoclingDocument)
  → Docling HybridChunker.chunk()                 (token-aware chunks + page/heading metadata)
  → LangChain Document(page_content, metadata)    (one per chunk: doc_id, page_number(s), heading, chunk_index)
  → PineconeVectorStore.add_documents()           (OpenAI text-embedding-3-small embeds each chunk automatically)
```

**Query path** (via `POST /query`):
```
user query
  → PineconeVectorStore.similarity_search_with_score()   (OpenAI embeds the query automatically; top-k, optional doc_id filter)
  → build context block (chunk text tagged with [p.X, "Heading"])
  → (ChatPromptTemplate | ChatOpenAI).invoke()            (LangChain LCEL chain; answers only from context, must cite page/heading)
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
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── pinecone_client.py     # LangChain PineconeVectorStore + OpenAI embeddings, index create
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── prompts.py             # system prompt + context-block builder, citation instructions
│   │   └── pipeline.py            # retrieve -> build context -> ChatPromptTemplate|ChatOpenAI -> answer+sources
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
│   └── test_pipeline_mocked.py    # mocks the Pinecone vector store and ChatOpenAI calls
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
OPENAI_API_KEY
OPENAI_EMBEDDING_MODEL  # text-embedding-3-small
OPENAI_CHAT_MODEL       # gpt-4o-mini
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

**⚠️ Rotate any key that passed through an insecure channel.** The Pinecone key originally shipped in this project's `.env` in plaintext before this repo existed — it's now gitignored, but should still be rotated in the Pinecone console. The OpenAI key used during development was shared directly in a chat conversation rather than a secrets manager — treat it as compromised and rotate it at platform.openai.com once real deployment keys are issued.

## 5. Pinecone Index Schema

- **Name:** `rag-app-index` (configurable via `PINECONE_INDEX_NAME`)
- **Type:** Serverless, `vector_type="dense"`
- **Dimension:** `1536` (matches OpenAI's `text-embedding-3-small`)
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
- `requirements.txt` pinning: `docling`, `fastapi`, `uvicorn[standard]`, `python-dotenv`, `pinecone`, `langchain`, `langchain-openai`, `langchain-pinecone`, `python-multipart`, `pydantic-settings`, `pytest`, `httpx`.
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

### Phase 2 — Vector store (Pinecone, via LangChain + OpenAI embeddings)
- `app/vectorstore/pinecone_client.py`:
  ```python
  from langchain_openai import OpenAIEmbeddings
  from langchain_pinecone import PineconeVectorStore
  from pinecone import Pinecone, ServerlessSpec

  pc = Pinecone(api_key=PINECONE_API_KEY)
  if not pc.has_index(PINECONE_INDEX_NAME):
      pc.create_index(
          name=PINECONE_INDEX_NAME, vector_type="dense", dimension=1536, metric="cosine",
          spec=ServerlessSpec(cloud="aws", region="us-east-1"), deletion_protection="disabled",
      )
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
  vectorstore = PineconeVectorStore(index=pc.Index(PINECONE_INDEX_NAME), embedding=embeddings)

  vectorstore.add_documents(documents=[...], ids=[...])                       # embeds + upserts
  vectorstore.similarity_search_with_score(query, k=5, filter={"doc_id": doc_id})  # embeds + retrieves
  ```
  `PineconeVectorStore` calls OpenAI's embeddings API automatically on both add and search — the app never builds or passes raw vectors. `requirements.txt` pins `langchain-pinecone`, which itself requires `pinecone>=6,<8`; watch for a stale `pinecone-plugin-inference` package left over from an older `pinecone` install in any existing venv — pinecone 6+ refuses to import if that deprecated plugin is present, and `pip install` alone won't remove it.
  Metadata payload per vector is capped around 40KB — chunk text should stay well under this given 512-token chunks, but add a defensive truncation/log-warning.
  **Gotcha (confirmed live):** Pinecone stores all numeric metadata as float internally and returns it that way on retrieval, even for values written as plain Python ints (e.g. a `page_number` of `7` comes back as `7.0`). Cast back to `int` explicitly wherever a page number is read back out — both in the API response and in the LLM-facing context block — rather than leaking a `p.7.0` citation.
- `scripts/ingest_sample.py`: CLI chaining Phase 1 + Phase 2 (via `app/retriever.py`) to ingest the sample PDF end-to-end, for manual testing before the API layer exists.

### Phase 3 — Retrieval & answer generation (LangChain + OpenAI)
- `app/rag/prompts.py`: system prompt instructing the model to answer **only** from the provided context and to cite every claim as `[p.X, "Heading"]`; a `build_context_block()` helper turns retrieved `(Document, score)` pairs into that labeled context text.
- `app/rag/pipeline.py`:
  ```python
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_openai import ChatOpenAI

  prompt = ChatPromptTemplate.from_messages([
      ("system", SYSTEM_PROMPT),
      ("human", "Context excerpts:\n\n{context}\n\nQuestion: {question}"),
  ])
  llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0, max_retries=3)
  chain = prompt | llm   # LCEL composition

  # vectorstore.similarity_search_with_score(query, k=top_k, filter=...) -> results
  # build_context_block(results) -> context string tagged [p.X, "Heading"]
  # chain.invoke({"context": context, "question": query}) -> response.content
  ```
  Uses LangChain's current LCEL pipe pattern rather than the older `create_retrieval_chain`/`create_stuff_documents_chain` helpers — those moved out of core `langchain` into the separate `langchain_classic` package as of LangChain 1.0, and a direct `prompt | llm` pipe is just as readable for a single-step "stuff the context in and ask" chain. `ChatOpenAI`'s built-in `max_retries` handles transient rate-limit/network errors, so no separate fallback-model logic is needed (that was specific to Groq's unusually low free-tier limits).
  Handle the zero-retrieval-result case explicitly (return "not found in this document" rather than calling the LLM at all).
- `app/retriever.py` becomes the thin façade the routers call (`ingest_pdf(...)`, `answer_query(...)`); `ingest_pdf` converts each `ChunkRecord` into a LangChain `Document(page_content, metadata)` before calling `vectorstore.add_documents`.

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

- **Test job requirements:** lint (`ruff`) + `pytest` (health-endpoint smoke test, the Phase-1 chunking-metadata validation test, and a mocked-pipeline test for `/query` that stubs the Pinecone vector store and `ChatOpenAI`). This job must pass before `build-and-deploy` runs — that's the "basic build/test/validation step" requirement.
- **Secrets handling requirement:** application secrets (Pinecone/OpenAI keys) are set as App Runner environment variables/secrets in production, never baked into the image; CI secrets are GitHub Actions repo secrets, and AWS auth uses OIDC role assumption rather than long-lived keys — satisfies "secure handling of credentials and secrets."

### Phase 7 — README & repo deliverables
- `README.md` covering: architecture overview (§2), why each cloud/AI service was chosen (§1 rationale column), local setup instructions, the assumptions/limitations list (§8 below), and the live App Runner URL once deployed.
- `gh repo create` (new repo), push to `main`, confirm the Actions run is green and the App Runner URL responds.

## 8. Assumptions & Risks

1. **OpenAI is a paid API** — no meaningful free tier beyond initial trial credit, unlike the originally-considered Groq/HF free tiers. Accepted as a cost trade-off for reliability and quality (no rate-limit juggling, no "is this model warm" uncertainty).
2. **Docling provenance accuracy** — page-number metadata can be inconsistent for chunks spanning multiple pages/tables (documented upstream issue); validate against the sample PDF rather than assuming correctness.
3. **Pinecone free-tier region must be pinned** to a supported combination (`aws`/`us-east-1`) rather than left to default.
4. **Pinecone returns numeric metadata as float**, not the original int — confirmed live during implementation (a `page_number` written as `7` comes back as `7.0`). The pipeline casts back to `int` explicitly wherever a page number is read; don't assume metadata round-trips its original type.
5. **AWS App Runner is not free** — it bills for provisioned vCPU/memory continuously (no scale-to-zero), roughly $5–25+/month depending on sizing. Accepted as a cost trade-off for deployment simplicity; worth disclosing explicitly.
6. **Docling's model-weight download** on first conversion affects cold start / image build time — mitigated by pre-baking weights into the Docker image.
7. **Synchronous `/ingest`** won't scale past demo-sized PDFs (large documents risk request timeouts) — explicitly a v1 limitation, not solved here.
8. **Rotate any key that passed through an insecure channel.** The original Pinecone key shipped in this project's `.env` in plaintext before this repo existed. The OpenAI key used during development was shared directly in a chat conversation rather than a secrets manager — both should be rotated once real deployment keys are issued, independent of anything else in this spec.
9. **LangChain 1.x moved `create_retrieval_chain`/`create_stuff_documents_chain` out of core** — into a separate `langchain_classic` package. The pipeline uses direct LCEL composition (`prompt | llm`) instead, which is both the current recommended pattern and simpler to follow for this project's single-step chain.
