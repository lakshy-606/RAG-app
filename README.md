# RAG App — PDF Q&A with Page/Section Citations

Given a PDF and a question, this app finds the pages and section headings
relevant to the question and returns an answer that cites them —
e.g. "The maximum allowed X is 42 [p.12, "Section 3.2 Limits"]."

Full step-by-step build spec, including the research behind each choice
below: see [`SPECS.md`](./SPECS.md).

## Approach

```
Ingest:  PDF → Docling (parse + chunk, keeping page/heading metadata)
             → HuggingFace Inference API (embed each chunk)
             → Pinecone (store vector + page/heading/text metadata)

Query:   question → HuggingFace Inference API (embed the question)
             → Pinecone (retrieve top-k relevant chunks)
             → Groq LLM (answer from retrieved context, citing page/heading)
```

- **Docling** is used specifically for its `HybridChunker`, which keeps
  page numbers and section-heading hierarchy attached to every chunk —
  that structure is what makes citation possible at all.
- **Groq** has no embeddings endpoint, so embeddings are generated
  separately via the **HuggingFace Inference API** free tier
  (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim).
- **Pinecone** (serverless, free tier) stores the vectors plus metadata
  (page number(s), heading, source text) needed to build citations.
- **Groq** (`llama-3.3-70b-versatile`, free tier) generates the final
  answer, instructed to answer only from retrieved context and cite
  every claim.
- **FastAPI** exposes `POST /ingest`, `POST /query`, `GET /health`, and
  serves a minimal static HTML/JS page at `/` for a browser-based demo.

## Cloud service selection

**AWS App Runner** was chosen for deployment: it takes a container image
and gives back a public HTTPS URL with no ALB/ECS task-definition/VPC
setup, and supports redeploying on every new image push — the simplest
path to "push code, get a live URL" on AWS. See SPECS.md §8 risk 6 for the
cost trade-off this implies (App Runner bills continuously, unlike a
scale-to-zero option).

**CI/CD**: GitHub Actions, triggered on push/merge to `main`
(`.github/workflows/ci-cd.yml`):
1. `test` job — lint (`ruff`) + `pytest` (mocked, no real API keys needed).
2. `build-and-deploy` job (only on `main`, only if tests pass) — builds the
   Docker image, pushes to Amazon ECR, deploys the new image to the
   existing App Runner service.

**Secrets handling**: application secrets (Pinecone/HF/Groq keys) are
read from environment variables — `.env` locally (gitignored, never
committed), App Runner's environment variable configuration in
production. AWS credentials for CI are never stored as long-lived keys:
GitHub Actions assumes an IAM role via OIDC (`AWS_ROLE_ARN`), scoped to
this repo's `main` branch.

## Project structure

See `SPECS.md` §3 for the full annotated tree.

## Local setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in real PINECONE_API_KEY / HF_API_TOKEN / GROQ_API_KEY

uvicorn app.main:app --reload
# open http://localhost:8000
```

Requires **Python 3.10+** (the codebase uses `X | None` union type syntax).

> **macOS note:** `docling`'s native PDF-parsing dependency (`docling-parse`)
> only ships prebuilt wheels for macOS 14+ on Apple Silicon. On an older
> macOS (13/Ventura or earlier), `pip install` will try to compile it from
> C++ source and can fail depending on your Xcode Command Line Tools
> version. This doesn't affect Docker or CI (both run on Linux, where
> prebuilt wheels exist) — it's only a local-dev-environment quirk on
> older Apple Silicon Macs.

Run tests: `pytest`
One-off end-to-end ingest of the sample PDF (bypassing the API):
`python scripts/ingest_sample.py`

## Deployment

- **Deployed URL**: _fill in after the first successful `build-and-deploy` run_
- **Repository**: _fill in with the GitHub repo URL once pushed_
- **CI/CD config**: [`.github/workflows/ci-cd.yml`](./.github/workflows/ci-cd.yml)

One-time AWS bootstrap required before the pipeline can deploy (not
automated by the workflow itself — see SPECS.md §7 Phase 6):
1. Create an ECR repository.
2. Register GitHub's OIDC provider in AWS IAM and create a role trusted
   for `repo:<org>/<repo>:ref:refs/heads/main`, with permissions to push
   to that ECR repo and manage the App Runner service.
3. Set the following in the GitHub repo's Settings → Secrets and
   variables → Actions:
   - Secrets: `AWS_ROLE_ARN`, `APP_RUNNER_ECR_ACCESS_ROLE_ARN`
   - Variables: `AWS_REGION`, `ECR_REPOSITORY`, `APP_RUNNER_SERVICE_NAME`
4. Set the app's own secrets (`PINECONE_API_KEY`, `HF_API_TOKEN`,
   `GROQ_API_KEY`, etc.) as App Runner environment variables.

## Assumptions & known limitations

- HuggingFace's free-tier serverless inference doesn't guarantee every
  model is hosted/warm; the embedding model is chosen with a same-dimension
  fallback in mind (see SPECS.md §8 risk 1).
- Groq's free tier has fairly low rate limits; the pipeline retries once
  on a smaller fallback model rather than failing outright.
- Docling's page-number metadata has documented edge cases for chunks
  spanning multiple pages — `tests/test_chunking.py` validates this
  against the real sample PDF rather than assuming correctness.
- `/ingest` is synchronous — fine for a demo-sized PDF, not built to scale
  to very large documents without a background job/queue.
- AWS App Runner bills continuously for provisioned capacity; it was
  chosen for deployment simplicity, not lowest cost.
- **The `.env` originally shipped in this project contained a real
  Pinecone API key in plaintext.** It has since been kept out of git via
  `.gitignore`, but that key should be rotated in the Pinecone console
  regardless, since it was exposed before this repo existed.
