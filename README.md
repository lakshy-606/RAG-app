# RAG App — PDF Q&A with Page/Section Citations

Given a PDF and a question, this app finds the pages and section headings
relevant to the question and returns an answer that cites them —
e.g. "The maximum allowed operating temperature is 85°C [p.7, "Section 4.1
Operating Limits"]."

Full step-by-step build spec, including the research behind each choice
below: see [`SPECS.md`](./SPECS.md).

## Approach

```
Ingest:  PDF → Docling (parse + chunk, keeping page/heading metadata)
             → LangChain Document objects
             → PineconeVectorStore.add_documents()
               (OpenAI text-embedding-3-small embeds each chunk automatically)

Query:   question → PineconeVectorStore.similarity_search_with_score()
               (OpenAI embeds the question automatically)
             → ChatPromptTemplate | ChatOpenAI  (LangChain LCEL chain)
               (answers from retrieved context, citing page/heading)
```

- **Docling** is used specifically for its `HybridChunker`, which keeps
  page numbers and section-heading hierarchy attached to every chunk —
  that structure is what makes citation possible at all.
- **LangChain** provides the framework layer: `Document` objects carry
  chunk text + page/heading metadata end-to-end, `PineconeVectorStore`
  wraps the vector DB, and a `ChatPromptTemplate | ChatOpenAI` pipe
  (LCEL) composes the citation-constrained answer step.
- **OpenAI** handles both embeddings (`text-embedding-3-small`, 1536-dim)
  and answer generation (`gpt-4o-mini`) — one provider for the whole
  model-facing side of the stack.
- **Pinecone** (serverless, free tier) stores the vectors plus metadata
  (page number(s), heading, source text) needed to build citations.
- **FastAPI** exposes `POST /ingest`, `POST /query`, `GET /health`, and
  serves a minimal static HTML/JS page at `/` for a browser-based demo.

## Cloud service selection

**AWS Lambda** (container image, behind a public **Function URL**) was
chosen for deployment — specifically to stay on AWS's genuinely-free
tier: Lambda gives 1M requests + 400,000 GB-seconds of compute every
month at $0, permanently, with no load balancer involved. (Two other
paths were tried and rejected first: **AWS App Runner**, the original
choice matching the assignment's own example, stopped accepting new
customers on 2026-04-30; its AWS-recommended replacement, **Amazon ECS
Express Mode**, works but requires an Application Load Balancer that
bills ~$16–20/month continuously regardless of traffic — not actually
free. Lambda avoids that entirely.) The trade-off is a cold-start delay
(~10–30s) after idle periods, since Docling/torch have to load into a
fresh execution environment — acceptable for a low-traffic demo at
genuinely $0/month. The container image runs completely unchanged
between local Docker, and Lambda: see the [AWS Lambda Web
Adapter](https://github.com/awslabs/aws-lambda-web-adapter) line in the
Dockerfile.

**CI/CD**: GitHub Actions, triggered on push/merge to `main`
(`.github/workflows/ci-cd.yml`):
1. `test` job — lint (`ruff`) + `pytest` (mocked, no real API keys needed).
2. `build-and-deploy` job (only on `main`, only if tests pass) — builds the
   Docker image, pushes to Amazon ECR, and creates/updates the Lambda
   function via the AWS CLI (already on GitHub's runners).

**Secrets handling**: application secrets (Pinecone/OpenAI keys) are read
from environment variables — `.env` locally (gitignored, never
committed), container environment variables set by the deploy step in
production. AWS credentials for CI are never stored as long-lived keys:
GitHub Actions assumes an IAM role via OIDC (`AWS_ROLE_ARN`), scoped to
this repo's `main` branch.

## Project structure

```
RAG_app/
├── .github/workflows/ci-cd.yml
├── app/
│   ├── main.py           # FastAPI app: routes (health/ingest/query) + static UI mount
│   ├── config.py         # Settings (env vars)
│   ├── ingestion.py      # Docling: PDF parsing + chunking
│   ├── vectorstore.py    # LangChain PineconeVectorStore + OpenAI embeddings
│   ├── rag.py            # LangChain ChatPromptTemplate | ChatOpenAI -> cited answer
│   └── retriever.py      # orchestration façade between main.py and the modules above
├── static/{index.html,app.js,style.css}
├── data/relevant_section_identification-sample.pdf
├── tests/{test_health,test_chunking,test_pipeline_mocked}.py
├── scripts/ingest_sample.py
├── Dockerfile, .dockerignore, .gitignore
├── requirements.txt, .env.example, .env (gitignored)
└── README.md
```

## Local setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in real PINECONE_API_KEY / OPENAI_API_KEY

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
- **Repository**: https://github.com/lakshy-606/RAG-app
- **CI/CD config**: [`.github/workflows/ci-cd.yml`](./.github/workflows/ci-cd.yml)

One-time AWS bootstrap required before the pipeline can deploy (not
automated by the workflow itself — see SPECS.md §7 Phase 6):
1. Create an ECR repository (`rag-app`).
2. Register GitHub's OIDC provider in AWS IAM. **Note:** for repos
   created after 2026-07-15, GitHub's OIDC `sub` claim uses an immutable
   `repo:<org>@<org_id>/<repo>@<repo_id>:ref:refs/heads/<branch>` format
   (org/repo *names* plus their numeric IDs) rather than the older
   plain-name format most existing tutorials assume — get the real IDs
   from a failed `AssumeRoleWithWebIdentity` CloudTrail event if the
   trust policy condition doesn't match on the first attempt.
3. Create a Lambda execution role, `rag-app-lambda-execution-role`
   (trusted by `lambda.amazonaws.com`, policy
   `AWSLambdaBasicExecutionRole`).
4. Create the GitHub Actions deploy role, trusted via OIDC for
   `repo:<org>/<repo>:ref:refs/heads/main`, with permissions to push to
   the ECR repo, `lambda:CreateFunction`/`UpdateFunctionCode`/
   `UpdateFunctionConfiguration`/`GetFunction`/`CreateFunctionUrlConfig`/
   `GetFunctionUrlConfig`/`AddPermission`, and `iam:PassRole` for the
   Lambda execution role above (nothing else — no static AWS keys
   anywhere).
5. Set the following in the GitHub repo's Settings → Secrets and
   variables → Actions:
   - Secrets: `AWS_ROLE_ARN`, `PINECONE_API_KEY`, `OPENAI_API_KEY`
   - Variables: `AWS_REGION`, `AWS_ACCOUNT_ID`, `ECR_REPOSITORY`, `LAMBDA_FUNCTION_NAME`

   The deploy step passes `PINECONE_API_KEY`, `OPENAI_API_KEY`, and
   `APP_ENV=prod` straight to the Lambda function's environment
   variables — that's how the *running app* gets its secrets, separate
   from the AWS credentials CI itself uses to deploy it.

## Assumptions & known limitations

- OpenAI is a paid API (no free tier beyond initial trial credit) —
  accepted as a cost trade-off for embedding/generation quality and
  reliability versus a free-tier provider.
- Docling's page-number metadata has documented edge cases for chunks
  spanning multiple pages — `tests/test_chunking.py` validates this
  against the real sample PDF rather than assuming correctness.
- Pinecone stores all numeric metadata as floats internally and returns
  it that way on retrieval (e.g. a page number written as `7` comes back
  as `7.0`) — the pipeline casts back to `int` explicitly rather than
  leaking that through the API response or into LLM-visible citations.
- `/ingest` is synchronous — fine for a demo-sized PDF, not built to scale
  to very large documents without a background job/queue.
- Lambda cold starts (~10–30s) are the accepted trade-off for staying at
  genuinely $0/month — Docling/torch have to load into a fresh execution
  environment after idle periods. A low-traffic demo can live with this;
  a production service handling steady traffic would outgrow it.
- GitHub Actions' OIDC subject-claim format changed for repos created
  after 2026-07-15 (see bootstrap step 2 above) — a real gotcha hit
  mid-implementation, not a hypothetical one.
- **Rotate every key used during development.** The Pinecone key
  originally shipped in this project's `.env` in plaintext before this
  repo existed. The OpenAI key and a GitHub PAT were both shared directly
  in a chat conversation while building this integration — treat any
  secret that passes through a chat channel as compromised and rotate it
  once real deployment keys are issued.
