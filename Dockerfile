# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Without this, Python fully buffers stdout/stderr when it isn't attached
# to a terminal (true of every container) — meaning nothing the app or
# uvicorn prints, including a startup traceback, would reach CloudWatch
# logs until the buffer happened to flush.
ENV PYTHONUNBUFFERED=1

# Docling's PDF/layout pipeline needs these system libs (fonts + rendering
# libs for its backend PDF/image processing).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY static/ static/

# The already-extracted sample chunks (text/page/heading, no PDF bytes) —
# lets app/retriever.reindex_with_model() re-embed with a different model
# without needing Docling or the source PDF at runtime at all.
COPY data/sample_chunks.json data/sample_chunks.json

# Redirect the HuggingFace/model cache into the image itself, at a fixed
# path, set *before* pre-warming below. This matters specifically for
# Lambda: its filesystem is read-only everywhere except /tmp, so caching
# to the default ~/.cache would bake models in at build time but then be
# unwritable at runtime — anything that tries to touch that directory (a
# lock file, an update check) would fail or hang against the read-only mount.
ENV HF_HOME=/app/.cache/huggingface
ENV XDG_CACHE_HOME=/app/.cache

# Pre-warm Docling's model cache at build time (it otherwise downloads
# layout/OCR model weights on the *first* real request in production,
# which would make someone's first query time out — doubly important on
# Lambda, where a cold start already has to load everything from scratch).
# Failure here is tolerated with `|| true`: if the sample PDF is ever
# removed, the build still succeeds and Docling just downloads on first
# real use instead.
COPY data/relevant_section_identification-sample.pdf /tmp/warm.pdf
RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter().convert('/tmp/warm.pdf')" || true
RUN rm -f /tmp/warm.pdf

# Force fully offline model loading — the models are already baked into
# the image above, so no network access is needed at runtime, and this
# guarantees Lambda cold starts can't stall on a slow/blocked "check for
# updates" call to huggingface.co.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false

# AWS Lambda Web Adapter: lets this same image run on Lambda (behind a
# Function URL) with zero changes to the app itself — the adapter runs
# alongside uvicorn as a Lambda extension and translates Lambda invoke
# events into real HTTP requests against localhost:8000. Inert outside
# Lambda, so this image still runs identically via plain `docker run`.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.1 /lambda-adapter /opt/extensions/lambda-adapter
ENV PORT=8000
ENV AWS_LWA_READINESS_CHECK_PATH=/health

EXPOSE 8000
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
