# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

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
