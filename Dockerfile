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
# which would make someone's first query time out). Failure here is
# tolerated with `|| true`: if the sample PDF is ever removed, the build
# still succeeds and Docling just downloads on first real use instead.
COPY data/relevant_section_identification-sample.pdf /tmp/warm.pdf
RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter().convert('/tmp/warm.pdf')" || true
RUN rm -f /tmp/warm.pdf

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
